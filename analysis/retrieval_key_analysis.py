"""Offline retrieval-key analysis for episodic transition memory.

Question under test: does keying Episodic Memory retrieval on the action
only (paper design) fetch better or worse evidence than keying on
state+action, and does the similarity function (token Jaccard vs
Qwen3-Embedding cosine) matter?

Design — same-store, four retrieval arms, no world-model calls:
  * Replay the Word2World Stage-2 stream in evaluation order (WE-Full
    protocol: one prequential store persisting across trajectories).
  * Rank the *identical* record sequence with four keys:
      - action_jaccard       action-only token Jaccard
                             (EpisodicLibrary._rank_by_action, paper design)
      - action_cosine        action-only embedding cosine (in-script index).
                             Key text is "action: {a}" — the SAME action
                             field scaffold as the state+action arm, so the
                             two cosine arms differ only in the presence of
                             the state field.
      - state_action_cosine  symmetric state+action embedding cosine:
                             memory and query both embed
                             "current state: {s}\\naction: {a}" — the
                             controlled key-content comparison (no
                             next-state in the memory text).
      - rawm_cosine          the RAWM Eq. 1/2 *key* (memory embeds
                             (s, a, s'), query embeds (s, a)) applied to the
                             same prequential store. NOT the full production
                             RAWM protocol: production Stage-2 RAWM uses a
                             static offline pool with current-trajectory
                             exclusion. This arm isolates only the effect of
                             RAWM's next-state-in-memory key relative to
                             state_action_cosine. (In a teacher-forced
                             stream the predecessor's stored s' equals the
                             current s, which inflates its similarity.)
  * At each step, rank with all arms BEFORE appending the step's transition,
    then score the top-1 retrieved transition's stored next-state against
    the current gold next observation (EM / token-F1).
  * The only model involved is the explicitly configured embedding endpoint.
    Every vector passes response validation (shape, row count, finiteness,
    and L2 norms), and a startup canary rejects grossly permuted or
    duplicated batch responses.
  * The RAWM arm uses exact dense-cosine ranking in this script, making the
    comparison independent of the installed approximate-vector backend.

Tie policy: the in-script cosine arms break similarity ties by earliest
record index (stable sort), matching EpisodicLibrary's stable descending
sort; top-1 tie frequency is reported per in-script arm. The RAWM exact
path uses NumPy argsort order for float ties (production behavior).

Retrieval-quality proxy: token-F1(retrieved next_state, gold next_state) —
how predictive the fetched evidence is of what actually happens next.

Statistics:
  * Per-arm means are reported conditional on a hit (`_given_hit`) and over
    all eligible steps with a predefined miss=0.0 policy
    (`next_f1_all_eligible`).
  * Every headline comparison is computed on its own pair-specific
    population: (i) steps where BOTH arms of that pair retrieved, and
    (ii) all eligible steps under the miss=0.0 policy — so no comparison
    is filtered by an unrelated arm's abstentions.
  * CIs are trajectory-resampled bootstrap intervals (10k, seed 42) over
    per-step deltas computed on the FIXED prequential replay. Because the
    shared store couples trajectories, these are fixed-stream descriptive
    intervals (step-sampling variability given this replay), not inference
    over alternative evaluation streams.
  * Equivalence is only claimed when a delta CI lies entirely inside the
    predeclared margin of ±0.05 F1 (`within_equivalence_margin`).
  * Raw scores are kept unrounded internally; rounding happens only at
    serialization.

Outputs (under --out-dir; stale outputs are removed at start, and
manifest.json with completed=true is written last):
  steps.jsonl        one row per step: per-arm retrieval indices + scores
  summary.json       per-arm + pair-specific stats with CIs
  failure_cases.json sign-filtered misleading-retrieval examples
  manifest.json      completion marker (absent/false => do not trust others)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from data.word2world import Word2WorldDataset  # noqa: E402
from world_model.modules.metrics import (  # noqa: E402
    configure_embedding_client,
    em_at_1,
    normalize,
    token_f1,
)
from world_model.modules.state_library import (  # noqa: E402
    EpisodicLibrary,
    RawmPhiLibrary,
    _action_jaccard,
    _action_tokens,
    _format_query_text,
)

ARMS = ("action_jaccard", "action_cosine", "state_action_cosine", "rawm_cosine")
HEADLINE_PAIRS = {
    "similarity_function__action_jaccard_vs_action_cosine": (
        "action_jaccard", "action_cosine",
    ),
    "key_content__action_cosine_vs_state_action_cosine": (
        "action_cosine", "state_action_cosine",
    ),
    "rawm_key__state_action_cosine_vs_rawm_cosine": (
        "state_action_cosine", "rawm_cosine",
    ),
    "paper_arm__action_jaccard_vs_state_action_cosine": (
        "action_jaccard", "state_action_cosine",
    ),
}
_TRUNC = 600
_MISLED_WINNER_MIN_F1 = 0.5  # "arm B misled" requires arm A's evidence be useful
_EQUIVALENCE_MARGIN_F1 = 0.05
_BOOT_N = 10_000
_BOOT_SEED = 42
_MIN_CLUSTERS = 5  # below this, trajectory resampling is meaningless
_OUTPUT_FILES = ("steps.jsonl", "summary.json", "failure_cases.json", "manifest.json")


def _validated_embeddings(mat: np.ndarray, expected_rows: int, context: str) -> np.ndarray:
    """Reject silently-corrupt embedder responses (audit finding: alignment)."""
    mat = np.asarray(mat, dtype="float32")
    if mat.ndim != 2 or mat.shape[0] != expected_rows:
        raise RuntimeError(
            f"{context}: expected {expected_rows} embedding rows, got shape {mat.shape}"
        )
    if not np.isfinite(mat).all():
        raise RuntimeError(f"{context}: non-finite values in embedding response")
    norms = np.linalg.norm(mat, axis=1)
    if np.any(norms < 0.99) or np.any(norms > 1.01):
        raise RuntimeError(
            f"{context}: embeddings not L2-normalized (norms in "
            f"[{norms.min():.4f}, {norms.max():.4f}])"
        )
    return mat


def _embedder_canary(encode: Callable[[List[str]], np.ndarray]) -> None:
    """Reject endpoints whose batch responses are permuted into duplicates.

    Three distinct words must embed to three distinct unit rows, and the
    batched response must match singleton encodings row-by-row (catching
    index permutations that a permutation-invariant check would miss). This
    cannot prove index integrity in general (see docstring), but catches
    gross failure modes before any result is computed.
    """
    texts = ["alpha", "bravo", "charlie"]
    mat = encode(texts)
    singles = np.vstack([encode([t]) for t in texts])
    if not np.allclose(mat, singles, atol=1e-3):
        raise RuntimeError(
            "embedder canary: batched rows differ from singleton encodings — "
            "batch response likely permuted"
        )
    sims = mat @ mat.T
    if not np.allclose(np.diag(sims), 1.0, atol=1e-3):
        raise RuntimeError("embedder canary: self-similarity != 1")
    off_diag = sims - np.diag(np.diag(sims))
    if float(off_diag.max()) > 0.999:
        raise RuntimeError(
            "embedder canary: distinct texts embedded (near-)identically — "
            "batch response likely duplicated/permuted"
        )


class _CheckedRawmLibrary(RawmPhiLibrary):
    """RawmPhiLibrary with validated embeddings and exact (engine-free) ranking."""

    def _encode_texts(self, texts):
        return _validated_embeddings(
            super()._encode_texts(texts), len(texts), "rawm_cosine encode"
        )

    def _rank_by_state_action(self, state, action, k):
        if k <= 0 or not self._ensure_transition_embeddings():
            return []
        query = self._encode_texts([_format_query_text(state, action)])[0]
        assert self._transition_mat is not None
        similarities = self._transition_mat @ query
        order = np.argsort(-similarities, kind="stable")[: min(k, len(self.records))]
        return [(int(index), float(similarities[index])) for index in order]


class _CosineIndex:
    """In-script embedding-cosine arm over a caller-defined key text.

    Memory and query use the SAME text template (symmetric key). Unique key
    texts are embedded once (identical text embeds identically, so the cache
    is numerically transparent). Similarity ties at top-1 break by earliest
    record index (stable sort), matching EpisodicLibrary's tie policy.
    """

    def __init__(
        self,
        encode_fn: Callable[[List[str]], np.ndarray],
        text_fn: Callable[[str, str], str],
    ) -> None:
        self._encode = encode_fn  # L2-normalizing + validated
        self._text_fn = text_fn
        self._cache: Dict[str, np.ndarray] = {}
        self._mat: Optional[np.ndarray] = None
        self._n = 0
        self.last_top_ties = 0

    @property
    def size(self) -> int:
        return self._n

    def _vec(self, text: str) -> np.ndarray:
        if text not in self._cache:
            self._cache[text] = self._encode([text])[0]
        return self._cache[text]

    def append(self, state: str, action: str) -> None:
        row = self._vec(self._text_fn(state, action))
        if self._mat is None:
            self._mat = np.zeros((1024, row.shape[0]), dtype=row.dtype)
        elif self._n == self._mat.shape[0]:
            grown = np.zeros(
                (self._mat.shape[0] * 2, self._mat.shape[1]), dtype=self._mat.dtype
            )
            grown[: self._n] = self._mat[: self._n]
            self._mat = grown
        self._mat[self._n] = row
        self._n += 1

    def rank(self, state: str, action: str, k: int) -> List[Tuple[int, float]]:
        if k <= 0 or self._n == 0 or self._mat is None:
            self.last_top_ties = 0
            return []
        sims = self._mat[: self._n] @ self._vec(self._text_fn(state, action))
        order = np.argsort(-sims, kind="stable")[: min(k, self._n)]
        self.last_top_ties = int((sims == sims[order[0]]).sum()) - 1
        return [(int(i), float(sims[i])) for i in order]


def _action_key_text(state: str, action: str) -> str:  # noqa: ARG001 — signature parity
    """Action-only key with the SAME field scaffold as the state+action key."""
    return f"action: {(action or '').strip()}"


def _clip(text: str) -> str:
    text = (text or "").strip()
    return text if len(text) <= _TRUNC else text[:_TRUNC] + " …[truncated]"


def _score_hit(
    record: Dict[str, Any],
    meta: Dict[str, Any],
    *,
    sim: float,
    idx: int,
    state: str,
    action: str,
    gold_next: str,
    traj_id: str,
    step_idx: int,
) -> Dict[str, Any]:
    r_action = str(record.get("action", ""))
    r_state = str(record.get("state_obs_raw", ""))
    r_next = str(record.get("next_observation_raw", ""))
    same_traj = meta["traj_id"] == traj_id
    return {
        "idx": idx,
        "sim": float(sim),
        "source_traj": meta["traj_id"],
        "source_step": meta["step_idx"],
        "same_traj": same_traj,
        "is_prev_step": same_traj and meta["step_idx"] == step_idx - 1,
        "action_match": normalize(r_action) == normalize(action),
        "action_jaccard": _action_jaccard(
            _action_tokens(action), _action_tokens(r_action)
        ),
        "state_f1": token_f1(r_state, state),
        "next_em": em_at_1(r_next, gold_next),
        "next_f1": token_f1(r_next, gold_next),
    }


def _mean(values: List[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def _summarize_arm(
    hits: List[Dict[str, Any]],
    *,
    n_eligible: int,
    top_ties: Optional[List[int]] = None,
) -> Dict[str, Any]:
    n_hits = len(hits)
    out: Dict[str, Any] = {
        "n_hits": n_hits,
        "n_missing": n_eligible - n_hits,
        "coverage": n_hits / n_eligible if n_eligible else None,
        # all-eligible estimand: a miss fetches no evidence and scores 0.0
        "next_f1_all_eligible": (
            float(sum(h["next_f1"] for h in hits)) / n_eligible if n_eligible else None
        ),
        "next_em_given_hit": _mean([h["next_em"] for h in hits]),
        "next_f1_given_hit": _mean([h["next_f1"] for h in hits]),
        "action_match_rate": _mean([1.0 if h["action_match"] else 0.0 for h in hits]),
        "state_f1": _mean([h["state_f1"] for h in hits]),
        "same_traj_rate": _mean([1.0 if h["same_traj"] else 0.0 for h in hits]),
        "prev_step_rate": _mean([1.0 if h["is_prev_step"] else 0.0 for h in hits]),
        # sensitivity: quality restricted to cross-trajectory retrievals
        "next_f1_given_other_traj": _mean(
            [h["next_f1"] for h in hits if not h["same_traj"]]
        ),
        # Same-action retrieval can still return the wrong outcome.
        "same_action_wrong_outcome_rate": _mean(
            [1.0 if h["action_match"] and h["next_em"] == 0.0 else 0.0 for h in hits]
        ),
        "next_f1_given_action_match": _mean(
            [h["next_f1"] for h in hits if h["action_match"]]
        ),
        "next_f1_given_action_mismatch": _mean(
            [h["next_f1"] for h in hits if not h["action_match"]]
        ),
    }
    if top_ties is not None:
        out["top1_tie_rate"] = _mean([1.0 if t > 0 else 0.0 for t in top_ties])
    return out


def _cluster_bootstrap_ci(
    pairs: List[Tuple[str, float]],
    *,
    n_boot: int = _BOOT_N,
    seed: int = _BOOT_SEED,
) -> Dict[str, Any]:
    """Trajectory-resampled bootstrap interval for the mean per-step delta.

    Fixed-stream descriptive interval: deltas were computed on one fixed
    prequential replay, so this quantifies step-sampling variability given
    that replay — not inference over alternative evaluation streams.
    """
    if not pairs:
        return {
            "mean": None, "ci95_lo": None, "ci95_hi": None,
            "n_steps": 0, "n_trajectories": 0,
        }
    groups: Dict[str, List[float]] = {}
    for traj_id, delta in pairs:
        groups.setdefault(traj_id, []).append(delta)
    sums = np.array([sum(v) for v in groups.values()], dtype="float64")
    counts = np.array([len(v) for v in groups.values()], dtype="float64")
    if len(sums) < _MIN_CLUSTERS:
        return {
            "mean": float(sums.sum() / counts.sum()),
            "ci95_lo": None, "ci95_hi": None,
            "status": "insufficient_clusters",
            "n_steps": int(counts.sum()),
            "n_trajectories": len(sums),
        }
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    boot_means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return {
        "mean": float(sums.sum() / counts.sum()),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "n_steps": int(counts.sum()),
        "n_trajectories": len(sums),
    }


def _with_equivalence(ci: Dict[str, Any]) -> Dict[str, Any]:
    if ci["ci95_lo"] is None:
        ci["within_equivalence_margin"] = None
    else:
        ci["within_equivalence_margin"] = bool(
            ci["ci95_lo"] > -_EQUIVALENCE_MARGIN_F1
            and ci["ci95_hi"] < _EQUIVALENCE_MARGIN_F1
        )
    return ci


def _rounded(obj: Any, ndigits: int = 4) -> Any:
    """Round floats recursively at serialization time only."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _rounded(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rounded(v, ndigits) for v in obj]
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, choices=["alfworld", "scienceworld"])
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--max-trajs", type=int, default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--embed-base-url", required=True)
    parser.add_argument(
        "--embed-model",
        default="Qwen/Qwen3-Embedding-8B",
    )
    parser.add_argument("--embed-api-key", default=None)
    parser.add_argument("--embed-timeout", type=float, default=120.0)
    args = parser.parse_args()

    configure_embedding_client(
        base_url=args.embed_base_url,
        model=args.embed_model,
        api_key=args.embed_api_key,
        timeout=args.embed_timeout,
    )
    if args.k != 1:
        parser.error("only --k 1 is supported: exactly the top-1 hit is scored")
    if args.max_trajs is not None and args.max_trajs < 1:
        parser.error("--max-trajs must be >= 1 when given")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in _OUTPUT_FILES:  # a crashed rerun must not leave stale outputs
        path = out_dir / name
        if path.exists():
            path.unlink()

    dataset = Word2WorldDataset(args.env, from_hf=False)
    jaccard_lib = EpisodicLibrary(args.env)
    rawm_lib = _CheckedRawmLibrary(args.env)
    encode = rawm_lib._encode_texts  # shared, validated, L2-normalizing
    _embedder_canary(encode)
    act_cos = _CosineIndex(encode, _action_key_text)
    sa_cos = _CosineIndex(encode, _format_query_text)
    store_meta: List[Dict[str, Any]] = []

    rows: List[Dict[str, Any]] = []
    hits_by_arm: Dict[str, List[Dict[str, Any]]] = {arm: [] for arm in ARMS}
    ties_by_arm: Dict[str, List[int]] = {"action_cosine": [], "state_action_cosine": []}
    steps_by_key: Dict[Tuple[str, int], Dict[str, str]] = {}
    n_eligible = 0
    started = time.time()

    for step in dataset.iter_steps(max_trajs=args.max_trajs):
        traj_id = str(step["traj_id"])
        step_idx = int(step["step_idx"])
        state = str(step["state"])
        action = str(step["action"])
        gold_next = str(step["gold_next_state"])

        key = (traj_id, step_idx)
        if key in steps_by_key:
            raise RuntimeError(f"duplicate (traj_id, step_idx) in stream: {key!r}")
        steps_by_key[key] = {"state": state, "action": action, "gold_next": gold_next}

        row: Dict[str, Any] = {
            "traj_id": traj_id,
            "step_idx": step_idx,
            "store_size": len(store_meta),
        }
        if store_meta:
            n_eligible += 1
            ranked_by_arm = {
                "action_jaccard": jaccard_lib._rank_by_action(action, args.k),
                "action_cosine": act_cos.rank(state, action, args.k),
                "state_action_cosine": sa_cos.rank(state, action, args.k),
                "rawm_cosine": rawm_lib._rank_by_state_action(state, action, args.k),
            }
            ties_by_arm["action_cosine"].append(act_cos.last_top_ties)
            ties_by_arm["state_action_cosine"].append(sa_cos.last_top_ties)
            for arm, ranked in ranked_by_arm.items():
                if not ranked:
                    continue
                idx, sim = ranked[0]
                hit = _score_hit(
                    jaccard_lib.records[idx], store_meta[idx], sim=sim, idx=idx,
                    state=state, action=action, gold_next=gold_next,
                    traj_id=traj_id, step_idx=step_idx,
                )
                row[arm] = hit
                hits_by_arm[arm].append(hit)
        rows.append(row)

        for lib in (jaccard_lib, rawm_lib):
            lib.append(
                task=str(step.get("task") or ""),
                state=state,
                action=action,
                gold_next_state=gold_next,
            )
        act_cos.append(state, action)
        sa_cos.append(state, action)
        store_meta.append({"traj_id": traj_id, "step_idx": step_idx})

        n = len(store_meta)
        if not (
            len(jaccard_lib.records) == len(rawm_lib.records)
            == act_cos.size == sa_cos.size == n
        ):
            raise RuntimeError(f"store desync at step {n}")
        mat = rawm_lib._transition_mat
        if mat is not None and mat.shape[0] != n:
            raise RuntimeError(
                f"rawm embedding matrix desync: {mat.shape[0]} rows vs {n} records"
            )

        if args.progress_every and len(rows) % args.progress_every == 0:
            print(
                f"[{args.env}] {len(rows)} steps, store={n}, "
                f"{time.time() - started:.0f}s",
                file=sys.stderr,
                flush=True,
            )

    eligible_rows = [r for r in rows if r["store_size"] > 0]
    all_arm_rows = [r for r in rows if all(arm in r for arm in ARMS)]

    def _f1_or_miss(row: Dict[str, Any], arm: str) -> float:
        return row[arm]["next_f1"] if arm in row else 0.0

    def _pair_stats(a: str, b: str) -> Dict[str, Any]:
        """Pair-specific populations: no filtering by unrelated arms."""
        both = [r for r in eligible_rows if a in r and b in r]
        wins = sum(1 for r in both if r[a]["next_f1"] > r[b]["next_f1"])
        losses = sum(1 for r in both if r[a]["next_f1"] < r[b]["next_f1"])
        return {
            "population_both_hit": {
                "n": len(both),
                "wins_a": wins,
                "wins_b": losses,
                "ties": len(both) - wins - losses,
                "mean_f1_a": _mean([r[a]["next_f1"] for r in both]),
                "mean_f1_b": _mean([r[b]["next_f1"] for r in both]),
                "delta_ci95": _with_equivalence(_cluster_bootstrap_ci(
                    [(r["traj_id"], r[a]["next_f1"] - r[b]["next_f1"]) for r in both]
                )),
            },
            "population_all_eligible_miss0": {
                "n": len(eligible_rows),
                "mean_f1_a": _mean([_f1_or_miss(r, a) for r in eligible_rows]),
                "mean_f1_b": _mean([_f1_or_miss(r, b) for r in eligible_rows]),
                "delta_ci95": _with_equivalence(_cluster_bootstrap_ci(
                    [
                        (r["traj_id"], _f1_or_miss(r, a) - _f1_or_miss(r, b))
                        for r in eligible_rows
                    ]
                )),
            },
        }

    summary = {
        "env": args.env,
        "k": args.k,
        "max_trajs": args.max_trajs,
        "n_steps": len(rows),
        "n_eligible": n_eligible,
        "embed_model": args.embed_model,
        "equivalence_margin_f1": _EQUIVALENCE_MARGIN_F1,
        "ci_interpretation": (
            "trajectory-resampled bootstrap over per-step deltas from ONE fixed "
            "prequential replay; descriptive for this stream, not inference over "
            "alternative evaluation orders (the shared store couples trajectories)"
        ),
        "arms": {
            arm: _summarize_arm(
                hits_by_arm[arm],
                n_eligible=n_eligible,
                top_ties=ties_by_arm.get(arm),
            )
            for arm in ARMS
        },
        "pairs": {
            name: _pair_stats(a, b) for name, (a, b) in HEADLINE_PAIRS.items()
        },
        "descriptive_all_arms_common_support": {
            "note": "steps where all four arms retrieved — cross-arm table only; "
            "headline comparisons use the pair-specific populations above",
            "n": len(all_arm_rows),
            "next_f1": {
                arm: _mean([r[arm]["next_f1"] for r in all_arm_rows]) for arm in ARMS
            },
            "next_em": {
                arm: _mean([r[arm]["next_em"] for r in all_arm_rows]) for arm in ARMS
            },
        },
    }

    def _case(row: Dict[str, Any]) -> Dict[str, Any]:
        step = steps_by_key[(row["traj_id"], row["step_idx"])]
        out = {
            "traj_id": row["traj_id"],
            "step_idx": row["step_idx"],
            "query_action": _clip(step["action"]),
            "query_state": _clip(step["state"]),
            "gold_next_state": _clip(step["gold_next"]),
        }
        for arm in ARMS:
            if arm not in row:
                continue
            hit = row[arm]
            src = jaccard_lib.records[hit["idx"]]
            out[arm] = {
                **{f: hit[f] for f in (
                    "sim", "source_traj", "source_step", "same_traj",
                    "action_match", "state_f1", "next_em", "next_f1",
                )},
                "retrieved_action": _clip(str(src.get("action", ""))),
                "retrieved_state": _clip(str(src.get("state_obs_raw", ""))),
                "retrieved_next_state": _clip(str(src.get("next_observation_raw", ""))),
            }
        return out

    # Key-content case study: the two COSINE arms, so the similarity function
    # is held fixed and only the key content differs. Sign-filtered; the
    # "winning" arm must have fetched objectively useful evidence.
    case_rows = [
        r for r in eligible_rows if "action_cosine" in r and "state_action_cosine" in r
    ]

    def _gap(r: Dict[str, Any]) -> float:
        return r["action_cosine"]["next_f1"] - r["state_action_cosine"]["next_f1"]

    state_key_misled_pop = [
        r for r in case_rows
        if _gap(r) > 0 and r["action_cosine"]["next_f1"] >= _MISLED_WINNER_MIN_F1
    ]
    action_key_misled_pop = [
        r for r in case_rows
        if _gap(r) < 0
        and r["state_action_cosine"]["next_f1"] >= _MISLED_WINNER_MIN_F1
    ]
    same_action_wrong_pop = [
        r for r in eligible_rows
        if "action_jaccard" in r
        and r["action_jaccard"]["action_match"]
        and r["action_jaccard"]["next_em"] == 0.0
    ]
    cases = {
        "criteria": {
            "state_key_misled": "state_action_cosine trails action_cosine "
            f"(same similarity function, gap > 0) and action_cosine next_f1 >= "
            f"{_MISLED_WINNER_MIN_F1}",
            "action_key_misled": "action_cosine trails state_action_cosine "
            f"(same similarity function, gap < 0) and state_action_cosine "
            f"next_f1 >= {_MISLED_WINNER_MIN_F1}",
            "same_action_wrong_outcome": "action_jaccard (the paper's arm) "
            "retrieved an exact action match whose stored outcome differs from "
            "gold (next_em == 0) - the state-dependence scenario",
        },
        "n_matching": {
            "state_key_misled": len(state_key_misled_pop),
            "action_key_misled": len(action_key_misled_pop),
            "same_action_wrong_outcome": len(same_action_wrong_pop),
        },
        "state_key_misled": [
            _case(r)
            for r in sorted(state_key_misled_pop, key=_gap, reverse=True)[:10]
        ],
        "action_key_misled": [
            _case(r) for r in sorted(action_key_misled_pop, key=_gap)[:10]
        ],
        "same_action_wrong_outcome": [_case(r) for r in same_action_wrong_pop[:10]],
    }

    with open(out_dir / "steps.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps(_rounded(row, 6)) + "\n")
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(_rounded(summary), fh, indent=2)
    with open(out_dir / "failure_cases.json", "w") as fh:
        json.dump(_rounded(cases), fh, indent=2)
    with open(out_dir / "manifest.json", "w") as fh:
        json.dump(
            {
                "completed": True,
                "env": args.env,
                "n_steps": len(rows),
                "n_eligible": n_eligible,
                "wall_seconds": round(time.time() - started, 1),
            },
            fh,
            indent=2,
        )

    print(json.dumps(_rounded(summary), indent=2))


if __name__ == "__main__":
    main()
