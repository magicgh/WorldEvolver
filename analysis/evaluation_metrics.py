#!/usr/bin/env python3
"""Offline metrics for Stage 2 prediction and Stage 3 planning logs."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        for idx in order[start:end]:
            ranks[idx] = rank
        start = end
    return ranks


def auroc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    positives = sum(int(v) for v in labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ranks = _ranks(scores)
    positive_rank_sum = sum(r for r, y in zip(ranks, labels) if y)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def adaptive_ece(
    labels: Sequence[int],
    confidences: Sequence[float],
    n_bins: int = 10,
) -> Optional[float]:
    if not labels:
        return None
    order = sorted(range(len(labels)), key=confidences.__getitem__)
    total = len(order)
    ece = 0.0
    for bin_idx in range(min(n_bins, total)):
        start = bin_idx * total // min(n_bins, total)
        end = (bin_idx + 1) * total // min(n_bins, total)
        indices = order[start:end]
        accuracy = _mean([float(labels[i]) for i in indices]) or 0.0
        confidence = _mean([float(confidences[i]) for i in indices]) or 0.0
        ece += len(indices) / total * abs(accuracy - confidence)
    return ece


def fixed_bin_ece(
    targets: Sequence[float],
    confidences: Sequence[float],
    n_bins: int = 10,
) -> Optional[float]:
    """Sample-weighted calibration error for a continuous quality target."""
    if len(targets) != len(confidences):
        raise ValueError("targets and confidences must have the same length")
    if not targets:
        return None
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    bins: list[list[int]] = [[] for _ in range(n_bins)]
    for index, confidence in enumerate(confidences):
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidences must lie in [0, 1]")
        bin_index = min(int(confidence * n_bins), n_bins - 1)
        bins[bin_index].append(index)

    total = len(targets)
    error = 0.0
    for indices in bins:
        if not indices:
            continue
        mean_target = sum(float(targets[i]) for i in indices) / len(indices)
        mean_confidence = sum(float(confidences[i]) for i in indices) / len(indices)
        error += len(indices) / total * abs(mean_confidence - mean_target)
    return error


def concordance_index(
    targets: Sequence[float],
    scores: Sequence[float],
) -> Optional[float]:
    """Rank continuous targets in O(n log n), ignoring tied target pairs."""
    if len(targets) != len(scores):
        raise ValueError("targets and scores must have the same length")
    if len(targets) < 2:
        return None

    score_values = sorted(set(float(score) for score in scores))
    score_ranks = {score: rank + 1 for rank, score in enumerate(score_values)}
    tree = [0] * (len(score_values) + 1)

    def add(index: int) -> None:
        while index < len(tree):
            tree[index] += 1
            index += index & -index

    def prefix_sum(index: int) -> int:
        total = 0
        while index > 0:
            total += tree[index]
            index -= index & -index
        return total

    ordered = sorted(
        zip((float(target) for target in targets), (float(score) for score in scores))
    )
    comparable = 0
    concordant = 0.0
    prior = 0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1

        for _, score in ordered[start:end]:
            rank = score_ranks[score]
            lower = prefix_sum(rank - 1)
            equal = prefix_sum(rank) - lower
            comparable += prior
            concordant += lower + 0.5 * equal
        for _, score in ordered[start:end]:
            add(score_ranks[score])
            prior += 1
        start = end

    return concordant / comparable if comparable else None


def spearman(
    values_a: Sequence[float],
    values_b: Sequence[float],
) -> Optional[float]:
    if len(values_a) < 2 or len(values_a) != len(values_b):
        return None
    a, b = _ranks(values_a), _ranks(values_b)
    mean_a, mean_b = _mean(a), _mean(b)
    assert mean_a is not None and mean_b is not None
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    denom_a = math.sqrt(sum((x - mean_a) ** 2 for x in a))
    denom_b = math.sqrt(sum((y - mean_b) ** 2 for y in b))
    return numerator / (denom_a * denom_b) if denom_a and denom_b else None


def bootstrap_mean_ci(
    values: Sequence[float],
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(n_bootstrap)
    )
    return _quantile(means, 0.025), _quantile(means, 0.975)


def _stage3_outcomes(payload: dict[str, Any]) -> dict[str, tuple[float, float]]:
    rows = payload.get("srs_per_trial") or []
    task_ids = (payload.get("evaluation_order") or {}).get("task_ids")
    if task_ids is None:
        raise ValueError("paired Stage 3 analysis requires evaluation_order.task_ids")
    task_ids = [str(task_id) for task_id in task_ids]
    if len(task_ids) != len(rows) or len(set(task_ids)) != len(task_ids):
        raise ValueError("Stage 3 task ids must be unique and match srs_per_trial")
    return {
        task_id: (
            float(bool(row and row[0])),
            float(any(row[:5])),
        )
        for task_id, row in zip(task_ids, rows)
    }


def paired_stage3_comparison(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare Stage 3 runs task by task, independent of evaluation order."""
    baseline_rows = _stage3_outcomes(baseline)
    candidate_rows = _stage3_outcomes(candidate)
    if set(baseline_rows) != set(candidate_rows):
        raise ValueError("paired Stage 3 runs must contain the same task ids")

    def compare(index: int) -> dict[str, Any]:
        before = [baseline_rows[task_id][index] for task_id in baseline_rows]
        after = [candidate_rows[task_id][index] for task_id in baseline_rows]
        deltas = [new - old for old, new in zip(before, after)]
        return {
            "baseline": _mean(before),
            "candidate": _mean(after),
            "delta": _mean(deltas),
            "delta_ci95": bootstrap_mean_ci(deltas, n_bootstrap, seed),
            "wins": sum(delta > 0 for delta in deltas),
            "losses": sum(delta < 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
        }

    return {
        "n_tasks": len(baseline_rows),
        "success_at_1": compare(0),
        "best_at_5": compare(1),
    }


def aggregate_stage3_orders(payloads: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Stage 3 metrics across randomized evaluation orders."""
    if not payloads:
        raise ValueError("order aggregation requires at least one Stage 3 result")
    task_sets = [set(_stage3_outcomes(payload)) for payload in payloads]
    if any(task_ids != task_sets[0] for task_ids in task_sets[1:]):
        raise ValueError("random-order runs must contain the same task ids")
    summaries = [stage3_summary(payload) for payload in payloads]

    def aggregate(key: str) -> dict[str, float]:
        values = [float(summary[key]) for summary in summaries]
        mean = sum(values) / len(values)
        denominator = len(values) - 1 if len(values) > 1 else 1
        std = math.sqrt(sum((value - mean) ** 2 for value in values) / denominator)
        return {"mean": mean, "std": std}

    return {
        "n_orders": len(payloads),
        "n_tasks": len(task_sets[0]),
        "success_at_1": aggregate("success_at_1"),
        "best_at_5": aggregate("best_at_5"),
        "seeds": [
            (payload.get("evaluation_order") or {}).get("seed")
            for payload in payloads
        ],
    }


def _numeric(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    out = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out.append(float(value))
    return out


def stage2_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    scored = [
        row
        for row in rows
        if row.get("mean_logprob") is not None and row.get("prediction") is not None
    ]
    labels = [int(float(row.get("em", 0.0)) == 1.0) for row in scored]
    token_f1 = [float(row.get("token_f1", 0.0)) for row in scored]
    logprobs = [float(row["mean_logprob"]) for row in scored]
    confidences = []
    for row in scored:
        confidence = row.get("confidence_q")
        if confidence is None:
            confidence = math.exp(float(row["mean_logprob"]))
        confidences.append(float(confidence))
    lengths = [
        float(row.get("prediction_token_length") or len(str(row["prediction"]).split()))
        for row in scored
    ]

    by_traj: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_traj[str(row.get("traj_id", ""))] = row
    final_sizes = _numeric(by_traj.values(), "episodic_store_size_after")

    def footprint(key: str, *, median: bool = False, p95: bool = False):
        values = _numeric(rows, key)
        out: dict[str, Any] = {"mean": _mean(values)}
        if median:
            out["median"] = _quantile(values, 0.5)
        if p95:
            out["p95"] = _quantile(values, 0.95)
        return out

    retrieved = _numeric(rows, "n_retrieved")
    active_rules = _numeric(rows, "active_rules")
    rendered_rules = _numeric(rows, "rendered_rules")
    retrieval_latency = _numeric(rows, "retrieval_latency_ms")
    return {
        "n_predictions": len(rows),
        "n_confidence_scored": len(scored),
        "auroc_em": auroc(labels, logprobs),
        "adaptive_ece_10": adaptive_ece(labels, confidences, 10),
        "ece_f1_10": fixed_bin_ece(token_f1, confidences, 10),
        "f1_c_index": concordance_index(token_f1, confidences),
        "mean_token_f1": _mean(token_f1),
        "mean_confidence": _mean(confidences),
        "length_confidence_spearman": spearman(lengths, confidences),
        "prompt_footprint": {
            "episodic_retrieved": {
                "mean": _mean(retrieved),
                "max": max(retrieved) if retrieved else None,
            },
            "episodic_block_tokens": footprint(
                "episodic_block_tokens",
                median=True,
                p95=True,
            ),
            "semantic_block_tokens": footprint(
                "semantic_block_tokens",
                median=True,
                p95=True,
            ),
            "total_prompt_tokens": footprint("total_prompt_tokens", p95=True),
            "active_rules": {
                "mean": _mean(active_rules),
                "max": max(active_rules) if active_rules else None,
            },
            "rendered_rules": {
                "mean": _mean(rendered_rules),
                "max": max(rendered_rules) if rendered_rules else None,
            },
        },
        "memory_growth": {
            "final_episodic_size_mean": _mean(final_sizes),
            "final_episodic_size_max": max(final_sizes) if final_sizes else None,
        },
        "retrieval_latency_ms": {
            "mean": _mean(retrieval_latency),
            "p95": _quantile(retrieval_latency, 0.95),
        },
    }


def aggregate_stage2_orders(
    row_sets: Sequence[Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    """Pool Stage 2 traces across seeded evaluation orders."""
    if not row_sets:
        raise ValueError("order aggregation requires at least one Stage 2 trace")
    rows = [row for row_set in row_sets for row in row_set]
    result = stage2_summary(rows)
    result["n_orders"] = len(row_sets)
    result["predictions_per_order"] = [len(row_set) for row_set in row_sets]
    return result


def stage3_summary(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("srs_per_trial") or []
    first = [float(bool(row and row[0])) for row in rows]
    best = [float(any(row[:5])) for row in rows]
    result = {
        "n_tasks": len(rows),
        "success_at_1": _mean(first),
        "success_at_1_ci95": bootstrap_mean_ci(first),
        "best_at_5": _mean(best),
        "best_at_5_ci95": bootstrap_mean_ci(best),
    }
    planning = payload.get("wm_planning_metrics") or {}
    calls = planning.get("wm_calls") or []
    growth = planning.get("memory_growth") or []
    if planning:
        result["planning_overhead"] = {
            "wm_invocations_per_agent_step": planning.get(
                "wm_invocations_per_agent_step"
            ),
            "wm_llm_calls": planning.get("wm_llm_calls"),
            "retrieval_latency_ms": planning.get("retrieval_latency_ms"),
            "sf_abstentions": sum(row.get("sf_gate") == "abstain" for row in calls),
            "prompt_tokens_mean": _mean(_numeric(calls, "total_prompt_tokens")),
            "episodic_store_size_max": max(_numeric(growth, "n_records"), default=None),
            "semantic_rule_count_max": max(_numeric(growth, "n_rules"), default=None),
        }
    return result


def action_change_summary(payloads: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Measure how often visible foresight changes the logged draft action."""
    exposed = 0
    changed = 0
    unchanged = 0
    failed_reruns = 0
    for payload in payloads:
        trajectory = payload.get("trajectory") or []
        for index, entry in enumerate(trajectory):
            if "Draft Action" not in entry:
                continue
            exposed += 1
            step_id = entry.get("id")
            final_action = next(
                (
                    later["Action"]
                    for later in trajectory[index + 1 :]
                    if later.get("id") == step_id and "Action" in later
                ),
                None,
            )
            if final_action is None:
                failed_reruns += 1
            elif str(final_action) == str(entry["Draft Action"]):
                unchanged += 1
            else:
                changed += 1
    completed = changed + unchanged
    return {
        "n_traces": len(payloads),
        "foresight_exposed_steps": exposed,
        "completed_decisions": completed,
        "action_changed": changed,
        "action_unchanged": unchanged,
        "failed_reruns": failed_reruns,
        "change_rate_exposed": changed / exposed if exposed else None,
        "change_rate_completed": changed / completed if completed else None,
    }


def select_k(rows: Sequence[dict[str, Any]], k_field: str) -> dict[str, Any]:
    if not rows:
        raise ValueError("validation selection requires at least one row")
    selected = min(
        rows,
        key=lambda row: (
            -float(row["macro_val_em"]),
            -float(row["macro_val_f1"]),
            float(row.get("mean_prompt_tokens", math.inf)),
            int(row[k_field]),
        ),
    )
    return {"k_field": k_field, "selected": selected, "candidates": list(rows)}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in (
        "stage2",
        "aggregate-stage2",
        "stage3",
        "aggregate-stage3",
        "action-changes",
    ):
        command = subparsers.add_parser(name)
        command.add_argument("paths", nargs="+", type=Path)
        command.add_argument("--output", type=Path)
    comparison = subparsers.add_parser("compare-stage3")
    comparison.add_argument("baseline", type=Path)
    comparison.add_argument("candidate", type=Path)
    comparison.add_argument("--output", type=Path)
    selection = subparsers.add_parser("select-k")
    selection.add_argument("path", type=Path)
    selection.add_argument("--k-field", required=True)
    selection.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.command == "stage2":
        result = {str(path): stage2_summary(_load_jsonl(path)) for path in args.paths}
    elif args.command == "aggregate-stage2":
        result = aggregate_stage2_orders([_load_jsonl(path) for path in args.paths])
    elif args.command == "stage3":
        result = {
            str(path): stage3_summary(json.loads(path.read_text()))
            for path in args.paths
        }
    elif args.command == "aggregate-stage3":
        result = aggregate_stage3_orders(
            [json.loads(path.read_text()) for path in args.paths]
        )
    elif args.command == "action-changes":
        result = action_change_summary(
            [json.loads(path.read_text()) for path in args.paths]
        )
    elif args.command == "compare-stage3":
        result = paired_stage3_comparison(
            json.loads(args.baseline.read_text()),
            json.loads(args.candidate.read_text()),
        )
    else:
        result = select_k(json.loads(args.path.read_text()), args.k_field)

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
