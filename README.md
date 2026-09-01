# WorldEvolver

Official implementation of the Findings of EMNLP 2026 paper
**[Self-Evolving World Models for LLM Agent Planning](https://arxiv.org/abs/2606.30639)**.

WorldEvolver is a training-free world model that evolves at test time while
keeping both the world model and downstream agent parameters frozen. It uses:

- **Episodic Memory** to retrieve realized environment transitions.
- **Semantic Memory** to turn prediction-observation mismatches into reusable
  rules.
- **Selective Foresight** to expose predictions to the agent only when their
  confidence passes a configurable threshold.

The implementation supports World Model Prediction on Word2World and Agent
Planning on ALFWorld and ScienceWorld.

## Repository Layout

```text
src/       WorldEvolver, agents, environments, prompts, and evaluation tasks
analysis/  Offline evaluation and retrieval-analysis utilities
```

Generated results, benchmark assets, deployment-specific configuration files,
and model-server launch scripts are intentionally not included.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

ALFWorld and ScienceWorld require their standard benchmark assets. Agent
Planning additionally expects the AgentBoard task labels referenced by the
runtime configuration.

## Model Endpoints

The implementation uses OpenAI-compatible chat-completions and embeddings
endpoints. Supply endpoint URLs, model names, and credentials through the
external runtime configuration. No credentials or deployment-specific values
are stored in this repository.

Set `base_url` and `api_key` in the selected `llm` entry. Embedding-backed
metrics and RAWM-$\phi$ accept `embed_base_url`, `embed_model`,
`embed_api_key`, and `embed_timeout` through `wm_config`.

## Running Evaluations

Evaluation configuration is supplied at runtime rather than committed to this
repository. A YAML file must contain the top-level sections `llm`, `agent`,
`env`, and `run`. The `llm` section maps the name passed to `--model` to an
OpenAI-compatible model configuration. Run the evaluator from `src/`
so prompt paths in the runtime configuration resolve consistently:

```bash
cd src
python eval_main.py \
  --cfg-path /absolute/path/to/evaluation.yaml \
  --tasks stage3_alfworld \
  --model local \
  --log_path ../results/stage3
```

Implemented evaluation tasks include:

| Stage | Task names |
|---|---|
| World Model Prediction | `stage2_word2world_alfworld`, `stage2_word2world_scienceworld` |
| Agent Planning | `stage3_alfworld`, `stage3_scienceworld` |

Implemented world-model settings are `wm-base`, `wm-episodic`, `wm-semantic`,
`wm-episodic-semantic`, `wm-rawm-phi`, and `wm-itp-i`. The paper setting uses
`wm-episodic-semantic` with `top_k: 5` and `batch_k: 1`. Set
`sf_confidence_pct` in `wm_config` to enable Selective Foresight.

### Evaluation Controls

The canonical protocol remains the default. The following optional settings
support the control analyses reported with the paper:

- `wm_config.reset_scope`: `none` (default), `trajectory`, or `task`.
- `wm_config.episodic_retriever`: `jaccard_topk` (default) or
  `uniform_random`.
- `run.shuffle_evaluation_order`: shuffle trajectories or tasks while
  preserving step order within a trajectory.
- `run.seed`: set the evaluation-order seed.
- `run.resume` and `run.checkpoint_path`: resume at a trajectory or task
  boundary.
- `run.trace_dir`: record per-step traces, memory resets, prompt footprint,
  retrieval latency, memory growth, world-model calls, and draft actions.

### Analysis Utilities

Run these commands from the repository root.

The metrics CLI summarizes prediction and planning traces, aggregates seeded
evaluation orders, compares paired planning runs, and measures action changes:

```bash
python analysis/evaluation_metrics.py stage2 TRACE.jsonl
python analysis/evaluation_metrics.py aggregate-stage2 ORDER_*.jsonl
python analysis/evaluation_metrics.py stage3 RESULT.json
python analysis/evaluation_metrics.py aggregate-stage3 ORDER_*.json
python analysis/evaluation_metrics.py action-changes TRACE_*.json
python analysis/evaluation_metrics.py compare-stage3 BASELINE.json CANDIDATE.json
```

The retrieval utility compares action- and state-action-centered keys over the
same transition store and extracts representative failure cases:

```bash
python analysis/retrieval_key_analysis.py \
  --env alfworld \
  --embed-base-url EMBEDDINGS_ENDPOINT \
  --out-dir results/retrieval_key_alfworld
```

## Citation

```bibtex
@article{zhang2026selfevolving,
  title   = {Self-Evolving World Models for LLM Agent Planning},
  author  = {Zhang, Xuan and Zhang, Wenxuan and Ng, See-Kiong and Deng, Yang},
  journal = {arXiv preprint arXiv:2606.30639},
  year    = {2026}
}
```
