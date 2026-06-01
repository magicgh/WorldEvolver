#!/usr/bin/env bash
set -euo pipefail
TASK="${TASK:-generate}"
if [[ "${TASK}" != "generate" && "${TASK}" != "embed" ]]; then
    echo "[vllm] FATAL: TASK must be 'generate' or 'embed'; got ${TASK}" >&2
    exit 1
fi
if [[ "${TASK}" == "embed" ]]; then
    MODEL="${MODEL:-Qwen/Qwen3-Embedding-8B}"
    PORT="${PORT:-30001}"
else
    MODEL="${MODEL:-google/gemma-4-26b-a4b-it}"
    PORT="${PORT:-30000}"
fi
GPU="${GPU:-0}"
HOST="${HOST:-0.0.0.0}"
DTYPE="${DTYPE:-float16}"
CTX_LEN="${CTX_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
API_KEY="${API_KEY:-EMPTY}"
LOG_FILE="${LOG_FILE:-/tmp/vllm_${TASK}_${PORT}.log}"
VLLM_BIN="${VLLM_BIN:-vllm}"
if ! command -v "${VLLM_BIN}" >/dev/null 2>&1; then
    echo "[vllm] FATAL: ${VLLM_BIN} not found on PATH. Install vLLM in an isolated serving env." >&2
    exit 127
fi
export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
ARGS=(
    serve "${MODEL}"
    --host "${HOST}"
    --port "${PORT}"
    --dtype "${DTYPE}"
    --max-model-len "${CTX_LEN}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --api-key "${API_KEY}"
    --trust-remote-code
)
if [[ "${TASK}" == "generate" ]]; then
    ARGS+=(--generation-config vllm)
else
    ARGS+=(--convert embed)
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    USER_ARGS=(${EXTRA_ARGS})
    ARGS+=("${USER_ARGS[@]}")
fi
echo "[vllm] selected GPU=${GPU}, task=${TASK}, model=${MODEL}, port=${PORT}"
echo "[vllm] command: ${VLLM_BIN} ${ARGS[*]}"
echo "[vllm] logs: ${LOG_FILE}"
nohup "${VLLM_BIN}" "${ARGS[@]}" > "${LOG_FILE}" 2>&1 &
PID=$!
echo "${PID}" > "/tmp/vllm_${TASK}_${PORT}.pid"
echo "[vllm] waiting for server readiness on port ${PORT} ..."
for i in $(seq 1 240); do
    if curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "[vllm] server ready after ${i}s"
        if [[ "${TASK}" == "embed" ]]; then
            echo "[vllm] export WORLDEVOLVER_EMBED_BASE_URL=http://localhost:${PORT}"
        else
            echo "[vllm] export VLLM_BASE_URL=http://localhost:${PORT}/v1"
        fi
        exit 0
    fi
    if ! kill -0 "${PID}" 2>/dev/null; then
        echo "[vllm] FATAL: server process died. tail of log:" >&2
        tail -n 60 "${LOG_FILE}" >&2 || true
        exit 1
    fi
    sleep 1
done
echo "[vllm] server did not become ready within 240s; tail of log:" >&2
tail -n 60 "${LOG_FILE}" >&2 || true
exit 1
