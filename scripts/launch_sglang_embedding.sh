#!/usr/bin/env bash
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen3-Embedding-8B}"
PORT="${PORT:-30001}"
GPU="${GPU:-0}"
DTYPE="${DTYPE:-auto}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
LOG_FILE="${LOG_FILE:-/tmp/sglang_embedding_${PORT}.log}"
WORLDEVOLVER_PY="${WORLDEVOLVER_PY:-${PYTHON:-python}}"
if [[ "${WORLDEVOLVER_PY}" == */* && ! -x "${WORLDEVOLVER_PY}" ]]; then
    echo "[embed] FATAL: python executable not found at ${WORLDEVOLVER_PY}" >&2
    exit 1
fi
if [[ "${WORLDEVOLVER_PY}" != */* ]] && ! command -v "${WORLDEVOLVER_PY}" >/dev/null 2>&1; then
    echo "[embed] FATAL: python executable ${WORLDEVOLVER_PY} not found on PATH" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
ARGS=(
    --model-path "${MODEL}"
    --is-embedding
    --host 0.0.0.0
    --port "${PORT}"
    --dtype "${DTYPE}"
    --mem-fraction-static "${MEM_FRACTION}"
    --trust-remote-code
)
echo "[embed] selected GPU=${GPU}, model=${MODEL}, port=${PORT}"
echo "[embed] command: ${WORLDEVOLVER_PY} -m sglang.launch_server ${ARGS[*]}"
echo "[embed] logs: ${LOG_FILE}"
nohup "${WORLDEVOLVER_PY}" -m sglang.launch_server "${ARGS[@]}" > "${LOG_FILE}" 2>&1 &
PID=$!
echo "${PID}" > "/tmp/sglang_embedding_${PORT}.pid"
echo "[embed] waiting for server readiness on port ${PORT} ..."
for i in $(seq 1 240); do
    if curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "[embed] server ready after ${i}s"
        echo "[embed] export WORLDEVOLVER_EMBED_BASE_URL=http://localhost:${PORT}"
        exit 0
    fi
    if ! kill -0 "${PID}" 2>/dev/null; then
        echo "[embed] FATAL: server process died. tail of log:" >&2
        tail -n 60 "${LOG_FILE}" >&2 || true
        exit 1
    fi
    sleep 1
done
echo "[embed] server did not become ready within 240s; tail of log:" >&2
tail -n 60 "${LOG_FILE}" >&2 || true
exit 1
