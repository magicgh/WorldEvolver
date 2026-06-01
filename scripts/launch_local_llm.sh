#!/usr/bin/env bash
set -euo pipefail
MODEL="${MODEL:-google/gemma-4-26b-a4b-it}"
PORT="${PORT:-30000}"
BACKEND="${BACKEND:-auto}"
DTYPE="${DTYPE:-float16}"
CTX_LEN="${CTX_LEN:-8192}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
LOG_FILE="${LOG_FILE:-/tmp/llm_${PORT}.log}"
WORLDEVOLVER_PY="${WORLDEVOLVER_PY:-${PYTHON:-python}}"
if [[ -z "${GPU:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
              | sort -t',' -k2 -n | head -1 | awk -F',' '{gsub(/ /,"",$1); print $1}')
    else
        GPU=0
    fi
fi
CC=$(nvidia-smi -i "${GPU}" --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null \
     | head -1 | tr -d '.' || true)
if [[ -z "${CC}" || "${CC}" == *"not a valid"* || "${CC}" =~ [^0-9] ]]; then
    CC=$(CUDA_VISIBLE_DEVICES="${GPU}" "${WORLDEVOLVER_PY}" - <<'PY' 2>/dev/null || echo "0"
import torch
try:
    major, minor = torch.cuda.get_device_capability(0)
    print(f"{major}{minor}")
except Exception:
    print("0")
PY
)
fi
CC="${CC:-0}"
echo "[launch] GPU=${GPU}  compute_cap=${CC}  model=${MODEL}  port=${PORT}"
if [[ "${BACKEND}" == "auto" ]]; then
    if [[ "${CC}" -ge 80 ]]; then
        BACKEND=sglang
    else
        BACKEND=hf
        echo "[launch] sm_70 / Volta detected — using HF fallback (sglang prebuilt kernels lack sm_70)"
    fi
fi
export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
case "${BACKEND}" in
    sglang)
        ARGS=(
            --model-path "${MODEL}"
            --host 0.0.0.0
            --port "${PORT}"
            --mem-fraction-static "${MEM_FRACTION}"
            --context-length "${CTX_LEN}"
            --dtype "${DTYPE}"
            --attention-backend triton
            --disable-cuda-graph
            --trust-remote-code
        )
        echo "[launch] (sglang) ${WORLDEVOLVER_PY} -m sglang.launch_server ${ARGS[*]}"
        nohup "${WORLDEVOLVER_PY}" -m sglang.launch_server "${ARGS[@]}" > "${LOG_FILE}" 2>&1 &
        ;;
    hf)
        REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
        echo "[launch] (hf) ${WORLDEVOLVER_PY} ${REPO_ROOT}/scripts/launch_hf_server.py --model ${MODEL} --port ${PORT} --dtype ${DTYPE} --max-context ${CTX_LEN}"
        nohup "${WORLDEVOLVER_PY}" "${REPO_ROOT}/scripts/launch_hf_server.py" \
            --model "${MODEL}" --host 0.0.0.0 --port "${PORT}" --dtype "${DTYPE}" \
            --max-context "${CTX_LEN}" --device cuda:0 \
            > "${LOG_FILE}" 2>&1 &
        ;;
    *)
        echo "[launch] FATAL: unknown BACKEND=${BACKEND}; expected sglang|hf|auto" >&2
        exit 1
        ;;
esac
PID=$!
echo "[launch] PID=${PID}; tail -f ${LOG_FILE} to watch progress"
echo "${PID}" > "/tmp/llm_${PORT}.pid"
echo "[launch] waiting for server readiness on port ${PORT} (up to 240s) ..."
for i in $(seq 1 240); do
    if curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "[launch] server ready after ${i}s"
        exit 0
    fi
    if ! kill -0 "${PID}" 2>/dev/null; then
        echo "[launch] FATAL: server process died. tail of log:" >&2
        tail -n 60 "${LOG_FILE}" >&2 || true
        exit 1
    fi
    sleep 1
done
echo "[launch] server did not become ready within 240s; tail of log:" >&2
tail -n 60 "${LOG_FILE}" >&2 || true
exit 1
