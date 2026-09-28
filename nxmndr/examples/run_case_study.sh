#!/usr/bin/env bash
# One-shot: deploy nxmndr-server on the GPU host, tunnel it, run all three case-study
# paths from the Mac, and (optionally) drop the new Figure-4 panels into both papers.
#
#   OPENAI_API_KEY=sk-... HF_TOKEN=hf_... bash run_case_study.sh
#
# Knobs (env): GPU_HOST=satej@pique.ischool.berkeley.edu  GPU_INDEX=1  PORT=50051
#              PATHS=gpt,sam3,dam   INSTALL_FIGS=1 (copy panels into arxiv/ and eccv/ figs)
#              SKIP_DEPLOY=1 (server already running on the host)
# Everything is idempotent; the remote server stays up in tmux after this exits.

set -euo pipefail
NX=/Users/satej/Documents/workspace/research/nxmndr/nxmndr          # server package (pyproject here)
ANX=/Users/satej/Documents/workspace/research/anaximander           # paper repo root
DAM=/Users/satej/Documents/workspace/research/nxmndr/models/DelineateAnything.pt
GPU_HOST=${GPU_HOST:-satej@pique.ischool.berkeley.edu}
GPU_INDEX=${GPU_INDEX:-1}
PORT=${PORT:-50051}
PATHS=${PATHS:-gpt,sam3,dam}
VENV=$HOME/venvs/nxmndr-mac
OUT=$ANX/nxmndr_outputs/case_study
SOCK=/tmp/nxmndr-tunnel.sock

: "${OPENAI_API_KEY:?set OPENAI_API_KEY}"
: "${HF_TOKEN:?set HF_TOKEN (facebook/sam3 is gated)}"
[ -f "$DAM" ] || { echo "missing $DAM"; exit 1; }
cd "$NX"

# ---- 1. local venv (torch/transformers/grpc for the client + proxy, ultralytics for DAM)
PY=$(command -v python3.12 || command -v python3.11 || command -v python3)
ver=$($PY -c 'import sys;print("%d.%d"%sys.version_info[:2])')
case "$ver" in 3.11|3.12|3.13) ;; *) echo "need python >= 3.11 on the Mac (found $ver): brew install python@3.12"; exit 1;; esac
[ -x "$VENV/bin/python" ] || $PY -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -c "import nxmndr, ultralytics, aiohttp, azure.identity" 2>/dev/null || {
  echo "==> installing into $VENV (first time: several minutes)"
  pip install -q -U pip
  pip install -q -e "$NX[server]" ultralytics rasterio
}

# ---- 2. remote server on the GPU host + tunnel
if [ -z "${SKIP_DEPLOY:-}" ]; then
  HF_TOKEN="$HF_TOKEN" bash "$NX/examples/deploy_gpu_server.sh" "$GPU_HOST" "$GPU_INDEX" "$PORT"
fi
ssh -S "$SOCK" -O check "$GPU_HOST" >/dev/null 2>&1 || \
  ssh -M -S "$SOCK" -f -N -o ExitOnForwardFailure=yes -L "$PORT:localhost:$PORT" "$GPU_HOST"
trap 'ssh -S "$SOCK" -O exit "$GPU_HOST" >/dev/null 2>&1 || true' EXIT
for i in $(seq 1 30); do nc -z localhost "$PORT" >/dev/null 2>&1 && break; sleep 1; done
nc -z localhost "$PORT" >/dev/null 2>&1 || { echo "tunnel to $GPU_HOST:$PORT not up"; exit 1; }
echo "==> tunnel up: 127.0.0.1:$PORT -> $GPU_HOST"

# ---- 3. the three paths (gpt via a local proxy the harness spawns; sam3 via the tunnel; dam local)
DEPLOYMENT=${VISION_DEPLOYMENT_NAME:-gpt-image-1}   # not exported: a bare *_DEPLOYMENT_NAME in the env would suppress the harness's OpenAI-platform proxy config
python "$NX/examples/case_study_harness.py" \
  --grpc-endpoint "127.0.0.1:$PORT" --paths "$PATHS" \
  --dam-weights "$DAM" --dam-device cpu --out "$OUT" --deployment "$DEPLOYMENT"
RUN=$(ls -td "$OUT"/*/ | head -1)
echo "==> run: $RUN"; ls "$RUN"

# ---- 4. optionally install the panels into both papers (arxiv + eccv), keeping the old ones
if [ -n "${INSTALL_FIGS:-}" ]; then
  for paper in arxiv eccv; do
    d=$ANX/$paper/figs/comparison; mkdir -p "$d/google_2025_superseded"
    for pair in input:input gpt:gpt sam3:sam dam:dam; do
      src=$RUN/panel_${pair%%:*}.png; dst=$d/comparison_${pair##*:}.png
      [ -f "$src" ] || continue
      [ -f "$dst" ] && [ ! -f "$d/google_2025_superseded/$(basename "$dst")" ] && cp "$dst" "$d/google_2025_superseded/"
      cp "$src" "$dst"; echo "installed $dst"
    done
  done
fi
echo "==> done. server still running on $GPU_HOST (tmux session nxmndr); tunnel closed."
