#!/usr/bin/env bash
# Deploy nxmndr-server (gRPC only) to a GPU host over ssh and open a local tunnel.
#
#   HF_TOKEN=hf_... bash deploy_gpu_server.sh satej@pique.ischool.berkeley.edu [GPU_INDEX] [PORT]
#
# What it does:
#   1. rsync this nxmndr package (src + pyproject, no .git/models) to ~/nxmndr on the host
#   2. create ~/venvs/nxmndr with the host's python3 (>= 3.11 required by the pins) and pip install -e
#   3. start `nxmndr-server --grpc-port PORT` in a tmux session "nxmndr", pinned to GPU_INDEX
#   4. print the tunnel command; with the tunnel up, the harness/plugin use 127.0.0.1:PORT
#
# Re-running is idempotent: rsync updates code, pip is a no-op if unchanged, tmux session is replaced.

set -euo pipefail
TARGET=${1:?usage: deploy_gpu_server.sh user@host [gpu_index] [port]}
GPU=${2:-1}
PORT=${3:-50051}
HERE=$(cd "$(dirname "$0")/.." && pwd)          # .../nxmndr/nxmndr (package root with pyproject.toml)
[ -f "$HERE/pyproject.toml" ] || { echo "cannot find pyproject.toml above $0"; exit 1; }
: "${HF_TOKEN:?set HF_TOKEN (facebook/sam3 is gated)}"

echo "==> rsync $HERE -> $TARGET:~/nxmndr"
rsync -az --delete --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' --exclude '/models/' \
      "$HERE/" "$TARGET:nxmndr/"

echo "==> remote: venv + install + start"
# SAM 3 instance selection on the server: 0 = score-free (area/IoU/group filters, used for the case-study
# figure because unprompted concept scores are ~0.01), or a score threshold such as 0.3 (transformers default).
SAM3_SCORE_THRESHOLD=${SAM3_SCORE_THRESHOLD:-0}
ssh -o StrictHostKeyChecking=accept-new "$TARGET" "HF_TOKEN='$HF_TOKEN' GPU='$GPU' PORT='$PORT' SAM3_SCORE_THRESHOLD='$SAM3_SCORE_THRESHOLD' bash -s" <<'EOS'
set -euo pipefail
PY=$(command -v python3.12 || command -v python3.11 || command -v python3)
ver=$($PY -c 'import sys;print("%d.%d"%sys.version_info[:2])')
case "$ver" in 3.11|3.12|3.13) ;; *) echo "python $ver too old for the server pins (need >=3.11); install one with: curl -LsSf https://astral.sh/uv/install.sh | sh && uv python install 3.12"; exit 1;; esac
mkdir -p ~/venvs
[ -x ~/venvs/nxmndr/bin/python ] || $PY -m venv ~/venvs/nxmndr
source ~/venvs/nxmndr/bin/activate
# home directories are quota-limited NFS; keep pip's cache and build temp on the local disk
export PIP_CACHE_DIR=/tmp/$USER-pip-cache TMPDIR=/tmp/$USER-tmp; mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
pip install -q -U pip
# torch first, from the CUDA-12.6 index: the PyPI default (+cu130) and the +cu128 wheels of torch 2.11
# carry no sm_70 kernels (arch list starts at sm_75), so the V100 (Volta) hosts get "no kernel image";
# only the +cu126 build still includes sm_70. Pin the local version tag explicitly, otherwise pip treats
# an installed 2.11.0+cu130 as satisfying torch==2.11.0. (server.py disables cuDNN >= 9.11 on SM < 7.5.)
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}
TORCH_TAG=${TORCH_TAG:-cu126}
pip install -q "torch==2.11.0+$TORCH_TAG" "torchvision==0.26.0+$TORCH_TAG" --index-url "$TORCH_INDEX"   # torchvision: transformers' Sam3Model imports it
pip install -q -e "$HOME/nxmndr[server]"   # transformers 5.3, grpcio, aiohttp (server.py imports it unconditionally); torch pin already satisfied
python - <<'PY'
import torch; print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} gpus={torch.cuda.device_count()}")
PY
mkdir -p ~/.config/nxmndr
printf 'export HF_TOKEN=%s\nexport HUGGINGFACE_TOKEN=%s\n' "$HF_TOKEN" "$HF_TOKEN" > ~/.config/nxmndr/env   # kept out of shell history
chmod 600 ~/.config/nxmndr/env
tmux kill-session -t nxmndr 2>/dev/null || true
# wait for the previous server to release the port, otherwise the readiness check below passes on the old process
for i in $(seq 1 30); do ss -ltn 2>/dev/null | grep ":$PORT " >/dev/null || break; sleep 1; done
# tmux runs commands with /bin/sh (dash on Ubuntu), so wrap in bash explicitly
cat > ~/.config/nxmndr/start.sh <<EOF2
#!/usr/bin/env bash
source ~/venvs/nxmndr/bin/activate
source ~/.config/nxmndr/env
export CUDA_VISIBLE_DEVICES=$GPU
export NXMNDR_SAM3_SCORE_THRESHOLD=$SAM3_SCORE_THRESHOLD
exec nxmndr-server --grpc-port $PORT --max-cores 1 --log-level INFO >> ~/nxmndr-server.log 2>&1
EOF2
chmod 700 ~/.config/nxmndr/start.sh
tmux new-session -d -s nxmndr "bash ~/.config/nxmndr/start.sh"
for i in $(seq 1 60); do
  if ss -ltn 2>/dev/null | grep ":$PORT " >/dev/null; then echo "listening on :$PORT (after ${i}s)"; break; fi
  if ! tmux has-session -t nxmndr 2>/dev/null; then echo "server exited; last log lines:"; tail -20 ~/nxmndr-server.log; exit 1; fi
  sleep 1
done
ss -ltn 2>/dev/null | grep ":$PORT " >/dev/null || { echo "not listening after 60s; log tail:"; tail -20 ~/nxmndr-server.log; exit 1; }
tail -3 ~/nxmndr-server.log
EOS

cat <<EOT

==> done. Open the tunnel in a separate terminal and leave it running:
    ssh -N -L $PORT:localhost:$PORT $TARGET

Then, on the Mac (server venv, from nxmndr/nxmndr):
    python examples/case_study_harness.py --grpc-endpoint 127.0.0.1:$PORT --paths sam3,dam \\
        --dam-weights ../models/DelineateAnything.pt --out ../../anaximander/nxmndr_outputs/case_study
(plugin: add a remote provider with endpoint 127.0.0.1:$PORT)

Logs on the host: ~/nxmndr-server.log ; console: ssh $TARGET -t tmux attach -t nxmndr
EOT
