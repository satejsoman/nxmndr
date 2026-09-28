#!/usr/bin/env bash
# Probe candidate GPU hosts for running nxmndr-server, from a machine on the ischool VPN.
#
# For each host it reports:
#   reach   - DNS + tcp/22 from this Mac
#   ssh     - key-based login works (BatchMode); user = $PROBE_USER (default: $USER)
#   gpu     - nvidia-smi summary (names, memory, driver) + CUDA visible from any python torch
#   python  - python3 version, whether `python3 -m venv` + pip index are usable
#   egress  - HTTP status from the host to huggingface.co, api.openai.com, pypi.org, github.com
#             (401/403/200 = reachable; 000 = blocked/no route), plus any proxy env vars
#   ingress - a throwaway HTTP listener on $PROBE_PORT (0.0.0.0) on the host, then a curl
#             from this Mac to host:$PROBE_PORT; this is exactly what the plugin/harness
#             needs for gRPC (default 50051)
#   fw      - firewall daemons active on the host, passwordless sudo available?
#
# Usage:  PROBE_USER=satej ./probe_gpu_hosts.sh [host ...]
#         (defaults to pedri fati pique @ ischool.berkeley.edu)

set -u
DOMAIN=${PROBE_DOMAIN:-ischool.berkeley.edu}
HOSTS=("$@"); [ ${#HOSTS[@]} -eq 0 ] && HOSTS=(pedri fati pique)
USER_=${PROBE_USER:-$USER}
PORT=${PROBE_PORT:-50051}
# a function rather than a string so this also works when run/sourced under zsh
SSH() { ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new "$@"; }

remote_script() {
cat <<'EOS'
set +e
echo "host=$(hostname -f 2>/dev/null || hostname)  os=$(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME")  kernel=$(uname -r)  uptime=$(uptime -p 2>/dev/null)"
echo "cpu=$(nproc 2>/dev/null) cores  mem=$(free -g 2>/dev/null | awk '/Mem:/{print $2"G total, "$7"G avail"}')  home_free=$(df -h "$HOME" 2>/dev/null | awk 'NR==2{print $4}')"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/gpu: /'
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/gpu-proc: /'
else
  echo "gpu: nvidia-smi not found"
fi
PY=$(command -v python3 || command -v python)
echo "python=${PY:-none} $($PY --version 2>&1)"
$PY - <<'PY' 2>/dev/null || echo "torch: not importable by system python (fine; venv will provide it)"
import torch; print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} devices={torch.cuda.device_count()}")
PY
$PY -m venv /tmp/_nx_venv_probe >/dev/null 2>&1 && echo "venv: ok" || echo "venv: FAILED (python3-venv missing?)"; rm -rf /tmp/_nx_venv_probe
for u in https://huggingface.co/api/models/facebook/sam3 https://api.openai.com/v1/models https://pypi.org/simple/pip/ https://github.com; do
  code=$(curl -sS -m 12 -o /dev/null -w '%{http_code}' "$u" 2>/dev/null); echo "egress: $code  $u"
done
env | grep -i -E '^(http|https|no)_proxy=' | sed 's/^/proxyenv: /'
for d in firewalld ufw nftables iptables; do s=$(systemctl is-active $d 2>/dev/null); [ -n "$s" ] && echo "fw: $d=$s"; done
sudo -n true 2>/dev/null && echo "sudo: passwordless OK" || echo "sudo: needs password / none"
ss -ltn 2>/dev/null | awk 'NR>1{print $4}' | grep -E ':(50051|8080)$' | sed 's/^/listening-already: /'
# ingress probe: bind a throwaway HTTP server on the gRPC port for 25 s
nohup $PY -m http.server __PORT__ --bind 0.0.0.0 >/dev/null 2>&1 &
echo "ingress-listener-pid=$!"
sleep 1
EOS
}

for h in "${HOSTS[@]}"; do
  fq="$h.$DOMAIN"; echo; echo "===== $fq ====="
  ip=$(dig +short "$fq" 2>/dev/null | tail -1); echo "reach: dns=${ip:-UNRESOLVED}"
  [ -z "$ip" ] && continue
  nc -z -G 5 "$ip" 22 >/dev/null 2>&1 && echo "reach: tcp/22 open" || { echo "reach: tcp/22 CLOSED"; continue; }
  out=$(SSH "$USER_@$fq" "bash -s" < <(remote_script | sed "s/__PORT__/$PORT/") 2>&1)
  rc=$?; [ $rc -ne 0 ] && { echo "ssh: FAILED (rc=$rc): $(echo "$out" | tail -2)"; continue; }
  echo "ssh: ok as $USER_"; echo "$out" | grep -v ingress-listener-pid | sed 's/^/  /'
  pid=$(echo "$out" | sed -n 's/^ingress-listener-pid=//p')
  code=$(curl -s -m 6 -o /dev/null -w '%{http_code}' "http://$fq:$PORT/" 2>/dev/null)
  if [ "$code" = "200" ]; then echo "  ingress: Mac -> $fq:$PORT OK (gRPC on this port will work)"; else echo "  ingress: Mac -> $fq:$PORT BLOCKED (code $code) - use an ssh tunnel: ssh -N -L $PORT:localhost:$PORT $USER_@$fq"; fi
  [ -n "$pid" ] && SSH "$USER_@$fq" "kill $pid 2>/dev/null" >/dev/null 2>&1
done
echo
echo "Verdict per host: need ssh ok + a GPU line + egress 200/401 for huggingface.co (gated model => 401 is fine) + ingress OK (or accept the tunnel)."
