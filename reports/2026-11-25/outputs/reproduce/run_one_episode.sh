#!/usr/bin/env bash
# Run the frozen upstream checkpoint in a fresh directory. This entry never modifies the training runs.
set -euo pipefail
: "${MODEL_DIR:?Set MODEL_DIR to the pinned, unpacked checkpoint}"
: "${POLICY_ENV:?Set POLICY_ENV to the vLLM / vLLM-Omni Python environment}"
ROOT=$(cd "$(dirname "$0")" && pwd)
export ROBOLAB_ROOT=${ROBOLAB_ROOT:-/home/robot_share/RoboLab}
RUN=$(mktemp -d "${EPISODE_PARENT:-/var/tmp}/cosmos-one-episode.XXXXXXXX")
PORT=${EP_PORT:-48136}
HOST=127.0.0.1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}" no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export VLLM_CACHE_ROOT="$RUN/vllm-cache" TRITON_CACHE_DIR="$RUN/triton" TORCHINDUCTOR_CACHE_DIR="$RUN/inductor"
mkdir "$RUN/logs"
echo "$RUN"
date --iso-8601=seconds > "$RUN/start.txt"
git --git-dir="$ROBOLAB_ROOT/.git" rev-parse HEAD > "$RUN/robolab_revision.txt"
(cd "$ROBOLAB_ROOT" && sha256sum policies/cosmos3/{run,client}.py robolab/eval/{runner,episode,summarize,base_client}.py) > "$RUN/robolab_entry_sha256.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.free --format=csv > "$RUN/gpu.csv"
"$POLICY_ENV/bin/python" - <<'PY' > "$RUN/policy_versions.txt"
import importlib.metadata as md
import sys
print(sys.version)
for name in ("vllm", "vllm-omni", "torch", "diffusers", "transformers", "safetensors"):
    print(name, md.version(name))
PY
"$POLICY_ENV/bin/python" - <<'PY' > "$RUN/policy_requirements.txt"
import importlib.metadata as md
for name, version in sorted((d.metadata["Name"], d.version) for d in md.distributions() if d.metadata["Name"]):
    print(f"{name}=={version}")
PY
"$ROBOLAB_ROOT/.venv/bin/python" - <<'PY' > "$RUN/simulation_versions.txt"
import importlib.metadata as md
import sys
print(sys.version)
for d in md.distributions():
    if d.metadata["Name"] and d.metadata["Name"].lower() in {"torch","isaacsim","isaaclab","numpy","gymnasium","h5py"}:
        print(d.metadata["Name"], d.version)
PY
MODEL_ARGS=()
if [[ -n "${MODEL_METADATA:-}" ]]; then MODEL_ARGS=(--metadata "$MODEL_METADATA"); fi
python3 "$ROOT/check_episode_model.py" "$MODEL_DIR" "$RUN/model.json" "${MODEL_ARGS[@]}" > "$RUN/logs/model_verify.log"
DEPLOY=$("$POLICY_ENV/bin/python" -c 'import importlib.util,pathlib;print(pathlib.Path(importlib.util.find_spec("vllm_omni").origin).parent/"deploy/cosmos3_policy_droid.yaml")')
cp "$DEPLOY" "$RUN/policy_deploy.yaml"
printf '%s\n' "$MODEL_DIR" > "$RUN/served_model_path.txt"
python3 - "$PORT" <<'PY'
import socket, sys
with socket.socket() as sock:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
PY
SERVER_CMD=("$POLICY_ENV/bin/vllm-omni" serve "$MODEL_DIR" --omni --host "$HOST" --port "$PORT"
            --init-timeout 1800 --deploy-config "$DEPLOY" --robot-openpi-idle-timeout 3600)
printf '%q ' "${SERVER_CMD[@]}" > "$RUN/server_command.txt"
CUDA_VISIBLE_DEVICES=${POLICY_GPU:-0} PATH="$POLICY_ENV/bin:$PATH" setsid "${SERVER_CMD[@]}" > "$RUN/logs/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill -TERM -- "-$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT
ready=0
for ((i=0; i<360; i++)); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "Policy startup failed. See $RUN/logs/server.log" >&2; exit 1; fi
    if curl --noproxy '*' --max-time 3 -fsS "http://$HOST:$PORT/v1/models" -o "$RUN/served_models.json" 2>/dev/null; then ready=1; break; fi
    sleep 5
done
if (( ! ready )); then echo "Policy did not become ready. See $RUN/logs/server.log" >&2; exit 1; fi
cd "$RUN"
export CUDA_VISIBLE_DEVICES=${SIM_GPU:-1} OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH="$ROBOLAB_ROOT"
export LD_LIBRARY_PATH="${SIM_VULKAN_LIB:-/home/robot_share/deps/deb-libs/vulkan/usr/lib/x86_64-linux-gnu}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$ROBOLAB_ROOT/.venv/bin/python" -B "$ROOT/episode_client.py" --task BananaInBowlTask \
  --headless --device cuda:0 --num-runs 1 --num-envs 1 --seed 2105981890 --init-pose-noise 0.1 60 \
  --instruction-type default --video-mode sensor --output-folder-name "$RUN/rollout" \
  --remote-uri "ws://$HOST:$PORT/v1/realtime/robot/openpi" > "$RUN/logs/simulator.log" 2>&1
date --iso-8601=seconds > "$RUN/done.txt"
"${ANALYSIS_PYTHON:-python3}" -B "$ROOT/verify_one_episode.py" "$RUN"
