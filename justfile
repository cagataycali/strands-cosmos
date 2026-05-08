# strands-cosmos — Full NVIDIA Cosmos ecosystem for Strands Agents
# All tools shell out to `just <recipe>`; operators can run them directly.
# Recipes sourced from thor-cosmos; adapted for strands-cosmos standalone usage.

set shell := ["bash", "-euo", "pipefail", "-c"]
set dotenv-load := true
set positional-arguments := true

# ── Environment defaults ──────────────────────────────────────────────────
VENV              := ".venv"
PYTHON            := "python3"

# Cosmos repos (clone alongside strands-cosmos or set env vars)
COSMOS_PREDICT_REPO   := env_var_or_default("COSMOS_PREDICT_REPO", "../cosmos-predict2.5")
COSMOS_TRANSFER_REPO  := env_var_or_default("COSMOS_TRANSFER_REPO", "../cosmos-transfer2.5")
COSMOS_REASON_REPO    := env_var_or_default("COSMOS_REASON_REPO", "../cosmos-reason2")
COSMOS_XENNA_REPO     := env_var_or_default("COSMOS_XENNA_REPO", "../cosmos-xenna")
COSMOS_RL_REPO        := env_var_or_default("COSMOS_RL_REPO", "../cosmos-rl")
COSMOS_COOKBOOK_REPO   := env_var_or_default("COSMOS_COOKBOOK_REPO", "../cosmos-cookbook")

# TensorRT-Edge-LLM binaries (Thor-side)
TRT_ROOT              := env_var_or_default("TRT_ROOT", "/opt/tensorrt-edge-llm")
SERVER_BIN            := env_var_or_default("COSMOS_SERVER_BIN", TRT_ROOT + "/build/examples/server/trt_edgellm_server")
LLM_BUILD_BIN         := env_var_or_default("TRT_LLM_BUILD_BIN", TRT_ROOT + "/build/examples/llm/llm_build")
VISUAL_BUILD_BIN      := env_var_or_default("TRT_VISUAL_BUILD_BIN", TRT_ROOT + "/build/examples/multimodal/visual_build")

# Serve config
VLM_HOST              := env_var_or_default("VLM_HOST", "127.0.0.1")
VLM_PORT              := env_var_or_default("VLM_PORT", "8080")
VLM_URL               := "http://" + VLM_HOST + ":" + VLM_PORT + "/v1/chat/completions"

# RTP / NATS
RTP_BIND              := env_var_or_default("RTP_BIND", "0.0.0.0")
RTP_PORT              := env_var_or_default("RTP_PORT", "5600")
NATS_URL              := env_var_or_default("NATS_URL", "nats://127.0.0.1:4222")

PID_FILE              := env_var_or_default("COSMOS_SERVER_PID", "/tmp/strands-cosmos-server.pid")
LOG_FILE              := env_var_or_default("COSMOS_SERVER_LOG", "/tmp/strands-cosmos-server.log")


# ── Git URLs for auto-clone ────────────────────────────────────────────────
# Override with env vars if you use forks or SSH URLs
COSMOS_PREDICT_GIT    := env_var_or_default("COSMOS_PREDICT_GIT", "https://github.com/nvidia-cosmos/cosmos-predict2.5.git")
COSMOS_TRANSFER_GIT   := env_var_or_default("COSMOS_TRANSFER_GIT", "https://github.com/nvidia-cosmos/cosmos-transfer2.5.git")
COSMOS_REASON_GIT     := env_var_or_default("COSMOS_REASON_GIT", "https://github.com/nvidia-cosmos/cosmos-reason2.git")
COSMOS_XENNA_GIT      := env_var_or_default("COSMOS_XENNA_GIT", "https://github.com/nvidia-cosmos/cosmos-curate.git")
COSMOS_RL_GIT         := env_var_or_default("COSMOS_RL_GIT", "https://github.com/nvidia-cosmos/cosmos-rl.git")
COSMOS_COOKBOOK_GIT    := env_var_or_default("COSMOS_COOKBOOK_GIT", "https://github.com/nvidia-cosmos/cosmos-cookbook.git")


# ── Top-level ─────────────────────────────────────────────────────────────
default:
    @just --list --unsorted

# Print the effective environment
env:
    @echo "COSMOS_PREDICT_REPO  = {{COSMOS_PREDICT_REPO}}"
    @echo "COSMOS_TRANSFER_REPO = {{COSMOS_TRANSFER_REPO}}"
    @echo "COSMOS_REASON_REPO   = {{COSMOS_REASON_REPO}}"
    @echo "COSMOS_XENNA_REPO    = {{COSMOS_XENNA_REPO}}"
    @echo "COSMOS_RL_REPO       = {{COSMOS_RL_REPO}}"
    @echo "COSMOS_COOKBOOK_REPO  = {{COSMOS_COOKBOOK_REPO}}"
    @echo "TRT_ROOT             = {{TRT_ROOT}}"
    @echo "VLM_URL              = {{VLM_URL}}"
    @echo "NATS_URL             = {{NATS_URL}}"


# ── Auto-clone / ensure repos ─────────────────────────────────────────────
# `just setup` clones all missing repos. Individual `ensure-*` recipes are
# called as deps by recipes that need a specific repo.

# Clone a repo if the target dir doesn't exist
[private]
_clone url dir:
    #!/usr/bin/env bash
    if [ -d "{{dir}}" ]; then
      echo "✓ {{dir}} exists"
    else
      echo "📥 cloning {{url}} → {{dir}}"
      git clone --depth 1 "{{url}}" "{{dir}}"
    fi

# Ensure individual repos exist (call before recipes that need them)
ensure-predict:
    @just _clone "{{COSMOS_PREDICT_GIT}}" "{{COSMOS_PREDICT_REPO}}"

ensure-transfer:
    @just _clone "{{COSMOS_TRANSFER_GIT}}" "{{COSMOS_TRANSFER_REPO}}"

ensure-reason:
    @just _clone "{{COSMOS_REASON_GIT}}" "{{COSMOS_REASON_REPO}}"

ensure-xenna:
    @just _clone "{{COSMOS_XENNA_GIT}}" "{{COSMOS_XENNA_REPO}}"

ensure-rl:
    @just _clone "{{COSMOS_RL_GIT}}" "{{COSMOS_RL_REPO}}"

ensure-cookbook:
    @just _clone "{{COSMOS_COOKBOOK_GIT}}" "{{COSMOS_COOKBOOK_REPO}}"

# Clone ALL repos (one-shot dev setup)
setup: ensure-predict ensure-transfer ensure-reason ensure-xenna ensure-rl ensure-cookbook
    @echo ""
    @echo "✅ All Cosmos repos cloned. Running doctor..."
    @echo ""
    @just doctor

# Full setup: system deps + python + repos + TRT (takes ~30min on Jetson)
setup-full:
    just install-system-deps
    just install-python-deps
    just setup
    @echo ""
    @echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    @echo "🎯 Next steps:"
    @echo "  For edge VLM (Jetson Thor):"
    @echo "    just install-trt-edge-llm"
    @echo "    just pipeline-edge-deploy"
    @echo ""
    @echo "  For HF inference (any GPU):"
    @echo "    just download reason2-2b"
    @echo "    python -c \"from strands_cosmos import CosmosVisionModel; print('ok')\""
    @echo ""

# Pull latest on all repos (useful after setup)
update:
    #!/usr/bin/env bash
    for dir in "{{COSMOS_PREDICT_REPO}}" "{{COSMOS_TRANSFER_REPO}}" "{{COSMOS_REASON_REPO}}" \
               "{{COSMOS_XENNA_REPO}}" "{{COSMOS_RL_REPO}}" "{{COSMOS_COOKBOOK_REPO}}"; do
      if [ -d "$dir/.git" ]; then
        echo "⬆ pulling $dir"
        git -C "$dir" pull --rebase 2>/dev/null || git -C "$dir" pull || true
      fi
    done

# Check which repos are present/missing
doctor:
    #!/usr/bin/env bash
    echo "🩺 strands-cosmos doctor"
    echo "========================"
    echo ""
    echo "📦 Cosmos Repos:"
    check() {
      if [ -d "$2" ]; then
        echo "  ✅ $1 → $2"
      else
        echo "  ❌ $1 → $2 (MISSING — run 'just setup' or 'just ensure-$3')"
      fi
    }
    check "Predict 2.5"  "{{COSMOS_PREDICT_REPO}}"  "predict"
    check "Transfer 2.5" "{{COSMOS_TRANSFER_REPO}}" "transfer"
    check "Reason 2"     "{{COSMOS_REASON_REPO}}"   "reason"
    check "Xenna/Curate" "{{COSMOS_XENNA_REPO}}"    "xenna"
    check "RL"           "{{COSMOS_RL_REPO}}"        "rl"
    check "Cookbook"      "{{COSMOS_COOKBOOK_REPO}}"  "cookbook"
    echo ""

    # Detect platform
    ARCH=$(uname -m)
    IS_JETSON=false
    [ -f /proc/device-tree/model ] && IS_JETSON=true
    IS_DOCKER=false
    [ -f /.dockerenv ] && IS_DOCKER=true

    echo "🖥  Platform:"
    echo "  arch: $ARCH"
    if $IS_JETSON; then
      echo "  type: 🟢 Jetson ($(cat /proc/device-tree/model 2>/dev/null | tr -d '\0'))"
    elif $IS_DOCKER; then
      echo "  type: 🐳 Docker container"
    else
      echo "  type: 🖥  Workstation / Cloud"
    fi
    echo ""

    # Core tools (needed everywhere)
    echo "🔧 Core Tools (needed on all platforms):"
    for bin in python3 pip git just hf curl jq; do
      if command -v "$bin" &>/dev/null; then
        echo "  ✅ $bin ($(command -v $bin))"
      else
        echo "  ❌ $bin — REQUIRED"
      fi
    done
    echo ""

    # Python packages
    echo "🐍 Python Packages:"
    for pkg in strands_agents strands_cosmos torch transformers accelerate; do
      if python3 -c "import $pkg" 2>/dev/null; then
        VER=$(python3 -c "import $pkg; print(getattr($pkg, '__version__', '?'))" 2>/dev/null)
        echo "  ✅ $pkg ($VER)"
      else
        echo "  ⚠️  $pkg (not installed)"
      fi
    done
    echo ""

    # Media / I/O tools (optional but useful)
    echo "📹 Media & I/O (optional):"
    for bin in ffmpeg ffprobe gst-launch-1.0 nats; do
      if command -v "$bin" &>/dev/null; then
        echo "  ✅ $bin"
      else
        echo "  ⚠️  $bin (not found — some tools will be limited)"
      fi
    done
    echo ""

    # TensorRT / Edge tools (only on Jetson or TRT docker)
    echo "⚡ TensorRT-Edge-LLM (Jetson or TRT Docker only):"
    TRT_TOOLS=(tensorrt-edgellm-quantize-llm tensorrt-edgellm-export-llm tensorrt-edgellm-export-visual)
    TRT_FOUND=0
    for bin in "${TRT_TOOLS[@]}"; do
      if command -v "$bin" &>/dev/null; then
        echo "  ✅ $bin"
        TRT_FOUND=$((TRT_FOUND + 1))
      else
        echo "  ⬜ $bin (not found)"
      fi
    done
    if [ -x "{{SERVER_BIN}}" ]; then
      echo "  ✅ trt_edgellm_server → {{SERVER_BIN}}"
      TRT_FOUND=$((TRT_FOUND + 1))
    else
      echo "  ⬜ trt_edgellm_server → {{SERVER_BIN}} (not found)"
    fi
    if [ -x "{{LLM_BUILD_BIN}}" ]; then
      echo "  ✅ llm_build → {{LLM_BUILD_BIN}}"
    else
      echo "  ⬜ llm_build → {{LLM_BUILD_BIN}} (not found)"
    fi
    if [ -x "{{VISUAL_BUILD_BIN}}" ]; then
      echo "  ✅ visual_build → {{VISUAL_BUILD_BIN}}"
    else
      echo "  ⬜ visual_build → {{VISUAL_BUILD_BIN}} (not found)"
    fi
    echo ""
    if [ $TRT_FOUND -eq 0 ]; then
      if $IS_JETSON; then
        echo "  ⚠️  TRT tools missing on Jetson! Install with:"
        echo "     just install-trt-edge-llm"
        echo "     (builds from source ~30min, or set TRT_ROOT if already built)"
      else
        echo "  ℹ️  TRT tools not found — EXPECTED on workstation."
        echo "     Quantize/export runs in the TRT docker container or on Jetson."
        echo "     Engine build + serve runs on Jetson Thor."
        echo "     Install: just install-trt-edge-llm /path/to/trt"
        echo "     Recipes: quantize, export-llm, export-visual, build-*-engine, serve-*"
        echo "     These will return exit 127 without TRT installed — that's normal."
      fi
    fi
    echo ""

    # CUDA
    echo "🎮 GPU/CUDA:"
    if command -v nvidia-smi &>/dev/null; then
      nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | while read line; do
        echo "  ✅ $line"
      done
    else
      echo "  ⬜ nvidia-smi not found (no GPU or driver not in PATH)"
    fi
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
      DEVICE=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
      echo "  ✅ torch.cuda → $DEVICE"
    else
      echo "  ⚠️  torch.cuda not available (CPU-only — VLM inference will be slow)"
    fi
    echo ""

    # Summary
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📋 What works on THIS machine:"
    echo "  • cosmos_reason_hf / cosmos_vision_invoke (HF direct inference)"
    echo "  • cosmos_model_download (HF downloads)"
    echo "  • video_probe / video_extract_frames / image_read"
    echo "  • cosmos_evaluate (if cookbook cloned)"
    echo "  • cosmos_curate (if xenna cloned + deps)"
    echo "  • cosmos_post_train (reason2 SFT/LoRA if GPU available)"
    if [ $TRT_FOUND -gt 0 ] || $IS_JETSON; then
      echo "  • cosmos_quantize / export / build_engine / serve (TRT available)"
      echo "  • cosmos_inference (against local TRT server)"
      echo "  • rtp_capture_frame (GStreamer RTP)"
    else
      echo ""
      echo "  ⚡ For TRT edge pipeline (quantize → export → build → serve):"
      echo "     Run on Jetson Thor or inside TRT docker."
      echo "     Recipes: just quantize, just export-llm, just build-engines, just serve-start"
    fi
    echo ""

# ── Install ───────────────────────────────────────────────────────────────
install:
    {{PYTHON}} -m venv {{VENV}} || true
    {{VENV}}/bin/pip install -U pip
    {{VENV}}/bin/pip install -e .
    @echo "✅ installed. Try: just smoke"

# Install TensorRT-Edge-LLM (builds from source — Jetson or x86 with CUDA)
# This provides: trt_edgellm_server, llm_build, visual_build,
#                tensorrt-edgellm-quantize-llm, tensorrt-edgellm-export-llm, etc.
install-trt-edge-llm trt_dir=TRT_ROOT:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "⚡ Installing TensorRT-Edge-LLM → {{trt_dir}}"
    echo ""

    # Prerequisites check
    for bin in cmake g++ ninja-build git; do
      if ! command -v "$bin" &>/dev/null; then
        echo "❌ $bin not found. Install build deps first:"
        echo "   sudo apt-get install -y cmake g++ ninja-build git python3-dev"
        exit 1
      fi
    done

    # Check TensorRT is available (JetPack provides it)
    if ! pkg-config --exists nvinfer 2>/dev/null && [ ! -f /usr/lib/aarch64-linux-gnu/libnvinfer.so ]; then
      echo "❌ TensorRT not found. On Jetson, ensure JetPack is installed."
      echo "   dpkg -l | grep libnvinfer"
      exit 1
    fi

    # Clone if not present
    if [ ! -d "{{trt_dir}}" ]; then
      echo "📥 Cloning TensorRT-Edge-LLM..."
      git clone --depth 1 https://github.com/NVIDIA/TensorRT-LLM.git "{{trt_dir}}"
    else
      echo "✓ {{trt_dir}} exists"
    fi

    cd "{{trt_dir}}"

    # Build
    echo "🔨 Building (this may take 20-60 min on Jetson)..."
    mkdir -p build && cd build
    cmake .. -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="87;90;100" \
      -DBUILD_EXAMPLES=ON \
      2>&1 | tail -5
    ninja -j$(nproc) 2>&1 | tail -20

    echo ""
    echo "✅ Build complete!"
    echo ""

    # Verify binaries exist
    for bin in examples/server/trt_edgellm_server examples/llm/llm_build examples/multimodal/visual_build; do
      if [ -x "build/$bin" ] || [ -x "$bin" ]; then
        echo "  ✅ $bin"
      else
        echo "  ⚠️  $bin not found (build may use different paths)"
      fi
    done

    # Install python tools if available
    if [ -f setup.py ] || [ -f pyproject.toml ]; then
      echo ""
      echo "📦 Installing Python tools (quantize, export)..."
      pip3 install -e . 2>&1 | tail -5 || true
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Set TRT_ROOT to use with strands-cosmos:"
    echo "  export TRT_ROOT={{trt_dir}}"
    echo "  # or add to .env:"
    echo "  echo 'TRT_ROOT={{trt_dir}}' >> .env"

# Install system deps (apt packages needed for build + runtime)
install-system-deps:
    #!/usr/bin/env bash
    echo "📦 Installing system dependencies..."
    sudo apt-get update
    sudo apt-get install -y \
      cmake g++ ninja-build git python3-dev \
      ffmpeg gstreamer1.0-tools gstreamer1.0-plugins-good \
      gstreamer1.0-plugins-bad gstreamer1.0-libav \
      libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
      jq curl
    echo ""
    echo "✅ System deps installed."
    echo "   For NATS: curl -sf https://binaries.nats.dev/nats-io/natscli/nats@latest | sh"

# Install Python deps (strands-cosmos + all extras)
install-python-deps:
    #!/usr/bin/env bash
    echo "🐍 Installing Python dependencies..."
    pip3 install -U pip
    pip3 install -e ".[all]"
    pip3 install strands-agents strands-agents-tools
    echo "✅ Python deps installed."

# ── Model / dataset download ──────────────────────────────────────────────
download name="reason2-2b" local_dir="":
    #!/usr/bin/env bash
    DEST="{{local_dir}}"
    [ -z "$DEST" ] && DEST="./checkpoints/{{name}}"
    mkdir -p "$DEST"
    case "{{name}}" in
      reason2-2b)        REPO="nvidia/Cosmos-Reason2-2B" ;;
      reason2-7b)        REPO="nvidia/Cosmos-Reason2-7B" ;;
      reason1-7b-reward) REPO="nvidia/Cosmos-Reason1-7B-Reward" ;;
      predict2.5-2b)     REPO="nvidia/Cosmos-Predict2.5-2B" ;;
      predict2.5-14b)    REPO="nvidia/Cosmos-Predict2.5-14B" ;;
      transfer2.5-2b)    REPO="nvidia/Cosmos-Transfer2.5-2B" ;;
      transfer2.5-edge)  REPO="nvidia/Cosmos-Transfer2.5-Edge" ;;
      transfer2.5-depth) REPO="nvidia/Cosmos-Transfer2.5-Depth" ;;
      transfer2.5-seg)   REPO="nvidia/Cosmos-Transfer2.5-Seg" ;;
      *)                 REPO="{{name}}" ;;
    esac
    hf download "$REPO" --local-dir "$DEST"

download-dataset name="gr1" local_dir="":
    #!/usr/bin/env bash
    DEST="{{local_dir}}"
    [ -z "$DEST" ] && DEST="./datasets/{{name}}"
    mkdir -p "$DEST"
    case "{{name}}" in
      gr1)         REPO="nvidia/PhysicalAI-Robotics-GR00T-GR1" ;;
      gr1-100)     REPO="nvidia/GR1-100" ;;
      gr00t-eval)  REPO="nvidia/PhysicalAI-Robotics-GR00T-Eval" ;;
      safe-unsafe) REPO="pjramg/Safe_Unsafe_Test" ;;
      *)           REPO="{{name}}" ;;
    esac
    hf download "$REPO" --repo-type dataset --local-dir "$DEST"


# ── Quantization + ONNX export (x86 GPU host) ─────────────────────────────
quantize model_dir="nvidia/Cosmos-Reason2-2B" output_dir="./quantized/Cosmos-Reason2-2B-fp8" dtype="fp16" quantization="fp8":
    mkdir -p "{{output_dir}}"
    tensorrt-edgellm-quantize-llm \
      --model_dir "{{model_dir}}" \
      --output_dir "{{output_dir}}" \
      --dtype "{{dtype}}" \
      --quantization "{{quantization}}"

export-llm model_dir output_dir:
    mkdir -p "{{output_dir}}"
    tensorrt-edgellm-export-llm \
      --model_dir "{{model_dir}}" \
      --output_dir "{{output_dir}}"

export-visual model_dir output_dir dtype="fp16" quantization="":
    mkdir -p "{{output_dir}}"
    #!/usr/bin/env bash
    CMD=(tensorrt-edgellm-export-visual \
      --model_dir "{{model_dir}}" \
      --output_dir "{{output_dir}}" \
      --dtype "{{dtype}}")
    [ -n "{{quantization}}" ] && CMD+=(--quantization "{{quantization}}")
    "${CMD[@]}"

prep-edge-model model="reason2-2b" out_root="./models/Cosmos-Reason2-2B-fp8":
    just download "{{model}}" "{{out_root}}/hf"
    just quantize "{{out_root}}/hf" "{{out_root}}/quantized" fp16 fp8
    just export-llm "{{out_root}}/quantized" "{{out_root}}/onnx"
    just export-visual "{{out_root}}/hf" "{{out_root}}/onnx/visual_enc_onnx" fp16 fp8
    @echo "✅ ONNX ready → {{out_root}}/onnx  (scp to Thor next)"


# ── TRT engine build (on Thor) ────────────────────────────────────────────
build-llm-engine onnx_dir engine_dir min_tokens="4" max_tokens="10240" max_input_len="1024":
    mkdir -p "{{engine_dir}}"
    "{{LLM_BUILD_BIN}}" \
      --onnxDir "{{onnx_dir}}" \
      --engineDir "{{engine_dir}}" \
      --vlm \
      --minImageTokens {{min_tokens}} \
      --maxImageTokens {{max_tokens}} \
      --maxInputLen {{max_input_len}}

build-visual-engine onnx_dir engine_dir:
    mkdir -p "{{engine_dir}}"
    "{{VISUAL_BUILD_BIN}}" \
      --onnxDir "{{onnx_dir}}" \
      --engineDir "{{engine_dir}}"

build-engines onnx_dir engine_root:
    just build-llm-engine    "{{onnx_dir}}" "{{engine_root}}/llm"
    just build-visual-engine "{{onnx_dir}}/visual_enc_onnx" "{{engine_root}}/visual"


# ── Inference server (on Thor) ────────────────────────────────────────────
serve-start llm_engine_dir visual_engine_dir port=VLM_PORT host=VLM_HOST:
    #!/usr/bin/env bash
    if [ -f "{{PID_FILE}}" ] && kill -0 "$(cat {{PID_FILE}})" 2>/dev/null; then
      echo "🟢 already running (pid=$(cat {{PID_FILE}}))"; exit 0
    fi
    nohup "{{SERVER_BIN}}" \
      --llmEngineDir "{{llm_engine_dir}}" \
      --visualEngineDir "{{visual_engine_dir}}" \
      --host "{{host}}" --port "{{port}}" \
      >> "{{LOG_FILE}}" 2>&1 &
    echo $! > "{{PID_FILE}}"
    sleep 1
    echo "▶ started pid=$(cat {{PID_FILE}})  http://{{host}}:{{port}}"

serve-stop:
    #!/usr/bin/env bash
    if [ ! -f "{{PID_FILE}}" ]; then echo "🔴 not running"; exit 0; fi
    PID=$(cat "{{PID_FILE}}")
    if kill -0 "$PID" 2>/dev/null; then kill "$PID" && echo "⏹ stopped pid=$PID"; fi
    rm -f "{{PID_FILE}}"

serve-status:
    #!/usr/bin/env bash
    if [ -f "{{PID_FILE}}" ] && kill -0 "$(cat {{PID_FILE}})" 2>/dev/null; then
      echo "🟢 running pid=$(cat {{PID_FILE}})  {{VLM_URL}}"
    else
      echo "🔴 not running"; rm -f "{{PID_FILE}}"
    fi

serve-logs lines="80":
    @tail -n {{lines}} "{{LOG_FILE}}" 2>/dev/null || echo "no log yet"

serve-restart llm_engine_dir visual_engine_dir:
    -just serve-stop
    just serve-start "{{llm_engine_dir}}" "{{visual_engine_dir}}"


# ── Inference (HTTP) ──────────────────────────────────────────────────────
infer image prompt="describe the scene" max_tokens="256" temperature="0.2" url=VLM_URL:
    #!/usr/bin/env bash
    IMG_B64=$(base64 < "{{image}}" | tr -d '\n')
    PROMPT='{{prompt}}'
    curl -sS -X POST "{{url}}" \
      -H "Content-Type: application/json" \
      -d @- <<EOF | jq -r '.choices[0].message.content // .'
    {
      "model": "trt-edgellm",
      "messages": [{"role":"user","content":[
        {"type":"text","text":"$PROMPT"},
        {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,$IMG_B64"}}
      ]}],
      "max_tokens": {{max_tokens}},
      "temperature": {{temperature}}
    }
    EOF


# ── RTP capture (GStreamer) ───────────────────────────────────────────────
rtp-capture port=RTP_PORT output="/tmp/cosmos_frame.jpg" width="800" height="600" timeout_s="5":
    #!/usr/bin/env bash
    timeout {{timeout_s}} gst-launch-1.0 -e \
      udpsrc address={{RTP_BIND}} port={{port}} \
        caps='application/x-rtp,media=video,encoding-name=H264,payload=96' ! \
      rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! \
      video/x-raw,width={{width}},height={{height}},format=I420 ! \
      nvjpegenc ! filesink location="{{output}}" || \
    timeout {{timeout_s}} gst-launch-1.0 -e \
      udpsrc address={{RTP_BIND}} port={{port}} \
        caps='application/x-rtp,media=video,encoding-name=H264,payload=96' ! \
      rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videoscale ! \
      video/x-raw,width={{width}},height={{height}} ! jpegenc ! \
      filesink location="{{output}}"
    @ls -la "{{output}}"


# ── NATS publish ──────────────────────────────────────────────────────────
nats-publish subject payload_json:
    #!/usr/bin/env bash
    echo '{{payload_json}}' | nats pub "{{subject}}" --server "{{NATS_URL}}"


# ── Generation (Predict 2.5 / Transfer 2.5) ───────────────────────────────
predict-generate input_json repo=COSMOS_PREDICT_REPO: ensure-predict
    cd "{{repo}}" && just run python examples/inference.py -i "{{input_json}}"

transfer-generate input_json control="edge" repo=COSMOS_TRANSFER_REPO: ensure-transfer
    cd "{{repo}}" && just run python examples/inference.py -i "{{input_json}}" "{{control}}"


# ── Post-training ─────────────────────────────────────────────────────────
post-train-reason2 config strategy="full":
    cosmos-cli train --config "{{config}}" --strategy "{{strategy}}"

post-train-reason2-rl config: ensure-rl
    cosmos-rl --config "{{config}}"

post-train-predict config num_gpus="8" repo=COSMOS_PREDICT_REPO: ensure-predict
    cd "{{repo}}" && torchrun --nproc-per-node={{num_gpus}} -m cosmos_predict2.train --config "{{config}}"

post-train-transfer config num_gpus="8" repo=COSMOS_TRANSFER_REPO: ensure-transfer
    cd "{{repo}}" && torchrun --nproc-per-node={{num_gpus}} -m cosmos_transfer2.train --config "{{config}}"


# ── Distillation ──────────────────────────────────────────────────────────
distill teacher student method="kd" family="transfer2_5" num_gpus="8":
    #!/usr/bin/env bash
    MODULE="cosmos_transfer2.distill"
    [ "{{family}}" = "predict2_5" ] && MODULE="cosmos_predict2.distill"
    torchrun --nproc-per-node={{num_gpus}} -m "$MODULE" \
      --method "{{method}}" \
      --teacher-ckpt "{{teacher}}" \
      --student-output "{{student}}"


# ── Data curation (Cosmos-Xenna) ──────────────────────────────────────────
curate input_dir output_dir="./outputs/curated" stages="all" workers="8" repo=COSMOS_XENNA_REPO: ensure-xenna
    cd "{{repo}}" && just run python -m cosmos_xenna.pipelines.v1.curate \
      --input-dir "{{input_dir}}" --output-dir "{{output_dir}}" \
      --stages "{{stages}}" --workers {{workers}}


# ── Evaluation ────────────────────────────────────────────────────────────
evaluate metric pred gt="" output_dir="./outputs/eval" repo=COSMOS_COOKBOOK_REPO: ensure-cookbook
    #!/usr/bin/env bash
    declare -A MAP=(
      [fid]=scripts/metrics/qualitative/compute_fid.py
      [fvd]=scripts/metrics/qualitative/compute_fvd.py
      [tse]=scripts/metrics/geometrical_consistency/compute_tse.py
      [cse]=scripts/metrics/geometrical_consistency/compute_cse.py
      [sampson]=scripts/metrics/geometrical_consistency/compute_sampson.py
      [blur_ssim]=scripts/metrics/control/compute_blur_ssim.py
      [canny_f1]=scripts/metrics/control/compute_canny_f1.py
      [depth_rmse]=scripts/metrics/control/compute_depth_rmse.py
      [seg_miou]=scripts/metrics/control/compute_seg_miou.py
      [dover]=scripts/metrics/control/compute_dover.py
      [reason_critic]=scripts/evaluation/reason_critic.py
      [reason_reward]=scripts/evaluation/cosmos-reason1-reward-7b/run.py
    )
    SCRIPT="${MAP[{{metric}}]}"
    if [ -z "$SCRIPT" ]; then echo "unknown metric: {{metric}}"; exit 2; fi
    mkdir -p "{{output_dir}}"
    CMD=(python "$SCRIPT" --pred "{{pred}}" --output "{{output_dir}}")
    [ -n "{{gt}}" ] && CMD+=(--gt "{{gt}}")
    cd "{{repo}}" && "${CMD[@]}"


# ── Video / image utils ───────────────────────────────────────────────────
video-probe video:
    ffprobe -v error -print_format json -show_format -show_streams "{{video}}"

video-frames video output_dir="/tmp/frames" fps="1.0" max_frames="0":
    #!/usr/bin/env bash
    mkdir -p "{{output_dir}}"
    CMD=(ffmpeg -y -hide_banner -loglevel warning -i "{{video}}" -vf fps={{fps}})
    [ "{{max_frames}}" != "0" ] && CMD+=(-frames:v {{max_frames}})
    CMD+=("{{output_dir}}/frame_%06d.jpg")
    "${CMD[@]}" 2>&1 || true
    ls "{{output_dir}}" | head -5


# ── System diagnostics ────────────────────────────────────────────────────
sysinfo:
    @echo "--- host ---"
    @hostname && uname -a
    @echo "--- jetson ---"
    @cat /proc/device-tree/model 2>/dev/null || echo "not a Jetson"
    @echo "--- nvidia-smi ---"
    @nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
    @echo "--- memory ---"
    @free -h 2>/dev/null || vm_stat
    @echo "--- thermal ---"
    @for z in /sys/class/thermal/thermal_zone*; do [ -r $z/temp ] && echo "$(cat $z/type 2>/dev/null || basename $z): $(awk '{printf "%.1fC\n", $1/1000}' $z/temp)"; done 2>/dev/null || true


# ── Pipelines (end-to-end) ────────────────────────────────────────────────
pipeline-edge-deploy model="reason2-2b" out_root="./models/Cosmos-Reason2-2B-fp8":
    @echo "🏗  prep on x86 host"
    just prep-edge-model "{{model}}" "{{out_root}}"
    @echo "📤  scp ONNX to Thor, then:"
    @echo "🔨  just build-engines {{out_root}}/onnx {{out_root}}/engines"
    @echo "▶️  just serve-start {{out_root}}/engines/llm {{out_root}}/engines/visual"

pipeline-gr00t-dreams dataset_dir="./datasets/gr1" config="configs/gr00t-dreams.yaml":
    just download-dataset gr1 "{{dataset_dir}}"
    just post-train-predict "{{config}}"

perception-loop subject="perception.vlm" prompt="Describe the scene; count people.":
    #!/usr/bin/env bash
    echo "🔁 perception-loop starting. Ctrl-C to stop."
    while true; do
      FRAME=/tmp/cosmos_perception.jpg
      just rtp-capture {{RTP_PORT}} "$FRAME" 800 600 5 >/dev/null || { sleep 1; continue; }
      RESULT=$(just infer "$FRAME" "{{prompt}}" 128 0.1 2>/dev/null || echo "")
      [ -z "$RESULT" ] && { sleep 1; continue; }
      PAYLOAD=$(printf '{"text":%s,"ts":%d}' "$(printf '%s' "$RESULT" | python3 -c 'import sys,json;print(json.dumps(sys.stdin.read()))')" "$(date +%s)")
      just nats-publish "{{subject}}" "$PAYLOAD" || true
      sleep 0.1
    done

# Smoke test
smoke:
    just env
    just sysinfo
    -just serve-status

# ── Development ───────────────────────────────────────────────────────────
test:
    {{PYTHON}} -m pytest -v tests/

lint:
    {{PYTHON}} -m ruff check strands_cosmos/

format:
    {{PYTHON}} -m ruff format strands_cosmos/
