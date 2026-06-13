# strands-cosmos — AGENTS.md

**Living dev contract for any agent (human, Claude, GPT, Gemini) working on strands-cosmos.**

---

## The 30-Second Pitch

**strands-cosmos = Full-lifecycle NVIDIA Cosmos toolkit for Strands Agents.**

- **Model Providers**: `CosmosModel` (text-only) + `CosmosVisionModel` (video+image+text) using Cosmos-Reason2 via HF Transformers
- **21 Tools**: Inference, video generation (Predict2.5), video-to-video (Transfer2.5), data curation (Xenna), post-training, distillation, quantization, edge deployment, evaluation
- **Edge-first**: Verified on Jetson AGX Thor (132GB), Orin, plus cloud GPUs
- **justfile as truth**: All pipeline commands flow through `just <recipe>`; tools are thin Python wrappers

---

## Core Principles

1. **RUN FIRST** — `pip install strands-cosmos` → `from strands_cosmos import CosmosVisionModel` works immediately.
2. **JUSTFILE IS TRUTH** — Every tool calls a `just` recipe. Change the pipeline? Edit the justfile.
3. **UPSTREAM UNTOUCHED** — Cosmos repos live alongside (`../cosmos-*`), never forked into this repo.
4. **EDGE + CLOUD** — Same code, different `just doctor` outputs. Graceful degradation when TRT unavailable.
5. **STRANDS NATIVE** — Model providers implement `strands.models.Model`; tools use `@tool` decorator.

---

## Repo Layout

```
strands-cosmos/
├── AGENTS.md                     # this file
├── README.md                     # install + quickstart + tool table
├── pyproject.toml                # v0.2.0, Apache-2.0
├── justfile                      # 50+ recipes (THE pipeline truth)
├── mkdocs.yml                    # GitHub Pages documentation site
├── strands_cosmos/               # core package
│   ├── __init__.py               # exports: 2 models + 21 tools
│   ├── cosmos_model.py           # CosmosModel (text-only Strands Model)
│   ├── cosmos_vision_model.py    # CosmosVisionModel (video+image+text)
│   ├── fix_cublas.py             # Jetson CUBLAS compatibility fix (CLI entry)
│   └── tools/                    # 21 tools (thin wrappers over justfile)
│       ├── __init__.py           # tool registry
│       ├── _common.py            # shared helpers (justfile runner)
│       ├── inference.py          # TRT server inference
│       ├── reason_hf.py          # HF Transformers direct inference
│       ├── serve.py              # TRT server lifecycle
│       ├── predict_generate.py   # Predict2.5 world model generation
│       ├── transfer_generate.py  # Transfer2.5 ControlNet video-to-video
│       ├── model_download.py     # HF model download
│       ├── quantize.py           # FP8 quantization
│       ├── export_onnx.py        # ONNX export
│       ├── build_engine.py       # TRT engine build
│       ├── post_train.py         # SFT/LoRA post-training
│       ├── distill.py            # Knowledge distillation
│       ├── curate.py             # Xenna data curation
│       ├── evaluate.py           # FID/FVD/CSE/CLIP benchmarks
│       ├── rtp.py                # GStreamer RTP frame capture
│       ├── nats_pub.py           # NATS messaging
│       ├── video_utils.py        # ffprobe + frame extraction
│       ├── image_read.py         # Base64 image read
│       ├── sysinfo.py            # GPU/platform diagnostics
│       ├── cosmos_invoke.py      # Legacy direct inference
│       └── cosmos_vision_invoke.py  # Legacy vision inference
├── examples/                     # 5 runnable examples
│   ├── 01_basic_text.py
│   ├── 02_video_caption.py
│   ├── 03_driving_analysis.py
│   ├── 04_embodied_reasoning.py
│   └── 05_tool_usage.py
├── tests/                        # pytest suite
│   ├── __init__.py
│   └── test_imports.py
├── docs/                         # MkDocs Material site
│   ├── index.md
│   ├── architecture.md
│   ├── api-reference.md
│   ├── getting-started/
│   ├── guide/
│   └── examples/
├── demo/                         # Demo GIF/video assets
├── LICENSE                       # Apache 2.0
└── sample.{mp4,png}             # Test media files
```

---

## Hardware & Platforms

| Platform | GPU | Primary Use |
|----------|-----|-------------|
| **Jetson AGX Thor** | 132GB unified | Edge deployment: TRT engines, serve, RTP capture |
| **Jetson Orin** | 32/64GB | Edge deployment (smaller models) |
| **Desktop/Cloud** | A100/H100/RTX 4090 | Training, quantization, ONNX export, HF inference |
| **Any machine** | CPU-only | Development, testing, documentation |

`just doctor` reveals what works on the current host.

---

## Dependencies Policy

**Core** (always installed): `strands-agents`, `transformers`, `accelerate`, `torch`, `torchvision`, `qwen-vl-utils`, `pillow`, `pyyaml`, `av`

**Optional extras**:
- `[vllm]` — vLLM + OpenAI client
- `[jetson]` — torchcodec companion
- `[dev]` — pytest, ruff
- `[all]` — everything

**External Cosmos repos** (cloned alongside, NOT bundled):
- `cosmos-predict2.5`, `cosmos-transfer2.5`, `cosmos-reason2`
- `cosmos-xenna` (data curation), `cosmos-rl`, `cosmos-cookbook`

Run `just setup` to auto-clone all six.

---

## Tool Architecture

Tools are **thin Python wrappers** over justfile recipes:

```
Agent calls tool → tool runs `just <recipe> <args>` → justfile executes pipeline
```

This means:
- **Operators can bypass Python** and run `just quantize ...` directly
- **Justfile is the single source of truth** for all CLI commands
- **Tools gracefully fail** with exit 127 when TRT binaries aren't available (expected on workstations)

---

## Key Workflows

### 1. HF Direct Inference (any GPU)
```bash
pip install strands-cosmos
python -c "
from strands import Agent
from strands_cosmos import CosmosVisionModel
agent = Agent(model=CosmosVisionModel())
agent('<video>sample.mp4</video> What is happening?')
"
```

### 2. Edge Deployment Pipeline (Jetson Thor)
```bash
just prep-edge-model reason2-2b ./models/reason2-2b-fp8
# scp ONNX to Thor, then on Thor:
just build-engines ./models/reason2-2b-fp8/onnx ./models/reason2-2b-fp8/engines
just serve-start ./models/reason2-2b-fp8/engines/llm ./models/reason2-2b-fp8/engines/visual
just infer /tmp/frame.jpg "describe the scene"
```

### 3. Perception Loop (RTP → VLM → NATS)
```bash
just perception-loop perception.vlm "Describe any safety hazards."
```

### 4. World Model Generation (Predict2.5)
```bash
just predict-generate configs/predict_config.json
```

---

## Development

```bash
git clone https://github.com/cagataycali/strands-cosmos && cd strands-cosmos
pip install -e ".[dev]"
just test           # pytest
just lint           # ruff check
just format         # ruff format
just doctor         # verify environment
just smoke          # quick sanity check
```

---

## Multi-Agent Coordination (Zenoh peers)

When multiple agents work on this repo concurrently:

1. `git fetch origin main` BEFORE starting any edit.
2. Broadcast your lane: `zenoh_peer(action='broadcast', message='[claim] <what you're doing>')`.
3. Wait 30s for collisions. Silence → proceed.
4. Commit atomically: `[<agent-id>] <area>: <change>`.
5. **Append-only to AGENTS.md** — never rewrite another agent's log entries.

### Lane Ownership

| Lane | Owner | Scope |
|------|-------|-------|
| Model providers | — | `cosmos_model.py`, `cosmos_vision_model.py` |
| Tools | — | `strands_cosmos/tools/*.py` |
| Justfile | — | `justfile` recipes |
| Docs | — | `docs/`, `mkdocs.yml` |
| Examples | — | `examples/*.py` |
| Tests | — | `tests/` |
| CI/Release | — | `pyproject.toml`, GitHub Actions |

*(Claim lanes in the Context Learning Log below)*

---

## Current Status

| Area | State | Notes |
|------|-------|-------|
| CosmosVisionModel | ✅ stable | Qwen3VL-based, video+image+text |
| CosmosModel | ✅ stable | Text-only provider |
| 21 Tools | ✅ shipped | All justfile-backed |
| Jetson Thor | ✅ verified | CUBLAS fix, TRT pipeline tested |
| PyPI | ✅ v0.2.0 | `pip install strands-cosmos` |
| Docs site | ✅ live | cagataycali.github.io/strands-cosmos |
| Tests | 🟡 minimal | Import tests only — needs expansion |
| CI | 🔴 missing | No GitHub Actions yet |

---

## What Needs Work

1. **Test expansion** — Unit tests for each tool, model provider edge cases, video processing
2. **CI pipeline** — GitHub Actions for lint + test + publish
3. **vLLM backend** — Alternative to TRT for cloud deployment
4. **Streaming improvements** — True token-level latency metrics
5. **Multi-GPU** — Tensor parallel for 8B model on multi-Orin setups
6. **Evaluation integration** — Run `cosmos_evaluate` results back into agent context

---

## Append-Only Context Learning Log

> New entries go AT THE TOP.

### 2026-06-13 — v0.7.0 security release (20→2 Dependabot alerts)

Bumped the dependency stack to clear 18 of 20 open Dependabot alerts on
strands-labs/strands-for-cosmos, then released v0.7.0 to both PyPI names.

**Bumps (uv.lock):** vllm 0.11.0→0.22.1, torch 2.8.0→2.11.0,
xgrammar 0.1.25→0.2.2, transformers 5.2.0→5.12.0.

**The non-obvious blocker:** `uv lock --upgrade-package vllm` did NOTHING —
vllm was frozen at 0.11.0 by a transitive knot:
  - vllm 0.11.0 HARD-PINS `torch==2.8.0` + xgrammar (so torch/xgrammar can't
    move until vllm does)
  - vllm >=0.19.1 requires `transformers>5.5.0`; vllm <0.19.1 requires
    `transformers<5`; cosmos-guardrail >=0.3 requires `transformers>=5.0`
  - our base pin was `transformers<5.3.0` → squeezed into an empty set.
**Fix:** relax the transformers cap to exclude only the band vllm forbids:
  `transformers>=4.57.0,!=5.0.*,!=5.1.*,!=5.2.*,!=5.3.*,!=5.4.*,!=5.5.0,<6.0.0`
  (matches vllm 0.22's own marker). Then bump vllm floor to 0.22.0 in the
  `vllm` and `all` extras. Clean resolve at transformers 5.12.0. Import tests 4/4.

**Cleared:** CRITICAL vllm video-RCE (#9), all 8 vllm highs (SSRF, auto_map RCE,
trust-override RCE, deserialization, shape DoS, CVE-2025-62164), all vllm
mediums, xgrammar DoS high (#11), torch memory-corruption (#18, #20).

**Remaining 2 (NO upstream patch exists — confirmed latest on PyPI):**
  - diskcache #10 (5.6.3 is latest; unsafe pickle; transitive via vllm)
  - torch #19 (jit.script; all <=2.12.0 affected; no fix released)
Both low/medium. Accept-risk until upstream ships.

**Release mechanics (dual-package, both from the for-cosmos tagged tree):**
  - module stays `strands_cosmos` for BOTH dist names (matches 0.6.0 layout)
  - build for-cosmos by `sed name = "strands-cosmos" -> "strands-for-cosmos"`
    on a clean copy + `SETUPTOOLS_SCM_PRETEND_VERSION=0.7.0` (dirty tree makes
    setuptools-scm emit 0.7.1.dev0 — always build from a clean/.git-less copy)
  - force-push to strands-labs/strands-for-cosmos via `--force-with-lease`,
    `git commit --amend` preserving the poetic HEAD message verbatim
    ("Strands for Cosmos — a verse for the machine that dreams in motion").
    NOTE: for-cosmos has CONTRIBUTING.md + CODE_OF_CONDUCT.md NOT in the
    cagataycali repo — rsync --delete will drop them; preserve explicitly.
  - twine upload both → pypi.org/project/strands-cosmos/0.7.0 + strands-for-cosmos/0.7.0
  - ~/.pypirc holds the token. Dependabot re-scanned within seconds → 20→2.

### 2026-06-04 — Cosmos 3 Phase 4 DONE (Action world-model via Cosmos Framework)

**Phase 4 GATE PASSED.** Forward dynamics verified end-to-end.

- `c3-setup-framework` FAST (FRAMEWORK_EXIT=0): clones cosmos-framework →
  cosmos/packages/cosmos3, `uv sync --all-extras --group=cu130-train`. venv at
  packages/cosmos3/.venv. `import cosmos_framework` OK.
- Forward dynamics (AV, av_traj_forward.json + av_0.jpg start frame):
  `python -m cosmos_framework.scripts.inference --parallelism-preset=latency
   -i spec.jsonl -o out --checkpoint-path Cosmos3-Nano --seed=0`.
  Loaded OmniMoTModel (Wan2.2 VAE + AVAE audio + action_gen), 30-step UniPC
  sampling in 31s → **/tmp/c3_action_out/av_forward/vision.mp4: 832x480, 61
  frames, H264, 7.5MB**. FD_EXIT=0. GPU 44.5GB.
- Input is a **JSONL spec** (one line per run): model_mode (forward_dynamics|
  inverse_dynamics|policy), vision_path, action_path, domain_name (av|
  bridge_orig_lerobot|...), action_chunk_size, fps, image_size, view_point,
  prompt, name, seed. Framework auto-downloads checkpoints (Cosmos3-Nano + Wan2.2
  VAE + AVAE) on first run.
- Action is NOT in the Diffusers Cosmos3OmniPipeline.__call__ (only video/sound).
  Confirmed: action requires Cosmos Framework. Updated `c3-action` recipe +
  cosmos3_forward_dynamics/inverse_dynamics/policy tools to take `input_jsonl`.
- Example 08_cosmos3_action.py added.

**ALL 4 capability surfaces now verified on local L40S (no NIM):**
1. Reasoner (caption/temporal/embodied/plausibility/situation) — vLLM ✅
2. Generator image/video — Diffusers ✅
3. Generator video+SOUND (AAC stereo 48kHz) — Diffusers in-proc ✅
4. Action forward-dynamics (world-model rollout) — Cosmos Framework ✅

**Remaining (optional polish):** inverse_dynamics + policy smoke (same recipe,
different spec), vLLM-Omni video2video (optional, in-proc covers most), docs site
pages, CI. Core integration COMPLETE.


### 2026-06-04 — Cosmos 3 Phase 2b + Phase 3 DONE (video + SOUND in-proc!)

**text2video PASS:** 29f @ 256p, 15 steps, **19.3s** → valid H264 320x192 MP4.

**text2video-with-SOUND PASS (Phase 3 done in-proc, NO vLLM-Omni needed!):**
- `Cosmos3OmniPipelineOutput` has BOTH `.video` (list of frames) AND `.sound`
  (torch.Tensor, stereo). Diffusers does sound generation in-process.
- Patched Cosmos3GeneratorModel.generate() + added `_mux_audio()`: writes silent
  video, writes sound tensor to WAV (soundfile), muxes via ffmpeg → AAC.
- Result MP4: **H264 video + AAC stereo @ 48kHz** (`has_audio: True`), 19.5s.
  Matches Cosmos 3 spec (stereo AAC 48kHz) exactly.
- Gen venv needs: soundfile (added). ffmpeg (system, present).

**Architecture update:** vLLM-Omni is now OPTIONAL — in-proc Diffusers covers
text2image/video/video+sound. vLLM-Omni only needed for video2video + the
OpenAI-server generation API (Phase 3b, lower priority). Action still needs
Cosmos Framework (Phase 4).

**Next:** Phase 4 (action: forward/inverse dynamics + policy — Cosmos3-Nano repo
ships example assets: example_action_fd_agibotworld_*, example_action_id_av_*).
Then Phase 5 (examples 06-12, docs, README tool table, CI).


### 2026-06-04 — Cosmos 3 Phase 2 DONE (Generator via Diffusers verified)

**Phase 2 GATE PASSED.** Cosmos3GeneratorModel text2image works in-proc.

- `c3-setup-gen` built .venv-c3-gen: diffusers 0.39.0.dev0 (main) has
  `Cosmos3OmniPipeline`. Also needs strands-agents+pydantic in gen venv
  (provider imports full strands_cosmos package).
- text2image smoke (256p, 10 steps): 93KB PNG in **33.3s** (incl 16B load),
  diffusion ~7.9 it/s. PASS.
- Pipeline load confirms **omnimodal single checkpoint**: Cosmos3OmniTransformer
  + AutoencoderKLWan (video VAE) + Cosmos3AVAEAudioTokenizer (audio) +
  action_proj layers (action modality). One Nano = text/img/video/audio/action gen.

**CRITICAL hardware finding — single L40S (46GB) can't run reasoner + generator
simultaneously.** vLLM reasoner holds ~40GB (weights+KV cache); Diffusers
generator needs ~16GB+ for the same 16B → OOM. **Must stop one before the other.**
For production: dedicate GPUs, or run reasoner OR generator per session. Documented
in COSMOS3_INTEGRATION.md risks. The two providers are correct; it's a memory
scheduling constraint, not a code bug.

**Verified reasoner suite (all via vLLM):** caption (6.6s), temporal (timestamps),
embodied (with <think>), plausibility (label), situation (next-action). All good.

**Daemonization for long GPU jobs:** /tmp/daemon_*.py double-fork pattern is the
only reliable way to survive harness timeouts. Logs: /tmp/c3reason2.log,
/tmp/c3gentest.log, /tmp/c3setupgen.log.

**Next:** Phase 3 (omni sound/v2v — needs vLLM-Omni or in-proc Diffusers sound),
Phase 4 (action: forward/inverse dynamics, policy via assets in Cosmos3-Nano repo),
Phase 5 (examples 06-12, docs, README, CI).


### 2026-06-04 — Cosmos 3 Phase 1 DONE (Reasoner via vLLM verified)

**Phase 1 GATE PASSED.** Cosmos3-Nano reasoner live on local vLLM (no NIM).

- `just c3-setup-reason` succeeded: vllm==0.21.0 + vllm-cosmos3==0.1.0 (cu130).
  Plugin registers `Cosmos3ReasonerForConditionalGeneration`. (~7GB uv cache dl.)
- Server: Cosmos3-Nano (16B) on L40S, single GPU, **--max-model-len 32768**
  (default 262144 needs 36GB KV cache > 21GB free → OOM; capped fixes it).
  `--gpu-memory-utilization 0.92`. Ready in ~2-3min (weights+torch.compile cached).
  Uses 41.9GB VRAM. justfile c3-serve-reason updated with max_len + gpu-mem-util params.
- `Cosmos3ReasonerModel` caption of strands-cosmos-demo-video.mp4: **6.6s**,
  accurate (front-end loader, church steeple, glass facade, urban setting).
  Beats Reason2-2B baseline. `just c3-reason ... temporal` also works (timestamps).

**Key learnings:**
- System python3.12 needs `pip install --break-system-packages openai` for the
  reasoner provider (separate from the venv).
- Backgrounding on this host: harness SIGKILLs commands with trailing `sleep`,
  taking child procs with them. Use a **double-fork daemon** (os.fork/setsid/fork,
  dup2 log fd) launched in a NO-sleep command → fully orphaned, survives. Pattern
  saved at /tmp/daemonize.py + /tmp/launch_c3_reason.sh.
- vllm "[ERROR] ... is part of ... signature but not documented" lines are
  harmless transformers docstring warnings, not failures.
- Server start logs: /tmp/c3reason.log. Health: curl localhost:8000/health.

**Next:** Phase 2 (Generator/Diffusers: c3-setup-gen, text2image/video), then
smoke remaining reasoner tools, Phase 3 (omni sound/v2v), Phase 4 (action), Phase 5.


### 2026-06-04 — Cosmos 3 integration (branch: feat/cosmos3-integration)

Started deep Cosmos 3 omnimodal support. **No NIM** — local compute only
(1×L40S 46GB, driver 580 → CUDA 13.0 → cu130 + vllm==0.21.0 pairing locked).

**Phase 0 DONE** (committed):
- `Cosmos3ReasonerModel` (vLLM OpenAI backend) — strands_cosmos/cosmos3_reasoner_model.py
  Reasoner path: text+vision→text. Transformers reasoner is "coming soon" upstream,
  so we go via local vLLM serving `Cosmos3ReasonerForConditionalGeneration`.
- `Cosmos3GeneratorModel` (Diffusers `Cosmos3OmniPipeline`, in-proc) — generator.
- 16 `cosmos3_*` tools (thin justfile wrappers): reason/caption/temporal/embodied/
  ground/plausibility/situation/action_cot + text2image/text2video/image2video/
  text2video_sound + forward_dynamics/inverse_dynamics/policy + serve.
- justfile c3-* recipes: c3-doctor, c3-setup-{reason,gen,omni,framework},
  c3-serve-{reason,omni,status,stop-*}, c3-reason, c3-gen, c3-action.
- tests/test_cosmos3_imports.py green (3/3). All 16 tools + 2 providers exported.
- `just` 1.51.0 installed at /tmp/justbin (symlinked ~/.local/bin/just). c3-doctor verified.

**Next (autonomous ambient driving):**
- Phase 1: `just c3-setup-reason` (uv venv, vllm 0.21.0 + vllm-cosmos3, cu130),
  `just c3-serve-reason`, then caption ../strands-cosmos-demo-video.mp4 via
  Cosmos3ReasonerModel; smoke all 8 reasoner tools. Needs HF_TOKEN/`hf auth login`.
- Phase 2: `just c3-setup-gen` (diffusers main), text2image→text2video→image2video.
- Phase 3: vLLM-Omni sound + video2video.  Phase 4: Cosmos Framework action.
- Phase 5: examples 06-12, docs, README/AGENTS update, CI.

**Gotchas learned:**
- Cosmos-Reason2 path needs transformers>=4.57 (Qwen3VL) + torchcodec==0.2.1
  (matches torch 2.6 + system FFmpeg 6 / libavutil.so.58). 0.10 fails to load.
- Use `just_run`/`proc_result` from tools/_common.py (NOT a `run_just` helper).
- pip needs `--break-system-packages` on this externally-managed host.


### 2026-05-08 — AGENTS.md created (initial)

First AGENTS.md generated from full repo audit. Package at v0.2.0 on PyPI.
21 tools, 2 model providers, justfile with 50+ recipes, MkDocs site live.
Verified platforms: Jetson AGX Thor, desktop GPU, CPU-only dev. Key gap:
test coverage is minimal (import-only) and no CI exists yet.

---
