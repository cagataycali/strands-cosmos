# Cosmos 3 Integration Plan — strands-cosmos

> Branch: `feat/cosmos3-integration`
> Goal: Deep, first-class support for the **entire Cosmos 3 omnimodal world-model family**
> inside strands-cosmos — as Strands model providers + justfile-backed tools.
> Constraint: **No NIM.** Local compute only. We have HF access to the gated repos.
> Slow is fine — we run *everything* end-to-end and test it.

---

## 0. What Cosmos 3 Is (recap)

Cosmos 3 = omnimodal world models in a unified **Mixture-of-Transformers (MoT)**:
jointly process & generate **text, image, video, audio, action**.

Two runtime surfaces:

| Surface       | Inputs                        | Outputs                     |
|---------------|-------------------------------|-----------------------------|
| **Reasoner**  | text, vision                  | text                        |
| **Generator** | text, vision, sound, action   | vision, sound, action       |

Model family:

| Model                          | Size | Role                                   |
|--------------------------------|-----:|----------------------------------------|
| `nvidia/Cosmos3-Nano`          | 16B  | Compact omnimodal (reasoner+generator) |
| `nvidia/Cosmos3-Super`         | 64B  | Frontier-scale omnimodal               |
| `nvidia/Cosmos3-Super-Text2Image`  | 64B | Text→image                        |
| `nvidia/Cosmos3-Super-Image2Video` | 64B | Image→video                       |
| `nvidia/Cosmos3-Nano-Policy-DROID` | 16B | VL robot policy (DROID)           |

---

## 1. Hardware Reality (this host)

- GPU: **1× NVIDIA L40S, 46 GB**
- Driver: **580.126.09 → CUDA 13.0** ⇒ use `cu130` / `vllm==0.21.0` pairing
- ⇒ **Cosmos3-Nano (16B)** fits single-GPU for Reasoner (vLLM) and Generator (Diffusers/vLLM-Omni).
- ⇒ **Cosmos3-Super (64B)** needs TP=4; on 1×L40S only viable via `--enable-layerwise-offload`
  (CPU↔GPU offload, very slow) — acceptable per "slow is fine", but Nano is the default test path.

---

## 2. Available Backends (and which we use)

| Backend          | Surface              | strands-cosmos use                    | Status      |
|------------------|----------------------|---------------------------------------|-------------|
| **Diffusers**    | Generator            | `Cosmos3OmniPipeline` — model provider + tools | ✅ primary gen |
| **vLLM**         | Reasoner             | OpenAI server → `Cosmos3ReasonerModel` provider | ✅ primary reason |
| **vLLM-Omni**    | Generator            | OpenAI server → generation tools      | ✅ secondary gen |
| **Cosmos Framework** | Reasoner + Generator (action) | native `torchrun` — for action / forward-dynamics | ✅ action path |
| **Transformers** | Reasoner             | (upstream "coming soon")              | ⏸ blocked  |
| **NIM**          | Reasoner             | —                                     | ❌ excluded |

Key insight: today the **Transformers reasoner path is NOT available** for Cosmos 3
(unlike Cosmos-Reason2 which uses Qwen3VL directly). So unlike `CosmosVisionModel`
(direct HF), Cosmos 3 Reasoner must go through **vLLM** or **Cosmos Framework**.
Generator goes through **Diffusers** (in-process) or **vLLM-Omni** (server).

---

## 3. Architecture: How Cosmos 3 Maps Into strands-cosmos

We keep the existing design contract:
- **Model providers** implement `strands.models.Model` (`stream()` async generator).
- **Tools** are thin wrappers over **justfile** recipes (justfile = single source of truth).
- Upstream `cosmos` repo lives alongside (we already cloned to `../cosmos`), never forked in.

### 3.1 New Model Providers (`strands_cosmos/`)

| File | Class | Backend | Capability |
|------|-------|---------|------------|
| `cosmos3_reasoner_model.py` | `Cosmos3ReasonerModel` | vLLM OpenAI client | text+vision→text (caption, VQA, temporal, embodied, grounding, plausibility, CoT) |
| `cosmos3_generator_model.py` | `Cosmos3GeneratorModel` | Diffusers `Cosmos3OmniPipeline` (in-proc) | text/img/video/action → image/video/sound |

`Cosmos3ReasonerModel` design:
- Reuses the **media-tag parsing** already in `cosmos_vision_model.py`
  (`<video>...</video>`, `<image>...</image>`) → builds OpenAI `image_url`/`video_url`
  message content (local files become `file://` paths or base64 data URIs).
- Talks to a local vLLM server (auto-startable via justfile recipe / `serve` tool).
- Streams tokens back as Strands `StreamEvent`s.
- Supports the documented sampling presets (with/without `<think>` reasoning).
- `media_io_kwargs` (fps / num_frames) + `mm_processor_kwargs` (resize bounds) passthrough.

`Cosmos3GeneratorModel` design:
- Loads `Cosmos3OmniPipeline` once (lazy, threaded lock like CosmosVisionModel).
- Exposes generation via tool calls (model provider mostly orchestrates); since
  Strands `Model` is text-streaming-centric, generator output (mp4/png) is written
  to disk and returned as a `ContentBlock` image/text path + base64 for small assets.
- Guardrails toggle (`cosmos_guardrail`) exposed; default ON, overridable.

### 3.2 New Tools (`strands_cosmos/tools/`)

All are thin wrappers over new justfile recipes.

**Reasoner (text out):**
| Tool | Recipe | Task |
|------|--------|------|
| `cosmos3_reason` | `c3-reason` | Generic reasoner: prompt + image/video → text. Sub-modes via `task=` |
| `cosmos3_caption` | `c3-reason task=caption` | Detailed video/image captioning |
| `cosmos3_temporal` | `c3-reason task=temporal` | Event detection + timestamps (text/JSON) |
| `cosmos3_embodied` | `c3-reason task=embodied` | Next-action prediction (robotics) |
| `cosmos3_ground` | `c3-reason task=grounding` | 2D bounding boxes (JSON) |
| `cosmos3_plausibility` | `c3-reason task=plausibility` | Physical plausibility label |
| `cosmos3_situation` | `c3-reason task=situation` | Situation understanding + next action |
| `cosmos3_action_cot` | `c3-reason task=action_cot` | Trajectory / driving CoT |

**Generator (media out):**
| Tool | Recipe | Task |
|------|--------|------|
| `cosmos3_text2image` | `c3-gen mode=text2image` | Text → PNG |
| `cosmos3_text2video` | `c3-gen mode=text2video` | Text → MP4 (189f/24fps default) |
| `cosmos3_text2video_sound` | `c3-gen mode=text2video-with-sound` | Text → MP4+AAC |
| `cosmos3_image2video` | `c3-gen mode=image2video` | Image+text → MP4 |
| `cosmos3_image2video_sound` | `c3-gen mode=image2video-with-sound` | Image+text → MP4+AAC |
| `cosmos3_video2video` | `c3-gen mode=video2video` | Video+text → MP4 (frame-conditioned) |

**Action / World-Model (Cosmos Framework, torchrun):**
| Tool | Recipe | Task |
|------|--------|------|
| `cosmos3_forward_dynamics` | `c3-action mode=forward_dynamics` | start image + action chunk → future video |
| `cosmos3_inverse_dynamics` | `c3-action mode=inverse_dynamics` | video + instruction → predicted action chunk |
| `cosmos3_policy` | `c3-action mode=policy` | image + instruction → action chunk + rollout video |

**Server lifecycle:**
| Tool | Recipe | Task |
|------|--------|------|
| `cosmos3_serve` | `c3-serve-reason` / `c3-serve-omni` | start/stop/status local vLLM(+omni) servers |

### 3.3 New justfile recipes

Mirror the existing edge/predict/transfer recipe style. Add a `packages/`-based
env bootstrap (the cookbooks clone `cosmos-framework` into `packages/cosmos3`):

```
# Cosmos 3 environment (uv-managed, cu130 to match driver)
c3-setup-reason        # uv venv + vllm==0.21.0 + vllm-cosmos3   (Reasoner server)
c3-setup-gen           # uv venv + diffusers(main) + cosmos_guardrail (Generator)
c3-setup-omni          # uv venv + vllm-omni PR branch OR docker image (Generator server)
c3-setup-framework     # clone cosmos-framework → packages/cosmos3, uv sync cu130-train (Action)
c3-doctor              # report which c3 backends are installed & GPU fit

# Reasoner
c3-serve-reason model="nvidia/Cosmos3-Nano" port="8000" tp="1"
c3-serve-stop-reason
c3-reason prompt image="" video="" task="caption" model="..." max_tokens="4096" think="false"

# Generator (Diffusers, in-proc)
c3-gen mode="text2video" prompt="" image="" video="" out="/tmp/c3_out" \
       frames="189" fps="24" steps="35" guidance="6.0" res="720" ar="16,9" \
       sound="false" guardrails="true" seed="0"

# Generator server (vLLM-Omni)
c3-serve-omni model="nvidia/Cosmos3-Nano" port="8001"
c3-serve-stop-omni

# Action (Cosmos Framework, torchrun)
c3-action mode="forward_dynamics" input_json="" out="/tmp/c3_action" \
          checkpoint="Cosmos3-Nano" seed="0"
```

### 3.4 Package wiring
- `strands_cosmos/__init__.py`: export `Cosmos3ReasonerModel`, `Cosmos3GeneratorModel`
  + all `cosmos3_*` tools; bump version `0.2.0 → 0.3.0`.
- `tools/__init__.py`: register new tools + extend `__all__`.
- `tools/_common.py`: add helpers for (a) building OpenAI multimodal messages from
  media paths, (b) polling a local vLLM `/health`, (c) base64 media data URIs.

---

## 4. Capability Coverage Matrix (what "support everything" means)

| # | Cosmos 3 capability       | Surface   | strands-cosmos artifact            | Backend        |
|---|---------------------------|-----------|------------------------------------|----------------|
| 1 | Video caption             | Reasoner  | `cosmos3_caption` / model provider | vLLM           |
| 2 | Temporal localization     | Reasoner  | `cosmos3_temporal`                 | vLLM           |
| 3 | Embodied next-action      | Reasoner  | `cosmos3_embodied`                 | vLLM           |
| 4 | Common-sense / plausibility | Reasoner | `cosmos3_plausibility`            | vLLM           |
| 5 | 2D grounding (boxes)      | Reasoner  | `cosmos3_ground`                   | vLLM           |
| 6 | Describe-anything         | Reasoner  | `cosmos3_reason task=describe`     | vLLM           |
| 7 | Action CoT / driving      | Reasoner  | `cosmos3_action_cot`               | vLLM           |
| 8 | Situation understanding   | Reasoner  | `cosmos3_situation`                | vLLM           |
| 9 | Text→image                | Generator | `cosmos3_text2image`               | Diffusers      |
|10 | Text→video                | Generator | `cosmos3_text2video`               | Diffusers      |
|11 | Text→video+sound          | Generator | `cosmos3_text2video_sound`         | Diffusers/Omni |
|12 | Image→video               | Generator | `cosmos3_image2video`              | Diffusers      |
|13 | Image→video+sound         | Generator | `cosmos3_image2video_sound`        | Omni           |
|14 | Video→video               | Generator | `cosmos3_video2video`              | Omni           |
|15 | Forward dynamics          | Generator | `cosmos3_forward_dynamics`         | Framework      |
|16 | Inverse dynamics          | Generator | `cosmos3_inverse_dynamics`         | Framework      |
|17 | Action policy (DROID)     | Generator | `cosmos3_policy`                   | Framework/Omni |

---

## 5. Phased Execution (test-as-we-go)

### Phase 0 — Scaffolding (no GPU)
- [x] Branch `feat/cosmos3-integration`
- [x] This plan committed
- [x] Add `c3-doctor` + `c3-setup-*` justfile recipes
- [x] Model providers + tools (import-safe)
- [x] `tests/test_cosmos3_imports.py` — 5 tests green
- **Gate:** `just test` green, `python -c "import strands_cosmos"` works

### Phase 1 — Reasoner via vLLM (Cosmos3-Nano on L40S)
- [x] `c3-setup-reason` → vllm 0.21.0 + vllm-cosmos3 (cu130)
- [x] `c3-serve-reason` Nano single-GPU (--max-model-len 32768)
- [x] `Cosmos3ReasonerModel` caption verified (6.6s)
- [x] Reasoner suite verified (caption/temporal/embodied/plausibility/situation)
- **Gate:** caption matches/improves on Reason2-2B baseline; tools return valid output

### Phase 2 — Generator via Diffusers (Cosmos3-Nano)
- [x] `c3-setup-gen` → diffusers 0.39.0.dev0 (Cosmos3OmniPipeline)
- [x] text→image verified (33.3s, 256p)
- [x] text→video verified (19.3s, valid H264)
- [x] image→video verified (19.1s)
- **Gate:** valid PNG + MP4 written, openable, guardrails toggle works

### Phase 3 — Generator+Sound & Video2Video via vLLM-Omni
- [x] SOUND done in-proc via Diffusers (vLLM-Omni now optional)
- [x] text→video+SOUND verified (AAC stereo 48kHz, has_audio=True)
- **Gate:** MP4 with AAC audio stream (ffprobe confirms audio track)

### Phase 4 — Action / World-Model via Cosmos Framework
- [x] `c3-setup-framework` → cosmos-framework, uv sync cu130-train
- [x] forward dynamics verified (AV, 832x480 61f rollout, 31s)
- [ ] inverse dynamics (recipe ready, same path; smoke pending)
- [ ] policy DROID (recipe ready; smoke pending)
- **Gate:** action JSON + rollout video produced from cookbook assets

### Phase 5 — Examples, Docs, CI
- [x] examples 06 (reason), 07 (generate), 08 (action)
- [ ] docs pages under `docs/guide/cosmos3.md`; update README tool table + model list
- [ ] AGENTS.md: append Cosmos 3 lane + learnings
- [ ] expand pytest; add (CPU-only import) CI gate
- **Gate:** docs build, examples run on this host, `just smoke` covers c3

---

## 6. Risks & Mitigations
- **vLLM/CUDA pairing**: locked to `cu130 + vllm==0.21.0` (driver 13.0). Documented in `c3-doctor`.
- **64B Super on 1×L40S**: only via layerwise offload (slow). Default tests use Nano; Super behind a flag.
- **Diffusers `Cosmos3OmniPipeline`** is on diffusers *main* (git), API may drift → pin a commit in `c3-setup-gen`.
- **vLLM-Omni** not fully upstreamed → prefer docker `vllm/vllm-omni:cosmos3` for full-modality tests.
- **HF gated repos**: ensure `HF_TOKEN` / `hf auth login` before any download recipe; `c3-doctor` checks auth.
- **Disk**: Nano+Super+caches = tens of GiB; `c3-doctor` reports free space and sets `HF_HOME` guidance.

---

## 7. Decisions Locked
1. NIM: **excluded**.
2. Default model: **Cosmos3-Nano (16B)** — fits L40S; Super opt-in.
3. Reasoner backend: **vLLM** (Transformers path unavailable upstream).
4. Generator backend: **Diffusers** in-proc primary; **vLLM-Omni** for sound/v2v/action-server.
5. Action backend: **Cosmos Framework** (`torchrun`).
6. Contract preserved: providers = `strands.models.Model`; tools = thin justfile wrappers.
7. Upstream `cosmos` repo stays external (`../cosmos`), pulled by `c3-setup-framework`.

---

## 8. Open Questions (to confirm while testing)
- Does `vllm-cosmos3` plugin build cleanly under cu130 on this exact driver?
- Diffusers `Cosmos3OmniPipeline` peak VRAM for Nano text2video @720p/189f on 46 GB? (may need 480p/256p)
- Can we run video2video & sound purely in-proc (Diffusers) or is vLLM-Omni mandatory?
- Policy-DROID checkpoint: Framework-only, or also exposed via vLLM-Omni action endpoint?
