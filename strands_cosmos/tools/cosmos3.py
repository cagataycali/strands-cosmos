"""Cosmos 3 tools — thin wrappers over justfile `c3-*` recipes.

Reasoner (text out, via local vLLM server):
    cosmos3_reason, cosmos3_caption, cosmos3_temporal, cosmos3_embodied,
    cosmos3_ground, cosmos3_plausibility, cosmos3_situation, cosmos3_action_cot
Generator (media out, via in-proc Diffusers):
    cosmos3_text2image, cosmos3_text2video, cosmos3_image2video,
    cosmos3_text2video_sound
Action (world-model, via Cosmos Framework torchrun):
    cosmos3_forward_dynamics, cosmos3_inverse_dynamics, cosmos3_policy
Server lifecycle:
    cosmos3_serve

All recipes live in the top-level justfile (single source of truth).
"""
from __future__ import annotations

from strands import tool

from ._common import err, just_run, ok, proc_result

# Long timeouts: video gen + model load can take many minutes.
_GEN_TIMEOUT = 60 * 60        # 1h for generation
_REASON_TIMEOUT = 60 * 20     # 20m for reasoning (includes first-call warmup)
_ACTION_TIMEOUT = 60 * 60     # 1h for action rollouts


# ----- Reasoner -----------------------------------------------------------
@tool
def cosmos3_reason(
    prompt: str,
    image: str = "",
    video: str = "",
    task: str = "",
    port: int = 8000,
    max_tokens: int = 4096,
    think: bool = False,
) -> dict:
    """Cosmos 3 Reasoner: text+vision -> text via local vLLM server.

    Requires a running reasoner server (`just c3-serve-reason` / cosmos3_serve).

    Args:
        prompt: User instruction.
        image: Image path or URL (optional).
        video: Video path or URL (optional).
        task: Optional built-in task hint (caption/temporal/embodied/...).
        port: vLLM server port.
        max_tokens: Output token cap.
        think: Enable explicit reasoning format.
    """
    proc = just_run("c3-reason", prompt, image, video, task, str(port),
                    str(max_tokens), "true" if think else "false",
                    timeout_s=_REASON_TIMEOUT)
    return proc_result(proc, success_text="cosmos3 reason result:",
                       fail_text=f"c3-reason failed: {proc.get('stderr','')[:200]}")


@tool
def cosmos3_caption(video: str = "", image: str = "", port: int = 8000, max_tokens: int = 4096) -> dict:
    """Detailed Cosmos 3 caption of a video or image."""
    proc = just_run("c3-reason", "Caption in detail.", image, video, "caption",
                    str(port), str(max_tokens), "false", timeout_s=_REASON_TIMEOUT)
    return proc_result(proc, "cosmos3 caption:", "c3 caption failed")


@tool
def cosmos3_temporal(video: str, port: int = 8000, max_tokens: int = 2048) -> dict:
    """Temporal localization: list notable events with approximate timestamps."""
    proc = just_run("c3-reason", "List the notable events with approximate timestamps.",
                    "", video, "temporal", str(port), str(max_tokens), "false",
                    timeout_s=_REASON_TIMEOUT)
    return proc_result(proc, "cosmos3 temporal:", "c3 temporal failed")


@tool
def cosmos3_embodied(video: str = "", image: str = "", port: int = 8000, max_tokens: int = 1024) -> dict:
    """Embodied reasoning: predict the next immediate action."""
    proc = just_run("c3-reason", "What can be the next immediate action?",
                    image, video, "embodied", str(port), str(max_tokens), "true",
                    timeout_s=_REASON_TIMEOUT)
    return proc_result(proc, "cosmos3 embodied:", "c3 embodied failed")


@tool
def cosmos3_ground(image: str, object_name: str, port: int = 8000, max_tokens: int = 1024) -> dict:
    """2D grounding: return bounding box JSON for object_name in an image."""
    proc = just_run("c3-reason", "Locate the bounding box of " + object_name + ". Return JSON.",
                    image, "", "grounding", str(port), str(max_tokens), "false",
                    timeout_s=_REASON_TIMEOUT)
    return proc_result(proc, "cosmos3 grounding:", "c3 grounding failed")


@tool
def cosmos3_plausibility(video: str, port: int = 8000, max_tokens: int = 1024) -> dict:
    """Physical plausibility: classify and explain (plausible / implausible)."""
    proc = just_run("c3-reason",
                    "Is this video physically plausible (object permanence, shape "
                    "constancy, continuous trajectories)? Answer plausible or "
                    "implausible, then explain.",
                    "", video, "plausibility", str(port), str(max_tokens), "true",
                    timeout_s=_REASON_TIMEOUT)
    return proc_result(proc, "cosmos3 plausibility:", "c3 plausibility failed")


@tool
def cosmos3_situation(video: str, question: str = "", port: int = 8000, max_tokens: int = 2048) -> dict:
    """Situation understanding + most likely next action."""
    p = question or "Describe the situation and predict the most likely next action."
    proc = just_run("c3-reason", p, "", video, "situation", str(port),
                    str(max_tokens), "true", timeout_s=_REASON_TIMEOUT)
    return proc_result(proc, "cosmos3 situation:", "c3 situation failed")


@tool
def cosmos3_action_cot(
    image: str = "",
    video: str = "",
    task_instruction: str = "complete the task",
    port: int = 8000,
    max_tokens: int = 2048,
) -> dict:
    """Action chain-of-thought: 2D end-effector trajectory / driving CoT (JSON)."""
    p = ('You are given the task "' + task_instruction + '". Specify the 2D '
         "trajectory your end effector should follow in pixel space. Return JSON "
         'like {"point_2d": [x, y], "label": "gripper trajectory"}.')
    proc = just_run("c3-reason", p, image, video, "action_cot", str(port),
                    str(max_tokens), "true", timeout_s=_REASON_TIMEOUT)
    return proc_result(proc, "cosmos3 action_cot:", "c3 action_cot failed")


# ----- Generator ----------------------------------------------------------
@tool
def cosmos3_text2image(prompt: str, out: str = "/tmp/c3_image.png", steps: int = 35,
                       guidance: float = 6.0, res: str = "480", seed: int = 0) -> dict:
    """Cosmos 3 Generator: text -> image (PNG) via Diffusers."""
    proc = just_run("c3-gen", "text2image", prompt, "", out, "1", "24", str(steps),
                    str(guidance), res, "false", str(seed), timeout_s=_GEN_TIMEOUT)
    return proc_result(proc, "cosmos3 text2image -> " + out, "c3 text2image failed")


@tool
def cosmos3_text2video(prompt: str, out: str = "/tmp/c3_t2v.mp4", frames: int = 189,
                       fps: int = 24, steps: int = 35, guidance: float = 6.0,
                       res: str = "480", seed: int = 0) -> dict:
    """Cosmos 3 Generator: text -> video (MP4) via Diffusers."""
    proc = just_run("c3-gen", "text2video", prompt, "", out, str(frames), str(fps),
                    str(steps), str(guidance), res, "false", str(seed),
                    timeout_s=_GEN_TIMEOUT)
    return proc_result(proc, "cosmos3 text2video -> " + out, "c3 text2video failed")


@tool
def cosmos3_image2video(prompt: str, image: str, out: str = "/tmp/c3_i2v.mp4",
                        frames: int = 189, fps: int = 24, steps: int = 35,
                        guidance: float = 6.0, res: str = "480", seed: int = 0) -> dict:
    """Cosmos 3 Generator: image + text -> video (MP4) via Diffusers."""
    proc = just_run("c3-gen", "image2video", prompt, image, out, str(frames), str(fps),
                    str(steps), str(guidance), res, "false", str(seed),
                    timeout_s=_GEN_TIMEOUT)
    return proc_result(proc, "cosmos3 image2video -> " + out, "c3 image2video failed")


@tool
def cosmos3_text2video_sound(prompt: str, out: str = "/tmp/c3_t2v_sound.mp4",
                             frames: int = 189, fps: int = 24, steps: int = 35,
                             guidance: float = 6.0, res: str = "480", seed: int = 0) -> dict:
    """Cosmos 3 Generator: text -> video + synchronized audio (MP4+AAC)."""
    proc = just_run("c3-gen", "text2video-with-sound", prompt, "", out, str(frames),
                    str(fps), str(steps), str(guidance), res, "true", str(seed),
                    timeout_s=_GEN_TIMEOUT)
    return proc_result(proc, "cosmos3 text2video+sound -> " + out, "c3 t2v-sound failed")


@tool
def cosmos3_image2video_sound(prompt: str, image: str, out: str = "/tmp/c3_i2v_sound.mp4",
                              frames: int = 189, fps: int = 24, steps: int = 35,
                              guidance: float = 6.0, res: str = "480", seed: int = 0) -> dict:
    """Cosmos 3 Generator: image + text -> video + synchronized audio (MP4+AAC) via Diffusers.

    Image-conditioned motion with synchronized stereo AAC@48kHz sound. Needs a
    sound-capable checkpoint (Cosmos3-Nano). In-proc Diffusers path (no server).
    """
    proc = just_run("c3-gen", "image2video-with-sound", prompt, image, out, str(frames),
                    str(fps), str(steps), str(guidance), res, "true", str(seed),
                    timeout_s=_GEN_TIMEOUT)
    return proc_result(proc, "cosmos3 image2video+sound -> " + out, "c3 i2v-sound failed")


@tool
def cosmos3_video2video(
    video: str,
    prompt: str,
    out: str = "/tmp/c3_v2v.mp4",
    port: int = 8001,
    steps: int = 35,
    guidance: float = 8.0,
    size: str = "832x480",
    frames: int = 29,
    fps: int = 16,
    seed: int = 0,
    negative: str = "blurry, distorted, low quality",
    guardrails: bool = True,
    condition_frames: str = "0",
    condition_keep: str = "last",
    generate_sound: bool = False,
    max_sequence_length: int = 512,
) -> dict:
    """Cosmos 3 video-to-video: re-render an input video with a new prompt.

    Structure-preserving transfer (day->night, recolor, restyle, change the scene)
    via the vLLM-Omni server's /v1/videos/sync endpoint. Start the server first
    with `just c3-omni-docker` (Docker image vllm/vllm-omni:cosmos3 — the only
    build with all modalities incl. video2video).

    How much the prompt changes the video is controlled by the conditioning:
    fewer/earlier conditioning frames + higher guidance = a stronger transform.

    Args:
        video: Path to the input video (local file).
        prompt: Target description (the transformation to apply). Be emphatic and
            pair with a `negative` prompt for strong restyles (e.g. day->night).
        out: Output MP4 path.
        port: Omni server port (default 8001).
        steps: Diffusion steps (35 recommended for a clean restyle).
        guidance: CFG scale. 6 = subtle/structure-faithful; 8-12 = strong restyle.
        size: Output resolution "WxH".
        frames: Frame count.
        fps: Frames per second.
        seed: Reproducibility seed.
        negative: Negative prompt (helps push away the original look).
        guardrails: Enable Cosmos 3 safety guardrails.
        condition_frames: Latent frame indexes kept as clean conditioning, as a
            comma-separated string. Default "0" (anchor only the first latent ->
            strongest transform). The model default is "0,1" (more faithful, weaker
            change). More indexes => closer to the original video.
        condition_keep: Which end of the clip the conditioning frames come from:
            "first" or "last" (default "last").
        generate_sound: Produce a synchronized soundtrack (stereo AAC@48kHz)
            alongside the transformed video (video-to-video-with-sound).
        max_sequence_length: Max prompt tokens kept for conditioning (Cosmos 3
            default 512); longer prompts are truncated with a warning.
    """
    import json as _json
    import os as _os

    try:
        import requests
    except ImportError:
        return err("requests required for cosmos3_video2video: pip install requests")

    path = _os.path.expanduser(video)
    if not _os.path.exists(path):
        return err("input video not found: " + path)

    if condition_keep not in ("first", "last"):
        return err("condition_keep must be 'first' or 'last'")
    try:
        cond_idx = [int(x.strip()) for x in str(condition_frames).split(",") if x.strip() != ""]
        if not cond_idx:
            raise ValueError
    except ValueError:
        return err("condition_frames must be comma-separated non-negative ints, e.g. '0' or '0,1'")

    extra = {
        "use_resolution_template": False,
        "use_duration_template": False,
        "guardrails": guardrails,
        "condition_frame_indexes_vision": cond_idx,
        "condition_video_keep": condition_keep,
    }
    if generate_sound:
        extra["generate_sound"] = True
    data = {
        "prompt": prompt,
        "negative_prompt": negative,
        "size": size,
        "num_frames": str(frames),
        "fps": str(fps),
        "num_inference_steps": str(steps),
        "guidance_scale": str(guidance),
        "flow_shift": "10.0",
        "seed": str(seed),
        "max_sequence_length": str(max_sequence_length),
        "extra_params": _json.dumps(extra),
    }
    if generate_sound:
        data["generate_sound"] = "true"
    try:
        with open(path, "rb") as f:
            resp = requests.post(
                f"http://localhost:{port}/v1/videos/sync",
                data=data,
                files={"input_reference": (_os.path.basename(path), f, "video/mp4")},
                timeout=60 * 30,
            )
        if resp.status_code != 200:
            return err(f"omni server returned {resp.status_code}: {resp.text[:200]}")
        _os.makedirs(_os.path.dirname(_os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "wb") as g:
            g.write(resp.content)
        return ok(
            f"cosmos3 video2video -> {out} ({len(resp.content)} bytes) "
            f"[guidance={guidance}, condition_frames={cond_idx}, keep={condition_keep}, "
            f"sound={generate_sound}]",
            data={"out": out, "bytes": len(resp.content),
                  "condition_frame_indexes_vision": cond_idx, "condition_video_keep": condition_keep,
                  "generate_sound": generate_sound},
        )
    except Exception as e:
        return err("video2video request failed: " + str(e))


# ----- Action / World-Model -----------------------------------------------
# All action tools take a JSONL spec (one line per run). The spec's `model_mode`
# field selects forward_dynamics / inverse_dynamics / policy. These thin wrappers
# call `just c3-action <input_jsonl>` (Cosmos Framework via torchrun). Sample
# specs + assets live in the cosmos repo cookbooks/cosmos3/generator/action.
@tool
def cosmos3_forward_dynamics(input_jsonl: str, out: str = "/tmp/c3_fd",
                             checkpoint: str = "Cosmos3-Nano", seed: int = 0) -> dict:
    """Forward dynamics: start image + action chunk -> future video (Cosmos Framework).

    Args:
        input_jsonl: JSONL spec with model_mode="forward_dynamics", vision_path,
            action_path, domain_name, action_chunk_size, fps, image_size, name.
        out: output dir (writes <out>/<name>/vision.mp4).
        checkpoint: Cosmos 3 checkpoint name.
        seed: reproducibility seed.
    """
    proc = just_run("c3-action", input_jsonl, out, checkpoint, str(seed),
                    timeout_s=_ACTION_TIMEOUT)
    return proc_result(proc, "cosmos3 forward_dynamics -> " + out, "c3 fd failed")


@tool
def cosmos3_inverse_dynamics(input_jsonl: str, out: str = "/tmp/c3_id",
                             checkpoint: str = "Cosmos3-Nano", seed: int = 0) -> dict:
    """Inverse dynamics: video + instruction -> predicted action chunk (Cosmos Framework).

    Args:
        input_jsonl: JSONL spec with model_mode="inverse_dynamics", vision_path
            (input video), domain_name, name.
        out: output dir.
        checkpoint: Cosmos 3 checkpoint name.
        seed: reproducibility seed.
    """
    proc = just_run("c3-action", input_jsonl, out, checkpoint, str(seed),
                    timeout_s=_ACTION_TIMEOUT)
    return proc_result(proc, "cosmos3 inverse_dynamics -> " + out, "c3 id failed")


@tool
def cosmos3_policy(input_jsonl: str, out: str = "/tmp/c3_policy",
                  checkpoint: str = "Cosmos3-Nano-Policy-DROID", seed: int = 0) -> dict:
    """Action policy: image + instruction -> action chunk + rollout video (Cosmos Framework).

    Args:
        input_jsonl: JSONL spec with model_mode="policy", vision_path, domain_name
            (e.g. bridge_orig_lerobot), prompt (instruction), name.
        out: output dir.
        checkpoint: Cosmos 3 policy checkpoint (default Cosmos3-Nano-Policy-DROID).
        seed: reproducibility seed.
    """
    proc = just_run("c3-action", input_jsonl, out, checkpoint, str(seed),
                    timeout_s=_ACTION_TIMEOUT)
    return proc_result(proc, "cosmos3 policy -> " + out, "c3 policy failed")


# ----- Server lifecycle ---------------------------------------------------
@tool
def cosmos3_serve(action: str = "status", surface: str = "reason",
                  model: str = "nvidia/Cosmos3-Nano", port: int = 0, tp: int = 1) -> dict:
    """Manage local Cosmos 3 servers (no NIM).

    Args:
        action: start | stop | status
        surface: reason (vLLM) | omni (vLLM-Omni generator)
        model: HF model id to serve.
        port: server port (0 = recipe default).
        tp: tensor-parallel size (reason only).
    """
    if action == "status":
        return proc_result(just_run("c3-serve-status"), "c3 server status:", "status failed")
    if surface == "reason":
        if action == "start":
            args = ["c3-serve-reason", model]
            if port:
                args += [str(port), str(tp)]
            return proc_result(just_run(*args, timeout_s=120), "c3 reason server starting", "start failed")
        if action == "stop":
            return proc_result(just_run("c3-serve-stop-reason"), "c3 reason server stopped", "stop failed")
    elif surface == "omni":
        if action == "start":
            args = ["c3-serve-omni", model]
            if port:
                args.append(str(port))
            return proc_result(just_run(*args, timeout_s=120), "c3 omni server starting", "start failed")
        if action == "stop":
            return proc_result(just_run("c3-serve-stop-omni"), "c3 omni server stopped", "stop failed")
    return err("unknown action/surface: " + action + "/" + surface)
