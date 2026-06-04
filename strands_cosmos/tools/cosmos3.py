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

from ._common import just_run, proc_result, err

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
                       guidance: float = 6.0, res: str = "720", seed: int = 0) -> dict:
    """Cosmos 3 Generator: text -> image (PNG) via Diffusers."""
    proc = just_run("c3-gen", "text2image", prompt, "", out, "1", "24", str(steps),
                    str(guidance), res, "false", str(seed), timeout_s=_GEN_TIMEOUT)
    return proc_result(proc, "cosmos3 text2image -> " + out, "c3 text2image failed")


@tool
def cosmos3_text2video(prompt: str, out: str = "/tmp/c3_t2v.mp4", frames: int = 189,
                       fps: int = 24, steps: int = 35, guidance: float = 6.0,
                       res: str = "720", seed: int = 0) -> dict:
    """Cosmos 3 Generator: text -> video (MP4) via Diffusers."""
    proc = just_run("c3-gen", "text2video", prompt, "", out, str(frames), str(fps),
                    str(steps), str(guidance), res, "false", str(seed),
                    timeout_s=_GEN_TIMEOUT)
    return proc_result(proc, "cosmos3 text2video -> " + out, "c3 text2video failed")


@tool
def cosmos3_image2video(prompt: str, image: str, out: str = "/tmp/c3_i2v.mp4",
                        frames: int = 189, fps: int = 24, steps: int = 35,
                        guidance: float = 6.0, res: str = "720", seed: int = 0) -> dict:
    """Cosmos 3 Generator: image + text -> video (MP4) via Diffusers."""
    proc = just_run("c3-gen", "image2video", prompt, image, out, str(frames), str(fps),
                    str(steps), str(guidance), res, "false", str(seed),
                    timeout_s=_GEN_TIMEOUT)
    return proc_result(proc, "cosmos3 image2video -> " + out, "c3 image2video failed")


@tool
def cosmos3_text2video_sound(prompt: str, out: str = "/tmp/c3_t2v_sound.mp4",
                             frames: int = 189, fps: int = 24, steps: int = 35,
                             guidance: float = 6.0, res: str = "720", seed: int = 0) -> dict:
    """Cosmos 3 Generator: text -> video + synchronized audio (MP4+AAC)."""
    proc = just_run("c3-gen", "text2video-with-sound", prompt, "", out, str(frames),
                    str(fps), str(steps), str(guidance), res, "true", str(seed),
                    timeout_s=_GEN_TIMEOUT)
    return proc_result(proc, "cosmos3 text2video+sound -> " + out, "c3 t2v-sound failed")


# ----- Action / World-Model -----------------------------------------------
@tool
def cosmos3_forward_dynamics(input_json: str, out: str = "/tmp/c3_fd",
                             checkpoint: str = "Cosmos3-Nano", seed: int = 0) -> dict:
    """Forward dynamics: start image + action chunk -> future video (Cosmos Framework)."""
    proc = just_run("c3-action", "forward_dynamics", input_json, out, checkpoint,
                    str(seed), timeout_s=_ACTION_TIMEOUT)
    return proc_result(proc, "cosmos3 forward_dynamics -> " + out, "c3 fd failed")


@tool
def cosmos3_inverse_dynamics(input_json: str, out: str = "/tmp/c3_id",
                             checkpoint: str = "Cosmos3-Nano", seed: int = 0) -> dict:
    """Inverse dynamics: video + instruction -> predicted action chunk (Cosmos Framework)."""
    proc = just_run("c3-action", "inverse_dynamics", input_json, out, checkpoint,
                    str(seed), timeout_s=_ACTION_TIMEOUT)
    return proc_result(proc, "cosmos3 inverse_dynamics -> " + out, "c3 id failed")


@tool
def cosmos3_policy(input_json: str, out: str = "/tmp/c3_policy",
                  checkpoint: str = "Cosmos3-Nano-Policy-DROID", seed: int = 0) -> dict:
    """Action policy: image + instruction -> action chunk + rollout video (Cosmos Framework)."""
    proc = just_run("c3-action", "policy", input_json, out, checkpoint,
                    str(seed), timeout_s=_ACTION_TIMEOUT)
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
