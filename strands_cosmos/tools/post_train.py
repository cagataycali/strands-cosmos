"""Wrapper around `just post-train-*` recipes."""
from __future__ import annotations

from pathlib import Path

from strands import tool
from ._common import just_run, proc_result, err


@tool
def cosmos_post_train(
    config_path: str,
    model_family: str = "reason2",
    strategy: str = "full",
    num_gpus: int = 1,
    dry_run: bool = False,
) -> dict:
    """Launch a Cosmos post-training job via just.

    Supports:
      - reason2 (full|lora) → `just post-train-reason2 <config> <strategy>`
      - reason2 rl          → `just post-train-reason2-rl <config>`
      - predict2_5          → `just post-train-predict <config> <num_gpus>`
      - transfer2_5         → `just post-train-transfer <config> <num_gpus>`

    Args:
        config_path: YAML / TOML training config.
        model_family: reason2 | predict2_5 | transfer2_5.
        strategy: full | lora | rl (rl is reason2 only).
        num_gpus: GPUs per node (predict/transfer only).
        dry_run: If True, just preview the recipe name.
    """
    if not Path(config_path).exists():
        return err(f"config not found: {config_path}")

    if model_family == "reason2" and strategy == "rl":
        recipe, args = "post-train-reason2-rl", (config_path,)
    elif model_family == "reason2":
        recipe, args = "post-train-reason2", (config_path, strategy)
    elif model_family == "predict2_5":
        recipe, args = "post-train-predict", (config_path, str(num_gpus))
    elif model_family == "transfer2_5":
        recipe, args = "post-train-transfer", (config_path, str(num_gpus))
    else:
        return err(f"unknown model_family: {model_family}")

    if dry_run:
        return proc_result(
            {"ok": True, "stdout": f"dry-run: just {recipe} " + " ".join(args),
             "returncode": 0, "cmd": f"just {recipe} {' '.join(args)}"},
            success_text=f"dry-run: just {recipe}",
        )

    proc = just_run(recipe, *args, timeout_s=60 * 60 * 12)
    return proc_result(
        proc,
        success_text=f"✅ post-training ({model_family}/{strategy}) finished",
        fail_text=f"post-training failed: {proc.get('stderr', '')[:300]}",
    )
