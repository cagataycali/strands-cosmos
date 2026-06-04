"""Phase 0 gate: Cosmos 3 providers and tools import without GPU/model load."""


def test_cosmos3_providers_import():
    from strands_cosmos import Cosmos3ReasonerModel, Cosmos3GeneratorModel

    r = Cosmos3ReasonerModel()
    assert r.get_config()["model_id"] == "nvidia/Cosmos3-Nano"
    g = Cosmos3GeneratorModel()
    assert g.get_config()["model_id"] == "nvidia/Cosmos3-Nano"


def test_cosmos3_reasoner_media_parsing():
    from strands_cosmos.cosmos3_reasoner_model import Cosmos3ReasonerModel

    m = Cosmos3ReasonerModel()
    msgs = [{"role": "user", "content": [
        {"text": "Caption: <video>https://x/y.mp4</video> and <image>https://x/z.jpg</image> please"}
    ]}]
    oai = m._extract_media_to_openai(msgs)
    assert len(oai) == 1
    parts = oai[0]["content"]
    types = [p["type"] for p in parts]
    assert "video_url" in types and "image_url" in types and "text" in types


def test_cosmos3_tools_import():
    from strands_cosmos import (
        cosmos3_reason, cosmos3_caption, cosmos3_temporal, cosmos3_embodied,
        cosmos3_ground, cosmos3_plausibility, cosmos3_situation, cosmos3_action_cot,
        cosmos3_text2image, cosmos3_text2video, cosmos3_image2video,
        cosmos3_text2video_sound, cosmos3_forward_dynamics, cosmos3_inverse_dynamics,
        cosmos3_policy, cosmos3_serve,
    )
    # All decorated tools expose a tool spec
    for t in [cosmos3_reason, cosmos3_caption, cosmos3_serve]:
        assert hasattr(t, "tool_spec") or callable(t)


def test_cosmos3_generator_modes_and_resolution():
    from strands_cosmos.cosmos3_generator_model import (
        Cosmos3GeneratorModel, RES_TIERS, GEN_DEFAULTS,
    )
    m = Cosmos3GeneratorModel()
    cfg = m.get_config()
    assert cfg["model_id"] == "nvidia/Cosmos3-Nano"
    assert cfg["guardrails"] is True
    # resolution tiers present
    assert RES_TIERS["256"] == (320, 192)
    assert RES_TIERS["480"] == (832, 480)
    assert RES_TIERS["720"] == (1280, 720)
    # has the audio mux helper
    assert hasattr(m, "_mux_audio")
    assert GEN_DEFAULTS["fps"] == 24


def test_cosmos3_reasoner_task_prompts():
    from strands_cosmos.cosmos3_reasoner_model import TASK_PROMPTS, SAMPLING_THINK, SAMPLING_NO_THINK
    for k in ["caption", "temporal", "embodied", "plausibility", "situation",
              "grounding", "describe", "action_cot", "driving"]:
        assert k in TASK_PROMPTS
    assert SAMPLING_THINK["temperature"] == 0.6
    assert SAMPLING_NO_THINK["temperature"] == 0.7
