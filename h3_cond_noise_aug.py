"""Cond Noise Aug for MiniMax H3 — 锚定强度控制节点.

Exposes ComfyUI's `minimax_visual_cond_noise_aug` / `minimax_audio_cond_noise_aug`
payload keys as a user-tunable strength. These keys are already consumed by the
H3 DiT (comfy/ldm/minimax/model.py `_cond_video_rows` / `_cond_audio_rows`) to
blend condition latents with noise, but no stock node sets them — the default is
`VISUAL_COND_TIMESTEP = 0.999` (visual) and `AUDIO_COND_TIMESTEP = 1.0` (audio),
i.e. the condition/keyframe rows are nearly locked.

Lowering `visual_strength` softens the anchor so the model is allowed to denoise
the condition rows, which removes the "frozen at the keyframe" stutter in chained
long videos while still retaining some drift resistance.

Nothing in ComfyUI or other custom nodes is modified.
"""

import node_helpers

from comfy_api.latest import io, ComfyExtension


class H3CondNoiseAug(io.ComfyNode):
    """Set the H3 condition noise-augmentation strength (anchor strength)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3CondNoiseAug",
            display_name="Cond Noise Aug for Minimax H3",
            category="kwnodes/H3",
            description=(
                "调节 MiniMax H3 锚定(keyframe/reference)的强度。visual_strength "
                "越低，锚定越软，人物动作越自然，能消除长视频拼接处的停顿感；但"
                "过低会漂移。audio_strength 同理控制音频锚定。"
            ),
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Float.Input(
                    "visual_strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    tooltip="视觉锚定强度。1.0=完全锁死(默认0.999,会停顿), 0.7~0.9=软锚定(动作自然+防漂移), 0.5以下=很松易漂移。",
                ),
                io.Float.Input(
                    "audio_strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    tooltip="音频锚定强度。默认1.0(完全锁死)。",
                ),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, conditioning, visual_strength=1.0, audio_strength=1.0) -> io.NodeOutput:
        values = {
            "minimax_visual_cond_noise_aug": float(visual_strength),
            "minimax_audio_cond_noise_aug": float(audio_strength),
        }
        return io.NodeOutput(node_helpers.conditioning_set_values(conditioning, values))
