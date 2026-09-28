"""comfyui-kwnodes-h3

Independent ComfyUI node pack for V3 (io.ComfyNode) MiniMax H3 clip-chain nodes.

Kept separate from comfyui-kwnodes so the V1 (NODE_CLASS_MAPPINGS) pack and the
V3 (comfy_entrypoint) pack never fight over one module entry point — ComfyUI
loads one or the other per directory, not both.

Provides:
- Multi Condition for Minimax H3 — continue a clip chain, redraw the seam, lock
  source audio, and anchor extra keyframe images via Autogrow (multi-keyframe).
- Save Latent for Minimax H3 — save the sampler's AV latent for the next clip.
- Video Trim for Minimax H3 — drop the pinned head off a decoded clip, picture
  and sound together, and match the audio tail to frames/fps.
"""

from comfy_api.latest import ComfyExtension

from .h3_multi_condition import H3MultiCondition
from .h3_save_latent import H3SaveLatent
from .h3_video_trim import H3VideoTrim

_NODES = [
    H3MultiCondition,
    H3SaveLatent,
    H3VideoTrim,
]


class KWNodesH3Extension(ComfyExtension):
    async def get_node_list(self):
        return _NODES


async def comfy_entrypoint():
    return KWNodesH3Extension()
