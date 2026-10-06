"""comfyui-kwnodes-h3

Independent ComfyUI node pack for V3 (io.ComfyNode) MiniMax H3 clip-chain nodes.

Kept separate from comfyui-kwnodes so the V1 (NODE_CLASS_MAPPINGS) pack and the
V3 (comfy_entrypoint) pack never fight over one module entry point — ComfyUI
loads one or the other per directory, not both.

Provides:
- Multi Condition for Minimax H3 — continue a clip chain, redraw the seam, lock
  source audio, and anchor extra keyframe images via Autogrow (multi-keyframe).
- Save Latent for Minimax H3 — save the sampler's AV latent for the next clip.
- Load Latent for Minimax H3 — load a <prefix>-<index> slot, the mirror of
  Save Latent.
- Load Latent File for Minimax H3 — load a saved AV latent by picking the file
  from a dropdown, from a directory with VHS-style autocomplete and an
  include-sub-directory toggle.
- Video Trim for Minimax H3 — drop the pinned head off a decoded clip, picture
  and sound together, and match the audio tail to frames/fps.
"""

import importlib

from comfy_api.latest import ComfyExtension

from .h3_multi_condition import H3MultiCondition
from .h3_save_latent import H3SaveLatent
from .h3_load_latent import H3LoadLatent, H3LoadLatentFile
from .h3_video_trim import H3VideoTrim
from .h3_cond_noise_aug import H3CondNoiseAug

WEB_DIRECTORY = "./web"

_NODES = [
    H3MultiCondition,
    H3SaveLatent,
    H3LoadLatent,
    H3LoadLatentFile,
    H3VideoTrim,
    H3CondNoiseAug,
]


class KWNodesH3Extension(ComfyExtension):
    async def get_node_list(self):
        return _NODES


async def comfy_entrypoint():
    return KWNodesH3Extension()


# Register the custom HTTP routes the Load Latent frontend calls (directory
# autocomplete + latent file list refresh).
importlib.import_module(".routes", __name__)
