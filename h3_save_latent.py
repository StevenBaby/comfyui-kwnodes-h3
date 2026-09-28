"""H3 Save Latent — save the sampler's AV latent for the next clip of a chain.

Naming: <prefix>_<index:05>.safetensors under the ComfyUI output folder.

  prefix  the chain's name (shared by every clip of the chain)
  index   which clip THIS is (0-based: 0 = first clip)

The Multi Condition node loads clip `index - 1` to continue from, so clip 0
saves `_00000`, clip 1 reads `_00000` and saves `_00001`, and so on. A re-roll
of clip N overwrites its own `_0000N` slot, so rejects never stack up.

This node ONLY saves the latent (video + audio streams). Video/mp4 encoding is
left to VHS_VideoCombine or the Director, which the user already wires.

trim_tail: drop the trailing anchor frames (the two-ended-anchoring tail zone)
from the latent BEFORE saving, so a middle clip (generated from both neighbors)
saves clean and re-rolls match the trimmed picture. Mirrors H3VideoTrim at the
latent level.
"""

import os

import torch

import folder_paths

from comfy_api.latest import io

try:
    from safetensors.torch import save_file as _st_save
except ImportError:
    _st_save = None


FRAME_PER_TOKEN = (1, 4, 4, 4, 4)


def _pixel_frames(latent_t):
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(latent_t))


def _steps_for_frames(n):
    k, covered = 0, 0
    while covered < n:
        covered += FRAME_PER_TOKEN[k % 5]
        k += 1
    return k if covered == n else None


def _streams_from_latent(latent):
    """Unpack an H3 AV latent into its contained [video, audio] streams."""
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    else:
        raise ValueError(
            "H3 Save Latent: expected a MiniMax H3 AV latent (nested video/audio "
            "pair), got %r" % type(samples)
        )
    if not parts:
        raise ValueError("H3 Save Latent: AV latent contains no streams")
    return parts


class H3SaveLatent(io.ComfyNode):
    """Save the sampler's H3 AV latent to a <prefix>_<index> slot."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3SaveLatent",
            display_name="Save Latent for Minimax H3",
            category="kwnodes/H3",
            description="Save the sampler's H3 AV latent to a <prefix>_<index> slot for the next clip of a chain.",
            is_output_node=True,
            inputs=[
                io.Latent.Input(
                    "latent",
                    tooltip="The sampler's AV latent output (the same one wired into the decode nodes).",
                ),
                io.String.Input(
                    "prefix",
                    default="h3_output/clip",
                    optional=True,
                    tooltip="Chain name + optional subfolder. Set it to the same value as Multi Condition's prefix.",
                ),
                io.Int.Input(
                    "index",
                    default=0,
                    min=0,
                    max=9999,
                    optional=True,
                    tooltip="Which clip THIS is (0-based). Set it to the same value as Multi Condition's index.",
                ),
                io.Int.Input(
                    "trim_head",
                    default=0,
                    min=0,
                    max=4096,
                    optional=True,
                    tooltip="Frames to trim off the head BEFORE saving (the head anchor zone). Wire Multi Condition's trim_head here. 0 = don't trim.",
                ),
                io.Int.Input(
                    "trim_tail",
                    default=0,
                    min=0,
                    max=4096,
                    optional=True,
                    tooltip="Frames to trim off the tail BEFORE saving (the two-ended-anchoring tail zone). Wire Multi Condition's trim_tail here. 0 = save the full latent.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="latent_path"),
                io.String.Output(display_name="basename"),
            ],
        )

    @classmethod
    def execute(
        cls, latent, prefix="h3_output/clip", index=0, trim_head=0, trim_tail=0
    ) -> io.NodeOutput:
        if _st_save is None:
            raise RuntimeError("H3 Save Latent: safetensors is not available.")

        parts = _streams_from_latent(latent)
        if len(parts) < 2:
            raise ValueError(
                "H3 Save Latent: latent has no audio stream; wire the sampler "
                "output of an H3 AV graph."
            )
        video = parts[0]
        audio = parts[1]

        total_t = int(video.shape[2])

        # Trim the head anchor frames (front) at the latent level.
        n_head = max(0, int(trim_head))
        head_cut = 0
        if n_head > 0:
            steps = _steps_for_frames(n_head)
            if steps is None:
                raise ValueError(
                    "H3 Save Latent: trim_head %d frames is not a whole number "
                    "of latent steps." % n_head
                )
            if steps >= total_t:
                raise ValueError(
                    "H3 Save Latent: trim_head %d frames (%d steps) >= latent "
                    "length %d steps." % (n_head, steps, total_t)
                )
            head_cut = steps

        # Trim the tail anchor frames (back) at the latent level.
        n_tail = max(0, int(trim_tail))
        tail_cut = 0
        if n_tail > 0:
            steps = _steps_for_frames(n_tail)
            if steps is None:
                raise ValueError(
                    "H3 Save Latent: trim_tail %d frames is not a whole number "
                    "of latent steps." % n_tail
                )
            if head_cut + steps >= total_t:
                raise ValueError(
                    "H3 Save Latent: trim_head + trim_tail (%d + %d frames) >= "
                    "latent length %d steps." % (n_head, n_tail, total_t)
                )
            tail_cut = steps

        if head_cut or tail_cut:
            end = total_t - tail_cut if tail_cut else total_t
            video = video[:, :, head_cut:end]

            # audio spans frames at 40 Hz; trim the matching duration on both ends
            head_audio = int(round(_pixel_frames(head_cut) / 24.0 * 40.0))
            frames_left = _pixel_frames(total_t - tail_cut)
            audio_end = int(round(frames_left / 24.0 * 40.0))
            if head_audio or audio_end < audio.shape[-1]:
                audio = audio[..., head_audio:audio_end]

        video = video.cpu().contiguous()
        audio = audio.cpu().contiguous()

        folder, filename, _counter, _sub, _ = folder_paths.get_save_image_path(
            prefix, folder_paths.get_output_directory()
        )
        os.makedirs(folder, exist_ok=True)

        # Fixed slot: <filename>-<index:05>.safetensors. Not the counter-based
        # name, because the loader resolves <prefix>-<index> deterministically.
        basename = "%s-%05d" % (filename, int(index))
        path = os.path.join(folder, basename + ".safetensors")

        tmp = path + ".tmp"
        _st_save({"video": video, "audio": audio}, tmp, metadata={"format": "h3_av_v1"})
        os.replace(tmp, path)

        return io.NodeOutput(path, basename)
