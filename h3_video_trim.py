"""Video Trim for MiniMax H3 — drop the pinned head and/or tail off a decoded
clip, picture and sound together.

The pinned frames occupy the start (and, in two-ended anchoring, the end) of
the delivered timeline, so they have to come off before concatenating. Trimming
only the images would leave the audio a full trim_frames longer than the video,
and muxing those puts the whole soundtrack ahead of the picture by
trim_frames/24 seconds.

So this takes both streams and removes the same span from each: whole frames
from the images, the matching number of samples from the waveform. Wire
trim_head / trim_tail from the Multi Condition node so the count follows
whatever the encoder actually produced. A value of 0 on either end means "don't
trim that end".

match_tail also evens out H3's 40 Hz audio grid against the 24 fps picture:
each clip ships ~8.3 ms more or less sound than picture, which accumulates at
every join in a chain. Matching the tail to exactly frames/fps stops it.
"""

import torch

from comfy_api.latest import io


class H3VideoTrim(io.ComfyNode):
    """Trim the leading (and optionally trailing) pinned frames from a decoded
    H3 clip, picture and sound together, and optionally match the audio tail to
    frames/fps."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3VideoTrim",
            display_name="Video Trim for Minimax H3",
            category="kwnodes/H3",
            description="Remove the leading (and optionally trailing) pinned frames from a decoded H3 clip, trimming picture and sound by the same duration.",
            inputs=[
                io.Image.Input(
                    "images", tooltip="Decoded frames of the clip, pinned head first."
                ),
                io.Int.Input(
                    "trim_head",
                    default=0,
                    min=0,
                    max=4096,
                    tooltip="Frames to trim off the head. Wire Multi Condition's trim_head here. 0 = don't trim.",
                ),
                io.Int.Input(
                    "trim_tail",
                    default=0,
                    min=0,
                    max=4096,
                    tooltip="Frames to trim off the tail. Wire Multi Condition's trim_tail here (two-ended anchoring). 0 = don't trim.",
                ),
                io.Audio.Input(
                    "audio",
                    optional=True,
                    tooltip="Decoded audio for the same clip. Trimmed by the matching duration so sound stays locked to picture. Leave unwired for silent clips.",
                ),
                io.Float.Input(
                    "fps",
                    default=24.0,
                    min=1.0,
                    max=240.0,
                    step=0.001,
                    optional=True,
                    tooltip="Frame rate used to convert the trim into an audio duration. Must match what you feed Create Video.",
                ),
                io.Boolean.Input(
                    "match_tail",
                    default=True,
                    optional=True,
                    tooltip="Make the audio duration equal frames/fps exactly, trimming a long tail or padding a short one with silence. H3 rounds its audio grid to the nearest step, so each clip carries about 8ms too much or too little sound, which accumulates at every join in a chain.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Audio.Output(display_name="audio"),
            ],
        )

    @classmethod
    def execute(
        cls, images, trim_head=0, trim_tail=0, audio=None, fps=24.0, match_tail=True
    ) -> io.NodeOutput:
        total = int(images.shape[0])
        n_head = max(0, int(trim_head))
        n_tail = max(0, int(trim_tail))

        if n_head + n_tail >= total:
            raise ValueError(
                "Video Trim: asked to trim %d (head) + %d (tail) frames from a "
                "%d frame clip" % (n_head, n_tail, total)
            )

        end = total - n_tail if n_tail else total
        out_images = images[n_head:end] if (n_head or n_tail) else images

        out_audio = audio
        if audio is not None:
            waveform = audio["waveform"]
            sr = int(audio["sample_rate"])
            head_seconds = n_head / float(fps)
            head_cut = int(round(head_seconds * sr))
            length = int(waveform.shape[-1])

            if n_head:
                if head_cut >= length:
                    raise ValueError(
                        "Video Trim: trimming %.3fs from %.3fs of audio would "
                        "leave nothing. Check that fps matches the clip."
                        % (head_seconds, length / sr)
                    )
                waveform = waveform[..., head_cut:]

            if n_tail:
                frames_left = total - n_head - n_tail
                want = int(round(frames_left / float(fps) * sr))
                if want < waveform.shape[-1]:
                    waveform = waveform[..., :want]

            if match_tail:
                frames_left = total - n_head - n_tail
                want = int(round(frames_left / float(fps) * sr))
                have = int(waveform.shape[-1])
                if have > want:
                    waveform = waveform[..., :want]
                elif have < want:
                    # H3 ships slightly LESS sound than picture for a third of
                    # clip lengths; zero is the honest fill (sound the model
                    # never generated), anything else would fabricate content.
                    missing = want - have
                    waveform = torch.nn.functional.pad(waveform, (0, missing))

            out_audio = {"waveform": waveform, "sample_rate": sr}

        return io.NodeOutput(out_images, out_audio)
