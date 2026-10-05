"""Multi Condition for MiniMax H3 — continue a clip chain by loading the
previous clip's latent from a <prefix>_<index> slot, pinning its tail video and
audio into the current conditioning, and optionally locking a source audio
track (lip-sync / audio drive).

The point over the stock H3 Motion Context pack: the previous clip's latent is
found automatically from `prefix` + `index` — no Load Latent node, no
clip_index bookkeeping.

  prefix  chain name (+ optional subfolder). Every clip shares it.
  index   which clip is being generated NOW (1-based).

Loads clip `index - 1`'s latent from <prefix>_<index-1:05>.safetensors (only
when index > 1), slices its video tail + audio tail, and pins them as
never-denoised keyframes at the head of the current clip — lossless
continuation. Clip 1 (index 1) has no predecessor and passes conditioning
through unchanged. The sibling H3 Save Latent node writes
<prefix>_<index:05>.safetensors from the sampler output.

lock_audio:
  off (default)  continuation audio: the previous clip's tail audio is pinned
                 so sound continues seamlessly (if a predecessor latent exists).
  on             audio drive: the whole source_audio track is encoded into the
                 AV latent and locked with a zero denoise mask, so the video
                 lip-syncs to it and the track passes through unchanged.
                 Forward-looking: it is THIS clip's audio, not the previous
                 clip's tail.

The H3 latent-slicing maths (frame-per-token grid, 40 Hz audio grid, tail
windows) follows MiniMax H3's published layout; reimplemented here to stay
self-contained.
"""

import os
import math

import torch

import folder_paths
import comfy.utils

from comfy_api.latest import io, ComfyExtension

try:
    from safetensors.torch import load_file as _st_load
except ImportError:
    _st_load = None


# --- H3 latent layout constants (MiniMax H3 model) -------------------------
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)  # video VAE: latent steps cover 1/4/4/4/4 frames
FPS = 24
FRAME_RESCALE = 5.0 / 3.0  # audio steps per pixel frame (40 Hz vs 24 fps)
AUDIO_HZ = 40.0
# run lengths the video VAE distinguishes: 17m+5 grid, downward-snapped
VIDEO_RUN_GRID = (124, 107, 90, 73, 56, 39, 22, 5, 1)


def _pixel_frames(latent_t):
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(latent_t))


def _resize_keyframe(image, width, height):
    """Resize a keyframe image [1,H,W,C] to the clip canvas (lanczos, center crop)."""
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", "center")
    return samples.movedim(1, -1)


def _step_offsets(latent_t):
    out, acc = [], 0
    for k in range(latent_t):
        out.append(acc)
        acc += FRAME_PER_TOKEN[k % 5]
    return out


def _steps_for_frames(n):
    k, covered = 0, 0
    while covered < n:
        covered += FRAME_PER_TOKEN[k % 5]
        k += 1
    return k if covered == n else None


def _frame_to_latent_step(frame_idx):
    """Map a pixel-frame index to its latent time step (inverse of _step_offsets).

    FRAME_PER_TOKEN = (1,4,4,4,4): latent step 0 covers frame 0, step 1 covers
    frames 1..4, step 2 covers 5..8, etc. Returns the latent step whose span
    contains frame_idx, or None if out of range.
    """
    if frame_idx < 0:
        return None
    acc = 0
    k = 0
    while True:
        span = FRAME_PER_TOKEN[k % 5]
        if frame_idx < acc + span:
            return k
        acc += span
        k += 1


def _streams_from_latent(latent):
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    else:
        raise ValueError(
            "Multi Condition: expected an H3 AV latent, got %r" % type(samples)
        )
    if not parts:
        raise ValueError("Multi Condition: AV latent contains no streams")
    return parts


def _video_from_latent(latent):
    video = _streams_from_latent(latent)[0]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError(
            "Multi Condition: expected video latent [B,C,T,H,W], got %s"
            % (tuple(video.shape),)
        )
    return video


def _video_tail_from_latent(latent, n):
    video = _video_from_latent(latent)
    total = int(video.shape[2])
    steps = _steps_for_frames(n)
    if steps is None:
        raise ValueError(
            "Multi Condition: %d frames is not a whole number of latent steps. Use 5, 22, 39 or 56."
            % n
        )
    if steps > total:
        raise ValueError(
            "Multi Condition: asked for %d latent steps, context has %d."
            % (steps, total)
        )
    start = total - steps
    if start % 5 != 0:
        raise RuntimeError(
            "Multi Condition: %d step tail of a %d step latent starts at cycle %d, not 0."
            % (steps, total, start % 5)
        )
    covered = _pixel_frames(steps)
    if covered != n:
        raise RuntimeError(
            "Multi Condition: %d steps cover %d frames, expected %d."
            % (steps, covered, n)
        )
    blocks = [video[:1, :, start + k : start + k + 1].clone() for k in range(steps)]
    return blocks, _step_offsets(steps), covered


def _video_head_from_latent(latent, n):
    """Take the FIRST `n` frames of a latent (the head), mirroring
    _video_tail_from_latent but from step 0. Used for the tail anchor (next
    clip's head)."""
    video = _video_from_latent(latent)
    total = int(video.shape[2])
    steps = _steps_for_frames(n)
    if steps is None:
        raise ValueError(
            "Multi Condition: %d frames is not a whole number of latent steps." % n
        )
    if steps > total:
        raise ValueError(
            "Multi Condition: asked for %d latent steps, context has %d."
            % (steps, total)
        )
    covered = _pixel_frames(steps)
    if covered != n:
        raise RuntimeError(
            "Multi Condition: %d steps cover %d frames, expected %d."
            % (steps, covered, n)
        )
    blocks = [video[:1, :, k : k + 1].clone() for k in range(steps)]
    return blocks, _step_offsets(steps), covered


def _audio_tail_from_latent(latent, a_frames):
    parts = _streams_from_latent(latent)
    if len(parts) < 2:
        raise ValueError("Multi Condition: context latent has no audio stream.")
    video, audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if audio.ndim != 4:
        raise ValueError(
            "Multi Condition: expected audio latent [B,C,2,T], got %s"
            % (tuple(audio.shape),)
        )
    total_t = int(audio.shape[-1])
    frames = _pixel_frames(int(video.shape[2]))
    overhang = total_t - FRAME_RESCALE * frames
    if not (-0.5 < overhang < 0.5):
        overhang = 0.0
    rt = int(round(a_frames / float(FPS) * AUDIO_HZ))
    if rt > total_t:
        rt = total_t
    if rt < 1:
        raise ValueError("Multi Condition: audio window is empty")
    tail = audio[:1, ..., total_t - rt :].clone()
    return tail, rt, float(overhang)


# --- redraw (guide + redraw seam) helpers --------------------------------
SEAM_MIN_MASK = 0.10
SEAM_FLOOR_MIN = 0.0
SEAM_FLOOR_MAX = 0.95
SEAM_TAPER_TOKENS = 4
_REMASK_KEY = "kwnodes_h3_redraw_prefix_remask"


def _clamp_seam_min(value) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        n = SEAM_MIN_MASK
    if not math.isfinite(n):
        n = SEAM_MIN_MASK
    return max(SEAM_FLOOR_MIN, min(SEAM_FLOOR_MAX, n))


def _prefix_token_weights(prefix_steps, taper_steps=SEAM_TAPER_TOKENS, seam_min=None):
    """1.0 on the disposable head, taper toward seam_min (0 = hard-lock seam)."""
    n = int(prefix_steps)
    if n < 1:
        return ()
    taper = max(1, min(int(taper_steps), n))
    head = n - taper
    floor = _clamp_seam_min(SEAM_MIN_MASK if seam_min is None else seam_min)
    weights = [1.0] * head
    weights.extend(
        1.0 + (floor - 1.0) * (float(i + 1) / float(taper)) for i in range(taper)
    )
    return tuple(weights)


def _schedule_values(sigmas):
    if torch.is_tensor(sigmas):
        raw = sigmas.detach().float().reshape(-1).cpu().tolist()
    else:
        raw = list(sigmas or ())
    values = [float(v) for v in raw if math.isfinite(float(v)) and float(v) >= 0.0]
    return tuple(sorted(set(values), reverse=True))


def _next_sigma_ratio(current, sigmas):
    cur = float(current)
    if not math.isfinite(cur) or cur <= 0.0:
        return 0.0
    tol = max(1e-7, abs(cur) * 1e-6)
    for cand in _schedule_values(sigmas):
        if cand < cur - tol:
            return max(0.0, min(1.0, cand / cur))
    return 0.0


def _spatial_video_mask(
    t_steps, prefix_steps, height, width, device, dtype, weights=None
):
    """H3 ``mask_row_values`` needs [B,1,T,H,W] so ``mask[0,0]`` is [T,H,W]."""
    t = int(t_steps)
    h = max(1, int(height))
    w = max(1, int(width))
    mask = torch.ones((1, 1, t, h, w), device=device, dtype=dtype)
    n = max(0, min(int(prefix_steps), t))
    if n < 1:
        return mask
    if weights is None:
        ramp = torch.tensor(_prefix_token_weights(n), device=device, dtype=dtype)
    else:
        ramp = weights[:n].to(device=device, dtype=dtype)
    mask[:, :, :n] = ramp.view(1, 1, n, 1, 1)
    return mask


class _VideoPrefixRemask:
    """Per-step video-prefix mask re-derivation (Director ``_PrefixRemask``, video-only)."""

    def __init__(self, prefix_steps, video_shape, seam_min=None):
        self.prefix_steps = int(prefix_steps)
        self.sigmas = ()
        self.video_shape = tuple(int(x) for x in video_shape)
        self.seam_min = _clamp_seam_min(SEAM_MIN_MASK if seam_min is None else seam_min)

    def _live_weights(self, sigma, sigmas=()):
        current = float(torch.as_tensor(sigma).detach().float().reshape(-1)[0])
        if (
            sigmas is None
            or (torch.is_tensor(sigmas) and sigmas.numel() == 0)
            or (not torch.is_tensor(sigmas) and len(sigmas) == 0)
        ):
            sigmas = self.sigmas
        ratio = _next_sigma_ratio(current, sigmas)
        live = []
        for base in _prefix_token_weights(self.prefix_steps, seam_min=self.seam_min):
            if base >= 0.999:
                live.append(1.0)
            elif self.seam_min <= 0.0:
                live.append(max(0.0, float(base) * max(float(ratio), 0.0)))
            else:
                live.append(max(self.seam_min, float(base) * max(float(ratio), 0.5)))
        return torch.tensor(live, dtype=torch.float32)

    def apply_model_wrapper(self, executor, *args, **kwargs):
        """APPLY_MODEL wrapper: re-derive the video denoise_mask per step.

        H3's ``apply_model`` runs BEFORE ``_apply_model`` -> ``process_timestep``,
        so replacing ``kwargs["denoise_mask"]`` here makes BOTH the per-patch
        timestep (process_timestep) and the per-row label (forward) see the new
        mask. Hanging on DIFFUSION_MODEL instead would only reach the forward
        half and leave process_timestep on the stale mask (the "half works" case).

        ``apply_model(x, t, c_concat, c_crossattn, control, transformer_options,
        **extra_conds)``: the sigma is ``t`` (args[1]) and the video mask is
        ``kwargs["denoise_mask"]``. The full schedule rides on
        ``transformer_options["sample_sigmas"]``.
        """
        denoise_mask = kwargs.get("denoise_mask")
        if denoise_mask is None:
            return executor(*args, **kwargs)

        timestep = args[1] if len(args) > 1 else kwargs.get("timestep")
        if timestep is None:
            return executor(*args, **kwargs)

        transformer_options = (
            args[5] if len(args) > 5 else kwargs.get("transformer_options", {})
        )
        sigmas = (transformer_options or {}).get("sample_sigmas", ())
        if (
            sigmas is None
            or (torch.is_tensor(sigmas) and sigmas.numel() == 0)
            or (not torch.is_tensor(sigmas) and len(sigmas) == 0)
        ):
            sigmas = self.sigmas

        t = int(self.video_shape[2])
        h = int(self.video_shape[3])
        w = int(self.video_shape[4])
        weights = self._live_weights(timestep, sigmas)
        new_video = _spatial_video_mask(
            t,
            self.prefix_steps,
            h,
            w,
            device=denoise_mask.device if torch.is_tensor(denoise_mask) else "cpu",
            dtype=torch.float32,
            weights=weights,
        )
        kwargs["denoise_mask"] = new_video
        return executor(*args, **kwargs)


def _install_video_prefix_remask(model, latent, prefix_steps, seam_min):
    """Clone ``model`` and attach a per-step video-prefix remask wrapper."""
    import comfy.patcher_extension

    if model is None or not callable(getattr(model, "clone", None)):
        return model
    if prefix_steps < 1:
        return model

    samples = latent.get("samples") if isinstance(latent, dict) else None
    if samples is None:
        return model
    if hasattr(samples, "unbind"):
        streams = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        streams = list(samples)
    else:
        return model
    if not streams or not torch.is_tensor(streams[0]) or streams[0].ndim != 5:
        return model
    if prefix_steps >= int(streams[0].shape[2]):
        return model

    patched = model.clone()
    state = _VideoPrefixRemask(prefix_steps, tuple(streams[0].shape), seam_min=seam_min)
    if callable(getattr(patched, "add_wrapper_with_key", None)):
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.APPLY_MODEL,
            _REMASK_KEY,
            state.apply_model_wrapper,
        )
    try:
        setattr(patched, "_kwnodes_h3_redraw_remask", state)
    except Exception:
        pass
    return patched


class H3MultiCondition(io.ComfyNode):
    """Continue / audio-drive a MiniMax H3 clip from a <prefix>_<index> slot."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3MultiCondition",
            display_name="Multi Condition for Minimax H3",
            category="kwnodes/H3",
            description="Continue an H3 clip chain from a <prefix>_<index> slot, pin the previous tail, redraw the seam, lock a source audio track, and anchor extra keyframe images.",
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="Sampler model. Clone gets a per-step video-prefix remask wrapper (breaks later-segment drift).",
                ),
                io.Conditioning.Input("conditioning"),
                io.Latent.Input("latent"),
                io.Vae.Input(
                    "vae",
                    optional=True,
                    tooltip="H3 video VAE. Required to anchor extra keyframe images.",
                ),
                io.Vae.Input(
                    "audio_vae",
                    optional=True,
                    tooltip="H3 audio VAE. Required when lock_audio is on.",
                ),
                io.Audio.Input(
                    "source_audio",
                    optional=True,
                    tooltip="The audio for THIS clip (lip-sync). Required when lock_audio is on.",
                ),
                io.Boolean.Input(
                    "lock_audio",
                    default=False,
                    tooltip="ON: lock a source audio track (forward lip-sync). OFF: continue the previous clip's tail audio.",
                ),
                io.Boolean.Input(
                    "split_audio",
                    default=True,
                    tooltip="ON (default): slice source_audio by index into this clip's overlapping segment. OFF: use source_audio as-is (no index-based splitting). Only applies when lock_audio is on.",
                ),
                io.Int.Input(
                    "length",
                    default=124,
                    min=5,
                    max=3600,
                    optional=True,
                    tooltip="Frame count of EVERY clip (at 24 fps), identical across the chain. Drives the audio segment length when lock_audio is on.",
                ),
                io.Combo.Input(
                    "context_length",
                    options=["22", "5", "39", "56"],
                    default="22",
                    tooltip="Frames of the previous clip's picture to pin. 22 is nearly seamless.",
                ),
                io.Int.Input(
                    "audio_context_length",
                    default=24,
                    min=0,
                    max=240,
                    tooltip="Frames of tail audio to pin (24 = one second).",
                ),
                io.Float.Input(
                    "seam_min",
                    default=0.8,
                    min=0.0,
                    max=0.95,
                    step=0.05,
                    tooltip="Redraw strength at the seam. 0.0 = hard-lock (copy tail bit-for-bit); higher = redraw the seam more.",
                ),
                io.Boolean.Input(
                    "ignore_head_latent",
                    default=False,
                    tooltip="ON: skip loading and pinning the HEAD anchor (the previous clip index-1's tail). Errors if the head latent is missing.",
                ),
                io.Boolean.Input(
                    "ignore_tail_latent",
                    default=False,
                    tooltip="ON: skip loading and pinning the TAIL anchor (the next clip index+1's head). Missing tail latent is silently ignored.",
                ),
                io.String.Input(
                    "keyframe_indices",
                    default="-1,0",
                    optional=True,
                    tooltip='Comma-separated frame indices, one per keyframe image in order (e.g. "0,60,120"). Negative = from the end. Empty = no keyframes.',
                ),
                io.String.Input(
                    "prefix",
                    default="h3_output/clip",
                    tooltip="Chain name + optional subfolder. Shared by every clip.",
                ),
                io.Int.Input(
                    "index",
                    default=0,
                    min=0,
                    max=9999,
                    tooltip="Which clip is generated NOW (0-based). Loads clip index-1 to continue from. 0 = first clip, no predecessor.",
                ),
                io.Autogrow.Input(
                    "keyframe_images",
                    optional=True,
                    tooltip="Extra keyframe images. Pair each with a matching index in keyframe_indices (comma-separated, in order). Linking the last slot adds another.",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "keyframe_image",
                            tooltip="One keyframe image to anchor at the matching frame index.",
                        ),
                        prefix="keyframe_",
                        min=0,
                        max=16,
                    ),
                ),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="conditioning"),
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="trim_head"),
                io.Int.Output(display_name="trim_tail"),
                io.Audio.Output(display_name="index_audio"),
            ],
        )

    @classmethod
    def _slot_path(cls, prefix, idx):
        folder, filename, _c, _s, _ = folder_paths.get_save_image_path(
            prefix, folder_paths.get_output_directory()
        )
        return os.path.join(folder, "%s-%05d.safetensors" % (filename, int(idx)))

    @classmethod
    def _load_latent(cls, path):
        if _st_load is None:
            raise RuntimeError("Multi Condition: safetensors is not available.")
        data = _st_load(path)
        if "video" not in data or "audio" not in data:
            raise ValueError("Multi Condition: %s is not an H3 AV latent." % path)
        video = data.pop("video").contiguous().clone()
        audio = data.pop("audio").contiguous().clone()
        return {"samples": [video, audio]}

    @classmethod
    def execute(
        cls,
        model=None,
        conditioning=None,
        latent=None,
        vae=None,
        audio_vae=None,
        source_audio=None,
        prefix="h3_output/clip",
        index=0,
        context_length="22",
        audio_context_length=24,
        seam_min=0.8,
        lock_audio=False,
        split_audio=True,
        ignore_head_latent=False,
        ignore_tail_latent=False,
        length=124,
        keyframe_images=None,
        keyframe_indices="-1,0",
    ) -> io.NodeOutput:
        index = int(index)

        # ---- anchor extra keyframe images (multi-keyframe) ----
        # Encoded into minimax_keyframes BEFORE pinning the tail, so _pin_tail's
        # merge keeps any of these that fall past the pinned head. Images pair
        # with keyframe_indices (comma-separated) by order; a negative index
        # counts back from the clip's end.
        extra_kfs = []
        video = _video_from_latent(latent)
        frame_count = _pixel_frames(int(video.shape[2]))
        width = int(video.shape[4]) * 16
        height = int(video.shape[3]) * 16
        images = dict(keyframe_images or {})
        # images dict keys are keyframe_0, keyframe_1, ... in order
        image_slots = sorted(
            [k for k in images if images[k] is not None],
            key=lambda k: int(k.rsplit("_", 1)[-1]),
        )
        # parse comma-separated indices
        idx_parts = [
            p.strip() for p in str(keyframe_indices or "").split(",") if p.strip()
        ]
        frame_list = [int(p) for p in idx_parts]
        for slot_key, fidx in zip(image_slots, frame_list):
            img = images[slot_key]
            if vae is None:
                raise ValueError("Multi Condition: keyframe images need vae wired.")
            resolved = fidx if fidx >= 0 else frame_count + fidx
            if resolved < 0 or resolved >= frame_count:
                raise ValueError(
                    "Multi Condition: keyframe frame %d is outside the clip's %d frames."
                    % (fidx, frame_count)
                )
            frame = img[:1]
            if frame.shape[1] != height or frame.shape[2] != width:
                frame = _resize_keyframe(frame, width, height)
            z = vae.encode(frame)
            extra_kfs.append(
                {"resolved_frame_index": resolved, "latent": z, "manual": True}
            )

        if extra_kfs:
            out = []
            for emb, extra in conditioning:
                d = extra.copy()
                prior = list(d.get("minimax_keyframes") or [])
                d["minimax_keyframes"] = prior + extra_kfs
                out.append([emb, d])
            conditioning = out

        # ---- load head (index-1) and tail (index+1) anchors ----
        # head = previous clip's tail (pins the current clip's HEAD).
        #   ignore_head_latent skips it; a missing head errors.
        # tail = next clip's head (pins the current clip's TAIL, for two-ended
        #   anchoring when generating a middle clip). ignore_tail_latent skips
        #   it; a missing tail is silently ignored.
        head_latent = None
        if index > 0 and not ignore_head_latent:
            path = cls._slot_path(prefix, index - 1)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    "Multi Condition: no head latent for clip %d at %s. Generate "
                    "clip %d first, or set ignore_head_latent."
                    % (index - 1, path, index - 1)
                )
            head_latent = cls._load_latent(path)

        tail_latent = None
        if not ignore_tail_latent:
            path = cls._slot_path(prefix, index + 1)
            if os.path.isfile(path):
                tail_latent = cls._load_latent(path)

        # ---- two-ended anchoring: pin the next clip's head into the tail ----
        # No latent extension: the tail anchor (next clip's head, context_length
        # frames) locks the last context_length frames of THIS clip's latent, so
        # the tail matches the next clip's true start. Those frames are trimmed
        # off afterward.
        tail_ctx = int(context_length) if tail_latent is not None else 0

        # ---- continuation: pin head (video + audio) + pin tail (video) ----
        trim_head = 0
        trim_tail = 0
        if head_latent is not None:
            conditioning, trim_head = cls._pin_tail(
                conditioning,
                latent,
                head_latent,
                int(context_length),
                int(audio_context_length),
            )
        elif ignore_head_latent and index > 0:
            trim_head = int(context_length)

        if tail_latent is not None:
            conditioning = cls._pin_tail_end(
                conditioning,
                latent,
                tail_latent,
                tail_ctx,
            )
            # The tail anchor (index3's head) is the true continuation content:
            # it stays in the output, so trim_tail stays 0.

        # ---- redraw: pin head+tail video into latent samples + taper masks ----
        prefix_steps = 0
        if head_latent is not None or tail_latent is not None:
            manual_frames = [kf["resolved_frame_index"] for kf in extra_kfs]
            latent, prefix_steps = cls._redraw_video(
                latent,
                head_latent,
                tail_latent,
                int(context_length),
                tail_ctx,
                seam_min,
                manual_frames=manual_frames,
            )

        # ---- audio drive: lock source audio (forward-looking) ----
        index_audio = None
        if lock_audio:
            if source_audio is None or audio_vae is None:
                raise ValueError(
                    "Multi Condition: lock_audio needs source_audio and audio_vae wired."
                )
            # This clip's audio segment, in frames at 24 fps, WITH an overlap
            # of `context_length` frames onto the previous clip's tail. The
            # overlapping head is the same frames the video pins as its anchor,
            # and MiniMaxH3MotionContextTrim cuts it off both streams together,
            # so every clip ships a full `length` frames and the chain joins
            # losslessly.
            #   index N -> [(length - ctx) * N, (length - ctx) * N + length)
            if split_audio:
                ctx = int(context_length)
                start_frame = (length - ctx) * index
                seg_len = length
                segment = cls._slice_audio_segment(source_audio, start_frame, seg_len)
            else:
                # Use the source audio as-is (no index-based splitting). The
                # caller supplies a per-clip track that already matches this
                # clip; _lock_audio still trims/pads it to `length` frames so it
                # stays in lockstep with the video.
                segment = source_audio
            latent, encoded_audio = cls._lock_audio(latent, segment, audio_vae, length)
            index_audio = segment
            # Inject the fitted audio latent as the conditioning's audio
            # reference so the DiT lip-syncs to THIS segment (not the full
            # track). Replaces the first kind=="audio" ref, or appends one.
            conditioning = cls._inject_audio_ref(conditioning, encoded_audio)

        # ---- dynamic remask: per-step video-prefix remask on a model clone ----
        patched_model = model
        if model is not None and prefix_steps > 0:
            patched_model = _install_video_prefix_remask(
                model,
                latent,
                prefix_steps,
                seam_min,
            )

        return io.NodeOutput(
            patched_model, conditioning, latent, trim_head, trim_tail, index_audio
        )

    # ------------------------------------------------------------------
    @classmethod
    def _inject_audio_ref(cls, conditioning, audio_latent):
        """Replace the first audio reference's latent with `audio_latent`.

        The <Audio 1> tag in the prompt was already baked in at tokenize time,
        so we can't renumber it; we only swap the DiT payload (the audio latent
        the model actually attends to) to point at this clip's segment.
        """
        ref_audio_t = int(audio_latent.shape[-1])
        out = []
        for emb, extra in conditioning:
            d = extra.copy()
            refs = d.get("minimax_refs") or []
            replaced = False
            new_refs = []
            for ref in refs:
                if not replaced and ref.get("kind") == "audio":
                    ref = dict(ref)
                    ref["audio_latent"] = audio_latent
                    ref["ref_audio_t"] = ref_audio_t
                    replaced = True
                new_refs.append(ref)
            if not replaced:
                new_refs.append(
                    {
                        "kind": "audio",
                        "ref_audio_t": ref_audio_t,
                        "audio_latent": audio_latent,
                    }
                )
            d["minimax_refs"] = new_refs
            out.append([emb, d])
        return out

    # ------------------------------------------------------------------
    @classmethod
    def _slice_audio_segment(cls, source_audio, start_frame, seg_len):
        """Cut this clip's audio segment out of the full source track.

        Returns an AUDIO dict for frames [start_frame, start_frame + seg_len)
        at 24 fps, converted to samples. start_frame is the cumulative frame
        offset of all previous clips; seg_len is this clip's effective length
        (length, or length - context_length after the first clip).
        """
        waveform = source_audio.get("waveform")  # [B, C, L]
        sample_rate = int(source_audio.get("sample_rate"))
        if waveform is None or sample_rate is None:
            raise ValueError(
                "Multi Condition: source_audio missing waveform/sample_rate."
            )

        start_sec = start_frame / float(FPS)
        end_sec = (start_frame + seg_len) / float(FPS)

        total = waveform.shape[-1]
        start = int(round(start_sec * sample_rate))
        end = int(round(end_sec * sample_rate))
        start = max(0, min(start, total))
        end = max(0, min(end, total))

        if end <= start:
            raise ValueError(
                "Multi Condition: audio segment [%.2fs, %.2fs) is empty or past "
                "the end of the source audio (%.2fs). Check length/context_length."
                % (start_sec, end_sec, total / sample_rate)
            )

        segment = waveform[..., start:end].clone()
        return {"waveform": segment, "sample_rate": sample_rate}

    # ------------------------------------------------------------------
    @classmethod
    def _pin_tail(
        cls, conditioning, latent, context_latent, context_length, audio_context_length
    ):
        video = _video_from_latent(latent)
        latent_t = int(video.shape[2])
        width = int(video.shape[4]) * 16
        height = int(video.shape[3]) * 16
        frame_count = _pixel_frames(latent_t)

        src_video = _video_from_latent(context_latent)
        src_w = int(src_video.shape[4]) * 16
        src_h = int(src_video.shape[3]) * 16
        if src_w != width or src_h != height:
            raise ValueError(
                "Multi Condition: context latent is %dx%d but this clip is %dx%d. "
                "A latent cannot be resized; regenerate the previous clip at this "
                "resolution." % (src_w, src_h, width, height)
            )

        available = _pixel_frames(int(src_video.shape[2]))
        n = min(context_length, available)
        if n < 1:
            raise ValueError("Multi Condition: no frames available to pin")

        blocks, offsets, covered = _video_tail_from_latent(context_latent, n)
        span = covered

        # Collect the frame indices already occupied by MANUAL keyframes (the
        # user-anchored authoritative frames). Tail frames that would land on
        # those indices are dropped so the manual keyframe owns that spot and
        # is not diluted by the drifted previous-tail frame.
        manual_indices = set()
        for emb, extra in conditioning:
            for kf in extra.get("minimax_keyframes") or []:
                if kf.get("manual"):
                    manual_indices.add(kf.get("resolved_frame_index", 0))

        keyframes = []
        for p, blk in zip(offsets, blocks):
            if p in manual_indices:
                continue
            keyframes.append({"resolved_frame_index": p, "latent": blk})

        # audio tail (end-aligned with the pinned video)
        audio_kf = None
        if audio_context_length > 0:
            a_frames = audio_context_length or span
            audio_latent, ref_audio_t, overhang = _audio_tail_from_latent(
                context_latent, a_frames
            )
            end_frame = float(span) + overhang / FRAME_RESCALE
            end_coord = round(FRAME_RESCALE * end_frame)
            end_frame = end_coord / FRAME_RESCALE
            audio_kf = {
                "resolved_frame_index": end_frame - ref_audio_t / FRAME_RESCALE,
                "audio_latent": audio_latent,
            }

        # merge into conditioning. Manual keyframes (user-anchored via
        # keyframe_images) always win — even inside the pinned head — because
        # they are the authoritative identity anchor. Non-manual anchors inside
        # the pinned head are dropped (they are stale tail frames).
        head_end = span
        tail_kfs = [audio_kf] if audio_kf is not None else []
        out = []
        for emb, extra in conditioning:
            d = extra.copy()
            prior = d.get("minimax_keyframes") or []
            kept = [
                dict(kf)
                for kf in prior
                if kf.get("manual") or kf.get("resolved_frame_index", 0) >= head_end
            ]
            d["minimax_keyframes"] = kept + keyframes + tail_kfs
            out.append([emb, d])

        return out, span

    # ------------------------------------------------------------------
    @classmethod
    def _pin_tail_end(cls, conditioning, latent, tail_latent, context_length):
        """Pin the NEXT clip's head (tail_latent) into the current clip's TAIL,
        as conditioning keyframes at the end of the timeline. This is the
        two-ended-anchoring mirror of _pin_tail: _pin_tail anchors the previous
        tail at the head, this anchors the next head at the tail."""
        video = _video_from_latent(latent)
        latent_t = int(video.shape[2])
        width = int(video.shape[4]) * 16
        height = int(video.shape[3]) * 16
        frame_count = _pixel_frames(latent_t)

        src_video = _video_from_latent(tail_latent)
        src_w = int(src_video.shape[4]) * 16
        src_h = int(src_video.shape[3]) * 16
        if src_w != width or src_h != height:
            raise ValueError(
                "Multi Condition: tail latent is %dx%d but this clip is %dx%d."
                % (src_w, src_h, width, height)
            )

        available = _pixel_frames(int(src_video.shape[2]))
        n = min(context_length, available)
        if n < 1:
            return conditioning

        # Take the HEAD of the tail latent (the next clip's first frames).
        blocks, offsets, covered = _video_head_from_latent(tail_latent, n)
        span = covered

        # Anchor these frames at the END of the current clip's timeline.
        end_start = frame_count - span
        keyframes = []
        for p, blk in zip(offsets, blocks):
            keyframes.append({"resolved_frame_index": end_start + p, "latent": blk})

        out = []
        for emb, extra in conditioning:
            d = extra.copy()
            prior = list(d.get("minimax_keyframes") or [])
            d["minimax_keyframes"] = prior + keyframes
            out.append([emb, d])
        return out

    # ------------------------------------------------------------------
    @classmethod
    def _redraw_video(
        cls,
        latent,
        head_latent,
        tail_latent,
        context_length,
        tail_ctx,
        seam_min,
        manual_frames=None,
    ):
        """Pin the previous clip's tail (head anchor) and, optionally, the next
        clip's head (tail anchor) into the current latent samples, and build a
        taper denoise mask on each end. The head/taper ends redraw gradually
        (head 1.0 -> seam_min) while the middle stays fully generated. Returns
        (latent, prefix_steps).

        manual_frames: pixel-frame indices that hold a MANUAL keyframe (the
        user's authoritative identity anchor). Those latent steps get mask 0
        (hard-lock) so the seam redraw never overwrites the manual keyframe.
        """
        import comfy.nested_tensor

        samples = latent["samples"]
        if not getattr(samples, "is_nested", False):
            raise ValueError("Multi Condition: redraw needs an H3 AV latent.")
        parts = list(samples.unbind())
        if len(parts) < 2:
            raise ValueError("Multi Condition: redraw latent has no audio stream.")
        video, audio = parts[0], parts[1]

        width = int(video.shape[4]) * 16
        height = int(video.shape[3]) * 16
        t = int(video.shape[2])
        h = int(video.shape[3])
        w = int(video.shape[4])

        patched_video = video.clone()
        seam = _clamp_seam_min(seam_min)

        head_steps = 0
        if head_latent is not None:
            src_video = _video_from_latent(head_latent)
            src_w = int(src_video.shape[4]) * 16
            src_h = int(src_video.shape[3]) * 16
            if src_w != width or src_h != height:
                raise ValueError(
                    "Multi Condition: head latent is %dx%d but this clip is %dx%d."
                    % (src_w, src_h, width, height)
                )
            available = _pixel_frames(int(src_video.shape[2]))
            n = min(int(context_length), available)
            if n >= 1:
                blocks, offsets, covered = _video_tail_from_latent(head_latent, n)
                head_tail_video = torch.cat(blocks, dim=2)
                head_steps = min(int(head_tail_video.shape[2]), t - 1)
                if head_steps >= 1:
                    patched_video[:, :, :head_steps] = head_tail_video[
                        :, :, :head_steps
                    ].to(device=patched_video.device, dtype=patched_video.dtype)

        tail_steps = 0
        if tail_latent is not None:
            src_video = _video_from_latent(tail_latent)
            src_w = int(src_video.shape[4]) * 16
            src_h = int(src_video.shape[3]) * 16
            if src_w != width or src_h != height:
                raise ValueError(
                    "Multi Condition: tail latent is %dx%d but this clip is %dx%d."
                    % (src_w, src_h, width, height)
                )
            available = _pixel_frames(int(src_video.shape[2]))
            n = min(int(tail_ctx), available)
            if n >= 1:
                blocks, offsets, covered = _video_head_from_latent(tail_latent, n)
                tail_head_video = torch.cat(blocks, dim=2)
                tail_steps = min(int(tail_head_video.shape[2]), t - 1)
                if tail_steps >= 1:
                    patched_video[:, :, t - tail_steps :] = tail_head_video[
                        :, :, :tail_steps
                    ].to(device=patched_video.device, dtype=patched_video.dtype)

        # video mask [B,1,T,H,W] so mask[0,0] is [T,H,W] (H3 mask_row_values layout)
        video_mask = torch.ones(
            (1, 1, t, h, w), device=patched_video.device, dtype=torch.float32
        )

        if head_steps >= 1:
            weights = _prefix_token_weights(head_steps, seam_min=seam)
            ramp = torch.tensor(
                weights, device=patched_video.device, dtype=torch.float32
            )
            video_mask[:, :, : len(ramp)] = ramp.view(1, 1, len(ramp), 1, 1)

        if tail_steps >= 1:
            # taper the tail end: the seam sits at the LAST step (the boundary
            # into the next clip), so lock it there and taper toward 1.0 toward
            # the interior — same orientation as the head, no flip.
            weights = _prefix_token_weights(tail_steps, seam_min=seam)
            ramp = torch.tensor(
                weights, device=patched_video.device, dtype=torch.float32
            )
            video_mask[:, :, t - len(ramp) :] = ramp.view(1, 1, len(ramp), 1, 1)

        # Hard-lock the latent steps that hold a manual keyframe: mask 0 keeps
        # those frames bit-for-bit so the seam redraw never touches them.
        for fidx in manual_frames or ():
            step = _frame_to_latent_step(int(fidx))
            if step is not None and 0 <= step < t:
                video_mask[:, :, step] = 0.0

        # audio stays untouched (locked by lock_audio if on); mask 0 keeps it.
        audio_mask = torch.zeros_like(audio, dtype=torch.float32)

        out = dict(latent)
        out["samples"] = comfy.nested_tensor.NestedTensor((patched_video, audio))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))

        return out, head_steps

    # ------------------------------------------------------------------
    @classmethod
    def _lock_audio(cls, latent, source_audio, audio_vae, length=124):
        import comfy.nested_tensor

        samples = latent["samples"]
        if not getattr(samples, "is_nested", False):
            raise ValueError("Multi Condition: lock_audio needs an H3 AV latent.")
        parts = list(samples.unbind())
        if len(parts) < 2:
            raise ValueError(
                "Multi Condition: could not find the audio half of the AV latent."
            )
        video_latent, template_audio = parts[0], parts[1]

        waveform = source_audio.get("waveform")
        sample_rate = source_audio.get("sample_rate")
        if waveform is None or sample_rate is None:
            raise ValueError(
                "Multi Condition: source_audio missing waveform/sample_rate."
            )
        if waveform.ndim != 3:
            raise ValueError(
                "Multi Condition: source_audio must be [batch, channels, samples]."
            )

        vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
        if int(sample_rate) != vae_sr:
            import comfy.audio

            waveform = comfy.audio.resample(waveform, int(sample_rate), vae_sr)

        encoded = audio_vae.encode(waveform[:1].movedim(1, -1))

        if encoded.ndim != 4 or template_audio.ndim != 4:
            raise ValueError("Multi Condition: audio latent layout mismatch.")
        if encoded.shape[1:-1] != template_audio.shape[1:-1]:
            raise ValueError(
                "Multi Condition: source audio latent layout does not match the H3 audio latent."
            )
        # The locked audio must span exactly `length` video frames (40 Hz grid),
        # NOT whatever length the template audio stream happens to carry. The
        # caller derives `length` from the same parameter that sizes the video
        # clip, so the two stay in lockstep even when `length` is not 124.
        target_t = int(round(int(length) * FRAME_RESCALE))
        current_t = encoded.shape[-1]
        if current_t > target_t:
            encoded = encoded[..., :target_t]
        elif current_t < target_t:
            pad = encoded.new_zeros((*encoded.shape[:-1], target_t - current_t))
            encoded = torch.cat((encoded, pad), dim=-1)
        encoded = encoded.to(device=template_audio.device, dtype=template_audio.dtype)

        output = latent.copy()
        output["samples"] = comfy.nested_tensor.NestedTensor((video_latent, encoded))
        # Preserve the video denoise mask already built by _redraw_video (taper
        # seam). lock_audio owns the audio stream only: keep the existing video
        # mask if one is present, otherwise fall back to all-ones (fully redraw).
        video_mask = None
        existing_mask = latent.get("noise_mask")
        if existing_mask is not None and getattr(existing_mask, "is_nested", False):
            mask_parts = list(existing_mask.unbind())
            if mask_parts and torch.is_tensor(mask_parts[0]):
                video_mask = mask_parts[0]
        if video_mask is None:
            video_mask = torch.ones_like(video_latent)
        output["noise_mask"] = comfy.nested_tensor.NestedTensor(
            (
                video_mask,
                torch.zeros_like(encoded),
            )
        )
        # Return the fitted audio latent too, so the caller can also inject it
        # as the conditioning's audio reference (lip-sync needs the DiT to
        # SEE this exact audio, not just the locked noise_mask to preserve it).
        return output, encoded
