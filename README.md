# comfyui-kwnodes-h3

MiniMax H3 clip-chain nodes for ComfyUI, using the V3 (`io.ComfyNode`) API.

Kept separate from `comfyui-kwnodes` (V1 `NODE_CLASS_MAPPINGS`) because ComfyUI
loads one API style per custom-node directory, not both.

## Nodes

### Multi Condition for Minimax H3

Continue an H3 clip chain from a `<prefix>-<index>` slot. Loads clip
`index - 1`'s latent, pins its tail as the HEAD anchor, and (optionally) loads
clip `index + 1`'s latent to pin its head as the TAIL anchor — two-ended
anchoring for generating a middle clip from both neighbours. Also locks a
source audio track (lip-sync) and anchors extra keyframe images.

Key inputs:

- `ignore_head_latent` — skip the head anchor (previous clip's tail). A missing
  head latent errors.
- `ignore_tail_latent` — skip the tail anchor (next clip's head). A missing
  tail latent is silently ignored. Generate the two end clips first, then the
  middle clip with both anchors enabled.
- `seam_min` — redraw strength at the seam (0.0 = hard-lock, higher = redraw).
  The seam redraws gradually so the join transitions instead of snapping.
- `keyframe_images` + `keyframe_indices` — anchor manual keyframes. Manual
  keyframes are authoritative: they win over the tail anchors and their latent
  steps are hard-locked (mask 0) so the seam redraw never overwrites them.
- `model` — wire the sampler model to install a per-step video-prefix remask
  wrapper (breaks later-segment drift by loosening the seam early and locking
  it late in denoising).

### Save Latent for Minimax H3

Save the sampler's AV latent to a `<prefix>-<index>.safetensors` slot for the
next clip. `trim_head` / `trim_tail` crop the anchor zones before saving, so a
saved latent starts at the true content (needed for the next clip's anchors to
match). 0 = save the full latent.

### Load Latent for Minimax H3

Address a slot by `prefix` + `index` — the mirror of Save Latent for Minimax H3.
Save clip N with prefix P, read it back with the same P and N. No directory to
point at and no list to refresh; it resolves `<prefix>-<index:05>.safetensors`
the same way Multi Condition does, so a chain's wiring stays symmetric.

### Load Latent File for Minimax H3

Pick the latent from a dropdown. Same shape as Load Image (Path):

- `directory` — where to look, relative to the ComfyUI root. VHS-style
  autocomplete confined to the root.
- `sub` (`include sub directory`) — also list latents in subdirectories, named
  relative to the directory.
- `latent` — the file dropdown, newest first.
- `↻ Refresh` — re-scan the directory and select the newest latent.

Use this one when you want to see what is on disk and choose by name.

### Video Trim for Minimax H3

Drop the leading (and optionally trailing) pinned frames from a decoded clip,
picture and sound together. `trim_head` / `trim_tail` are frame counts (0 =
don't trim that end). `match_tail` evens H3's 40 Hz audio grid against the 24
fps picture so the audio tail matches frames/fps exactly.

## Two-ended anchoring (the middle-clip workflow)

To fight gradual identity/content drift across a long chain, generate the two
end clips first, then the middle clip anchored to both:

```
index1 (clean) ──┐
                 ├──> index2 (middle, two-ended anchored)
index3 (clean) ──┘
```

- `ignore_head_latent = False`, `ignore_tail_latent = False` on index2.
- The head anchor pins index1's tail; the tail anchor pins index3's head.
- index3 is generated with `ignore_head_latent = True` and saved with
  `trim_head` so its saved latent starts at the true content, which is what
  index2's tail anchor locks onto — the join fuses cleanly.
- `trim_tail` stays 0: the tail anchor is the true continuation content, not a
  disposable seam.

Post-production: use DaVinci Resolve's cross dissolve at the joins to smooth
any residual colour jump between clips.
