"""H3 Load Latent — two ways to pull a saved AV latent back off disk.

Load Latent File for Minimax H3
  Pick the file from a dropdown, same shape as Load Image (Path): a `directory`
  field with VHS-style autocomplete confined to the project root, an `include
  sub directory` toggle, and a refresh button. Files are sorted newest first,
  so the slot just saved is at the top. Use this when you want to see what is
  on disk and choose it by name.

Load Latent for Minimax H3
  Address a slot by `prefix` + `index`, the mirror image of Save Latent for
  Minimax H3: save clip N with prefix P and read it back with the same P and N.
  No directory to point at and no list to refresh — it resolves
  <prefix>-<index:05>.safetensors the exact way Multi Condition does, so a
  chain's wiring stays symmetric. Use this inside a chain.

Both return the latent as the graph originally carried it: the video and audio
streams repacked into an H3 AV latent, ready for the sampler, Multi Condition's
latent input, or the upscale / trim nodes. The round trip is lossless — it is
exactly what Save Latent wrote.
"""

import os

from comfy.nested_tensor import NestedTensor
from comfy_api.latest import io

try:
    from safetensors.torch import load_file as _st_load
except ImportError:
    _st_load = None

LATENT_EXTENSIONS = (".safetensors",)

# Directory ceiling: ComfyUI's working directory. The directory picker is
# confined to this subtree (no escaping upward via ".."). Same ROOT the other
# kwnodes path nodes use, so one directory field behaves the same everywhere.
ROOT = os.path.realpath(os.getcwd())

# H3 chain latents are written under the output folder by Save Latent. The
# field defaults to `output/h3_output`; point it at whatever folder the chain
# actually saves into.
DEFAULT_DIRECTORY = "output/h3_output"

NO_LATENTS = "(no latents)"

SLOT_PREFIX_DEFAULT = "h3_output/clip"


def _hint_directories(directory, sub):
    """Directories under ROOT that DO hold latents, for the empty-state hint."""
    found = []
    for root, dirs, files in os.walk(_abs(directory if sub else ROOT)):
        dirs.sort()
        if any(f.lower().endswith(LATENT_EXTENSIONS) for f in files):
            found.append(os.path.relpath(root, ROOT))
        if len(found) >= 8:
            break
    return found


def _is_within(path, root):
    """True if `path` is `root` or inside it (both realpath'd, no `..` escape)."""
    p = os.path.realpath(path)
    r = os.path.realpath(root)
    return p == r or p.startswith(r + os.sep)


def _abs(directory):
    """Resolve a directory (relative to ROOT) to an absolute path, confined to
    ROOT. A leading '/' input is treated as relative anyway (stripped), so the
    field always shows a ROOT-relative path like 'output' or 'output/sub'.

    A None directory (ComfyUI validates before widgets carry values) falls back
    to the default rather than raising on os.path.join.
    """
    directory = os.path.expanduser(directory or "").strip() or DEFAULT_DIRECTORY
    directory = directory.lstrip("/")
    path = os.path.realpath(os.path.join(ROOT, directory))
    if not _is_within(path, ROOT):
        return ROOT
    return path


def _list_latents(directory, sub=True):
    """Return latent files under `directory` (ROOT-relative), newest first.

    With `sub`, walk subdirectories and prefix names with the subpath, so a
    chain living in its own folder still shows every slot in one list.
    """
    directory = _abs(directory)

    if not sub:
        try:
            entries = [
                e
                for e in os.listdir(directory)
                if e.lower().endswith(LATENT_EXTENSIONS)
                and os.path.isfile(os.path.join(directory, e))
            ]
        except OSError:
            return [NO_LATENTS]
        entries.sort(
            key=lambda e: os.path.getmtime(os.path.join(directory, e)),
            reverse=True,
        )
        return entries if entries else [NO_LATENTS]

    found = []  # (relpath, abspath)
    for root, dirs, files in os.walk(directory):
        dirs.sort()
        for f in files:
            if not f.lower().endswith(LATENT_EXTENSIONS):
                continue
            full = os.path.join(root, f)
            found.append((os.path.relpath(full, directory), full))
    if not found:
        return [NO_LATENTS]
    found.sort(key=lambda x: os.path.getmtime(x[1]), reverse=True)
    return [rel for rel, _ in found]


def _slot_path(prefix, index):
    """The <prefix>-<index:05>.safetensors slot Save Latent writes.

    Resolved through the same folder_paths call Save Latent and Multi Condition
    use, so all three agree on where a chain's slots live.
    """
    import folder_paths

    # ComfyUI validates a node before its widgets carry values, so prefix can
    # arrive as None/"" here; folder_paths.get_save_image_path would raise
    # "argument of type 'NoneType' is not iterable".
    if not prefix:
        prefix = SLOT_PREFIX_DEFAULT
    folder, filename, _c, _s, _f = folder_paths.get_save_image_path(
        prefix, folder_paths.get_output_directory()
    )
    return os.path.join(folder, "%s-%05d.safetensors" % (filename, int(index or 0)))


def _read_av_latent(path, node_name):
    """Load an H3 AV latent file and repack it as the graph's LATENT."""
    if _st_load is None:
        raise RuntimeError("%s: safetensors is not available." % node_name)

    data = _st_load(path)
    if "video" not in data or "audio" not in data:
        raise ValueError(
            "%s: %s is not an H3 AV latent (missing video/audio streams). It "
            "must be one written by Save Latent for Minimax H3."
            % (node_name, path)
        )

    # Clone off the mmap so a later save can overwrite this slot.
    video = data.pop("video").contiguous().clone()
    audio = data.pop("audio").contiguous().clone()

    # Must be a NestedTensor, not a plain list: the H3 AV latent contract is a
    # nested (video, audio) pair, and consumers such as Separate AV Latent call
    # .unbind() on it. A plain list loads fine but fails downstream.
    return {"samples": NestedTensor((video, audio))}


class H3LoadLatentFile(io.ComfyNode):
    """Load an H3 AV latent by picking the file from a dropdown."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3LoadLatentFile",
            display_name="Load Latent File for Minimax H3",
            category="kwnodes/H3",
            description=(
                "Load an H3 AV latent from a directory inside the project root "
                "and output it as LATENT. The directory field has VHS-style "
                "autocomplete confined to the root. With 'sub' on (default), "
                "latents in subdirectories are listed too, named relative to "
                "the directory. Newest files first."
            ),
            inputs=[
                io.String.Input(
                    "directory",
                    default=DEFAULT_DIRECTORY,
                    tooltip="Where to look for latents, relative to the ComfyUI root. Has VHS-style autocomplete.",
                ),
                io.Boolean.Input(
                    "sub",
                    default=True,
                    tooltip="ON: also list latents in subdirectories, named relative to the directory.",
                ),
                io.Combo.Input(
                    "latent",
                    options=_list_latents(DEFAULT_DIRECTORY, True),
                    tooltip="The latent file to load. Newest first. Press ↻ Refresh after saving a new one.",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.String.Output(display_name="latent_path"),
            ],
        )

    @classmethod
    def _resolve(cls, directory, latent):
        """Resolve (directory, latent) to an absolute path.

        `latent` may carry a subpath when recursion is on (e.g.
        "sub/slot-00001.safetensors"), so join it rather than treating it as a
        bare filename.
        """
        return os.path.join(_abs(directory), str(latent or "").strip())

    @classmethod
    def _empty_message(cls, directory, sub):
        """Say where the latents ARE when the chosen folder has none."""
        hint = _hint_directories(directory, sub)
        if hint:
            return "no latents in %s. Latents found in: %s" % (
                directory,
                ", ".join(hint[:5]),
            )
        return "no latents in %s" % directory

    @classmethod
    def validate_inputs(cls, directory=None, sub=None, latent=None, **kwargs):
        # Input names must stay visible to getfullargspec, and a LINKED input is
        # unresolvable at validation time (nothing has executed), so only a real
        # widget value is worth checking. See H3LoadLatent.validate_inputs.
        if directory is None or latent is None:
            return True
        files = _list_latents(directory, True if sub is None else sub)
        if latent == NO_LATENTS:
            return "H3 Load Latent File: %s" % cls._empty_message(directory, sub)
        if latent not in files:
            return "H3 Load Latent File: latent '%s' not found in %s" % (
                latent,
                directory,
            )
        return True

    @classmethod
    def fingerprint_inputs(cls, directory=DEFAULT_DIRECTORY, sub=True, latent=NO_LATENTS, **kwargs):
        """Cache on the file's identity, not its name.

        A slot keeps its name while its contents are overwritten, so without
        this ComfyUI would serve a stale latent forever after a re-roll.
        """
        path = cls._resolve(directory, latent)
        try:
            stat = os.stat(path)
        except OSError:
            return float("NaN")
        return "%s:%d:%d" % (path, stat.st_mtime_ns, stat.st_size)

    @classmethod
    def execute(cls, directory, sub, latent) -> io.NodeOutput:
        if latent == NO_LATENTS:
            raise ValueError(
                "H3 Load Latent File: %s" % cls._empty_message(directory, sub)
            )

        path = cls._resolve(directory, latent)
        if not _is_within(path, ROOT):
            raise ValueError(
                "H3 Load Latent File: path %s is outside the project root %s"
                % (path, ROOT)
            )
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "H3 Load Latent File: no latent at %s. Press ↻ Refresh to "
                "re-scan the directory." % path
            )

        return io.NodeOutput(_read_av_latent(path, "H3 Load Latent File"), path)


class H3LoadLatent(io.ComfyNode):
    """Load an H3 AV latent by chain slot (prefix + index)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3LoadLatent",
            display_name="Load Latent for Minimax H3",
            category="kwnodes/H3",
            description=(
                "Load an H3 AV latent from a <prefix>-<index> slot, the mirror "
                "of Save Latent for Minimax H3. Give it the same prefix and "
                "index the Save node used to write the latent."
            ),
            inputs=[
                io.String.Input(
                    "prefix",
                    default=SLOT_PREFIX_DEFAULT,
                    tooltip="Chain name + optional subfolder. Set it to the same value Save Latent used.",
                ),
                io.Int.Input(
                    "index",
                    default=0,
                    min=0,
                    max=9999,
                    tooltip="Which clip's slot to load (0-based). Set it to the same value Save Latent used when it wrote the latent.",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.String.Output(display_name="latent_path"),
            ],
        )

    @classmethod
    def validate_inputs(cls, prefix=None, index=None, **kwargs):
        # The input names must stay visible to getfullargspec (no collapsing to
        # just 'cls'), or ComfyUI never passes the workflow's values at all.
        #
        # A LINKED input has no resolvable value at validation time: nothing has
        # executed yet, so execution.get_input_data marks it missing and hands us
        # None. Failing here would reject every chained workflow -- the exact
        # case this node exists for -- so defer to execute(), which runs with
        # real values and raises a proper error if the slot is genuinely absent.
        if prefix is None or index is None:
            return True
        path = _slot_path(prefix, index)
        if not os.path.isfile(path):
            return "H3 Load Latent: no latent at %s" % path
        return True

    @classmethod
    def fingerprint_inputs(cls, prefix=SLOT_PREFIX_DEFAULT, index=0, **kwargs):
        """Cache on the file's identity, not the prefix/index pair.

        Re-rolling a clip overwrites the same slot, so keying on the name alone
        would serve the previous take forever.
        """
        path = _slot_path(prefix, index)
        try:
            stat = os.stat(path)
        except OSError:
            return float("NaN")
        return "%s:%d:%d" % (path, stat.st_mtime_ns, stat.st_size)

    @classmethod
    def execute(cls, prefix, index) -> io.NodeOutput:
        path = _slot_path(prefix, index)
        if not _is_within(path, ROOT):
            raise ValueError(
                "H3 Load Latent: path %s is outside the project root %s"
                % (path, ROOT)
            )
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "H3 Load Latent: no latent at %s. Generate clip %d first, or "
                "check the prefix." % (path, int(index))
            )

        return io.NodeOutput(_read_av_latent(path, "H3 Load Latent"), path)
