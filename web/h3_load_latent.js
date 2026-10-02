import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Load Latent File for Minimax H3 frontend:
// - directory field uses a VHS-style path search dialog (the same
//   KWNODES_LATENT_PATH widget PromptFilePicker / Load Image (Path) use).
// - latent dropdown refreshes when the directory or `sub` changes; a refresh
//   button re-scans the directory.
//
// Registered from its own route (/kwnodes/getpath_latent) and its own widget
// type, so this pack does not depend on comfyui-kwnodes being installed.
//
// Load Latent for Minimax H3 (prefix + index) is backend-only and needs none
// of this.

const NODE_TYPE = "H3LoadLatentFile";

// The backend owns the ceiling (ComfyUI's working directory); the dialog only
// ever asks one level at a time, so it never walks above it.
const PATH_ROUTE = "/kwnodes/getpath_latent";

function pathStem(value) {
  // ROOT-relative path: the whole value is the directory being browsed, with
  // an optional trailing segment being typed. "" = the ROOT itself.
  const v = String(value || "");
  const i = v.lastIndexOf("/");
  if (i < 0) return ["", v];
  return [v.slice(0, i + 1), v.slice(i + 1)];
}

// VHS-style path search dialog bound to a path widget's mouse.
function pathSearchBox(event, pos, node) {
  const pathWidget = this;
  if (pathWidget.prompt) return;
  pathWidget.prompt = true;

  const dialog = document.createElement("div");
  dialog.className = "litegraph litesearchbox graphdialog rounded";
  dialog.innerHTML =
    '<span class="name">Directory</span> <input autofocus type="text" class="value">' +
    '<button class="rounded">OK</button><div class="helper"></div>';
  dialog.close = () => {
    dialog.remove();
    pathWidget.prompt = false;
  };
  document.body.append(dialog);

  const input = dialog.querySelector(".value");
  const optionsEl = dialog.querySelector(".helper");
  input.value = pathWidget.value;

  let timeout = null;
  let lastPath = null;
  let options = [];

  const commit = () => {
    pathWidget.value = input.value;
    pathWidget.callback?.(pathWidget.value);
    dialog.close();
    node.graph?.setDirtyCanvas?.(true);
  };

  const updateOptions = async () => {
    timeout = null;
    const [path, remainder] = pathStem(input.value);
    if (lastPath !== path) {
      try {
        const url = api.apiURL(
          PATH_ROUTE + "?" + new URLSearchParams({ path }),
        );
        const resp = await fetch(url);
        options = (await resp.json()) || [];
        options.sort();
      } catch (_) {
        options = [];
      }
      lastPath = path;
    }
    optionsEl.innerHTML = "";
    for (const option of options) {
      const name = option.endsWith("/") ? option.slice(0, -1) : option;
      if (!name.startsWith(remainder)) continue;
      const el = document.createElement("div");
      el.innerText = option;
      el.className = "litegraph lite-search-item is-dir";
      el.addEventListener("click", () => {
        input.value = path + option;
        if (timeout) clearTimeout(timeout);
        timeout = setTimeout(updateOptions, 10);
      });
      optionsEl.appendChild(el);
    }
  };

  input.addEventListener("keydown", (e) => {
    if (e.keyCode === 27) {
      dialog.close();
    } else if (e.keyCode === 13 && e.target.localName !== "textarea") {
      commit();
    } else if (e.keyCode === 9) {
      const first = optionsEl.firstChild;
      if (first) {
        const [path] = pathStem(input.value);
        input.value = path + first.innerText;
        e.preventDefault();
        e.stopPropagation();
      }
    } else {
      if (timeout) clearTimeout(timeout);
      timeout = setTimeout(updateOptions, 10);
      return;
    }
    e.preventDefault();
    e.stopPropagation();
  });

  dialog.querySelector("button").addEventListener("click", commit);

  const rect = app.canvas?.canvas?.getBoundingClientRect?.();
  if (rect) {
    dialog.style.left = (event?.clientX || 0) - rect.left - 20 + "px";
    dialog.style.top = (event?.clientY || 0) - rect.top - 20 + "px";
  }
  setTimeout(() => {
    input.focus();
    updateOptions();
  }, 10);
}

app.registerExtension({
  name: "kwnodes.H3LoadLatentFile",

  // Register a KWNODES_LATENT_PATH widget type (VHS-style), a ROOT-relative
  // path picker for the latent directory field.
  async getCustomWidgets() {
    return {
      KWNODES_LATENT_PATH(node, inputName, inputData) {
        const w = {
          name: inputName,
          type: "KWNODES_LATENT_PATH",
          value: "",
          options: inputData?.[1] || {},
          draw(ctx, node, widgetWidth, y, H) {
            const scale = app.canvas.ds.scale;
            const showText =
              scale >= (app.canvas.low_quality_zoom_threshold ?? 0.5);
            const margin = 15;
            ctx.textAlign = "left";
            ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
            ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
            ctx.beginPath();
            if (showText)
              ctx.roundRect(margin, y, widgetWidth - margin * 2, H, [H * 0.5]);
            else ctx.rect(margin, y, widgetWidth - margin * 2, H);
            ctx.fill();
            if (showText) {
              if (!this.disabled) ctx.stroke();
              ctx.save();
              ctx.beginPath();
              ctx.rect(margin, y, widgetWidth - margin * 2, H);
              ctx.clip();
              const label = this.label || this.name;
              if (label != null) {
                ctx.fillStyle = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;
                ctx.fillText(label, margin * 2, y + H * 0.7);
              }
              ctx.fillStyle = this.value ? LiteGraph.WIDGET_TEXT_COLOR : "#777";
              ctx.textAlign = "right";
              ctx.fillText(
                String(this.value || this.options.placeholder || ""),
                widgetWidth - margin * 2,
                y + H * 0.7,
              );
              ctx.restore();
            }
          },
          mouse: pathSearchBox,
        };
        if (w.options.default != null) w.value = w.options.default;
        node.widgets = node.widgets || [];
        node.widgets.push(w);
        return w;
      },
    };
  },

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeType.comfyClass !== NODE_TYPE) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);

      // Replace the directory STRING widget with a KWNODES_LATENT_PATH
      // autocomplete, then re-scan the file list when it changes.
      const dirWidget = this.widgets?.find((w) => w.name === "directory");
      if (dirWidget) {
        const idx = this.widgets.indexOf(dirWidget);
        const defaultVal = dirWidget.value;
        const placeholder = dirWidget.options?.placeholder;
        this.widgets.splice(idx, 1);
        const custom = app.widgets.KWNODES_LATENT_PATH(this, "directory", [
          "STRING",
          { default: defaultVal, placeholder },
        ]);
        this.widgets.splice(this.widgets.length - 1, 1);
        this.widgets.splice(idx, 0, custom);
        custom.callback = (value) => this._refreshLatents(value);
      }

      // Refresh button: re-scan the directory and select the newest latent.
      // The selection change alone is enough to re-read: the backend
      // fingerprints the file (path + mtime + size), so picking a fresh slot
      // or re-queueing after a re-roll always reloads.
      const refreshBtn = this.addWidget(
        "button",
        "refresh",
        "↻ Refresh",
        () => {
          const dirW = this.widgets?.find((w) => w.name === "directory");
          if (dirW) this._refreshLatents(dirW.value, true);
        },
        { serialize: false },
      );
      refreshBtn.label = "↻ Refresh";

      // Toggling `sub` re-scans the file list (recursive vs flat).
      const subW = this.widgets?.find((w) => w.name === "sub");
      if (subW) {
        subW.label = "include sub directory";
        const node = this;
        const origSubCb = subW.callback;
        subW.callback = function (value) {
          origSubCb?.call(this, value);
          const dirW = node.widgets?.find((w) => w.name === "directory");
          if (dirW) node._refreshLatents(dirW.value);
        };
      }

      // Auto-refresh the file list once after the node is created (covers page
      // load, when saved workflows are reconstructed).
      const dirW0 = this.widgets?.find((w) => w.name === "directory");
      if (dirW0) {
        setTimeout(() => this._refreshLatents(dirW0.value), 0);
      }

      return r;
    };

    nodeType.prototype._refreshLatents = async function (directory, selectLatest) {
      const fileWidget = this.widgets?.find((w) => w.name === "latent");
      if (!fileWidget || !directory) return;
      const subW = this.widgets?.find((w) => w.name === "sub");
      const sub = subW?.value ? 1 : 0;
      try {
        const url = api.apiURL(
          "/kwnodes/list_latent_files?" +
            new URLSearchParams({ directory, sub }),
        );
        const resp = await fetch(url);
        if (!resp.ok) {
          // A failed scan must not look like "nothing changed". Surface it and
          // clear the list so a stale selection cannot be queued by accident.
          const text = await resp.text().catch(() => "");
          console.error("[H3LoadLatent] list_latent_files failed", resp.status, text.slice(0, 300));
          fileWidget.options.values = ["(no latents)"];
          fileWidget.value = "(no latents)";
          this.setDirtyCanvas?.(true, true);
          return;
        }
        const data = await resp.json();
        const files = data.files || [];
        // An empty result is meaningful: the folder holds no latents, so show
        // the placeholder rather than leaving a stale list from the last
        // directory in place.
        fileWidget.options.values = files.length ? files : ["(no latents)"];
        if (!files.length) {
          fileWidget.value = "(no latents)";
          this.setDirtyCanvas?.(true, true);
          return;
        }
        // On manual refresh, always jump to the newest file (files are sorted
        // by mtime descending, so [0] is the latest). Otherwise keep the
        // current selection if it still exists.
        if (selectLatest) {
          fileWidget.value = files[0];
        } else if (!files.includes(fileWidget.value)) {
          fileWidget.value = files[0];
        }
        this.setDirtyCanvas?.(true, true);
      } catch (e) {
        console.error("[H3LoadLatent] could not list latents", e);
      }
    };
  },
});
