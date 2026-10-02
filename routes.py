"""Custom HTTP routes for comfyui-kwnodes-h3.

Only what the Load Latent frontend needs: one-level subdirectory listing for the
VHS-style directory dialog, and the recursive latent file list for the dropdown.
Self-contained so this pack does not require comfyui-kwnodes to be installed.
"""

import os

from aiohttp import web
from server import PromptServer

from .h3_load_latent import _abs, _list_latents

@PromptServer.instance.routes.get("/kwnodes/getpath_latent")
async def get_path_latent(request):
    """Return subdirectory names (with trailing '/') directly under a path, for
    the directory dialog. Lists only ONE level, confined to the project root."""
    query = request.rel_url.query
    if "path" not in query:
        return web.Response(status=204)

    path = _abs(query["path"])
    if not os.path.isdir(path):
        return web.json_response([])

    items = []
    try:
        for item in os.scandir(path):
            try:
                if item.is_dir():
                    items.append(item.name + "/")
            except OSError:
                pass
    except OSError:
        return web.json_response([])
    items.sort()
    return web.json_response(items)


@PromptServer.instance.routes.get("/kwnodes/list_latent_files")
async def list_latent_files(request):
    """Return the latent files in a ROOT-relative directory (recursive when sub=1)."""
    directory = request.query.get("directory", "")
    sub = request.query.get("sub", "1") in ("1", "true", "True")
    try:
        files = _list_latents(directory, sub)
    except OSError as e:
        return web.json_response({"files": [], "error": str(e)}, status=500)
    return web.json_response({"files": files})
