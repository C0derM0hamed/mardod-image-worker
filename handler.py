"""RunPod serverless worker: background removal (BiRefNet via rembg) + deterministic compositing.

Input (all images are base64 strings):
{
  "source_b64":  "...",            # original supplier photo (jpeg/png/webp). Triggers background removal.
  "cutout_b64":  "...",            # OR an already-removed RGBA PNG (compose only, no model call).
  "output":      "white" | "template" | "none",
  "template":    {"png_b64": "...", "box": [x, y, w, h], "shadow": true},   # fractions of the canvas
  "canvas":      1200,
  "fill":        0.88,             # max fraction of the box the product may occupy (white output)
  "shadow":      true,
  "model":       "birefnet-general"
}
Output:
{ "cutout_b64": "...png" (only when source_b64 was given), "image_b64": "...jpeg", "mime": "image/jpeg",
  "timings": {"remove_ms": .., "compose_ms": .., "total_ms": ..} }
"""

from __future__ import annotations

import base64
import io
import os
import time

from PIL import Image, ImageFilter, ImageOps

MAX_SOURCE_SIDE = int(os.environ.get("MAX_SOURCE_SIDE", "2000"))
DEFAULT_MODEL = os.environ.get("REMBG_MODEL", "birefnet-general")
DEFAULT_BOX = (0.10, 0.12, 0.80, 0.78)
# Catalog framing for the white background: even margins, with room under the product for the shadow.
WHITE_BOX = (0.07, 0.07, 0.86, 0.82)

_sessions: dict = {}


def _session(model: str):
    from rembg import new_session

    if model not in _sessions:
        _sessions[model] = new_session(model)
    return _sessions[model]


def _providers(session) -> list:
    """ONNX Runtime execution providers actually in use, so a response proves whether the GPU is
    really being used rather than us assuming it from the image tag."""
    try:
        return list(session.inner_session.get_providers())
    except Exception:  # noqa: BLE001 - diagnostics must never break a job
        return []


def _decode(b64: str) -> Image.Image:
    image = Image.open(io.BytesIO(base64.b64decode(b64)))
    image.load()
    image = ImageOps.exif_transpose(image)
    return image


def _encode(image: Image.Image, fmt: str, **kwargs) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **kwargs)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def remove_background(source: Image.Image, model: str) -> tuple:
    from rembg import remove

    source = source.convert("RGB")
    if max(source.size) > MAX_SOURCE_SIDE:
        source.thumbnail((MAX_SOURCE_SIDE, MAX_SOURCE_SIDE), Image.LANCZOS)
    session = _session(model)
    cutout = remove(source, session=session, post_process_mask=True)
    return _trim(cutout.convert("RGBA")), _providers(session)


def _trim(cutout: Image.Image) -> Image.Image:
    alpha = cutout.getchannel("A")
    bbox = alpha.point(lambda a: 255 if a > 8 else 0).getbbox()
    return cutout.crop(bbox) if bbox else cutout


def _shadow(cutout: Image.Image, canvas: int) -> Image.Image:
    """Soft contact shadow: squashed, blurred alpha placed under the product."""
    alpha = cutout.getchannel("A")
    w, h = cutout.size
    squash = max(1, int(h * 0.12))
    layer = Image.new("L", (w, squash), 0)
    layer.paste(alpha.resize((w, squash), Image.BILINEAR))
    pad = max(8, canvas // 40)
    shadow = Image.new("L", (w + pad * 2, squash + pad * 2), 0)
    shadow.paste(layer, (pad, pad))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=max(4, canvas // 60)))
    shadow = shadow.point(lambda v: int(v * 0.42))
    return shadow


def compose(cutout: Image.Image, background: Image.Image, box: tuple, canvas: int, fill: float, shadow: bool) -> Image.Image:
    background = background.convert("RGB")
    if background.size != (canvas, canvas):
        background = ImageOps.fit(background, (canvas, canvas), Image.LANCZOS)
    bx, by, bw, bh = box
    box_w, box_h = int(canvas * bw * fill), int(canvas * bh * fill)
    scale = min(box_w / cutout.width, box_h / cutout.height)
    product = cutout.resize((max(1, int(cutout.width * scale)), max(1, int(cutout.height * scale))), Image.LANCZOS)
    x = int(canvas * bx + (canvas * bw - product.width) / 2)
    y = int(canvas * by + canvas * bh - product.height)  # bottom-anchored inside the box

    if shadow:
        layer = _shadow(product, canvas)
        sx = x + (product.width - layer.width) // 2
        sy = y + product.height - layer.height + layer.height // 3
        black = Image.new("RGB", layer.size, (0, 0, 0))
        background.paste(black, (sx, sy), layer)

    background.paste(product, (x, y), product)
    return background


def handler(job: dict) -> dict:
    started = time.perf_counter()
    payload = job.get("input") or {}
    output = payload.get("output", "white")
    canvas = int(payload.get("canvas", 1200))
    fill = float(payload.get("fill", 0.88))
    shadow = bool(payload.get("shadow", True))
    model = payload.get("model") or DEFAULT_MODEL
    timings: dict = {}
    result: dict = {"timings": timings}

    if payload.get("source_b64"):
        t = time.perf_counter()
        cutout, providers = remove_background(_decode(payload["source_b64"]), model)
        timings["remove_ms"] = int((time.perf_counter() - t) * 1000)
        result["providers"] = providers
        result["cutout_b64"] = _encode(cutout, "PNG", optimize=True)
    elif payload.get("cutout_b64"):
        cutout = _trim(_decode(payload["cutout_b64"]).convert("RGBA"))
    else:
        return {"error": "source_b64 or cutout_b64 is required"}

    if output in ("white", "template"):
        t = time.perf_counter()
        if output == "template":
            template = payload.get("template") or {}
            if not template.get("png_b64"):
                return {"error": "template.png_b64 is required for template output"}
            background = _decode(template["png_b64"])
            box = tuple(template.get("box") or DEFAULT_BOX)
            use_shadow = bool(template.get("shadow", shadow))
            image = compose(cutout, background, box, canvas, 1.0, use_shadow)
        else:
            background = Image.new("RGB", (canvas, canvas), (255, 255, 255))
            # A margin keeps the catalog framing tidy and leaves room for the contact shadow.
            image = compose(cutout, background, WHITE_BOX, canvas, fill, shadow)
        timings["compose_ms"] = int((time.perf_counter() - t) * 1000)
        result["image_b64"] = _encode(image, "JPEG", quality=90, optimize=True)
        result["mime"] = "image/jpeg"

    timings["total_ms"] = int((time.perf_counter() - started) * 1000)
    return result


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
