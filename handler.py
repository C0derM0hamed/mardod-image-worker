"""RunPod serverless worker: background removal (BiRefNet via rembg) + deterministic compositing.

Input (all images are base64 strings):
{
  "source_b64":  "...",            # original supplier photo (jpeg/png/webp). Triggers background removal.
  "cutout_b64":  "...",            # OR an already-removed RGBA PNG (compose only, no model call).
  "output":      "white" | "template" | "none",
  "template":    {"png_b64": "...", "box": [x, y, w, h], "shadow": true,
                   "light": "left" | "right" | "front",  # shadow direction
                   "shadow_strength": 0.45,               # shadow opacity, 0..1
                   "harmonize": true,                     # capped RGB harmonization
                   "scale": 1.0},                          # template size hint, 0.5..1.3
                   # box values are fractions of the canvas
  "canvas":      1200,
  "fill":        0.88,             # max fraction of the box the product may occupy (white output)
  "shadow":      true,
  "model":       "birefnet-general"
}
Output:
{ "cutout_b64": "...png" (only when source_b64 was given, after stray-speck cleanup),
  "cutout_raw_b64": "...png" (only when cleanup actually removed something, so it can be re-run),
  "image_b64": "...jpeg", "mime": "image/jpeg",
  "removed_components": .., "kept_components": ..,
  "timings": {"remove_ms": .., "compose_ms": .., "total_ms": ..},
  "diagnostics": {"light": .., "shadow": .., "shadow_strength": ..,
                  "harmonize": .., "harmonize_strength": .., "scale": ..} }
"""

from __future__ import annotations

import base64
import io
import os
import time

import numpy as np
from PIL import Image, ImageFilter, ImageOps

MAX_SOURCE_SIDE = int(os.environ.get("MAX_SOURCE_SIDE", "2000"))
DEFAULT_MODEL = os.environ.get("REMBG_MODEL", "birefnet-general")
# Stray supplier text/logos survive background removal as tiny specks. Measured on real cutouts:
# a "(1+4)" marker's glyphs were 0.06%-0.17% of the product mass, while every real product tested
# (including five-piece sets) came through as one connected mass. 0.5% clears the text with a wide
# margin and leaves headroom for a genuinely detached small accessory. 0 disables the filter.
MIN_COMPONENT_RATIO = float(os.environ.get("MIN_COMPONENT_RATIO", "0.005"))
DEFAULT_BOX = (0.10, 0.12, 0.80, 0.78)
# Catalog framing for the white background: even margins, with room under the product for the shadow.
WHITE_BOX = (0.07, 0.07, 0.86, 0.82)
DEFAULT_LIGHT = "front"
DEFAULT_SHADOW_STRENGTH = 0.45
MAX_HARMONIZE_STRENGTH = 0.6


def _clamp_float(value, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(value):
        return default
    return max(minimum, min(maximum, value))


# This is deliberately capped at import time as well as at the call site. A bad environment value
# must not be able to wash a product out, even before a request reaches the handler.
HARMONIZE_STRENGTH = _clamp_float(
    os.environ.get("HARMONIZE_STRENGTH", "0.25"),
    0.25,
    0.0,
    MAX_HARMONIZE_STRENGTH,
)

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


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Return an 8-connected component label image and the component areas."""
    height, width = mask.shape
    labels = np.zeros(mask.shape, dtype=np.int32)
    areas: list[int] = []

    # A flat integer stack avoids allocating a tuple for every pixel in a large product.
    for seed in np.flatnonzero(mask):
        seed = int(seed)
        if labels.flat[seed] != 0:
            continue

        label = len(areas) + 1
        labels.flat[seed] = label
        stack = [seed]
        area = 0

        while stack:
            index = stack.pop()
            y, x = divmod(index, width)
            area += 1

            for neighbor_y in range(max(0, y - 1), min(height, y + 2)):
                for neighbor_x in range(max(0, x - 1), min(width, x + 2)):
                    if neighbor_y == y and neighbor_x == x:
                        continue
                    neighbor = neighbor_y * width + neighbor_x
                    if mask[neighbor_y, neighbor_x] and labels.flat[neighbor] == 0:
                        labels.flat[neighbor] = label
                        stack.append(neighbor)

        areas.append(area)

    return labels, areas


def _filter_small_components(cutout: Image.Image) -> tuple[Image.Image, int, int]:
    """Remove tiny disconnected alpha components while preserving real product pieces."""
    alpha = np.asarray(cutout.getchannel("A"), dtype=np.uint8)
    labels, areas = _label_components(alpha > 0)
    if not areas:
        return cutout, 0, 0

    # A zero ratio disables erasure but still reports the observed component count.
    if MIN_COMPONENT_RATIO == 0:
        return cutout, 0, len(areas)

    largest_area = max(areas)
    threshold = largest_area * MIN_COMPONENT_RATIO

    # Do not keep only the largest component: legitimate multi-piece furniture sets can
    # be disconnected, so every component at or above the threshold must remain intact.
    removed_labels = [label for label, area in enumerate(areas, start=1) if area < threshold]
    kept_components = len(areas) - len(removed_labels)
    if not removed_labels:
        return cutout, 0, kept_components

    filtered_alpha = alpha.copy()
    filtered_alpha[np.isin(labels, np.asarray(removed_labels, dtype=np.int32))] = 0
    filtered = cutout.copy()
    filtered.putalpha(Image.fromarray(filtered_alpha, mode="L"))
    return filtered, len(removed_labels), kept_components


def remove_background(source: Image.Image, model: str) -> tuple:
    from rembg import remove

    source = source.convert("RGB")
    if max(source.size) > MAX_SOURCE_SIDE:
        source.thumbnail((MAX_SOURCE_SIDE, MAX_SOURCE_SIDE), Image.LANCZOS)
    session = _session(model)
    raw_cutout = remove(source, session=session, post_process_mask=True).convert("RGBA")
    cutout, removed_components, kept_components = _filter_small_components(raw_cutout)
    return (
        _trim(cutout),
        # The unfiltered cutout is returned as well and cached by the caller, so component cleanup
        # can be re-run at a different threshold (or undone) without paying for segmentation again.
        _trim(raw_cutout),
        _providers(session),
        {"removed_components": removed_components, "kept_components": kept_components},
    )


def _trim(cutout: Image.Image) -> Image.Image:
    alpha = cutout.getchannel("A")
    bbox = alpha.point(lambda a: 255 if a > 8 else 0).getbbox()
    return cutout.crop(bbox) if bbox else cutout


def _backdrop_parameters(template: dict | None) -> dict:
    """Read optional sidecar values without making older sidecars invalid."""
    template = template if isinstance(template, dict) else {}
    light = template.get("light", DEFAULT_LIGHT)
    if light not in {"left", "right", "front"}:
        light = DEFAULT_LIGHT

    harmonize = template.get("harmonize", True)
    if not isinstance(harmonize, bool):
        harmonize = True

    return {
        "light": light,
        "shadow_strength": _clamp_float(
            template.get("shadow_strength", DEFAULT_SHADOW_STRENGTH),
            DEFAULT_SHADOW_STRENGTH,
            0.0,
            1.0,
        ),
        "harmonize": harmonize,
        "scale": _clamp_float(template.get("scale", 1.0), 1.0, 0.5, 1.3),
    }


def _effective_harmonize_strength(strength: float | None = None) -> float:
    return _clamp_float(
        HARMONIZE_STRENGTH if strength is None else strength,
        0.25,
        0.0,
        MAX_HARMONIZE_STRENGTH,
    )


def _aspect_width_fraction(aspect: float) -> float:
    """Return the template width cap fraction for a cutout aspect ratio."""
    # Keep square and near-square pieces at the conservative small-piece cap. This is the useful
    # boundary for the furniture catalog (and prevents a square nightstand from filling a tall
    # placement); the remaining anchors interpolate linearly.
    # Anchors were tuned on real cutouts: a square nightstand at 0.50 of the box reads as a small
    # piece, a bed at aspect ~1.5 needs ~0.78 to stay believably large, and sofas/sets at 2.2+ may
    # use the whole box.
    if aspect <= 1.0:
        return 0.50
    if aspect >= 2.2:
        return 1.0
    if aspect <= 1.4:
        return 0.50 + (aspect - 1.0) * (0.74 - 0.50) / (1.4 - 1.0)
    return 0.74 + (aspect - 1.4) * (1.0 - 0.74) / (2.2 - 1.4)


def _shadow(cutout: Image.Image, canvas: int, light: str = DEFAULT_LIGHT, strength: float = DEFAULT_SHADOW_STRENGTH) -> Image.Image:
    """Build a directional two-pass shadow from the product's per-column base contour.

    The returned mask is positioned at the product's top-left by ``compose``. Its horizontal
    padding lets the shadow drift away from the light. Its height includes the whole product plus
    the shadow depth, so each column can start at its own base row instead of sharing one global
    alpha-bbox bottom.
    """
    alpha_array = np.asarray(cutout.getchannel("A"), dtype=np.uint8)
    w, h = cutout.size

    # A base is the lowest sufficiently opaque pixel in each column. Transparent columns stay
    # inactive, which preserves separate contact patches for disconnected furniture pieces.
    opaque = alpha_array > 128
    has_base = opaque.any(axis=0)
    if not np.any(has_base):
        return Image.new("L", (w + 16, max(8, h + int(min(max(1, h), max(1, canvas)) * 0.16))), 0)
    rows = np.arange(h, dtype=np.int32)[:, None]
    base_rows = np.where(opaque, rows, -1).max(axis=0).astype(np.int32)

    # A window around 1.5% of the product width removes one-pixel alpha fringes without pulling
    # a contour upward into the product. Invalid columns are excluded rather than bridged.
    smooth_radius = max(1, int(round(w * 0.0075)))
    base_values = np.where(has_base, base_rows.astype(np.float32), np.nan)
    padded_base = np.pad(base_values, (smooth_radius, smooth_radius), mode="constant", constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded_base, 2 * smooth_radius + 1)
    smoothed = np.full(w, np.nan, dtype=np.float32)
    valid_columns = np.flatnonzero(has_base)
    smoothed[valid_columns] = np.nanmedian(windows[valid_columns], axis=1)
    smoothed_rows = base_rows.copy()
    smoothed_rows[has_base] = np.maximum(
        base_rows[has_base],
        np.rint(smoothed[has_base]).astype(np.int32),
    )

    # Retain the alpha at the real opaque base for a softer edge when a base pixel is only just
    # above the threshold. Smoothing moves the row only downward, so it never samples above it.
    columns = np.arange(w)
    base_alpha = np.zeros(w, dtype=np.float32)
    base_alpha[has_base] = alpha_array[base_rows[has_base], columns[has_base]]

    light = light if light in {"left", "right", "front"} else DEFAULT_LIGHT
    direction = {"left": 1, "right": -1, "front": 0}[light]
    strength = _clamp_float(strength, DEFAULT_SHADOW_STRENGTH, 0.0, 1.0)
    depth = max(8, int(min(max(1, h), max(1, canvas)) * 0.16))
    max_offset = int(round(min(max(1, w) * 0.16, max(1, canvas) * 0.10))) * direction
    halo_radius = max(1, int(round(min(max(1, w), max(1, canvas)) * 0.014)))
    core_radius = max(1, int(round(min(max(1, w), max(1, canvas)) * 0.006)))
    ao_radius = max(1, int(round(min(max(1, w), max(1, canvas)) * 0.002)))
    expand_radius = max(1, int(round(min(max(1, w), max(1, canvas)) * 0.018)))
    pad = max(8, abs(max_offset) + expand_radius + halo_radius * 3 + 2)
    layer_width = w + pad * 2
    layer_height = h + depth
    columns = np.arange(w)
    shifted_columns = pad + columns

    def dilate_1d(values: np.ndarray) -> np.ndarray:
        # PIL's MaxFilter needs a 2-D image, and feeding it a 1-D array silently produces a w x w
        # result. Keep the widening as shifted maxima over the one-dimensional row instead.
        padded = np.pad(values, (expand_radius, expand_radius), mode="constant")
        widened = values.copy()
        for shift in range(1, expand_radius + 1):
            left = padded[expand_radius - shift:expand_radius - shift + values.size]
            right = padded[expand_radius + shift:expand_radius + shift + values.size]
            widened = np.maximum(widened, np.maximum(left, right))
        return widened

    def render_pass(pass_depth: int, peak: float, fade_power: float) -> np.ndarray:
        result = np.zeros((layer_height, layer_width), dtype=np.float32)
        for row in range(pass_depth):
            progress = row / max(1, pass_depth - 1)
            fade = max(0.0, (1.0 - progress) ** fade_power)
            row_values = np.zeros(w, dtype=np.float32)
            active = has_base & (smoothed_rows + row < layer_height)
            row_values[active] = base_alpha[active] * (peak * fade / 255.0)

            # The direct row follows the contour; its widened counterpart is blended in as the
            # shadow travels away from the contact line, matching the old tight-core/soft-halo idea.
            wide_values = dilate_1d(row_values)
            blend = min(1.0, progress * 1.5)
            row_values = row_values * (1.0 - blend) + wide_values * blend
            target_y = smoothed_rows[active] + row
            shifted_x = shifted_columns[active] + int(round(max_offset * progress))
            result[target_y, shifted_x] = row_values[active]
        return result

    # The near-base core carries most of the opacity; the full-depth halo is wider and much
    # fainter. The short AO pass reinforces the immediate contact line without exceeding the old
    # 220 * strength peak when the masks are combined.
    core_depth = max(3, int(depth * 0.45))
    core = Image.fromarray(
        np.clip(render_pass(core_depth, 210.0 * strength, 1.35), 0.0, 255.0).astype(np.uint8),
        mode="L",
    ).filter(ImageFilter.GaussianBlur(core_radius))
    halo = Image.fromarray(
        np.clip(render_pass(depth, 105.0 * strength, 1.7), 0.0, 255.0).astype(np.uint8),
        mode="L",
    ).filter(ImageFilter.GaussianBlur(halo_radius))
    ao_depth = max(1, int(round(canvas * 0.004)))
    ao = Image.fromarray(
        np.clip(render_pass(ao_depth, 220.0 * strength, 0.8), 0.0, 255.0).astype(np.uint8),
        mode="L",
    ).filter(ImageFilter.GaussianBlur(ao_radius))
    core_array = np.asarray(core, dtype=np.float32)
    halo_array = np.asarray(halo, dtype=np.float32)
    ao_array = np.asarray(ao, dtype=np.float32)
    combined = core_array + halo_array * (1.0 - core_array / 255.0)
    combined = np.maximum(combined, ao_array)
    combined = np.clip(combined, 0.0, 220.0 * strength).astype(np.uint8)
    return Image.fromarray(combined, mode="L")


def _floor_mean(background: Image.Image, box: tuple, canvas: int) -> np.ndarray:
    """Return mean RGB from the backdrop floor below the placement box."""
    scene = np.asarray(background.convert("RGB"), dtype=np.float32)
    height, width = scene.shape[:2]
    bx, by, bw, bh = box
    x1 = max(0, min(width - 1, int(np.floor(width * bx))))
    x2 = max(x1 + 1, min(width, int(np.ceil(width * (bx + bw)))))
    floor_start = int(np.ceil(height * (by + bh)))
    # A box whose bottom lands exactly at the canvas edge still needs a small floor sample.
    floor_start = max(0, min(height - 1, floor_start))
    floor = scene[floor_start:, x1:x2]
    if floor.size == 0:
        floor = scene[max(0, height - max(1, canvas // 50)):, x1:x2]
    if floor.size == 0:
        floor = scene
    return floor.reshape(-1, 3).mean(axis=0)


def harmonize_product(product: Image.Image, background: Image.Image, box: tuple, canvas: int, strength: float | None = None) -> Image.Image:
    """Nudge only product RGB means toward the backdrop floor; alpha and shape are untouched."""
    effective_strength = _effective_harmonize_strength(strength)
    if effective_strength == 0.0:
        return product.copy()

    rgba = np.array(product.convert("RGBA"), dtype=np.uint8, copy=True)
    alpha = rgba[:, :, 3].copy()
    own_pixels = alpha > 0
    if not np.any(own_pixels):
        return product.copy()

    rgb = rgba[:, :, :3].astype(np.float32)
    weights = alpha[own_pixels].astype(np.float32) / 255.0
    product_mean = np.average(rgb[own_pixels], axis=0, weights=weights)
    floor_mean = _floor_mean(background, box, canvas)
    # This one additive per-channel delta is both the colour-temperature nudge and the exposure
    # nudge: its luminance moves by the same fraction toward the floor luminance.
    delta = effective_strength * (floor_mean - product_mean)
    rgb[own_pixels] = np.clip(rgb[own_pixels] + delta, 0.0, 255.0)
    rgba[:, :, :3] = np.rint(rgb).astype(np.uint8)
    rgba[:, :, 3] = alpha
    return Image.fromarray(rgba, mode="RGBA")


def _clip_shadow_from_product(layer: Image.Image, product: Image.Image, shadow_xy: tuple[int, int], product_xy: tuple[int, int]) -> Image.Image:
    """Defensive overlap mask: a shadow must never cover an alpha-bearing product pixel."""
    layer_array = np.array(layer, dtype=np.uint8, copy=True)
    product_alpha = np.asarray(product.getchannel("A"), dtype=np.uint8) > 0
    shadow_x, shadow_y = shadow_xy
    product_x, product_y = product_xy
    intersection_left = max(shadow_x, product_x)
    intersection_top = max(shadow_y, product_y)
    intersection_right = min(shadow_x + layer.width, product_x + product.width)
    intersection_bottom = min(shadow_y + layer.height, product_y + product.height)
    if intersection_left < intersection_right and intersection_top < intersection_bottom:
        layer_slice = (
            slice(intersection_top - shadow_y, intersection_bottom - shadow_y),
            slice(intersection_left - shadow_x, intersection_right - shadow_x),
        )
        product_slice = (
            slice(intersection_top - product_y, intersection_bottom - product_y),
            slice(intersection_left - product_x, intersection_right - product_x),
        )
        overlap = layer_array[layer_slice]
        overlap[product_alpha[product_slice]] = 0
        layer_array[layer_slice] = overlap
    return Image.fromarray(layer_array, mode="L")


def compose(
    cutout: Image.Image,
    background: Image.Image,
    box: tuple,
    canvas: int,
    fill: float,
    shadow: bool,
    light: str = DEFAULT_LIGHT,
    shadow_strength: float = DEFAULT_SHADOW_STRENGTH,
    harmonize: bool = False,
    harmonize_strength: float | None = None,
    scale: float = 1.0,
    aspect_cap: bool = False,
) -> Image.Image:
    background = background.convert("RGB")
    if background.size != (canvas, canvas):
        background = ImageOps.fit(background, (canvas, canvas), Image.LANCZOS)
    bx, by, bw, bh = box
    box_w, box_h = int(canvas * bw * fill), int(canvas * bh * fill)
    product_scale = min(box_w / cutout.width, box_h / cutout.height)
    if aspect_cap:
        aspect = cutout.width / cutout.height
        cap_fraction = _aspect_width_fraction(aspect)
        cap_width = canvas * bw * cap_fraction
        product_scale = min(product_scale, cap_width / cutout.width)
        product_scale *= _clamp_float(scale, 1.0, 0.5, 1.3)
    product = cutout.resize(
        (max(1, int(cutout.width * product_scale)), max(1, int(cutout.height * product_scale))),
        Image.LANCZOS,
    )
    x = int(canvas * bx + (canvas * bw - product.width) / 2)
    y = int(canvas * by + canvas * bh - product.height)  # bottom-anchored inside the box

    if harmonize:
        product = harmonize_product(product, background, box, canvas, harmonize_strength)

    if shadow and _clamp_float(shadow_strength, DEFAULT_SHADOW_STRENGTH, 0.0, 1.0) > 0:
        layer = _shadow(product, canvas, light, shadow_strength)
        pad = (layer.width - product.width) // 2
        sx = x - pad
        layer = _clip_shadow_from_product(layer, product, (sx, y), (x, y))
        black = Image.new("RGB", layer.size, (0, 0, 0))
        background.paste(black, (sx, y), layer)

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
        cutout, raw_cutout, providers, component_diagnostics = remove_background(_decode(payload["source_b64"]), model)
        timings["remove_ms"] = int((time.perf_counter() - t) * 1000)
        result["providers"] = providers
        result.update(component_diagnostics)
        result["cutout_b64"] = _encode(cutout, "PNG", optimize=True)
        if component_diagnostics.get("removed_components"):
            result["cutout_raw_b64"] = _encode(raw_cutout, "PNG", optimize=True)
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
            backdrop = _backdrop_parameters(template)
            applied_harmonize_strength = _effective_harmonize_strength() if backdrop["harmonize"] else 0.0
            image = compose(
                cutout,
                background,
                box,
                canvas,
                1.0,
                use_shadow,
                light=backdrop["light"],
                shadow_strength=backdrop["shadow_strength"],
                harmonize=backdrop["harmonize"],
                harmonize_strength=applied_harmonize_strength,
                scale=backdrop["scale"],
                aspect_cap=True,
            )
            result["diagnostics"] = {
                "light": backdrop["light"],
                "shadow": use_shadow,
                "shadow_strength": backdrop["shadow_strength"] if use_shadow else 0.0,
                "harmonize": backdrop["harmonize"],
                "harmonize_strength": applied_harmonize_strength,
                "scale": backdrop["scale"],
            }
        else:
            background = Image.new("RGB", (canvas, canvas), (255, 255, 255))
            # A margin keeps the catalog framing tidy and leaves room for the contact shadow.
            image = compose(
                cutout,
                background,
                WHITE_BOX,
                canvas,
                fill,
                shadow,
                light=DEFAULT_LIGHT,
                shadow_strength=DEFAULT_SHADOW_STRENGTH,
                harmonize=False,
            )
            result["diagnostics"] = {
                "light": DEFAULT_LIGHT,
                "shadow": shadow,
                "shadow_strength": DEFAULT_SHADOW_STRENGTH if shadow else 0.0,
                "harmonize": False,
                "harmonize_strength": 0.0,
            }
        timings["compose_ms"] = int((time.perf_counter() - t) * 1000)
        result["image_b64"] = _encode(image, "JPEG", quality=90, optimize=True)
        result["mime"] = "image/jpeg"

    timings["total_ms"] = int((time.perf_counter() - started) * 1000)
    return result


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
