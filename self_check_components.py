"""Plain-Python safety checks for component cleanup and image compositing.

Run from the repository root with:
    python runpod/image-worker/self_check_components.py
"""

import base64
import io
import os

os.environ["MIN_COMPONENT_RATIO"] = "0.02"
os.environ["HARMONIZE_STRENGTH"] = "0.9"  # prove the import-time cap is enforced

import numpy as np
from PIL import Image

import handler


def _png_b64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def main() -> None:
    alpha = np.zeros((80, 80), dtype=np.uint8)
    alpha[20:50, 20:50] = 255  # product mass: 900 pixels
    alpha[55:67, 5:17] = 255  # a disconnected legitimate product piece: 144 pixels
    alpha[5, 5] = 255
    alpha[8:10, 8:10] = 255
    alpha[70:73, 70:73] = 255

    cutout = Image.fromarray(np.dstack((np.zeros((80, 80, 3), dtype=np.uint8), alpha)), mode="RGBA")
    filtered, removed, kept = handler._filter_small_components(cutout)
    filtered_alpha = np.asarray(filtered.getchannel("A"))

    assert removed == 3, (removed, kept)
    assert kept == 2, (removed, kept)
    assert np.all(filtered_alpha[20:50, 20:50] == 255)
    assert np.all(filtered_alpha[55:67, 5:17] == 255)
    assert filtered_alpha[5, 5] == 0
    assert np.all(filtered_alpha[8:10, 8:10] == 0)
    assert np.all(filtered_alpha[70:73, 70:73] == 0)

    handler.MIN_COMPONENT_RATIO = 0
    unfiltered, removed, kept = handler._filter_small_components(cutout)
    unfiltered_alpha = np.asarray(unfiltered.getchannel("A"))
    assert removed == 0
    assert kept == 5
    assert unfiltered_alpha[5, 5] == 255
    assert np.all(unfiltered_alpha[8:10, 8:10] == 255)
    assert np.all(unfiltered_alpha[70:73, 70:73] == 255)

    # Harmonization is allowed to change RGB only on product pixels. It must not alter alpha,
    # opaque-pixel count, or the alpha bounding box.
    product_alpha = np.zeros((32, 32), dtype=np.uint8)
    product_alpha[5:25, 7:23] = 255
    product_alpha[5, 7] = 120  # retain a fractional anti-aliased edge in the safety check
    product_rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    product_rgb[:, :] = (70, 85, 105)
    product_rgb[product_alpha == 0] = (3, 4, 5)  # transparent RGB must also stay untouched
    product = Image.fromarray(np.dstack((product_rgb, product_alpha)), mode="RGBA")
    backdrop_array = np.full((64, 64, 3), (245, 245, 245), dtype=np.uint8)
    backdrop_array[48:, 13:52] = (180, 105, 55)  # floor below the placement box
    backdrop = Image.fromarray(backdrop_array, mode="RGB")

    before = np.asarray(product, dtype=np.uint8)
    before_alpha = product.getchannel("A").tobytes()
    before_opaque_count = int(np.count_nonzero(before[:, :, 3] == 255))
    before_bbox = product.getchannel("A").getbbox()
    harmonized = handler.harmonize_product(product, backdrop, (0.20, 0.10, 0.60, 0.65), 64, strength=0.4)
    after = np.asarray(harmonized, dtype=np.uint8)
    assert harmonized.getchannel("A").tobytes() == before_alpha
    assert int(np.count_nonzero(after[:, :, 3] == 255)) == before_opaque_count
    assert harmonized.getchannel("A").getbbox() == before_bbox
    assert np.array_equal(after[before[:, :, 3] == 0, :3], before[before[:, :, 3] == 0, :3])
    assert not np.array_equal(after[before[:, :, 3] > 0, :3], before[before[:, :, 3] > 0, :3])

    unchanged = handler.harmonize_product(product, backdrop, (0.20, 0.10, 0.60, 0.65), 64, strength=0)
    assert np.array_equal(np.asarray(unchanged), before)
    assert handler.HARMONIZE_STRENGTH == 0.6, handler.HARMONIZE_STRENGTH
    assert handler._effective_harmonize_strength(1.0) == 0.6

    # The shadow follows the per-column alpha base, and light direction moves its weighted center
    # in the opposite horizontal direction. The explicit overlap check mirrors compose placement.
    shadow_alpha = np.zeros((30, 40), dtype=np.uint8)
    shadow_alpha[5:25, 10:30] = 255
    shadow_rgb = np.full((30, 40, 3), (220, 40, 30), dtype=np.uint8)
    shadow_product = Image.fromarray(np.dstack((shadow_rgb, shadow_alpha)), mode="RGBA")
    canvas = 120
    box = (0.20, 0.20, 0.60, 0.50)
    box_w, box_h = int(canvas * box[2]), int(canvas * box[3])
    scale = min(box_w / shadow_product.width, box_h / shadow_product.height)
    rendered = shadow_product.resize((int(shadow_product.width * scale), int(shadow_product.height * scale)), Image.LANCZOS)
    product_x = int(canvas * box[0] + (canvas * box[2] - rendered.width) / 2)
    product_y = int(canvas * box[1] + canvas * box[3] - rendered.height)
    product_bbox = rendered.getchannel("A").getbbox()
    assert product_bbox is not None
    shadow_left = handler._shadow(rendered, canvas, "left", 0.9)
    shadow_right = handler._shadow(rendered, canvas, "right", 0.9)
    shadow_pad = (shadow_left.width - rendered.width) // 2
    shadow_xy = (product_x - shadow_pad, product_y)
    shadow_left = handler._clip_shadow_from_product(shadow_left, rendered, shadow_xy, (product_x, product_y))
    shadow_right = handler._clip_shadow_from_product(shadow_right, rendered, shadow_xy, (product_x, product_y))
    shadow_pixels = np.argwhere(np.asarray(shadow_left) > 0)
    rendered_alpha = np.asarray(rendered.getchannel("A")) > 0
    for shadow_y, shadow_x in shadow_pixels:
        local_y = shadow_y + shadow_xy[1] - product_y
        local_x = shadow_x + shadow_xy[0] - product_x
        if 0 <= local_y < rendered.height and 0 <= local_x < rendered.width:
            assert not rendered_alpha[local_y, local_x], (local_x, local_y)

    left_pixels = np.asarray(shadow_left, dtype=np.float32)
    right_pixels = np.asarray(shadow_right, dtype=np.float32)
    left_center = float((left_pixels * np.arange(shadow_left.width)[None, :]).sum() / left_pixels.sum())
    right_center = float((right_pixels * np.arange(shadow_right.width)[None, :]).sum() / right_pixels.sum())
    assert left_center > right_center, (left_center, right_center)

    # Disconnected pieces with different bottoms each receive a contact shadow. This exercises
    # compose rather than only the local mask, so a regression to one global footprint is caught.
    contour_canvas = 240
    contour_alpha = np.zeros((180, 200), dtype=np.uint8)
    contour_alpha[40:80, 20:70] = 255
    contour_alpha[100:140, 120:170] = 255
    contour_rgb = np.full((180, 200, 3), (220, 40, 30), dtype=np.uint8)
    contour_cutout = Image.fromarray(np.dstack((contour_rgb, contour_alpha)), mode="RGBA")
    flat_background = Image.new("RGB", (contour_canvas, contour_canvas), (180, 180, 180))
    contour_composite = handler.compose(
        contour_cutout,
        flat_background,
        (0.0, 0.0, 1.0, 1.0),
        contour_canvas,
        1.0,
        True,
        light="front",
        shadow_strength=0.45,
        harmonize=False,
    )
    contour_scale = min(contour_canvas / contour_cutout.width, contour_canvas / contour_cutout.height)
    contour_rendered = contour_cutout.resize(
        (int(contour_cutout.width * contour_scale), int(contour_cutout.height * contour_scale)),
        Image.Resampling.LANCZOS,
    )
    contour_x = int((contour_canvas - contour_rendered.width) / 2)
    contour_y = contour_canvas - contour_rendered.height
    contour_rendered_alpha = np.asarray(contour_rendered.getchannel("A"))
    contour_composite_rgb = np.asarray(contour_composite)
    for local_x in (int(55 * contour_scale), int(145 * contour_scale)):
        base = int(np.flatnonzero(contour_rendered_alpha[:, local_x] > 128)[-1])
        assert contour_composite_rgb[contour_y + base + 2, contour_x + local_x, 0] < 180

    # A shadow is clipped wherever the resized product is opaque, so the product RGB remains
    # byte-identical after compositing (harmonization is disabled for this check).
    opaque_pixels = contour_rendered_alpha == 255
    assert np.array_equal(
        contour_composite_rgb[contour_y:contour_y + contour_rendered.height, contour_x:contour_x + contour_rendered.width][opaque_pixels],
        np.asarray(contour_rendered)[:, :, :3][opaque_pixels],
    )

    # Template-only aspect caps keep a square/small piece from filling the placement height while
    # a genuinely wide set can still use the full placement width.
    def exact_product_width(image: Image.Image, color: tuple[int, int, int]) -> int:
        pixels = np.asarray(image)
        matches = np.all(pixels == color, axis=2)
        columns = np.flatnonzero(matches.any(axis=0))
        return int(columns[-1] - columns[0] + 1)

    sizing_background = Image.new("RGB", (1200, 1200), (180, 180, 180))
    square_cutout = Image.new("RGBA", (100, 100), (230, 30, 40, 255))
    square_1 = handler.compose(
        square_cutout, sizing_background, (0.1, 0.1, 0.8, 0.6), 1200, 1.0, False,
        harmonize=False, scale=1.0, aspect_cap=True,
    )
    square_13 = handler.compose(
        square_cutout, sizing_background, (0.1, 0.1, 0.8, 0.6), 1200, 1.0, False,
        harmonize=False, scale=1.3, aspect_cap=True,
    )
    square_05 = handler.compose(
        square_cutout, sizing_background, (0.1, 0.1, 0.8, 0.6), 1200, 1.0, False,
        harmonize=False, scale=0.5, aspect_cap=True,
    )
    square_width = exact_product_width(square_1, (230, 30, 40))
    assert square_width < 0.50 * 0.8 * 1200 + 1
    assert exact_product_width(square_13, (230, 30, 40)) > square_width
    assert exact_product_width(square_05, (230, 30, 40)) < square_width

    wide_cutout = Image.new("RGBA", (400, 100), (230, 30, 40, 255))
    wide = handler.compose(
        wide_cutout, sizing_background, (0.1, 0.1, 0.8, 0.6), 1200, 1.0, False,
        harmonize=False, scale=1.0, aspect_cap=True,
    )
    assert exact_product_width(wide, (230, 30, 40)) >= int(0.98 * 0.8 * 1200)
    assert handler._backdrop_parameters({"scale": 2.0})["scale"] == 1.3
    assert handler._backdrop_parameters({"scale": 0.1})["scale"] == 0.5

    # A legacy sidecar with only box/shadow must use safe defaults for the new fields.
    cutout = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    cutout.putalpha(Image.new("L", (16, 16), 255))
    template = Image.new("RGB", (64, 64), (90, 100, 110))
    result = handler.handler({
        "input": {
            "cutout_b64": _png_b64(cutout),
            "output": "template",
            "canvas": 64,
            "template": {"png_b64": _png_b64(template), "box": [0.10, 0.10, 0.80, 0.70], "shadow": True},
        },
    })
    assert "error" not in result, result
    assert result["mime"] == "image/jpeg"
    assert result["diagnostics"] == {
        "light": "front",
        "shadow": True,
        "shadow_strength": 0.45,
        "harmonize": True,
        "harmonize_strength": 0.6,
        "scale": 1.0,
    }, result["diagnostics"]

    print("component/image-worker self-check passed: removed=3 kept=2; alpha-safe harmonization; contour shadows; aspect sizing; legacy sidecar")


if __name__ == "__main__":
    main()
