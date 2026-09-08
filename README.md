# Mardod image worker (RunPod Serverless)

Background removal (BiRefNet-general via `rembg`, MIT) plus deterministic Pillow compositing.
One call with `source_b64` returns **both** the transparent cutout and the requested output image;
later calls pass the cached `cutout_b64` and only compose.

## Build & push

```bash
cd runpod/image-worker
docker build -t ghcr.io/c0derm0hamed/mardod-image-worker:1.0.0 .
docker push ghcr.io/c0derm0hamed/mardod-image-worker:1.0.0
```

The package must be public on GHCR (or add registry credentials in RunPod).

## Endpoint settings

- Image: `ghcr.io/c0derm0hamed/mardod-image-worker:<tag>`
- GPU: 16 GB tier (fallback 24 GB), workers 0–2, idle timeout 120 s, FlashBoot on, execution timeout 120 s
- No network volume needed (weights are baked into the image)

## Local test (needs a GPU or CPU fallback of onnxruntime)

```bash
python3 -c "import json,handler; print(handler.handler(json.load(open('test_input.json')))['timings'])"
```

## Request / response

See the docstring at the top of `handler.py`.
