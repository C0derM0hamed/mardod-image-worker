# GPU build. The endpoint runs on GPU workers, so BiRefNet-lite runs through the CUDA execution
# provider rather than wasting the card. The image is multi-GB, which is fine because GitHub Actions
# builds and pushes it (see .github/workflows/build.yml in the mardod-image-worker repo) — it is
# never uploaded from a developer machine.
#
# onnxruntime-gpu 1.20.x targets CUDA 12.x with cuDNN 9, which is what this base provides.
# The handler reports the ONNX Runtime providers actually in use in every response that runs the
# model, so "is the GPU really being used" is answered by data, not by the image tag.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    U2NET_HOME=/models \
    REMBG_MODEL=birefnet-general-lite

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Baked so cold starts never download. The lite (swin-tiny) variant is 214 MB against 928 MB for
# full birefnet-general; upgrade by changing this line and REMBG_MODEL together, then rebuilding.
RUN python3 -c "from rembg import new_session; new_session('birefnet-general-lite')"

COPY handler.py ./

CMD ["python3", "-u", "handler.py"]
