# CPU-only on purpose. BiRefNet-lite segments a 1200px product photo in a few seconds on CPU, which
# sits well inside this worker's 100s HTTP budget for a queued job, and it avoids the CUDA runtime
# entirely: the image is roughly ten times smaller to ship, the endpoint costs less, and it never
# waits for a GPU to be free. That mattered concretely — the workstation that builds this image
# uploads at about 42 KB/s, which makes a multi-gigabyte CUDA image undeliverable.
# To move to GPU later: swap the base for nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04, install
# python3/pip, and use onnxruntime-gpu in requirements.txt.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    U2NET_HOME=/models \
    OMP_NUM_THREADS=4 \
    REMBG_MODEL=birefnet-general-lite

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Baked so cold starts never download. The lite (swin-tiny) variant is 214 MB against 928 MB for
# full birefnet-general; upgrade by changing this line and REMBG_MODEL together, then rebuilding.
RUN python -c "from rembg import new_session; new_session('birefnet-general-lite')"

COPY handler.py ./

CMD ["python", "-u", "handler.py"]
