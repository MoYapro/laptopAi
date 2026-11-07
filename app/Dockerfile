FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/hf

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Face detection: YuNet from opencv_zoo, checksum-pinned.
ADD --checksum=sha256:8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4 \
    https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
    /opt/face-models/yunet.onnx
# Face recognition: ArcFace w600k_r50 from InsightFace's buffalo_l pack (only that model is kept).
# Note: InsightFace's pretrained models are licensed for non-commercial use.
RUN python - <<'EOF'
import hashlib, io, urllib.request, zipfile
url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
data = urllib.request.urlopen(url).read()
assert hashlib.sha256(data).hexdigest() == "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f", "checksum mismatch"
model = zipfile.ZipFile(io.BytesIO(data)).read("w600k_r50.onnx")
assert hashlib.sha256(model).hexdigest() == "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"
open("/opt/face-models/w600k_r50.onnx", "wb").write(model)
EOF
RUN chmod 644 /opt/face-models/*.onnx

# Model files and data live in volumes; make them writable for the non-root runtime user.
RUN mkdir -p /models /data && chmod 777 /models /data

COPY . .
EXPOSE 8080
# Short graceful-shutdown timeout so the shutdown event is written well within Docker's stop grace period.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-graceful-shutdown", "5"]
