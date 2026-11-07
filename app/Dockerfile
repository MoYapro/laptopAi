FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/hf

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Face detection (YuNet) + recognition (SFace) models from opencv_zoo, checksum-pinned.
ADD --checksum=sha256:8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4 \
    https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
    /opt/face-models/yunet.onnx
ADD --checksum=sha256:0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79 \
    https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx \
    /opt/face-models/sface.onnx
RUN chmod 644 /opt/face-models/*.onnx

# Model files and data live in volumes; make them writable for the non-root runtime user.
RUN mkdir -p /models /data && chmod 777 /models /data

COPY . .
EXPOSE 8080
# Short graceful-shutdown timeout so the shutdown event is written well within Docker's stop grace period.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-graceful-shutdown", "5"]
