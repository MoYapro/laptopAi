FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/hf

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Face models are downloaded on first start into the models volume (see faces.ensure_face_models).

# Model files and data live in volumes; make them writable for the non-root runtime user.
RUN mkdir -p /models /data && chmod 777 /models /data

COPY . .
EXPOSE 8080
# Short graceful-shutdown timeout so the shutdown event is written well within Docker's stop grace period.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-graceful-shutdown", "5"]
