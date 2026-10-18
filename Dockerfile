# SGLang Workbench — portable web app image.
#
# The workbench is GPU-agnostic: it talks to any OpenAI-compatible SGLang server
# over HTTP. That is what makes one image work on macOS, Windows and Linux — the
# Metal/CUDA stack stays on the host (or in the optional GPU compose service).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WORKBENCH_DATA=/data \
    WORKBENCH_ROOT=/app \
    HF_HOME=/hf-cache

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY harness2_lite.py ./
COPY static ./static
COPY mcp_demo_server.py ./
COPY README.md ./

RUN useradd --system --uid 10001 --create-home app \
 && mkdir -p /data /hf-cache \
 && chown -R app:app /app /data /hf-cache
USER app

EXPOSE 8090

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8090/healthz', timeout=3).status==200 else 1)"

CMD ["python", "app.py", "--port", "8090", "--host", "0.0.0.0"]
