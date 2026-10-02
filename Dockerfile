FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md README.zh-CN.md ./
COPY duplexomni ./duplexomni
COPY web ./web
RUN pip install --no-cache-dir ".[serve]"

EXPOSE 8765
ENV DUPLEX_HOST=0.0.0.0 \
    DUPLEX_PORT=8765 \
    DUPLEX_DB=/data/duplexomni.sqlite
VOLUME ["/data"]
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health')"
CMD ["python", "-m", "duplexomni", "serve"]
