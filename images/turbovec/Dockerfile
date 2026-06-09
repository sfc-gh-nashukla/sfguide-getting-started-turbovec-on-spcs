FROM python:3.11-slim

RUN pip install --no-cache-dir turbovec==0.7.0 fastapi uvicorn[standard] numpy pydantic

WORKDIR /app
COPY service.py /app/service.py

ENV TURBOVEC_DIM=1536
ENV TURBOVEC_BIT_WIDTH=4
ENV TURBOVEC_INDEX_PATH=/data/turbovec_index.tvim

RUN mkdir -p /data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
