# Video Processing Service - CUDA-enabled FFmpeg + yt-dlp + FastAPI
# Built on NVIDIA PyTorch base for DGX Spark GPU support
FROM nvcr.io/nvidia/pytorch:25.11-py3

WORKDIR /app

# Install FFmpeg with NVIDIA hardware acceleration
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    yt-dlp \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    aiofiles

# Copy the API server
COPY main.py .

# Create directories
RUN mkdir -p /downloads /outputs && chmod 777 /downloads /outputs

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
