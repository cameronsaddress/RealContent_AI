# Video Processing Service - CUDA-enabled FFmpeg + yt-dlp + FastAPI
# Using existing 'cuda social container' (n8n-video-processor) as base to ensure correct yt-dlp/net config
FROM n8n-video-processor:latest

WORKDIR /app

# Install ImageMagick for MoviePy TextClip
# Unblock ImageMagick security policy for TextClip
RUN apt-get update && apt-get install -y imagemagick fonts-dejavu fonts-liberation && \
    sed -i 's/none/read,write/g' /etc/ImageMagick-6/policy.xml && \
    rm -rf /var/lib/apt/lists/*

# Install custom fonts for viral TikTok-style captions
COPY fonts/ /usr/share/fonts/truetype/custom/
RUN fc-cache -fv

# Install NEW dependencies for Viral Clips (Whisper, MoviePy)
# Existing image already contains: ffmpeg, yt-dlp, fastapi, curl_cffi
RUN pip install --no-cache-dir \
    openai-whisper \
    moviepy \
    scipy \
    numpy

# Copy the API server and assets
COPY main.py .
COPY chi_rho.png /app/chi_rho.png

# Create directories
RUN mkdir -p /downloads /outputs && chmod 777 /downloads /outputs

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
