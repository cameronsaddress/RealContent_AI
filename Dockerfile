# n8n with FFmpeg static build for ARM64
FROM docker.n8n.io/n8nio/n8n

USER root

# Download and install static FFmpeg build for ARM64 (no dependencies needed)
RUN wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz \
    && tar xf ffmpeg-release-arm64-static.tar.xz \
    && mv ffmpeg-*-arm64-static/ffmpeg /usr/local/bin/ffmpeg \
    && mv ffmpeg-*-arm64-static/ffprobe /usr/local/bin/ffprobe \
    && rm -rf ffmpeg-* \
    && chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe

# Create directories for assets
RUN mkdir -p /home/node/assets/fonts /home/node/assets/logos /home/node/assets/templates \
    && mkdir -p /home/node/.n8n-files/assets/videos \
    && chown -R node:node /home/node/assets /home/node/.n8n-files

USER node

WORKDIR /home/node
