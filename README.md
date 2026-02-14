# RealContent AI

**AI-powered video content pipeline for real estate marketing** — end-to-end automation from content discovery to multi-platform publishing.

Built with FastAPI, Celery, PostgreSQL, FFmpeg (GPU-accelerated), React, and Docker.

---

## Overview

RealContent AI automates the entire real estate video content lifecycle:

1. **Discover** trending real estate content across social platforms
2. **Analyze** transcripts with LLM-powered insight extraction
3. **Generate** scripts tailored to your brand persona and content pillars
4. **Produce** AI avatar videos with professional voiceover and captions
5. **Compose** final videos with GPU-accelerated rendering (NVIDIA NVENC)
6. **Publish** to TikTok, Instagram Reels, YouTube Shorts, and more

Each video costs approximately **$2-5 to produce**, fully autonomously.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        REALCONTENT AI PIPELINE                         │
│                                                                        │
│  ┌──────────┐    ┌─────────┐   ┌─────────┐   ┌─────────┐             │
│  │  React   │◄───│ FastAPI │◄──│ Celery  │──►│  Redis  │             │
│  │Dashboard │    │ Backend │   │ Workers │   │  Queue  │             │
│  └──────────┘    └────┬────┘   └────┬────┘   └─────────┘             │
│                       │             │                                  │
│                       ▼             ▼                                  │
│                  ┌─────────┐  ┌──────────────────────────┐            │
│                  │PostgreSQL│  │   GPU Video Processor    │            │
│                  │ Database │  │  FFmpeg + NVENC encoding │            │
│                  └─────────┘  │  Whisper transcription   │            │
│                               │  Karaoke caption burning │            │
│                               └──────────────────────────┘            │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                     EXTERNAL SERVICES                            │  │
│  │  Apify (Discovery) | LLM (Script Gen) | ElevenLabs (TTS)        │  │
│  │  HeyGen (Avatar)   | Whisper (STT)    | Pexels (B-Roll)         │  │
│  │  Dropbox (Storage) | Blotato (Publishing)                       │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **API** | FastAPI (Python 3.9+), Uvicorn |
| **Task Queue** | Celery + Redis |
| **Database** | PostgreSQL 16, SQLAlchemy ORM |
| **Video Processing** | FFmpeg, NVIDIA NVENC (h264_nvenc), OpenCV |
| **AI / ML** | OpenAI Whisper (STT), ElevenLabs (TTS), HeyGen (Avatars), LLM via OpenRouter |
| **Frontend** | React 18, React Router, TanStack Query |
| **Infrastructure** | Docker Compose (7 containers), NVIDIA Container Toolkit |
| **Storage** | Dropbox API, local volume mounts |

---

## Features

### Content Generation Pipeline (9 Stages)

| Stage | Description | Technology |
|-------|------------|------------|
| 1. Content Discovery | Scrape trending real estate content from TikTok/Instagram | Apify |
| 2. Source Download | Download source videos for reference | yt-dlp |
| 3. Transcription | Extract and transcribe source audio | OpenAI Whisper |
| 4. Script Generation | LLM writes branded scripts with hooks and CTAs | Grok via OpenRouter |
| 5. Voice Generation | Text-to-speech with selected voice profile | ElevenLabs |
| 6. Avatar Generation | AI avatar video on green screen | HeyGen |
| 7. Video Composition | Chromakey overlay, GPU-accelerated encoding | FFmpeg + NVENC |
| 8. Caption Burning | Karaoke-style word-by-word captions | Whisper + FFmpeg |
| 9. Multi-Platform Publish | Distribute to social platforms | Blotato |

### Content Discovery Pipeline

Monitors real estate content creators and extracts high-engagement clips:

- **Download** full videos via yt-dlp with resume support
- **Transcribe** with word-level timestamps (Whisper GPU)
- **Analyze** transcripts with LLM to identify top moments
- **Render** clips with GPU effects pipeline (18 visual effects, B-roll insertion, karaoke captions)

### Smart Resource Management

- **GPU Semaphore**: Redis-based distributed locking prevents GPU memory exhaustion
- **Whisper Queue**: Single-instance lock ensures transcription stability
- **Asset Reuse**: Checks for existing files before making expensive API calls
- **Render Concurrency**: Lua-scripted atomic semaphore for parallel renders

### Brand Persona System

- Configurable agent identity (name, tone, bio, values)
- Four content pillars: Market Intelligence, Educational Tips, Lifestyle/Local, Brand Humanization
- Custom hooks, CTAs, and platform-specific captions
- Per-persona LLM prompt templates

### React Dashboard

- Pipeline overview with real-time status tracking
- Content idea management (approve/reject workflow)
- Brand persona editor
- Music and B-roll library management
- Publishing queue with approval workflow
- API credit monitoring (ElevenLabs, HeyGen, OpenRouter)

---

## Project Structure

```
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── models.py               # SQLAlchemy models (20+ tables)
│   ├── schemas.py              # Pydantic request/response schemas
│   ├── config.py               # Settings management
│   ├── celery_app.py           # Celery configuration
│   ├── routers/
│   │   ├── pipeline.py         # Content pipeline endpoints
│   │   ├── viral.py            # Content discovery endpoints
│   │   └── publishing.py       # Publishing queue management
│   ├── services/
│   │   ├── avatar.py           # HeyGen avatar generation
│   │   ├── captions.py         # Whisper + ASS subtitle generation
│   │   ├── clip_analyzer.py    # LLM-powered clip analysis
│   │   ├── pexels.py           # Stock B-roll search
│   │   ├── publisher.py        # Multi-platform distribution
│   │   ├── script_generator.py # LLM script generation
│   │   ├── scraper.py          # Trend discovery (Apify)
│   │   ├── video.py            # FFmpeg composition
│   │   ├── voice.py            # ElevenLabs TTS
│   │   └── storage.py          # Dropbox integration
│   ├── tasks/
│   │   ├── pipeline.py         # 9-stage Celery pipeline
│   │   ├── viral.py            # Content discovery tasks
│   │   └── scrape.py           # Scraping scheduler
│   └── utils/
│       ├── logging.py          # Structured logging
│       ├── paths.py            # Asset path management
│       └── retry.py            # Retry decorator
├── video-downloader/
│   ├── main.py                 # GPU video processing service (FastAPI)
│   ├── Dockerfile              # CUDA-enabled container
│   └── fonts/                  # Custom caption fonts
├── frontend/
│   ├── src/
│   │   ├── App.js              # React Router (13 pages)
│   │   ├── api.js              # API client
│   │   └── pages/              # Dashboard, Settings, Pipeline views
│   └── Dockerfile
├── docker-compose.yml          # 7-container orchestration
└── .env.example                # Required environment variables
```

---

## Getting Started

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with NVENC support
- NVIDIA Container Toolkit installed

### Setup

```bash
# Clone the repository
git clone https://github.com/cameronsaddress/realcontent-ai.git
cd realcontent-ai

# Configure environment
cp .env.example .env
# Edit .env with your API keys (see .env.example for required variables)

# Build and start all services
docker compose up --build -d

# Access the dashboard
open http://localhost:3000
```

### Required API Keys

| Service | Purpose | Get Key |
|---------|---------|---------|
| OpenRouter | LLM script generation and analysis | [openrouter.ai](https://openrouter.ai) |
| ElevenLabs | Text-to-speech voiceover | [elevenlabs.io](https://elevenlabs.io) |
| HeyGen | AI avatar video generation | [heygen.com](https://heygen.com) |
| OpenAI | Whisper transcription | [platform.openai.com](https://platform.openai.com) |
| Apify | Social media content discovery | [apify.com](https://apify.com) |
| Pexels | Stock video B-roll | [pexels.com/api](https://www.pexels.com/api/) |

---

## Infrastructure

**Docker Services (7 containers):**

| Container | Port | Purpose |
|-----------|------|---------|
| `realcontent_backend` | 8000 | FastAPI REST API |
| `realcontent_celery_worker` | -- | Pipeline task execution (GPU) |
| `realcontent_celery_beat` | -- | Scheduled task scheduler |
| `realcontent_frontend` | 3000 | React dashboard |
| `realcontent_video_processor` | 8080 | GPU video encoding (NVENC) |
| `realcontent_postgres` | 5433 | PostgreSQL 16 database |
| `realcontent_redis` | 6380 | Redis message broker |

**GPU Configuration:**
- Celery workers and video processor run with full NVIDIA GPU access
- IPC host mode + 8GB shared memory for video processing
- Persistent Whisper model cache via Docker volumes

---

## License

MIT
