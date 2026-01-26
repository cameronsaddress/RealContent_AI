# SocialGen - AI Video Content Pipeline

Automated video production system for real estate content creators. Discovers trending content, generates scripts, creates AI avatar videos with karaoke-style captions, and publishes to multiple social platforms - all autonomously.

## System Architecture

```
                            ┌─────────────────────────────────────────────────────────────────┐
                            │                    SOCIALGEN PIPELINE                            │
                            │                                                                  │
┌──────────────┐           │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐          │
│   Frontend   │◄──────────┼──│ FastAPI │◄──│ Celery  │──►│  Redis  │◄──│  Beat   │          │
│   React UI   │           │  │ Backend │   │ Worker  │   │  Queue  │   │Scheduler│          │
└──────────────┘           │  └────┬────┘   └────┬────┘   └─────────┘   └─────────┘          │
                            │       │            │                                             │
                            │       ▼            │                                             │
                            │  ┌─────────┐       │    ┌───────────────────────────────────┐   │
                            │  │PostgreSQL│       │    │         GPU Video Processor       │   │
                            │  │ Database │       └───►│  FFmpeg + NVENC (h264_nvenc)     │   │
                            │  └─────────┘            │  - Chromakey composition          │   │
                            │                          │  - Karaoke caption burning        │   │
                            │                          └───────────────────────────────────┘   │
                            │                                                                  │
                            │  ┌─────────────────────────────────────────────────────────────┐ │
                            │  │                    EXTERNAL SERVICES                         │ │
                            │  │  Apify (Scraping) │ Grok 4.1 (LLM) │ ElevenLabs (TTS)       │ │
                            │  │  HeyGen (Avatar)  │ Whisper (STT)  │ Blotato (Publishing)   │ │
                            │  │  Dropbox (Storage)│ Pexels (B-Roll)                         │ │
                            │  └─────────────────────────────────────────────────────────────┘ │
                            └──────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Orchestration** | Celery + Redis | Task queue and pipeline orchestration |
| **Backend API** | FastAPI + PostgreSQL | REST API and data persistence |
| **Frontend** | React | Dashboard for monitoring and control |
| **GPU Processing** | FFmpeg + NVENC | Hardware-accelerated video encoding |
| **LLM** | Grok 4.1 (OpenRouter) | Script generation and trend analysis |
| **Voice** | ElevenLabs | Text-to-speech generation |
| **Avatar** | HeyGen | AI avatar video creation (green screen) |
| **Transcription** | OpenAI Whisper | Word-level timing for karaoke captions |
| **Scraping** | Apify | TikTok/Instagram trend discovery |
| **Publishing** | Blotato | Multi-platform distribution |
| **Storage** | Dropbox | Video hosting for publishing |

## Quick Start

### Prerequisites
- Docker with NVIDIA GPU support (nvidia-container-toolkit)
- NVIDIA GPU with NVENC support
- API keys for external services

### 1. Clone and Configure

```bash
cd /home/canderson/n8n
cp .env.example .env  # Edit with your API keys
```

### 2. Start Services

```bash
docker compose up -d
```

### 3. Access Interfaces

| Service | URL | Purpose |
|---------|-----|---------|
| Frontend | http://100.83.153.43:3000 | Dashboard UI |
| Backend API | http://100.83.153.43:8000 | REST API |
| API Docs | http://100.83.153.43:8000/docs | Swagger documentation |
| Video Processor | http://100.83.153.43:8080 | GPU encoding service |

### 4. Run Your First Pipeline

```bash
# Trigger pipeline for a content idea
curl -X POST http://100.83.153.43:8000/api/pipeline/trigger \
  -H "Content-Type: application/json" \
  -d '{"content_idea_id": 141}'

# Check task status
curl http://100.83.153.43:8000/api/pipeline/status/{task_id}
```

## Pipeline Stages

The pipeline processes content through 9 stages, with each stage checking for existing assets to avoid redundant API calls:

### Stage 1: Content Retrieval
- Fetches approved content idea from database
- Extracts source URL, pillar, and metadata

### Stage 2: Source Video Download (Apify)
- **Reuses existing:** Checks for `idea_{content_idea_id}_source.mp4`
- Downloads original TikTok/Instagram video via Apify
- Handles watermark removal and CDN authentication
- Saves to `/app/assets/videos/`
- **Must happen BEFORE script generation** to get transcription

### Stage 3: Source Video Transcription (OpenAI Whisper)
- **Reuses existing:** Checks `content_ideas.source_transcription`
- Extracts audio from source video (ffmpeg → 16kHz mono MP3)
- Transcribes what they SAID in the viral video
- Saves transcription to database for script context
- **Critical:** Gives LLM context about video content, not just caption

### Stage 4: Script Generation (Grok 4.1)
- **Reuses existing:** Checks `scripts` table for `content_idea_id`
- **Now receives:** Caption + Source Transcription + Brand Persona
- Generates hook, body, CTA that REACT to what was said in the video
- Creates platform-specific captions (TikTok, Instagram, YouTube, etc.)
- Content pillars:
  - `market_intelligence` - Data, trends, analysis
  - `educational_tips` - How-tos, tutorials
  - `lifestyle_local` - Community content
  - `brand_humanization` - Personal stories

### Stage 5: Voice Generation (ElevenLabs)
- **Reuses existing:** Checks for `{script_id}_voice.mp3`
- Generates TTS audio from full script
- Saves to `/app/assets/audio/`

### Stage 6: Avatar Generation (HeyGen)
- **Reuses existing:** Checks for `{script_id}_avatar.mp4`
- **Resume support:** Stores `heygen_video_id` in assets table
- Uploads audio, creates video job, polls for completion
- Downloads green screen avatar video to `/app/assets/avatar/`

### Stage 7: Video Composition (GPU FFmpeg)
- **Reuses existing:** Checks for `{script_id}_combined.mp4`
- Calls GPU video-processor service via HTTP
- Chromakey removes green screen from avatar
- Overlays avatar on source video with configurable position
- Uses NVENC hardware encoding (h264_nvenc)

### Stage 8: Caption Burning (GPU FFmpeg)
- **Reuses existing:** Checks for `{script_id}_final.mp4`
- Transcribes OUR voice audio with Whisper (word-level timing)
- Generates ASS file with karaoke effects (`\kf` tags)
- Burns captions via GPU video-processor service

### Stage 9: Upload & Publish
- Uploads final video to Dropbox
- Publishes to platforms via Blotato API
- Records publish status in database

## Asset Reuse Logic

Each stage checks for existing assets before making expensive API calls:

| Stage | Check Method | File Pattern |
|-------|--------------|--------------|
| Source Video | File exists check | `videos/idea_{content_idea_id}_source.mp4` |
| Transcription | DB field `source_transcription` | N/A (stored in content_ideas table) |
| Script | DB query `scripts.content_idea_id` | N/A |
| Voice | `voice_exists(script_id)` | `audio/{script_id}_voice.mp3` |
| Avatar | `avatar_exists(script_id)` | `avatar/{script_id}_avatar.mp4` |
| Combined | `combined_video_exists(script_id)` | `output/{script_id}_combined.mp4` |
| Final | `final_video_exists(script_id)` | `output/{script_id}_final.mp4` |

**To force regeneration:** Delete the corresponding file before running the pipeline.

## Source Video Transcription

The pipeline now transcribes the source video BEFORE generating the script. This gives Grok context about what was actually SAID in the viral video:

```
=== WHAT GROK RECEIVES ===

Post Caption: "Day 68, saving for your house"

WHAT THEY ACTUALLY SAY IN THE VIDEO:
"""
Day 68 of saving up money every single day until we buy our first home.
Houses are quite expensive which means we need a lot of money. A hundred
thousand is the goal I'm trying to achieve and I'm already at fifty one
thousand...
"""
```

This enables scripts that actually REACT to the video content, not just the caption.

## GPU Video Processing

The video-processor container provides hardware-accelerated encoding:

### Compose Endpoint
```bash
POST http://video-processor:8080/compose
{
  "script_id": "33",
  "avatar_path": "/avatar/33_avatar.mp4",
  "background_path": "/downloads/33_source.mp4",
  "audio_path": "/audio/33_voice.mp3",
  "avatar_scale": 0.75,
  "avatar_offset_x": -250,
  "avatar_offset_y": 600,
  "greenscreen_color": "0x00FF00",
  "use_gpu": true
}
```

### Caption Endpoint
```bash
POST http://video-processor:8080/caption
{
  "script_id": "33",
  "video_path": "/outputs/33_combined.mp4",
  "ass_path": "/captions/33_captions.ass",
  "use_gpu": true
}
```

### Performance Comparison
| Encoder | File Size | Encoding Time |
|---------|-----------|---------------|
| h264_nvenc (GPU) | ~14 MB | ~30 seconds |
| libx264 (CPU) | ~200 MB | ~5 minutes |

## Karaoke Caption System

Captions use ASS format with karaoke fill effects:

```ass
[V4+ Styles]
Style: Default,Arial,75,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,...

[Events]
Dialogue: 0,0:00:00.00,0:00:01.51,Default,,0,0,0,,{\kf30}Hey {\kf30}neighbors
```

- **Primary Color:** White (`&H00FFFFFF`)
- **Secondary Color:** Yellow (`&H0000FFFF`) - fills as word is spoken
- **Effect:** `\kf` (karaoke fill) with duration in centiseconds
- **Words per line:** 5 (configurable)
- **Position:** Center of screen (MarginV: 960 for 1920px height)

## Settings Configuration

Settings are stored in `system_settings` table and read by the pipeline:

### Video Settings
```json
{
  "avatar_scale": 0.75,
  "avatar_offset_x": -250,
  "avatar_offset_y": 600,
  "greenscreen_enabled": true,
  "greenscreen_color": "#00FF00",
  "caption_style": "karaoke",
  "caption_font": "Arial",
  "caption_font_size": 75,
  "caption_color": "#FFFFFF",
  "caption_highlight_color": "#FFFF00",
  "caption_position_y": 960
}
```

### Update Settings
```bash
curl -X PUT http://100.83.153.43:8000/api/settings/video \
  -H "Content-Type: application/json" \
  -d '{"avatar_offset_x": -250, "avatar_offset_y": 600}'
```

## Docker Services

| Service | Container | Port | Purpose |
|---------|-----------|------|---------|
| PostgreSQL | `SocialGen_postgres` | 5433 | Database |
| Redis | `SocialGen_redis` | 6380 | Task queue |
| Backend | `SocialGen_backend` | 8000 | FastAPI API |
| Celery Worker | `SocialGen_celery_worker` | - | Pipeline execution |
| Celery Beat | `SocialGen_celery_beat` | - | Scheduled tasks |
| Frontend | `SocialGen_frontend` | 3000 | React dashboard |
| Video Processor | `SocialGen_video_processor` | 8080 | GPU encoding |

## Directory Structure

```
/home/canderson/n8n/
├── docker-compose.yml          # Service orchestration
├── .env                        # Environment variables
├── CLAUDE.md                   # AI assistant instructions
├── README.md                   # This file
│
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── celery_app.py           # Celery configuration
│   ├── config.py               # Settings from environment
│   ├── models.py               # SQLAlchemy models
│   ├── schemas.py              # Pydantic schemas
│   ├── routers/                # API route handlers
│   │   └── viral.py            # Viral Clip Factory API
│   ├── services/               # Business logic
│   │   ├── avatar.py           # HeyGen integration
│   │   ├── captions.py         # Whisper + ASS generation
│   │   ├── clip_analyzer.py    # Grok AI clip analysis + effects director
│   │   ├── publisher.py        # Blotato publishing
│   │   ├── scraper.py          # Apify trend discovery
│   │   ├── script_generator.py # Grok LLM integration
│   │   ├── storage.py          # Dropbox upload
│   │   ├── video.py            # FFmpeg composition
│   │   ├── video_downloader.py # TikTok download
│   │   └── voice.py            # ElevenLabs TTS
│   ├── tasks/
│   │   ├── pipeline.py         # Main pipeline task
│   │   ├── viral.py            # Viral clip analysis + rendering tasks
│   │   └── scrape.py           # Scraping tasks
│   ├── utils/
│   │   ├── logging.py          # Structured logging
│   │   ├── paths.py            # Asset path management
│   │   └── retry.py            # Retry decorator
│   └── tests/                  # Unit tests
│
├── video-downloader/
│   ├── main.py                 # FastAPI GPU service (effects engine, 18 effects)
│   ├── fonts/                  # Custom fonts for viral captions
│   └── Dockerfile              # GPU-enabled container
│
├── frontend/
│   └── src/
│       ├── App.js              # Main React component
│       ├── api.js              # API client
│       └── pages/              # Dashboard pages
│
├── assets/                     # Media storage (volume mounted)
│   ├── audio/                  # Voice files ({script_id}_voice.mp3)
│   ├── avatar/                 # HeyGen output ({script_id}_avatar.mp4)
│   ├── broll/                  # B-roll clips for viral montages
│   ├── luts/                   # .cube LUT files for color grading
│   ├── videos/                 # Source downloads (video_{id}.mp4)
│   ├── captions/               # ASS/SRT files ({script_id}_captions.ass)
│   ├── output/                 # Combined/Final/Rendered clips
│   └── music/                  # Background music tracks
│
└── archive/                    # Deprecated n8n workflows
```

## Monitoring & Debugging

### View Pipeline Logs
```bash
# Real-time monitoring
docker logs -f SocialGen_celery_worker

# Find errors
docker logs SocialGen_celery_worker 2>&1 | grep -i error

# Check which stage failed
docker logs SocialGen_celery_worker 2>&1 | grep "stage="
```

### Database Access
```bash
docker exec -it SocialGen_postgres psql -U n8n -d content_pipeline

# View recent scripts
SELECT id, content_idea_id, status FROM scripts ORDER BY id DESC LIMIT 5;

# View asset status
SELECT script_id, avatar_video_path, final_video_path FROM assets ORDER BY id DESC LIMIT 5;
```

### Task Queue Status
```bash
# Pending tasks
docker exec SocialGen_redis redis-cli LLEN celery

# Worker status
docker logs SocialGen_celery_worker 2>&1 | grep "ready"
```

## API Reference

### Pipeline
```bash
# Trigger pipeline
POST /api/pipeline/trigger
{"content_idea_id": 141}

# Get status
GET /api/pipeline/status/{task_id}

# Get stats
GET /api/pipeline/stats
```

### Content Ideas
```bash
# List
GET /api/content-ideas?status=approved

# Create
POST /api/content-ideas
{"source_url": "https://tiktok.com/...", "status": "approved"}

# Update
PATCH /api/content-ideas/{id}
{"status": "approved"}
```

### Settings
```bash
# Get all settings
GET /api/settings/all

# Update video settings
PUT /api/settings/video
{"avatar_offset_x": -250, "avatar_offset_y": 600}

# Get character config
GET /api/settings/character
```

## Viral Clip Factory

The Viral Clip Factory is a second pipeline that monitors influencer channels, downloads full-length videos, uses Grok AI to identify up to 20 viral-worthy segments per video, and renders them with cinematic effects.

### Viral Pipeline Flow
```
Add Influencer → Fetch Videos → Download (yt-dlp) → Transcribe (Whisper)
    → Analyze (Grok AI identifies clips) → Render (GPU effects pipeline)
```

### Grok as Movie Director

Grok acts as a "movie director" - choosing color grade, motion effects, glitch style, caption animation, transitions, and audio reactivity per clip based on content energy, topic, and emotional arc.

### Effects System (18 Effects)

| Category | Effects | Description |
|----------|---------|-------------|
| **Color Grades** | 8 LUT presets + vibrant/bw | Cinema-quality 3D LUT color grading (Kodak, Teal Orange, Film Noir, etc.) |
| **Motion** | Camera Shake, Speed Ramps, Temporal Trail | Handheld feel, slow-mo emphasis, ghosting streaks |
| **Glitch** | Retro Glow, Wave Displacement, Heavy VHS | Neon bloom, melting distortion, VHS tracking noise |
| **Captions** | Standard, Pop Scale, Shake, Blur Reveal | Word-by-word karaoke with animated trigger styles |
| **Audio Reactive** | Beat Sync, Audio Saturation | Zoom pulses on beats, color boost on loud moments |
| **Transitions** | Pixelize, Radial, Dissolve, Slide, Fade | B-roll cut transitions via FFmpeg xfade |
| **Rare** | Datamosh, Pixel Sort | Frame-melt glitch, brightness-sorted glitch art |

### Render Pipeline (GPU)
```
Extract Segment → Mega-Filter Chain (CUDA) → Speed Ramps → RGB Glitch
    → VHS Effects → Template FX → Datamosh/Pixel Sort → B-Roll Montage
    → 9-Grid Outro → BGM Mix → Final Output (1080x1920 H.264)
```

### Mega-Filter Chain Order
```
hwdownload → scale(-2:1920) → scale(zoom+pulse) → crop(1080:1920 + face + shake)
    → [LUT or eq grade] → [audio saturation] → [retro glow] → [temporal trail]
    → [wave displacement] → [VHS grain] → ASS captions → hwupload_cuda
```

### Viral API Endpoints
```bash
# Influencer management
GET/POST /api/viral/influencers
POST /api/viral/influencers/{id}/fetch

# Video analysis
POST /api/viral/videos/{id}/analyze
GET /api/viral/influencers/{id}/videos

# Clip rendering
POST /api/viral/viral-clips/{id}/render
GET /api/viral/viral-clips

# Effects
GET /api/viral/effects-catalog
PUT /api/viral/viral-clips/{id}/effects

# B-Roll management
GET /api/viral/broll
POST /api/viral/broll/upload-youtube
```

### Viral Factory Cost
| Service | Cost | Notes |
|---------|------|-------|
| OpenRouter (Grok) | ~$0.02-0.05 | Per video analysis |
| GPU Rendering | Free | Local NVIDIA DGX |
| Storage | Free | Local disk |

### B-Roll Category System

Grok assigns B-roll insertions using a semantic 28-category system. Categories map to local clips via AI tagging or filename prefixes, with YouTube fallback for missing content.

**Valid Categories:**
| Group | Categories |
|-------|------------|
| Destruction | `war`, `chaos`, `explosions`, `storms`, `fire` |
| Money | `money`, `luxury`, `wealth`, `city` |
| Fitness | `gym`, `sports`, `boxing`, `strength` |
| Faith | `patriotic`, `crowd`, `faith`, `cathedrals` |
| Animals | `lions`, `eagles`, `wolves`, `nature` |
| Military | `jets`, `navy`, `helicopters` |
| Vehicles | `cars`, `racing` |
| Other | `history`, `people`, `victory`, `power` |

**AI Tagging:** Run `docker exec -it SocialGen_video_processor python /app/scripts/tag_broll.py` to tag local clips with BLIP+CLIP.

---

## Cost Estimates (Main Pipeline)

| Service | Cost Per Run | Notes |
|---------|--------------|-------|
| HeyGen | ~$1-3 | Per avatar video |
| ElevenLabs | ~$0.01-0.05 | Per voice generation |
| Apify | ~$0.30-1.50 | Per scrape run |
| OpenRouter (Grok) | ~$0.01 | Per script |
| **Total** | **~$2-5** | Per published video |

## Environment Variables

```bash
# Database
DATABASE_URL=postgresql://n8n:password@postgres:5432/content_pipeline
POSTGRES_USER=n8n
POSTGRES_PASSWORD=password
POSTGRES_DB=content_pipeline

# Network
N8N_HOST=100.83.153.43

# API Keys
OPENROUTER_API_KEY=sk-or-v1-...
OPENAI_API_KEY=sk-...
ELEVENLABS_API_KEY=sk_...
ELEVENLABS_VOICE_ID=...
HEYGEN_API_KEY=...
HEYGEN_AVATAR_ID=...
APIFY_API_KEY=apify_api_...
BLOTATO_API_KEY=...
DROPBOX_APP_KEY=...
DROPBOX_APP_SECRET=...
DROPBOX_REFRESH_TOKEN=...
```

## Development

### Rebuild Services
```bash
# Rebuild specific service
docker compose up -d --build celery-worker

# Rebuild all
docker compose up -d --build
```

### Run Tests
```bash
docker exec SocialGen_backend python3 -m pytest /app/tests/ -v
```

### Hot Reload
- **Backend:** Mounted volume with `--reload` flag
- **Frontend:** Mounted `src/` directory

---

*Last updated: January 26, 2026*
