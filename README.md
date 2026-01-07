# AI Video Content Pipeline

Automated video production system for real estate content creators. Discovers trending content, generates scripts, creates AI avatar videos with captions, and publishes to multiple social platforms - all autonomously.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           COMPLETE PIPELINE                                  │
│                                                                             │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐       │
│  │ SCRAPE  │ → │  AUTO   │ → │ SCRIPT  │ → │   GET   │ → │ CREATE  │       │
│  │ TRENDS  │   │ TRIGGER │   │   GEN   │   │  VIDEO  │   │  VOICE  │       │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘       │
│       │             │             │             │             │             │
│       ▼             ▼             ▼             ▼             ▼             │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐       │
│  │ CREATE  │ ← │ COMBINE │ ← │ CAPTION │ ← │ PUBLISH │ ← │ANALYTICS│       │
│  │ AVATAR  │   │  VIDS   │   │  (SRT)  │   │(Blotato)│   │ (Stats) │       │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Workflow Engine | n8n | Orchestrates the entire pipeline |
| Backend API | FastAPI + PostgreSQL | Data persistence & REST API |
| Frontend | React | Dashboard for monitoring & control |
| LLM | Grok 4.1 (OpenRouter) | Script generation & trend analysis |
| Voice | ElevenLabs | Text-to-speech generation |
| Avatar | HeyGen | AI avatar video creation |
| B-Roll | Pexels | Stock video footage |
| Captions | OpenAI Whisper | Automatic transcription |
| Video Processing | FFmpeg | Compositing & caption burn |
| Scraping | Apify | TikTok/Instagram trend discovery |
| Publishing | Blotato | Multi-platform distribution |

## Quick Start

### Prerequisites
- Docker & Docker Compose
- API keys for: OpenRouter, ElevenLabs, HeyGen, Pexels, OpenAI, Apify, Blotato

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
| n8n | http://100.83.153.43:5678 | Workflow editor |
| Frontend | http://100.83.153.43:3000 | Dashboard UI |
| Backend API | http://100.83.153.43:8000 | REST API |
| PostgreSQL | localhost:5433 | Database |

### 4. Import Workflow

```bash
docker exec n8n n8n import:workflow --input=/home/node/workflows/COMPLETE_PIPELINE.json
```

### 5. Import Credentials

```bash
docker exec n8n n8n import:credentials --input=/home/node/workflows/credentials.json
```

### 6. Configure Credentials in n8n (Manual Method)

If import fails, navigate to Settings → Credentials and create:

Navigate to Settings → Credentials and create:

| Credential Name | Type | Configuration |
|-----------------|------|---------------|
| `openrouter` | HTTP Header Auth | Name: `Authorization`, Value: `Bearer sk-or-v1-...` |
| `elevenlabs` | HTTP Header Auth | Name: `xi-api-key`, Value: `sk_...` |
| `heygen` | HTTP Header Auth | Name: `X-Api-Key`, Value: `...` |
| `pexels` | HTTP Header Auth | Name: `Authorization`, Value: `...` |
| `openai` | HTTP Header Auth | Name: `Authorization`, Value: `Bearer sk-...` |
| `apify` | HTTP Header Auth | Name: `Authorization`, Value: `Bearer apify_api_...` |
| `blotato` | HTTP Header Auth | Name: `Authorization`, Value: `Bearer ...` |

## Pipeline Sections

### 1. SCRAPE - Trend Discovery
**Triggers:** Daily at 6am UTC OR UI webhook

Discovers viral content from TikTok and Instagram using Apify scrapers. Analyzes with Grok 4.1 to score virality and categorize by content pillar.

```
Webhook: POST /webhook/scrape-trends
Body: {
  "niche": "real estate",
  "hashtags": ["realestate", "homebuying"],
  "platforms": ["tiktok", "instagram"]
}
```

### 2. AUTO TRIGGER - Pipeline Orchestrator
**Triggers:** Every 15 minutes OR UI webhook

Polls for approved content ideas and kicks off the full production pipeline.

```
Webhook: POST /webhook/trigger-pipeline
Body: { "content_idea_id": 123 }  // Optional - processes specific idea
```

### 3. SCRIPT GEN - AI Scriptwriting
**Input:** Approved content idea
**Output:** Hook, body, CTA, duration estimate, suggested B-roll

Uses Grok 4.1 to create viral scripts tailored to content pillars:
- `market_intelligence` - Data, trends, market analysis
- `educational_tips` - How-tos, tutorials, tips
- `lifestyle_local` - Community, local content
- `brand_humanization` - Personal stories, behind-the-scenes

### 4. GET VIDEO - B-Roll Acquisition
**Input:** Script with suggested scenes
**Output:** Downloaded MP4 files

Searches Pexels for portrait-orientation stock videos matching scene descriptions.

### 5. CREATE VOICE - TTS Generation
**Input:** Full script text
**Output:** MP3 voice file

ElevenLabs text-to-speech with configurable voice ID and settings.

### 6. CREATE AVATAR - AI Video
**Input:** Audio file URL
**Output:** Avatar video with green screen

HeyGen generates lip-synced avatar video. Polls for completion then downloads.

### 7. COMBINE VIDS - FFmpeg Compositing
**Input:** Avatar video + B-roll clips
**Output:** Combined video

Chromakey removes green screen, overlays avatar on B-roll footage.

### 8. CAPTION - Whisper + Burn
**Input:** Combined video + audio
**Output:** Final video with burned captions

OpenAI Whisper generates SRT, FFmpeg burns styled subtitles.

### 9. PUBLISH - Multi-Platform
**Input:** Final video + generated captions
**Output:** Published posts

Blotato API distributes to TikTok, Instagram Reels, YouTube Shorts, etc.

### 10. FILE SERVER - Asset Delivery
Webhooks serve audio/video files to external APIs (HeyGen, Blotato):

```
GET /webhook/serve-audio/:script_id  → MP3
GET /webhook/serve-video/:script_id  → MP4
```

## Database Schema

### Core Tables

```sql
content_ideas     -- Scraped/imported content with status tracking
scripts           -- Generated scripts linked to content ideas
assets            -- Media files (audio, video, avatar)
published         -- Published post records with platform URLs
scrape_runs       -- Trend scrape history and results
niche_presets     -- Saved niche configurations
```

### Status Flow

```
content_ideas: pending → approved → script_generating → script_ready → published
scripts:       pending → script_ready → voice_generating → voice_ready
assets:        pending → voice_ready → avatar_generating → avatar_ready → assembling → captioning → ready_to_publish → published
```

## API Reference

### Content Ideas

```bash
# List ideas (with filters)
GET /api/content-ideas?status=pending&pillar=educational_tips

# Create idea
POST /api/content-ideas
{
  "source_url": "https://tiktok.com/...",
  "source_platform": "tiktok",
  "original_text": "...",
  "pillar": "educational_tips",
  "viral_score": 8,
  "suggested_hook": "...",
  "status": "pending"
}

# Update status
PATCH /api/content-ideas/{id}
{ "status": "approved" }
```

### Scrape Operations

```bash
# Trigger scrape
POST /api/scrape/run
{
  "niche": "real estate",
  "hashtags": ["realestate", "homebuying"],
  "platforms": ["tiktok", "instagram"]
}

# List scrape runs
GET /api/scrape/runs

# Get niche presets
GET /api/niche-presets

# Create preset
POST /api/niche-presets
{
  "name": "Luxury Homes",
  "keywords": ["luxury real estate", "million dollar homes"],
  "hashtags": ["luxuryrealestate", "luxuryhomes"]
}
```

### Pipeline Stats

```bash
GET /api/pipeline/stats
→ { "pending": 5, "approved": 2, "published": 47, ... }
```

## Directory Structure

```
/home/canderson/n8n/
├── docker-compose.yml      # Service orchestration
├── Dockerfile              # n8n with FFmpeg
├── .env                    # Environment variables (secrets)
├── MASTERPLAN.md           # Detailed implementation guide
├── README.md               # This file
│
├── workflows/
│   └── COMPLETE_PIPELINE.json  # Unified n8n workflow
│
├── backend/
│   ├── main.py             # FastAPI application
│   ├── models.py           # SQLAlchemy models
│   ├── schemas.py          # Pydantic schemas
│   ├── init.sql            # Database initialization
│   ├── migrations/         # Database migrations
│   ├── Dockerfile          # Backend container
│   └── requirements.txt    # Python dependencies
│
├── frontend/
│   ├── src/
│   │   ├── App.js          # Main React component
│   │   ├── components/     # UI components
│   │   └── api.js          # API client
│   ├── Dockerfile          # Frontend container
│   └── package.json        # Node dependencies
│
└── assets/                 # Media storage (volume mounted)
    ├── audio/              # Voice files
    ├── avatar/             # HeyGen output
    ├── videos/             # B-roll downloads
    ├── captions/           # SRT files
    └── output/             # Final videos
```

## Environment Variables

```bash
# Database
DATABASE_URL=postgresql://n8n:n8n_password@postgres:5432/content_pipeline
POSTGRES_USER=n8n
POSTGRES_PASSWORD=n8n_password
POSTGRES_DB=content_pipeline

# n8n
N8N_HOST=100.83.153.43
N8N_PORT=5678
N8N_PUBLIC_URL=http://100.83.153.43:5678

# API Keys
OPENROUTER_API_KEY=sk-or-v1-...
OPENAI_API_KEY=sk-...
ELEVENLABS_API_KEY=sk_...
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
HEYGEN_API_KEY=...
HEYGEN_AVATAR_ID=Kristin_pubblic_3_20240108
PEXELS_API_KEY=...
APIFY_API_KEY=apify_api_...
BLOTATO_API_KEY=...
```

## Troubleshooting

### Common Issues

**n8n can't connect to backend**
- Ensure containers are on same Docker network
- Use `http://backend:8000` not `localhost:8000`

**HeyGen video stuck "processing"**
- Check audio file is accessible via public URL
- Verify N8N_PUBLIC_URL is reachable from internet

**FFmpeg fails**
- Check file paths use `/home/node/assets/` (container path)
- Verify assets volume is mounted correctly

**Scrape returns empty**
- Check Apify API key is valid
- Verify hashtags exist and have content

**Scrape Error: "Expecting value: line 1 column 1"**
- **Cause**: Workflow crashing immediately due to missing credentials.
- **Fix**: Run `docker exec n8n n8n import:credentials --input=/home/node/workflows/credentials.json`

**n8n Webhook 404**
- **Cause**: Workflow not active or webhook not registered in DB.
- **Fix**: Ensure workflow is active in n8n UI. If CLI activation fails, use manual DB update (see internal docs).

### Logs

```bash
# n8n logs
docker logs n8n -f

# Backend logs
docker logs backend -f

# All services
docker compose logs -f
```

### Database Access

```bash
docker exec -it n8n_postgres psql -U n8n -d content_pipeline
```

## Development

### Rebuild after changes

```bash
docker compose build --no-cache
docker compose build --no-cache
docker compose up -d
```

### Hot Reloading

- **Frontend**: Local `frontend/src` is mounted to `/app/src`. Changes reflect immediately.
- **Backend**: Local `backend` is mounted to `/app`. `uvicorn` runs with `--reload` to auto-restart on changes.

### Re-import workflow

```bash
docker exec n8n n8n import:workflow --input=/home/node/workflows/COMPLETE_PIPELINE.json
```

### Run database migrations

```bash
docker exec -it n8n_postgres psql -U n8n -d content_pipeline -f /docker-entrypoint-initdb.d/migrations/001_add_scrape_tables.sql
```

## License

Private/Internal Use
