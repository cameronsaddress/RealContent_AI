# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal: Automated AI Video Content Pipeline

This n8n instance automates end-to-end viral video content creation and distribution:

### Pipeline Stages

1. **Content Scraping** - Apify to pull trending/viral ideas from Reddit, TikTok, YouTube, X (Twitter), etc.
2. **Script Generation** - OpenRouter (Grok 4.1) to create unique scripts from scraped ideas
3. **Voiceover Creation** - ElevenLabs for AI voice cloning or custom recordings
4. **AI Avatar Videos** - HeyGen (or similar) for lip-synced avatar videos over background content
5. **Captions/Overlays** - Whisper AI for auto-generated subtitles and text overlays
6. **Editing & Management** - PostgreSQL + React UI for asset management, FFmpeg for video processing
7. **Multi-Platform Publishing** - Blotato to auto-post to X, YouTube, TikTok, Instagram, LinkedIn, Threads, Facebook, Pinterest

### Required Integrations

| Service | Purpose |
|---------|---------|
| Apify | Web scraping (Reddit, TikTok, YouTube, X) |
| OpenRouter | LLM gateway - using Grok 4.1 for script gen, FFmpeg commands |
| OpenAI | Whisper API for transcription only |
| ElevenLabs | AI voiceover/voice cloning |
| HeyGen | AI avatar video generation |
| FFmpeg | Video processing/editing (self-hosted in container) |
| PostgreSQL | Database for content pipeline management |
| FastAPI | Backend API for content pipeline |
| React | Frontend dashboard for content management |
| Blotato | Multi-platform social media posting |

### OpenRouter Configuration

```
Base URL: https://openrouter.ai/api/v1
Model: x-ai/grok-4-1106
```

**HTTP Request Node for LLM calls:**
```json
{
  "method": "POST",
  "url": "https://openrouter.ai/api/v1/chat/completions",
  "headers": {
    "Authorization": "Bearer {{$credentials.openRouter.apiKey}}",
    "HTTP-Referer": "http://100.83.153.43:5678",
    "X-Title": "n8n-video-pipeline"
  },
  "body": {
    "model": "x-ai/grok-4-1106",
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."}
    ]
  }
}
```

## n8n Setup

n8n is running via Docker on DGX Spark. Access the web UI at **http://100.83.153.43:5678**

### Docker Commands
```bash
# Start n8n
docker compose up -d

# Stop n8n
docker compose down

# View logs
docker compose logs -f n8n

# Restart n8n
docker compose restart

# Update to latest version
docker compose pull && docker compose up -d
```

## Workflow Management

### Workflow File Format
n8n workflows are JSON files with this structure:
```json
{
  "name": "Workflow Name",
  "nodes": [...],
  "connections": {...},
  "settings": {...}
}
```

### Import/Export Workflows
- **Export from UI**: Workflow menu → Download
- **Import from UI**: Workflows → Import from File
- **CLI export**: `docker exec n8n n8n export:workflow --all --output=/home/node/workflows/`
- **CLI import**: `docker exec n8n n8n import:workflow --input=/home/node/workflows/workflow.json`

### Working with Workflow JSON

**Node structure:**
```json
{
  "id": "uuid",
  "name": "Node Display Name",
  "type": "n8n-nodes-base.nodeName",
  "position": [x, y],
  "parameters": {...}
}
```

**Connection structure:**
```json
{
  "Source Node Name": {
    "main": [[{"node": "Target Node Name", "type": "main", "index": 0}]]
  }
}
```

### Common Node Types
- `n8n-nodes-base.manualTrigger` - Manual execution trigger
- `n8n-nodes-base.scheduleTrigger` - Cron-based scheduling
- `n8n-nodes-base.webhook` - HTTP webhook trigger
- `n8n-nodes-base.httpRequest` - Make HTTP requests
- `n8n-nodes-base.code` - Custom JavaScript/Python code
- `n8n-nodes-base.if` - Conditional branching
- `n8n-nodes-base.merge` - Combine data from multiple branches
- `n8n-nodes-base.set` - Set/modify data fields

### Expressions
n8n uses expressions with `{{ }}` syntax:
- `{{ $json.fieldName }}` - Access current item's field
- `{{ $('Node Name').item.json.field }}` - Access another node's output
- `{{ $input.all() }}` - Get all input items
- `{{ $now }}` - Current datetime
- `{{ $env.VAR_NAME }}` - Environment variable

## Directory Structure
```
/home/canderson/n8n/
├── docker-compose.yml    # Docker configuration (n8n, postgres, backend, frontend)
├── Dockerfile            # Custom n8n image with FFmpeg
├── workflows/            # Store exported workflow JSON files
├── assets/               # Fonts, logos, generated media
│   ├── fonts/
│   └── logos/
├── backend/              # FastAPI backend
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py           # FastAPI app with all routes
│   ├── models.py         # SQLAlchemy models
│   ├── schemas.py        # Pydantic schemas
│   └── init.sql          # PostgreSQL schema
├── frontend/             # React frontend
│   ├── Dockerfile
│   ├── package.json
│   └── src/
│       ├── App.js
│       ├── api.js        # API client
│       └── pages/        # Dashboard, ContentIdeas, Scripts, Assets, Published
├── CLAUDE.md
├── MASTERPLAN.md
└── readme.md
```

## Environment Variables

Set in docker-compose.yml under `environment:`:
- `N8N_HOST` - Hostname for n8n
- `WEBHOOK_URL` - Base URL for webhooks
- `GENERIC_TIMEZONE` - Timezone setting
- `N8N_ENCRYPTION_KEY` - For credential encryption (add for production)

## API Access

### n8n REST API
Available at `http://100.83.153.43:5678/api/v1/`
- Enable API access in Settings → API
- Generate API key for authentication

### Content Pipeline API (FastAPI)
Available at `http://100.83.153.43:8000`

**Endpoints:**
- `GET /api/content-ideas` - List content ideas
- `POST /api/content-ideas` - Create content idea
- `PATCH /api/content-ideas/{id}` - Update content idea
- `POST /api/content-ideas/bulk-approve` - Bulk approve ideas
- `GET /api/scripts` - List scripts
- `PATCH /api/scripts/{id}` - Update script
- `GET /api/assets` - List assets
- `GET /api/published` - List published content
- `GET /api/analytics` - Get analytics data
- `GET /api/pipeline/overview` - Full pipeline status
- `GET /api/pipeline/stats` - Pipeline statistics

### Frontend Dashboard
Available at `http://100.83.153.43:3000`

**Pages:**
- `/` - Dashboard with stats and recent activity
- `/ideas` - Content ideas management (approve/reject, bulk actions)
- `/scripts` - Script viewing and editing
- `/assets` - Asset tracking (voiceovers, videos, captions)
- `/published` - Published content with platform links

## Database Schema

PostgreSQL database with the following tables:
- `content_ideas` - Scraped content with pillar, viral score, status
- `scripts` - Generated scripts with platform-specific captions
- `assets` - File paths for voiceovers, videos, captions
- `published` - Platform URLs and IDs for published content
- `analytics` - Views, likes, comments, shares per platform
- `pipeline_runs` - Workflow execution tracking

**View:** `pipeline_overview` - Joins all tables for full status view

## Starting the Full Stack

```bash
# Build and start all services
docker compose up -d --build

# Check service status
docker compose ps

# View logs
docker compose logs -f backend
docker compose logs -f frontend

# Access services
# n8n: http://100.83.153.43:5678
# Backend API: http://100.83.153.43:8000
# Frontend: http://100.83.153.43:3000
```
