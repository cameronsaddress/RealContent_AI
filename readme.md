# AI Video Content Pipeline

**Fully Autonomous Video Production System for Real Estate Content Creators**

An end-to-end automated pipeline that transforms trending content ideas into polished, published videos across 8 social media platforms. Built with n8n workflow automation, powered by AI, and managed through a custom React dashboard.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Technology Stack](#technology-stack)
4. [Quick Start](#quick-start)
5. [Service URLs](#service-urls)
6. [Pipeline Stages](#pipeline-stages)
7. [Content Strategy](#content-strategy)
8. [Database Schema](#database-schema)
9. [API Reference](#api-reference)
10. [n8n Workflows](#n8n-workflows)
11. [Environment Variables](#environment-variables)
12. [Frontend Dashboard](#frontend-dashboard)
13. [Directory Structure](#directory-structure)
14. [Deployment](#deployment)
15. [Cost Analysis](#cost-analysis)
16. [Troubleshooting](#troubleshooting)

---

## System Overview

This pipeline automates the complete lifecycle of viral video content creation:

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           AI VIDEO CONTENT PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   DISCOVER          GENERATE           CREATE            ASSEMBLE          PUBLISH │
│                                                                                     │
│  ┌─────────┐      ┌─────────┐      ┌─────────────┐      ┌─────────┐      ┌───────┐ │
│  │ Apify   │ ──▶  │ Grok    │ ──▶  │ ElevenLabs  │ ──▶  │ FFmpeg  │ ──▶  │Blotato│ │
│  │ Scraper │      │ 4.1     │      │ + HeyGen    │      │         │      │       │ │
│  └─────────┘      └─────────┘      └─────────────┘      └─────────┘      └───────┘ │
│       │                │                  │                  │               │     │
│       ▼                ▼                  ▼                  ▼               ▼     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                         PostgreSQL Database                                  │   │
│  │   content_ideas → scripts → assets → published → analytics                  │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                             │
│                                      ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                       React Dashboard (UI Control)                           │   │
│  │   • Approve/reject content ideas                                             │   │
│  │   • Monitor pipeline status                                                  │   │
│  │   • View published content                                                   │   │
│  │   • Track analytics                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### What It Does

1. **Scrapes trending content** from TikTok, Reddit, YouTube, and X/Twitter using Apify
2. **Analyzes and scores** content for viral potential using Grok 4.1 via OpenRouter
3. **Generates unique scripts** tailored to real estate brand voice and content pillars
4. **Creates AI voiceovers** using ElevenLabs voice cloning
5. **Produces avatar videos** with HeyGen lip-sync technology
6. **Assembles final videos** using FFmpeg for compositing, overlays, and captions
7. **Publishes simultaneously** to 8 platforms via Blotato
8. **Tracks analytics** and generates performance reports

### Key Features

- **Fully Autonomous**: Once content is approved, the entire pipeline runs without intervention
- **UI-Controlled**: Approve content, trigger processing, and monitor progress from the dashboard
- **Status-Driven**: Each workflow triggers the next based on database status changes
- **Error Resilient**: Comprehensive error handling with retry logic and status tracking
- **Cost-Effective**: ~$0.75-1.40 per video, leveraging self-hosted video processing

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                 DOCKER COMPOSE                                       │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐   │
│  │     n8n         │  │    Backend      │  │    Frontend     │  │  PostgreSQL  │   │
│  │  :5678          │  │    :8000        │  │    :3000        │  │   :5433      │   │
│  │                 │  │                 │  │                 │  │              │   │
│  │  • 10 Workflows │  │  • FastAPI      │  │  • React        │  │  • 6 Tables  │   │
│  │  • Webhooks     │  │  • SQLAlchemy   │  │  • React Query  │  │  • 1 View    │   │
│  │  • FFmpeg       │  │  • Pydantic     │  │  • Dashboard    │  │  • Triggers  │   │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  └──────┬───────┘   │
│           │                    │                    │                  │           │
│           └────────────────────┴────────────────────┴──────────────────┘           │
│                                        │                                           │
│                              Docker Network: n8n_default                           │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
          ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
          │  External APIs  │  │  Asset Storage  │  │   File System   │
          │                 │  │                 │  │                 │
          │  • OpenRouter   │  │  • /assets/     │  │  • ./workflows/ │
          │  • ElevenLabs   │  │    audio/       │  │  • ./assets/    │
          │  • HeyGen       │  │    videos/      │  │  • ./backend/   │
          │  • OpenAI       │  │    avatar/      │  │  • ./frontend/  │
          │  • Apify        │  │    captions/    │  │                 │
          │  • Blotato      │  │    output/      │  │                 │
          │  • Pexels       │  │                 │  │                 │
          └─────────────────┘  └─────────────────┘  └─────────────────┘
```

### Data Flow

```
                                   USER ACTION
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              CONTENT DISCOVERY                                    │
│                                                                                  │
│   Apify Scrapers ──▶ Grok 4.1 Analysis ──▶ content_ideas (status='pending')     │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                       │
                         User approves in React Dashboard
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              SCRIPT GENERATION                                    │
│                                                                                  │
│   content_ideas ──▶ Grok 4.1 Script ──▶ scripts (status='script_ready')         │
│                     + Platform Captions                                          │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              MEDIA CREATION                                       │
│                                                                                  │
│   scripts ──▶ Background Video (Pexels/Original) ──▶ assets                     │
│            ──▶ ElevenLabs Voiceover ──▶ Whisper SRT                              │
│            ──▶ HeyGen Avatar Video                                               │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              VIDEO ASSEMBLY                                       │
│                                                                                  │
│   assets ──▶ FFmpeg: Chroma key + Overlay + Captions ──▶ final video            │
│              (status='ready_to_publish')                                         │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              PUBLISHING                                           │
│                                                                                  │
│   assets ──▶ Blotato API ──▶ 8 Platforms ──▶ published (status='published')     │
│              TikTok, Instagram, YouTube, LinkedIn, X, Facebook, Threads, Pinterest│
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              ANALYTICS                                            │
│                                                                                  │
│   published ──▶ Platform APIs ──▶ analytics ──▶ Weekly AI Report                │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

### Core Infrastructure

| Component | Technology | Purpose |
|-----------|------------|---------|
| Workflow Automation | n8n | Orchestrates all 10 pipeline workflows |
| Backend API | FastAPI + SQLAlchemy | REST API for data management |
| Frontend | React + React Query | Dashboard for pipeline control |
| Database | PostgreSQL 16 | Content and asset storage |
| Video Processing | FFmpeg | Compositing, overlays, captions |
| Containerization | Docker Compose | Service orchestration |

### AI Services

| Service | Provider | Purpose | Cost |
|---------|----------|---------|------|
| LLM | OpenRouter (Grok 4.1) | Script generation, analysis, FFmpeg commands | ~$0.01-0.03/video |
| Voice | ElevenLabs | AI voice cloning & text-to-speech | $0.05-0.15/video |
| Transcription | OpenAI Whisper | Word-level timestamps for captions | $0.003/video |
| Avatar | HeyGen | Lip-synced talking head videos | $0.50-1.00/video |

### Content Services

| Service | Purpose | Cost |
|---------|---------|------|
| Apify | Web scraping (TikTok, Reddit, YouTube, X) | $0.01-0.05/video |
| Pexels | Stock video backgrounds | Free |
| Blotato | Multi-platform social publishing | ~$0.15/video |

### Platform Support

Videos are published to:
- TikTok
- Instagram (Feed + Reels)
- YouTube Shorts
- LinkedIn
- X (Twitter)
- Facebook
- Threads
- Pinterest

---

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- DGX Spark or ARM64-compatible host (for HeyGen avatar generation)
- API keys for all external services (see [Environment Variables](#environment-variables))

### Installation

1. **Clone and navigate to the project**
   ```bash
   cd /home/canderson/n8n
   ```

2. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Build and start all services**
   ```bash
   docker compose up -d --build
   ```

4. **Verify services are running**
   ```bash
   docker compose ps
   ```

5. **Initialize the database** (if tables don't exist)
   ```bash
   docker exec -i n8n_postgres psql -U n8n -d content_pipeline < backend/init.sql
   ```

6. **Access the services**
   - n8n: http://100.83.153.43:5678
   - Frontend: http://100.83.153.43:3000
   - Backend API: http://100.83.153.43:8000

### First Run

1. Open n8n at http://100.83.153.43:5678
2. Create credentials for each external service (see MASTERPLAN.md Appendix D)
3. Import workflows from `./workflows/` directory
4. Activate all workflows
5. Open the React dashboard at http://100.83.153.43:3000
6. Wait for the daily scrape or trigger manually

---

## Service URLs

| Service | URL | Purpose |
|---------|-----|---------|
| **n8n** | http://100.83.153.43:5678 | Workflow automation interface |
| **Frontend** | http://100.83.153.43:3000 | React dashboard for pipeline control |
| **Backend API** | http://100.83.153.43:8000 | FastAPI REST endpoints |
| **PostgreSQL** | localhost:5433 | Database (internal access) |
| **API Docs** | http://100.83.153.43:8000/docs | Swagger UI documentation |

**Host**: DGX Spark (accessible via Tailscale VPN at 100.83.153.43)

---

## Pipeline Stages

### Stage 1: Content Discovery (Workflow 1: SCRAPE)

**Trigger**: Daily at 6:00 AM UTC

**Process**:
1. Parallel Apify scrapers pull trending content from TikTok, Reddit, YouTube, X
2. Results are merged and sent to Grok 4.1 for analysis
3. Each piece is scored for viral potential (1-10) and categorized by content pillar
4. Items with score >= 7 are saved to `content_ideas` table with status='pending'

**Output**: Content ideas awaiting human approval

### Stage 2: Script Generation (Workflows 2-3: AUTO TRIGGER + SCRIPT GEN)

**Trigger**:
- Every 15 minutes (polls for approved content)
- Webhook trigger from UI

**Process**:
1. Picks next approved content idea
2. Grok 4.1 generates viral script with hook, body, and CTA
3. Separate LLM call generates platform-specific captions
4. Script saved to `scripts` table with all captions

**Output**: Ready-to-produce script with captions for 8 platforms

### Stage 3: Video Acquisition (Workflow 4: GET VIDEO)

**Trigger**: Automatic (chained from script generation)

**Process**:
1. Determines video source based on original platform
2. TikTok/YouTube: Downloads original via Apify
3. Reddit/X: Searches Pexels for relevant stock video
4. Saves video to `/assets/videos/`
5. Creates `assets` record

**Output**: Background video ready for compositing

### Stage 4: Voiceover Creation (Workflow 5: CREATE VOICE)

**Trigger**: Automatic (chained from video acquisition)

**Process**:
1. Fetches full script from database
2. Sends to ElevenLabs for voice synthesis
3. Saves audio to `/assets/audio/`
4. Sends audio to OpenAI Whisper for word-level transcription
5. Generates SRT file with precise timestamps
6. Saves SRT to `/assets/captions/`

**Output**: Voiceover audio + SRT subtitles with word timing

### Stage 5: Avatar Generation (Workflow 6: CREATE AVATAR)

**Trigger**: Automatic (chained from voiceover creation)

**Process**:
1. Uploads audio to HeyGen
2. Triggers avatar video generation with green screen background
3. Polls HeyGen API until complete (typically 1-3 minutes)
4. Downloads avatar video to `/assets/avatar/`

**Output**: Lip-synced avatar video with green screen

### Stage 6: Video Assembly (Workflow 7: COMBINE VIDS)

**Trigger**: Automatic (chained from avatar generation)

**Process**:
1. Grok 4.1 generates optimal FFmpeg command based on inputs
2. FFmpeg executes:
   - Scales/crops background to 1080x1920 portrait
   - Loops background if shorter than audio
   - Chroma keys avatar (removes green screen)
   - Overlays avatar at bottom center
3. Combined video saved to `/assets/output/`

**Output**: Assembled video with avatar over background

### Stage 7: Caption Burn-in (Workflow 8: CAPTION)

**Trigger**: Automatic (chained from video assembly)

**Process**:
1. Converts SRT to ASS format (styled subtitles)
2. Applies custom styling: Montserrat Bold, 72px, white with black outline
3. FFmpeg burns subtitles into video
4. Final video saved to `/assets/output/`

**Output**: Final video with styled captions, ready to publish

### Stage 8: Publishing (Workflow 9: PUBLISH)

**Trigger**:
- Hourly schedule (polls for ready content)
- Webhook trigger from UI

**Process**:
1. Reads final video and platform-specific captions
2. Uploads to Blotato with per-platform settings
3. Blotato distributes to all 8 platforms
4. Records platform URLs and IDs in `published` table

**Output**: Live content across 8 social platforms

### Stage 9: Analytics Collection (Workflow 10: ANALYTICS)

**Trigger**: Weekly on Sunday at midnight

**Process**:
1. Fetches all published content from the past week
2. Collects analytics from each platform (views, likes, comments, shares)
3. Calculates engagement rates
4. Grok 4.1 generates performance insights and recommendations
5. Stores in `analytics` table

**Output**: Performance metrics and AI-generated weekly report

---

## Content Strategy

### Four Content Pillars

All content is categorized into four strategic pillars:

| Pillar | Focus | Example Topics |
|--------|-------|----------------|
| **Market Intelligence** | Data, trends, insights | Housing prices, interest rates, market predictions |
| **Educational Tips** | Practical guidance | Home buying steps, negotiation tips, financing explained |
| **Lifestyle & Local** | Community focus | Neighborhood guides, local events, quality of life |
| **Brand Humanization** | Personal connection | Behind-the-scenes, faith, family, client testimonials |

### Viral Hook Patterns

Scripts are generated using proven hook patterns:

- "Here's what no one tells you about buying a home..."
- "POV: You just found out [surprising fact]"
- "3 things I wish I knew before [action]"
- "The truth about [topic] in [city]"
- "Stop doing this when [buying/selling]"
- "The biggest mistake [buyers/sellers] make is..."

### Scraping Sources by Pillar

| Pillar | Primary Sources | Search Terms |
|--------|-----------------|--------------|
| Market Intelligence | Reddit, YouTube, X | housing market, real estate trends, home prices, mortgage rates |
| Educational Tips | TikTok, YouTube, Reddit | home buying tips, realtor advice, first time buyer |
| Lifestyle & Local | TikTok, Instagram | [city] lifestyle, best neighborhoods, things to do |
| Brand Humanization | TikTok, Instagram | day in the life realtor, realtor vlog, client stories |

---

## Database Schema

### Tables

```sql
-- Content ideas scraped from social media
content_ideas (
    id SERIAL PRIMARY KEY,
    source_url TEXT,
    source_platform TEXT,        -- tiktok, reddit, youtube, twitter
    original_text TEXT,
    pillar TEXT,                 -- market_intelligence, educational_tips, etc.
    viral_score INTEGER,         -- 1-10 score from LLM analysis
    suggested_hook TEXT,
    status TEXT,                 -- pending, approved, rejected, script_generating, etc.
    created_at TIMESTAMP,
    updated_at TIMESTAMP
)

-- Generated scripts with platform-specific captions
scripts (
    id SERIAL PRIMARY KEY,
    content_idea_id INTEGER REFERENCES content_ideas(id),
    hook TEXT,
    body TEXT,
    cta TEXT,
    full_script TEXT,
    duration_estimate INTEGER,
    tiktok_caption TEXT,
    ig_caption TEXT,
    yt_title TEXT,
    yt_description TEXT,
    linkedin_text TEXT,
    x_text TEXT,
    facebook_text TEXT,
    threads_text TEXT,
    status TEXT,
    created_at TIMESTAMP
)

-- Asset tracking (files, paths, status)
assets (
    id SERIAL PRIMARY KEY,
    script_id INTEGER REFERENCES scripts(id),
    background_video_path TEXT,
    voiceover_path TEXT,
    voiceover_duration FLOAT,
    srt_path TEXT,
    avatar_video_path TEXT,
    combined_video_path TEXT,
    ass_path TEXT,
    final_video_path TEXT,
    status TEXT,                 -- pending, voice_ready, avatar_ready, etc.
    error_message TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
)

-- Published content with platform URLs
published (
    id SERIAL PRIMARY KEY,
    asset_id INTEGER REFERENCES assets(id),
    tiktok_url TEXT,
    tiktok_id TEXT,
    ig_url TEXT,
    ig_id TEXT,
    yt_url TEXT,
    yt_id TEXT,
    linkedin_url TEXT,
    linkedin_id TEXT,
    x_url TEXT,
    x_id TEXT,
    facebook_url TEXT,
    facebook_id TEXT,
    threads_url TEXT,
    threads_id TEXT,
    pinterest_url TEXT,
    pinterest_id TEXT,
    published_at TIMESTAMP
)

-- Performance analytics per platform
analytics (
    id SERIAL PRIMARY KEY,
    published_id INTEGER REFERENCES published(id),
    platform TEXT,
    views INTEGER,
    likes INTEGER,
    comments INTEGER,
    shares INTEGER,
    engagement_rate FLOAT,
    fetched_at TIMESTAMP
)

-- Workflow execution tracking
pipeline_runs (
    id SERIAL PRIMARY KEY,
    workflow_name TEXT,
    status TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    run_metadata JSONB
)
```

### Status Flow

```
content_ideas: pending → approved → script_generating → script_ready → published | error
scripts:       pending → script_ready
assets:        pending → voice_ready → avatar_generating → avatar_ready →
               assembling → captioning → ready_to_publish → publishing → published | error
```

---

## API Reference

### Base URL

```
http://100.83.153.43:8000/api
```

### Content Ideas

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/content-ideas` | List all content ideas |
| GET | `/content-ideas?status=pending` | Filter by status |
| GET | `/content-ideas/{id}` | Get single idea |
| POST | `/content-ideas` | Create new idea |
| PATCH | `/content-ideas/{id}` | Update idea (status, etc.) |
| POST | `/content-ideas/bulk-approve` | Approve multiple ideas |

### Scripts

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/scripts` | List all scripts |
| GET | `/scripts?content_idea_id={id}` | Get scripts for idea |
| GET | `/scripts/{id}` | Get single script |
| POST | `/scripts` | Create new script |
| PATCH | `/scripts/{id}` | Update script |

### Assets

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/assets` | List all assets |
| GET | `/assets?status=ready_to_publish` | Filter by status |
| GET | `/assets/{id}` | Get single asset |
| POST | `/assets` | Create asset record |
| PATCH | `/assets/{id}` | Update asset (paths, status) |

### Published

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/published` | List published content |
| GET | `/published/{id}` | Get single published item |
| POST | `/published` | Create published record |

### Analytics

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/analytics` | List all analytics |
| GET | `/analytics?published_id={id}` | Get analytics for item |
| POST | `/analytics` | Create analytics record |

### Pipeline

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/pipeline/stats` | Dashboard statistics |
| GET | `/pipeline/overview` | Full pipeline status view |

---

## n8n Workflows

### Workflow Summary

| # | Name | Trigger | Purpose |
|---|------|---------|---------|
| 1 | SCRAPE | Daily 6am | Discover trending content |
| 2 | AUTO TRIGGER | 15min poll + webhook | Initiate processing |
| 3 | SCRIPT GEN | Chained from WF2 | Generate scripts |
| 4 | GET VIDEO | Chained from WF3 | Acquire background video |
| 5 | CREATE VOICE | Chained from WF4 | Generate voiceover + SRT |
| 6 | CREATE AVATAR | Chained from WF5 | Generate avatar video |
| 7 | COMBINE VIDS | Chained from WF6 | FFmpeg compositing |
| 8 | CAPTION | Chained from WF7 | Burn in subtitles |
| 9 | PUBLISH | Hourly + webhook | Multi-platform publishing |
| 10 | ANALYTICS | Weekly Sunday | Collect performance data |

### Webhook Endpoints

| Path | Method | Purpose |
|------|--------|---------|
| `/webhook/trigger-scrape` | POST | Manual content scraping |
| `/webhook/trigger-pipeline` | POST | Start processing `{ content_idea_id: number }` |
| `/webhook/publish-video` | POST | Publish video `{ asset_id?: number }` |
| `/webhook/retry-asset` | POST | Retry failed `{ asset_id, from_stage }` |

### Importing Workflows

1. Open n8n at http://100.83.153.43:5678
2. Go to Workflows → Import
3. Import each file from `./workflows/` directory
4. Configure credentials in each workflow
5. Activate all workflows

---

## Environment Variables

### Required Variables

```bash
# API Keys for Content Pipeline

# ElevenLabs - Voice cloning & text-to-speech
ELEVENLABS_API_KEY=sk_...
ELEVENLABS_VOICE_ID=your_cloned_voice_id

# OpenRouter - LLM gateway (Grok 4.1)
OPENROUTER_API_KEY=sk-or-v1-...

# OpenAI - Whisper API for transcription
OPENAI_API_KEY=sk-proj-...

# HeyGen - AI avatar video generation
HEYGEN_API_KEY=...
HEYGEN_AVATAR_ID=your_avatar_id

# Apify - Web scraping
APIFY_API_KEY=...

# Pexels - Stock video
PEXELS_API_KEY=...

# Blotato - Multi-platform publishing
BLOTATO_API_KEY=...

# Database
POSTGRES_USER=n8n
POSTGRES_PASSWORD=n8n_password
POSTGRES_DB=content_pipeline
DATABASE_URL=postgresql://n8n:n8n_password@postgres:5432/content_pipeline

# n8n Configuration
N8N_HOST=100.83.153.43
N8N_PORT=5678
```

### Variable Usage by Workflow

| Workflow | Required Variables |
|----------|-------------------|
| WF1: Scrape | `OPENROUTER_API_KEY`, `APIFY_API_KEY` |
| WF3: Script Gen | `OPENROUTER_API_KEY` |
| WF4: Get Video | `APIFY_API_KEY`, `PEXELS_API_KEY`, `OPENROUTER_API_KEY` |
| WF5: Create Voice | `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`, `OPENAI_API_KEY` |
| WF6: Create Avatar | `HEYGEN_API_KEY`, `HEYGEN_AVATAR_ID` |
| WF7: Combine Vids | `OPENROUTER_API_KEY` |
| WF9: Publish | `BLOTATO_API_KEY` |
| WF10: Analytics | `BLOTATO_API_KEY` |

---

## Frontend Dashboard

### Pages

| Route | Purpose |
|-------|---------|
| `/` | Dashboard with stats and recent activity |
| `/ideas` | Content ideas management (approve/reject, bulk actions) |
| `/scripts` | Script viewing and editing |
| `/assets` | Asset tracking with status indicators |
| `/published` | Published content with platform links |

### Features

- **Dashboard**: Pipeline statistics, recent items, status overview
- **Content Ideas**: Approve/reject scraped content, bulk approve, trigger processing
- **Scripts**: View generated scripts, edit if needed, see platform captions
- **Assets**: Track progress through pipeline, view file paths, retry failed items
- **Published**: See all platform URLs, view analytics

### Triggering Workflows from UI

```javascript
// Trigger processing for specific idea
await fetch('http://100.83.153.43:5678/webhook/trigger-pipeline', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ content_idea_id: 123 })
});

// Publish specific asset
await fetch('http://100.83.153.43:5678/webhook/publish-video', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ asset_id: 456 })
});
```

---

## Directory Structure

```
/home/canderson/n8n/
├── docker-compose.yml          # Docker service configuration
├── Dockerfile                  # Custom n8n image with FFmpeg (ARM64)
├── .env                        # API keys and configuration
├── .gitignore                  # Git ignore rules
├── CLAUDE.md                   # Claude Code guidance document
├── MASTERPLAN.md               # Detailed workflow specifications
├── readme.md                   # This file
│
├── backend/                    # FastAPI backend
│   ├── Dockerfile              # Python container config
│   ├── requirements.txt        # Python dependencies
│   ├── main.py                 # FastAPI app with all routes
│   ├── models.py               # SQLAlchemy ORM models
│   ├── schemas.py              # Pydantic request/response schemas
│   └── init.sql                # PostgreSQL database schema
│
├── frontend/                   # React frontend
│   ├── Dockerfile              # Node.js container config
│   ├── package.json            # Node dependencies
│   └── src/
│       ├── App.js              # Main app with routing
│       ├── api.js              # API client functions
│       ├── index.js            # React entry point
│       └── pages/
│           ├── Dashboard.js    # Stats and overview
│           ├── ContentIdeas.js # Idea management
│           ├── Scripts.js      # Script viewing
│           ├── Assets.js       # Asset tracking
│           └── Published.js    # Published content
│
├── workflows/                  # n8n workflow JSON exports
│   ├── 01-scrape.json
│   ├── 02-auto-trigger.json
│   ├── 03-script-gen.json
│   ├── 04-get-video.json
│   ├── 05-create-voice.json
│   ├── 06-create-avatar.json
│   ├── 07-combine-vids.json
│   ├── 08-caption.json
│   ├── 09-publish.json
│   └── 10-analytics.json
│
└── assets/                     # Generated media (Docker volume)
    ├── fonts/
    │   └── Montserrat-Bold.ttf
    ├── logos/
    │   └── logo.png
    ├── audio/                  # ElevenLabs voiceovers
    ├── videos/                 # Background videos
    ├── avatar/                 # HeyGen avatar videos
    ├── captions/               # SRT and ASS files
    └── output/                 # Final videos
```

---

## Deployment

### Docker Commands

```bash
# Build and start all services
docker compose up -d --build

# Stop all services
docker compose down

# View logs
docker compose logs -f n8n
docker compose logs -f backend
docker compose logs -f frontend

# Check service status
docker compose ps

# Restart specific service
docker compose restart backend

# Rebuild specific service
docker compose up -d --build backend

# Access container shell
docker exec -it n8n /bin/sh
docker exec -it backend /bin/bash

# Database access
docker exec -it n8n_postgres psql -U n8n -d content_pipeline
```

### Health Checks

```bash
# Check n8n
curl http://100.83.153.43:5678/healthz

# Check backend API
curl http://100.83.153.43:8000/api/pipeline/stats

# Check frontend
curl http://100.83.153.43:3000

# Check PostgreSQL
docker exec n8n_postgres pg_isready -U n8n
```

### Backup

```bash
# Backup database
docker exec n8n_postgres pg_dump -U n8n content_pipeline > backup.sql

# Backup n8n workflows
docker exec n8n n8n export:workflow --all --output=/home/node/workflows/

# Restore database
docker exec -i n8n_postgres psql -U n8n -d content_pipeline < backup.sql
```

---

## Cost Analysis

### Per-Video Costs

| Service | Cost | Notes |
|---------|------|-------|
| Apify | $0.01-0.05 | Scraping |
| OpenRouter (Grok 4.1) | $0.01-0.03 | 3-5 LLM calls |
| ElevenLabs | $0.05-0.15 | 45 sec voice |
| OpenAI Whisper | $0.003 | 45 sec transcription |
| HeyGen | $0.50-1.00 | Avatar video |
| Pexels | Free | Stock video |
| FFmpeg | Free | Self-hosted |
| Blotato | ~$0.15 | Per post |

**Total per video: ~$0.75-1.40**

### Monthly Projections

| Volume | Daily Cost | Monthly Cost |
|--------|------------|--------------|
| 1 video/day | $0.75-1.40 | $23-42 |
| 2 videos/day | $1.50-2.80 | $45-85 |
| 3 videos/day | $2.25-4.20 | $68-126 |

### Platform Monthly Fees

| Service | Plan | Cost |
|---------|------|------|
| ElevenLabs | Starter | $5/mo |
| HeyGen | Creator | $24/mo |
| Blotato | Starter | $29/mo |
| Apify | Free tier | $5/mo credits |

**Base monthly overhead: ~$63/mo**

---

## Troubleshooting

### Common Issues

#### FFmpeg Errors

**Error**: `ffmpeg: command not found`
- **Solution**: Rebuild n8n container: `docker compose up -d --build n8n`

**Error**: `No such file or directory`
- **Solution**: Ensure paths use `/home/node/assets/` (container path)

**Error**: `colorkey filter failed`
- **Solution**: Verify HeyGen video uses exact green #00FF00 background

#### API Errors

**Error**: `401 Unauthorized` on OpenRouter
- **Solution**: Check `OPENROUTER_API_KEY` format: `sk-or-v1-...`

**Error**: `402 Payment Required` on ElevenLabs
- **Solution**: Check character quota, upgrade plan if needed

**Error**: `HeyGen video stuck in processing`
- **Solution**: Videos may fail silently; check HeyGen dashboard

#### Database Errors

**Error**: `connection refused` to backend
- **Solution**: Use `http://backend:8000` (Docker network), not localhost

**Error**: `relation does not exist`
- **Solution**: `docker exec -i n8n_postgres psql -U n8n -d content_pipeline < backend/init.sql`

#### Workflow Errors

**Error**: Workflow stops without error
- **Solution**: Check IF nodes have both TRUE and FALSE branches connected

**Error**: Data not passing between workflows
- **Solution**: Ensure Execute Workflow nodes have "Wait for completion" enabled

### Logs

```bash
# n8n logs
docker compose logs -f n8n

# Backend logs
docker compose logs -f backend

# All logs
docker compose logs -f

# Specific time range
docker compose logs --since 1h
```

### Reset Pipeline

```bash
# Clear all data and restart
docker compose down -v
docker compose up -d --build

# Reinitialize database
docker exec -i n8n_postgres psql -U n8n -d content_pipeline < backend/init.sql
```

---

## Resources

- [n8n Documentation](https://docs.n8n.io/)
- [OpenRouter API](https://openrouter.ai/docs)
- [ElevenLabs API](https://elevenlabs.io/docs/api-reference)
- [HeyGen API](https://docs.heygen.com/)
- [OpenAI Whisper API](https://platform.openai.com/docs/guides/speech-to-text)
- [Blotato API](https://docs.blotato.com/)
- [Apify Store](https://apify.com/store)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
- [Pexels API](https://www.pexels.com/api/)

---

## License

Private repository. All rights reserved.

---

*Last updated: January 2026*
