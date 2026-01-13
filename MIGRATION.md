# n8n to Python Migration Plan

## Overview

This document provides a complete migration plan from n8n workflow orchestration to a pure Python implementation. Every n8n node is mapped to its Python equivalent with code snippets.

**Current State:**
- 107 n8n nodes total
- 38 Code nodes (JavaScript)
- 25 HTTP Request nodes
- 12 If nodes (conditionals)
- 10 Sticky Notes (comments only)
- 22 Other nodes (webhooks, triggers, file ops, etc.)

**Target State:**
- Single FastAPI application with Celery task queue
- All services in Python
- Same PostgreSQL database
- Same React frontend (no changes needed)

---

## Table of Contents

1. [Architecture Changes](#architecture-changes)
2. [New File Structure](#new-file-structure)
3. [Dependency Requirements](#dependency-requirements)
4. [Service Implementations](#service-implementations)
5. [Node-by-Node Migration](#node-by-node-migration)
6. [Celery Task Definitions](#celery-task-definitions)
7. [API Endpoint Changes](#api-endpoint-changes)
8. [Docker Compose Changes](#docker-compose-changes)
9. [Migration Checklist](#migration-checklist)

---

## Architecture Changes

### Before (n8n)
```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│   Frontend  │────▶│   Backend   │────▶│   PostgreSQL    │
│   (React)   │     │  (FastAPI)  │     │                 │
└─────────────┘     └─────────────┘     └─────────────────┘
                           │
                           ▼
                    ┌─────────────┐     ┌─────────────────┐
                    │    n8n      │────▶│ video-processor │
                    │  (Node.js)  │     │    (Python)     │
                    └─────────────┘     └─────────────────┘
```

### After (Python)
```
┌─────────────┐     ┌─────────────────────────────────────┐
│   Frontend  │────▶│            Backend (FastAPI)         │
│   (React)   │     │  ┌─────────────┐  ┌──────────────┐  │
└─────────────┘     │  │   Routes    │  │   Services   │  │
                    │  └─────────────┘  └──────────────┘  │
                    │         │                │          │
                    │         ▼                ▼          │
                    │  ┌─────────────────────────────┐   │
                    │  │      Celery Workers         │   │
                    │  │  (Pipeline, Scrape, etc.)   │   │
                    │  └─────────────────────────────┘   │
                    └─────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
             ┌───────────┐  ┌───────────┐  ┌───────────┐
             │ PostgreSQL│  │   Redis   │  │  Assets   │
             └───────────┘  └───────────┘  └───────────┘
```

---

## New File Structure

```
backend/
├── main.py                      # FastAPI app (existing, extended)
├── models.py                    # SQLAlchemy models (existing)
├── schemas.py                   # Pydantic schemas (existing)
├── config.py                    # NEW: Configuration management
├── celery_app.py                # NEW: Celery configuration
│
├── services/                    # NEW: Service layer
│   ├── __init__.py
│   ├── base.py                  # Base service class with HTTP client
│   ├── scraper.py               # Apify scrapers + LLM analysis
│   ├── script_generator.py      # Script generation with Grok
│   ├── voice.py                 # ElevenLabs TTS
│   ├── avatar.py                # HeyGen avatar generation
│   ├── video.py                 # FFmpeg video processing
│   ├── captions.py              # Whisper transcription + SRT
│   ├── publisher.py             # Blotato publishing
│   └── storage.py               # Dropbox upload
│
├── tasks/                       # NEW: Celery tasks
│   ├── __init__.py
│   ├── pipeline.py              # Main pipeline orchestration
│   ├── scrape.py                # Trend scraping task
│   └── scheduled.py             # Scheduled task definitions
│
├── utils/                       # NEW: Utilities
│   ├── __init__.py
│   ├── paths.py                 # Asset path management
│   ├── retry.py                 # Retry decorators
│   └── logging.py               # Structured logging
│
├── Dockerfile                   # Updated to include Celery
└── requirements.txt             # Updated dependencies
```

---

## Dependency Requirements

Add to `backend/requirements.txt`:

```txt
# Existing
fastapi==0.109.0
uvicorn[standard]==0.27.0
sqlalchemy==2.0.25
psycopg2-binary==2.9.9
python-multipart==0.0.6
aiofiles==23.2.1

# NEW: Async HTTP client
httpx==0.26.0

# NEW: Task queue
celery[redis]==5.3.4
redis==5.0.1

# NEW: Scheduling
celery-beat==2.5.0

# NEW: Retry logic
tenacity==8.2.3

# NEW: Better logging
structlog==24.1.0

# NEW: FFmpeg Python bindings (optional, can use subprocess)
ffmpeg-python==0.2.0

# NEW: File type detection
python-magic==0.4.27

# NEW: Environment management
python-dotenv==1.0.0
pydantic-settings==2.1.0
```

---

## Service Implementations

### Base Service Class

```python
# backend/services/base.py
import httpx
from typing import Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

logger = structlog.get_logger()

class BaseService:
    """Base class for all services with shared HTTP client and retry logic."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=60.0)

    async def close(self):
        await self.client.aclose()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict] = None,
        json: Optional[Dict] = None,
        data: Optional[Any] = None,
        files: Optional[Dict] = None
    ) -> httpx.Response:
        """Make HTTP request with automatic retry."""
        logger.info("http_request", method=method, url=url)
        response = await self.client.request(
            method=method,
            url=url,
            headers=headers,
            json=json,
            data=data,
            files=files
        )
        response.raise_for_status()
        return response
```

---

## Node-by-Node Migration

### SECTION 1: SCRAPE FLOW (19 nodes)

#### 1.1 Daily 6am Scrape (scheduleTrigger)
**n8n Node ID:** `scrape-schedule`
**Type:** `n8n-nodes-base.scheduleTrigger`
**Purpose:** Triggers scraping at 6am UTC daily

**Python Implementation:**
```python
# backend/tasks/scheduled.py
from celery import Celery
from celery.schedules import crontab

app = Celery('tasks')

app.conf.beat_schedule = {
    'daily-scrape-6am': {
        'task': 'tasks.scrape.run_scrape',
        'schedule': crontab(hour=6, minute=0),
        'args': ({
            'niche': 'real estate',
            'platforms': ['tiktok', 'instagram'],
            'hashtags': ['realestate', 'homebuying']
        },)
    },
}
```

---

#### 1.2 Scrape Webhook (webhook)
**n8n Node ID:** `scrape-webhook`
**Type:** `n8n-nodes-base.webhook`
**Path:** `POST /webhook/scrape-trends`
**Purpose:** Manual trigger for scraping

**Python Implementation:**
```python
# backend/main.py (add to existing)
from tasks.scrape import run_scrape

@app.post("/webhook/scrape-trends")
async def trigger_scrape(request: ScrapeRequest, background_tasks: BackgroundTasks):
    """Trigger trend scraping - replaces n8n webhook."""
    task = run_scrape.delay(request.dict())
    return {"success": True, "task_id": task.id, "message": "Scrape started"}
```

---

#### 1.3 Parse Scrape Params (code)
**n8n Node ID:** `scrape-parse-params`
**Purpose:** Parse incoming niche/platform parameters

**Original JavaScript:**
```javascript
const body = $input.first().json.body || $input.first().json;
return [{
  json: {
    niche: body.niche || 'real estate',
    platforms: body.platforms || ['tiktok', 'instagram'],
    hashtags: body.hashtags || ['realestate', 'homebuying', 'realtor'],
    resultsPerPlatform: body.resultsPerPlatform || 20
  }
}];
```

**Python Implementation:**
```python
# backend/services/scraper.py
from pydantic import BaseModel
from typing import List, Optional

class ScrapeParams(BaseModel):
    niche: str = "real estate"
    platforms: List[str] = ["tiktok", "instagram"]
    hashtags: List[str] = ["realestate", "homebuying", "realtor"]
    results_per_platform: int = 20

class ScraperService(BaseService):
    def parse_params(self, raw_input: dict) -> ScrapeParams:
        """Parse and validate scrape parameters."""
        return ScrapeParams(**raw_input)
```

---

#### 1.4 TikTok? / Instagram? / YouTube? (if nodes)
**n8n Node IDs:** `check-tiktok`, `check-instagram`, `check-youtube`
**Purpose:** Check if platform is in requested platforms list

**Python Implementation:**
```python
# backend/services/scraper.py
async def scrape_platforms(self, params: ScrapeParams) -> List[dict]:
    """Scrape all requested platforms in parallel."""
    tasks = []

    if 'tiktok' in params.platforms:
        tasks.append(self.scrape_tiktok(params.hashtags, params.results_per_platform))

    if 'instagram' in params.platforms:
        tasks.append(self.scrape_instagram(params.hashtags, params.results_per_platform))

    if 'youtube' in params.platforms:
        tasks.append(self.scrape_youtube(params.hashtags, params.results_per_platform))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if not isinstance(r, Exception)]
```

---

#### 1.5 Apify TikTok (httpRequest)
**n8n Node ID:** `apify-tiktok`
**URL:** `https://api.apify.com/v2/acts/clockworks~free-tiktok-scraper/run-sync-get-dataset-items`
**Method:** POST

**Python Implementation:**
```python
# backend/services/scraper.py
from config import settings

class ScraperService(BaseService):
    APIFY_TIKTOK_ACTOR = "clockworks~free-tiktok-scraper"

    async def scrape_tiktok(self, hashtags: List[str], limit: int = 20) -> List[dict]:
        """Scrape TikTok videos using Apify."""
        url = f"https://api.apify.com/v2/acts/{self.APIFY_TIKTOK_ACTOR}/run-sync-get-dataset-items"

        response = await self._request(
            method="POST",
            url=url,
            headers={"Authorization": f"Bearer {settings.APIFY_API_KEY}"},
            json={
                "hashtags": hashtags,
                "resultsPerPage": limit,
                "shouldDownloadVideos": False
            }
        )

        items = response.json()
        return [self._normalize_tiktok_item(item) for item in items]

    def _normalize_tiktok_item(self, item: dict) -> dict:
        """Normalize TikTok item to common format."""
        return {
            "platform": "tiktok",
            "url": item.get("webVideoUrl", ""),
            "title": item.get("text", ""),
            "views": item.get("playCount", 0),
            "likes": item.get("diggCount", 0),
            "shares": item.get("shareCount", 0),
            "author": item.get("authorMeta", {}).get("name", ""),
            "hashtags": [h.get("name", "") for h in item.get("hashtags", [])]
        }
```

---

#### 1.6 Apify Instagram Reels (httpRequest)
**n8n Node ID:** `apify-instagram`
**URL:** `https://api.apify.com/v2/acts/apify~instagram-hashtag-scraper/run-sync-get-dataset-items`

**Python Implementation:**
```python
# backend/services/scraper.py
class ScraperService(BaseService):
    APIFY_INSTAGRAM_ACTOR = "apify~instagram-hashtag-scraper"

    async def scrape_instagram(self, hashtags: List[str], limit: int = 20) -> List[dict]:
        """Scrape Instagram reels using Apify."""
        url = f"https://api.apify.com/v2/acts/{self.APIFY_INSTAGRAM_ACTOR}/run-sync-get-dataset-items"

        response = await self._request(
            method="POST",
            url=url,
            headers={"Authorization": f"Bearer {settings.APIFY_API_KEY}"},
            json={
                "hashtags": hashtags,
                "resultsLimit": limit,
                "mediaType": "reels"
            }
        )

        items = response.json()
        return [self._normalize_instagram_item(item) for item in items]

    def _normalize_instagram_item(self, item: dict) -> dict:
        """Normalize Instagram item to common format."""
        return {
            "platform": "instagram",
            "url": item.get("url", ""),
            "title": item.get("caption", ""),
            "views": item.get("videoViewCount", 0),
            "likes": item.get("likesCount", 0),
            "shares": 0,
            "author": item.get("ownerUsername", ""),
            "hashtags": item.get("hashtags", [])
        }
```

---

#### 1.7 Apify YouTube (httpRequest)
**n8n Node ID:** `apify-youtube`
**URL:** `https://api.apify.com/v2/acts/apify~youtube-scraper/run-sync-get-dataset-items`

**Python Implementation:**
```python
# backend/services/scraper.py
class ScraperService(BaseService):
    APIFY_YOUTUBE_ACTOR = "apify~youtube-scraper"

    async def scrape_youtube(self, hashtags: List[str], limit: int = 20) -> List[dict]:
        """Scrape YouTube shorts using Apify."""
        url = f"https://api.apify.com/v2/acts/{self.APIFY_YOUTUBE_ACTOR}/run-sync-get-dataset-items"

        # YouTube uses search queries, not hashtags
        search_query = " ".join(f"#{h}" for h in hashtags) + " shorts"

        response = await self._request(
            method="POST",
            url=url,
            headers={"Authorization": f"Bearer {settings.APIFY_API_KEY}"},
            json={
                "searchKeywords": search_query,
                "maxResults": limit,
                "type": "video"
            }
        )

        items = response.json()
        return [self._normalize_youtube_item(item) for item in items]

    def _normalize_youtube_item(self, item: dict) -> dict:
        """Normalize YouTube item to common format."""
        return {
            "platform": "youtube",
            "url": item.get("url", ""),
            "title": item.get("title", ""),
            "views": item.get("viewCount", 0),
            "likes": item.get("likes", 0),
            "shares": 0,
            "author": item.get("channelName", ""),
            "hashtags": []
        }
```

---

#### 1.8 Skip TikTok / Skip Instagram / Skip YouTube (noOp)
**n8n Node IDs:** `skip-tiktok`, `skip-instagram`, `skip-youtube`
**Purpose:** Placeholder for skipped platforms

**Python Implementation:** Not needed - handled by conditional in `scrape_platforms()`

---

#### 1.9 Merge Results (merge)
**n8n Node ID:** `scrape-merge`
**Purpose:** Combine results from all platforms

**Python Implementation:**
```python
# Already handled in scrape_platforms() with asyncio.gather()
# Results are flattened into single list
```

---

#### 1.10 Normalize Data (code)
**n8n Node ID:** `scrape-normalize`
**Purpose:** Normalize scraped items from different platforms

**Original JavaScript:**
```javascript
const items = $input.all();
const params = $('Parse Scrape Params').first().json;
// Flatten and normalize all items...
```

**Python Implementation:**
```python
# backend/services/scraper.py
def normalize_all_results(self, results: List[List[dict]]) -> List[dict]:
    """Flatten and deduplicate results from all platforms."""
    all_items = []
    seen_urls = set()

    for platform_results in results:
        for item in platform_results:
            if item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                all_items.append(item)

    # Sort by engagement (views + likes)
    all_items.sort(key=lambda x: x.get("views", 0) + x.get("likes", 0), reverse=True)

    return all_items
```

---

#### 1.11 Analyze with Grok (httpRequest)
**n8n Node ID:** `scrape-analyze-grok`
**URL:** `https://openrouter.ai/api/v1/chat/completions`
**Purpose:** Use LLM to score virality and categorize content

**Python Implementation:**
```python
# backend/services/scraper.py
class ScraperService(BaseService):
    async def analyze_trends(self, items: List[dict], niche: str) -> List[dict]:
        """Analyze trends using Grok LLM via OpenRouter."""

        prompt = f"""Analyze these {niche} social media posts for viral potential.

For each post, provide:
1. viral_score (1-10): How likely to go viral
2. pillar: One of [market_intelligence, educational_tips, lifestyle_local, brand_humanization]
3. suggested_hook: A compelling hook for recreating this content
4. why_viral: Brief explanation of why this works

Posts to analyze:
{json.dumps(items[:10], indent=2)}

Return as JSON array with original post data plus your analysis fields."""

        response = await self._request(
            method="POST",
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "x-ai/grok-4.1-fast",
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"}
            }
        )

        result = response.json()
        content = result["choices"][0]["message"]["content"]
        analyzed = json.loads(content)

        return analyzed.get("trends", analyzed) if isinstance(analyzed, dict) else analyzed
```

---

#### 1.12 Format Response (code)
**n8n Node ID:** `scrape-format-response`
**Purpose:** Parse LLM response and format for API

**Python Implementation:**
```python
# backend/services/scraper.py
def format_response(self, analyzed_items: List[dict], params: ScrapeParams) -> dict:
    """Format final scrape response."""
    return {
        "success": True,
        "niche": params.niche,
        "scraped_at": datetime.utcnow().isoformat(),
        "total_scraped": len(analyzed_items),
        "analyzed_count": len(analyzed_items),
        "trends": analyzed_items
    }
```

---

#### 1.13 Process & Save Trends (code)
**n8n Node ID:** `scrape-process-save`
**Purpose:** Save analyzed trends to database

**Original JavaScript:** Uses `this.helpers.request()` to POST to backend

**Python Implementation:**
```python
# backend/services/scraper.py
async def save_trends(self, trends: List[dict]) -> List[dict]:
    """Save analyzed trends to database."""
    saved = []

    for trend in trends:
        try:
            # Direct database insert (no HTTP call needed - we ARE the backend!)
            idea = ContentIdea(
                source_url=trend.get("url", ""),
                source_platform=trend.get("platform", "unknown"),
                original_text=trend.get("title", ""),
                pillar=trend.get("pillar", "educational_tips"),
                viral_score=trend.get("viral_score", 7),
                suggested_hook=trend.get("suggested_hook", ""),
                why_viral=trend.get("why_viral", ""),
                status="pending"
            )
            self.db.add(idea)
            await self.db.commit()
            await self.db.refresh(idea)
            saved.append(idea)
        except Exception as e:
            logger.error("save_trend_failed", error=str(e), trend=trend)
            continue

    return saved
```

---

#### 1.14 Respond to Webhook (respondToWebhook)
**n8n Node ID:** `scrape-respond`
**Purpose:** Return response to webhook caller

**Python Implementation:**
```python
# Already handled by FastAPI route return statement
# The endpoint returns the response directly
```

---

### SECTION 2: PIPELINE TRIGGER (10 nodes)

#### 2.1 Schedule Trigger (scheduleTrigger)
**n8n Node ID:** `schedule-trigger`
**Purpose:** Run pipeline every 15 minutes

**Python Implementation:**
```python
# backend/tasks/scheduled.py
app.conf.beat_schedule['pipeline-every-15min'] = {
    'task': 'tasks.pipeline.run_pipeline',
    'schedule': crontab(minute='*/15'),
    'args': (None,)  # No specific ID = get approved idea
}
```

---

#### 2.2 Webhook (webhook)
**n8n Node ID:** `webhook-trigger`
**Path:** `POST /webhook/trigger-pipeline`
**Purpose:** Manual pipeline trigger

**Python Implementation:**
```python
# backend/main.py
@app.post("/webhook/trigger-pipeline")
async def trigger_pipeline(request: Optional[PipelineRequest] = None):
    """Trigger the video production pipeline."""
    content_idea_id = request.content_idea_id if request else None
    task = run_pipeline.delay(content_idea_id)
    return {"success": True, "task_id": task.id}
```

---

#### 2.3 Respond Immediately (respondToWebhook)
**n8n Node ID:** `trigger-respond-immediate`
**Purpose:** Return 200 immediately, process in background

**Python Implementation:**
```python
# Already handled - Celery .delay() returns immediately
# The actual work happens in background worker
```

---

#### 2.4 Specific ID? (if)
**n8n Node ID:** `if-specific-id`
**Purpose:** Check if specific content_idea_id was provided

**Python Implementation:**
```python
# backend/tasks/pipeline.py
@celery_app.task
def run_pipeline(content_idea_id: Optional[int] = None):
    """Main pipeline task."""
    if content_idea_id:
        idea = get_specific_idea(content_idea_id)
    else:
        idea = get_next_approved_idea()

    if not idea:
        return {"success": False, "message": "No content idea to process"}

    # Continue with pipeline...
```

---

#### 2.5 Get Approved Idea (httpRequest)
**n8n Node ID:** `get-approved-idea`
**URL:** `http://backend:8000/api/content-ideas?status=approved&limit=1`

**Python Implementation:**
```python
# backend/tasks/pipeline.py
def get_next_approved_idea() -> Optional[ContentIdea]:
    """Get the next approved content idea from database."""
    with get_db() as db:
        return db.query(ContentIdea)\
            .filter(ContentIdea.status == "approved")\
            .order_by(ContentIdea.created_at)\
            .first()
```

---

#### 2.6 Get Specific Idea (httpRequest)
**n8n Node ID:** `get-specific-idea`
**URL:** `http://backend:8000/api/content-ideas/{id}`

**Python Implementation:**
```python
# backend/tasks/pipeline.py
def get_specific_idea(idea_id: int) -> Optional[ContentIdea]:
    """Get specific content idea by ID."""
    with get_db() as db:
        return db.query(ContentIdea).filter(ContentIdea.id == idea_id).first()
```

---

#### 2.7 Merge (merge)
**n8n Node ID:** `merge-idea-sources`
**Purpose:** Combine results from Get Approved / Get Specific

**Python Implementation:** Not needed - handled by if/else in `run_pipeline()`

---

#### 2.8 Normalize Idea (code)
**n8n Node ID:** `normalize-idea`
**Purpose:** Validate and normalize content idea data

**Python Implementation:**
```python
# backend/tasks/pipeline.py
def normalize_idea(idea: ContentIdea) -> dict:
    """Normalize content idea for pipeline processing."""
    return {
        "content_idea_id": idea.id,
        "source_url": idea.source_url,
        "source_platform": idea.source_platform,
        "original_text": idea.original_text,
        "pillar": idea.pillar,
        "suggested_hook": idea.suggested_hook,
        "has_script": idea.script_id is not None
    }
```

---

#### 2.9 Fetch Settings (code)
**n8n Node ID:** `fetch-settings`
**Purpose:** Get audio/video settings from backend

**Original JavaScript:** Makes HTTP request to `/api/settings/all`

**Python Implementation:**
```python
# backend/tasks/pipeline.py
def fetch_settings() -> dict:
    """Get all pipeline settings from database."""
    with get_db() as db:
        audio = db.query(AudioSettings).first()
        video = db.query(VideoSettings).first()

        return {
            "audio": {
                "original_volume": audio.original_volume if audio else 0.7,
                "avatar_volume": audio.avatar_volume if audio else 1.0,
                "music_volume": audio.music_volume if audio else 0.3,
                "ducking_enabled": audio.ducking_enabled if audio else True,
                "avatar_delay_seconds": audio.avatar_delay_seconds if audio else 3,
                "music_autoduck": audio.music_autoduck if audio else True
            },
            "video": {
                "output_width": video.output_width if video else 1080,
                "output_height": video.output_height if video else 1920,
                "greenscreen_enabled": video.greenscreen_enabled if video else True,
                "greenscreen_color": video.greenscreen_color if video else "#00FF00"
            }
        }
```

---

#### 2.10 Has Content? (if)
**n8n Node ID:** `if-has-content`
**Purpose:** Check if idea has script or needs generation

**Python Implementation:**
```python
# backend/tasks/pipeline.py
def run_pipeline(content_idea_id: Optional[int] = None):
    # ... get idea ...

    normalized = normalize_idea(idea)
    settings = fetch_settings()

    if normalized["has_script"]:
        # Skip script generation
        script = get_script(idea.script_id)
    else:
        # Generate new script
        script = await generate_script(idea)

    # Continue with voice generation...
```

---

### SECTION 3: SCRIPT GENERATION (7 nodes)

#### 3.1 Update Status (httpRequest)
**n8n Node ID:** `update-status-processing`
**URL:** `PATCH http://backend:8000/api/content-ideas/{id}`
**Purpose:** Update status to "script_generating"

**Python Implementation:**
```python
# backend/tasks/pipeline.py
def update_idea_status(idea_id: int, status: str):
    """Update content idea status."""
    with get_db() as db:
        idea = db.query(ContentIdea).filter(ContentIdea.id == idea_id).first()
        idea.status = status
        db.commit()
```

---

#### 3.2 Generate Script (Grok) (httpRequest)
**n8n Node ID:** `generate-script-llm`
**URL:** `https://openrouter.ai/api/v1/chat/completions`
**Purpose:** Generate video script using LLM

**Python Implementation:**
```python
# backend/services/script_generator.py
class ScriptGenerator(BaseService):
    async def generate(self, idea: ContentIdea, character_config: dict) -> dict:
        """Generate video script using Grok LLM."""

        prompt = self._build_prompt(idea, character_config)

        response = await self._request(
            method="POST",
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "x-ai/grok-4.1-fast",
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"}
            }
        )

        result = response.json()
        content = result["choices"][0]["message"]["content"]
        return json.loads(content)

    def _build_prompt(self, idea: ContentIdea, config: dict) -> str:
        """Build the script generation prompt."""
        return f"""You are a viral video scriptwriter for {config.get('niche', 'real estate')}.

Create a TikTok/Reels script based on this trending content:
- Original: {idea.original_text}
- Hook suggestion: {idea.suggested_hook}
- Content pillar: {idea.pillar}

Character: {config.get('name', 'Beth')} - {config.get('bio', 'Real estate expert')}
Location: {config.get('location', 'Coeur d\'Alene, Idaho')}

Return JSON with:
{{
  "hook": "Attention-grabbing first line",
  "body": "Main script content (30-45 seconds when spoken)",
  "cta": "Call to action",
  "full_script": "Complete script to read",
  "suggested_broll": ["scene1", "scene2", "scene3"],
  "hashtags": ["hashtag1", "hashtag2"],
  "estimated_duration": 45
}}"""
```

---

#### 3.3 Parse Script (code)
**n8n Node ID:** `parse-script`
**Purpose:** Parse and validate LLM script response

**Python Implementation:**
```python
# backend/services/script_generator.py
def parse_response(self, raw_response: dict) -> Script:
    """Parse LLM response into Script object."""
    return Script(
        hook=raw_response.get("hook", ""),
        body=raw_response.get("body", ""),
        cta=raw_response.get("cta", ""),
        full_script=raw_response.get("full_script", ""),
        suggested_broll=raw_response.get("suggested_broll", []),
        hashtags=raw_response.get("hashtags", []),
        estimated_duration=raw_response.get("estimated_duration", 45)
    )
```

---

#### 3.4 Save Script (httpRequest)
**n8n Node ID:** `save-script`
**URL:** `POST http://backend:8000/api/scripts`

**Python Implementation:**
```python
# backend/services/script_generator.py
async def save_script(self, script: Script, idea_id: int) -> Script:
    """Save script to database and link to content idea."""
    with get_db() as db:
        script.content_idea_id = idea_id
        db.add(script)
        db.commit()
        db.refresh(script)

        # Update content idea with script reference
        idea = db.query(ContentIdea).filter(ContentIdea.id == idea_id).first()
        idea.script_id = script.id
        idea.status = "script_ready"
        db.commit()

        return script
```

---

#### 3.5 Create Asset Record (httpRequest)
**n8n Node ID:** `create-asset-record`
**URL:** `POST http://backend:8000/api/assets`

**Python Implementation:**
```python
# backend/tasks/pipeline.py
def create_asset_record(script_id: int) -> Asset:
    """Create asset record for tracking media files."""
    with get_db() as db:
        asset = Asset(
            script_id=script_id,
            voice_path=None,
            avatar_path=None,
            combined_path=None,
            final_path=None,
            status="pending"
        )
        db.add(asset)
        db.commit()
        db.refresh(asset)
        return asset
```

---

### SECTION 4: VIDEO DOWNLOAD (3 nodes)

#### 4.1 Download Source Video (code)
**n8n Node ID:** `download-source-video`
**Purpose:** Download source video from URL using yt-dlp

**Original JavaScript:** Calls video-processor `/download` endpoint

**Python Implementation:**
```python
# backend/services/video.py
class VideoService(BaseService):
    VIDEO_PROCESSOR_URL = "http://video-processor:8080"

    async def download_source(self, url: str, script_id: int) -> Path:
        """Download source video using video-processor service."""

        response = await self._request(
            method="POST",
            url=f"{self.VIDEO_PROCESSOR_URL}/download",
            json={
                "url": url,
                "filename": f"{script_id}_source.mp4"
            }
        )

        result = response.json()
        if not result.get("success"):
            raise Exception(f"Download failed: {result.get('error')}")

        return Path(f"/app/assets/videos/{script_id}_source.mp4")
```

---

#### 4.2 Verify Download (code)
**n8n Node ID:** `verify-download`
**Purpose:** Verify downloaded video exists and is valid

**Python Implementation:**
```python
# backend/services/video.py
def verify_download(self, path: Path) -> bool:
    """Verify video file exists and has content."""
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    if path.stat().st_size < 1000:
        raise ValueError(f"Video file too small: {path}")

    # Get video info to verify it's valid
    info = self.get_video_info(path)
    if info["duration"] < 1:
        raise ValueError(f"Video duration invalid: {path}")

    return True
```

---

### SECTION 5: VOICE GENERATION (9 nodes)

#### 5.1 Get Character Config (httpRequest)
**n8n Node ID:** `get-character-config`
**URL:** `GET http://backend:8000/api/config/character`

**Python Implementation:**
```python
# backend/tasks/pipeline.py
def get_character_config() -> dict:
    """Get character configuration from database."""
    with get_db() as db:
        config = db.query(CharacterConfig).first()
        return {
            "name": config.name,
            "voice_id": config.voice_id,
            "avatar_id": config.avatar_id,
            "avatar_type": config.avatar_type,
            "image_url": config.image_url,
            "bio": config.bio,
            "location": config.location
        } if config else {}
```

---

#### 5.2 Prepare TTS Text (code)
**n8n Node ID:** `prepare-tts-text`
**Purpose:** Prepare text and metadata for TTS generation

**Python Implementation:**
```python
# backend/services/voice.py
class VoiceService(BaseService):
    def prepare_tts(self, script: Script, config: dict, asset_id: int) -> dict:
        """Prepare data for text-to-speech generation."""
        return {
            "text": script.full_script,
            "voice_id": config.get("voice_id", settings.ELEVENLABS_VOICE_ID),
            "script_id": script.id,
            "asset_id": asset_id,
            "output_path": f"/app/assets/audio/{script.id}_voice.mp3"
        }
```

---

#### 5.3 Check Voice Exists (code)
**n8n Node ID:** `check-voice-exists`
**Purpose:** Check if voice file already exists (skip regeneration)

**Python Implementation:**
```python
# backend/services/voice.py
def voice_exists(self, script_id: int) -> Tuple[bool, Optional[Path]]:
    """Check if voice file already exists."""
    path = Path(f"/app/assets/audio/{script_id}_voice.mp3")
    if path.exists() and path.stat().st_size > 1000:
        return True, path
    return False, None
```

---

#### 5.4 Voice Exists? (if)
**n8n Node ID:** `if-voice-exists`
**Purpose:** Branch based on voice file existence

**Python Implementation:**
```python
# backend/services/voice.py
async def get_or_generate_voice(self, script: Script, config: dict, asset_id: int) -> Path:
    """Get existing voice or generate new one."""
    exists, path = self.voice_exists(script.id)

    if exists:
        logger.info("voice_exists_skipping", script_id=script.id)
        return path

    return await self.generate_voice(script, config, asset_id)
```

---

#### 5.5 Generate Voice (ElevenLabs) (httpRequest)
**n8n Node ID:** `generate-voice`
**URL:** `POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}`

**Python Implementation:**
```python
# backend/services/voice.py
async def generate_voice(self, script: Script, config: dict, asset_id: int) -> Path:
    """Generate voice using ElevenLabs API."""
    voice_id = config.get("voice_id", settings.ELEVENLABS_VOICE_ID)

    response = await self._request(
        method="POST",
        url=f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        headers={
            "xi-api-key": settings.ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg"
        },
        json={
            "text": script.full_script,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
    )

    # Save audio file
    output_path = Path(f"/app/assets/audio/{script.id}_voice.mp3")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)

    return output_path
```

---

#### 5.6 Save Voice (writeBinaryFile)
**n8n Node ID:** `save-voice-file`
**Purpose:** Write voice audio to file

**Python Implementation:** Already included in `generate_voice()` above

---

#### 5.7 Get Duration (code)
**n8n Node ID:** `get-voice-duration`
**Purpose:** Get audio duration using ffprobe

**Python Implementation:**
```python
# backend/services/voice.py
def get_duration(self, audio_path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path)
        ],
        capture_output=True,
        text=True
    )

    return float(result.stdout.strip())
```

---

#### 5.8 Update Asset Voice (httpRequest)
**n8n Node ID:** `update-asset-voice`
**URL:** `PATCH http://backend:8000/api/assets/{id}`

**Python Implementation:**
```python
# backend/services/voice.py
def update_asset_voice(self, asset_id: int, voice_path: Path, duration: float):
    """Update asset record with voice file info."""
    with get_db() as db:
        asset = db.query(Asset).filter(Asset.id == asset_id).first()
        asset.voice_path = str(voice_path)
        asset.voice_duration = duration
        asset.status = "voice_ready"
        db.commit()
```

---

#### 5.9 Read Audio File (readBinaryFile)
**n8n Node ID:** `read-audio-file`
**Purpose:** Read audio file for HeyGen upload

**Python Implementation:**
```python
# backend/services/avatar.py
def read_audio_file(self, path: Path) -> bytes:
    """Read audio file content."""
    return path.read_bytes()
```

---

### SECTION 6: AVATAR GENERATION (19 nodes)

#### 6.1 Avatar Already Exists? (code)
**n8n Node ID:** `check-avatar-exists-early`
**Purpose:** Check if avatar video already exists or has HeyGen ID

**Python Implementation:**
```python
# backend/services/avatar.py
class AvatarService(BaseService):
    def check_avatar_status(self, script_id: int, asset: Asset) -> dict:
        """Check if avatar already exists or can be resumed."""
        avatar_path = Path(f"/app/assets/avatar/{script_id}_avatar.mp4")

        return {
            "file_exists": avatar_path.exists() and avatar_path.stat().st_size > 10000,
            "has_heygen_id": bool(asset.heygen_video_id),
            "heygen_video_id": asset.heygen_video_id,
            "avatar_path": str(avatar_path) if avatar_path.exists() else None
        }
```

---

#### 6.2 Avatar File Exists? / Has HeyGen ID? (if nodes)
**n8n Node IDs:** `if-avatar-file-exists`, `if-heygen-id-exists`
**Purpose:** Route based on avatar file or HeyGen job existence

**Python Implementation:**
```python
# backend/services/avatar.py
async def get_or_create_avatar(
    self,
    script_id: int,
    audio_path: Path,
    config: dict,
    asset: Asset,
    settings: dict
) -> Path:
    """Get existing avatar or create new one."""

    status = self.check_avatar_status(script_id, asset)

    # Option 1: File already exists
    if status["file_exists"]:
        logger.info("avatar_exists_skipping", script_id=script_id)
        return Path(status["avatar_path"])

    # Option 2: HeyGen job exists, resume polling
    if status["has_heygen_id"]:
        logger.info("resuming_heygen_job", video_id=status["heygen_video_id"])
        return await self.poll_and_download(status["heygen_video_id"], script_id)

    # Option 3: Create new avatar
    return await self.create_avatar(script_id, audio_path, config, asset, settings)
```

---

#### 6.3 Use Existing Avatar (code)
**n8n Node ID:** `mock-heygen-data`
**Purpose:** Use existing avatar file, skip HeyGen

**Python Implementation:** Already handled in `get_or_create_avatar()` above

---

#### 6.4 Resume HeyGen Check / Route Resume Status (httpRequest + code)
**n8n Node IDs:** `resume-heygen-check`, `route-resume-status`
**Purpose:** Check status of existing HeyGen job and route accordingly

**Python Implementation:**
```python
# backend/services/avatar.py
async def check_heygen_status(self, video_id: str) -> dict:
    """Check HeyGen video generation status."""
    response = await self._request(
        method="GET",
        url=f"https://api.heygen.com/v1/video_status.get?video_id={video_id}",
        headers={"X-Api-Key": settings.HEYGEN_API_KEY}
    )
    return response.json()
```

---

#### 6.5 Upload HeyGen Audio (httpRequest)
**n8n Node ID:** `upload-heygen-audio-http`
**URL:** `POST https://upload.heygen.com/v1/asset`

**Python Implementation:**
```python
# backend/services/avatar.py
async def upload_audio(self, audio_path: Path, script_id: int) -> str:
    """Upload audio file to HeyGen."""
    audio_data = audio_path.read_bytes()

    response = await self._request(
        method="POST",
        url=f"https://upload.heygen.com/v1/asset?name={script_id}_voice.mp3",
        headers={
            "X-Api-Key": settings.HEYGEN_API_KEY,
            "Content-Type": "audio/mpeg"
        },
        data=audio_data
    )

    result = response.json()
    return result["data"]["url"]
```

---

#### 6.6 Is Talking Photo? (if)
**n8n Node ID:** `check-is-photo`
**Purpose:** Check if using talking photo vs video avatar

**Python Implementation:**
```python
# backend/services/avatar.py
def is_talking_photo(self, config: dict) -> bool:
    """Check if avatar type is talking photo."""
    return config.get("avatar_type") == "talking_photo"
```

---

#### 6.7 Download Character Image / Upload Talking Photo (httpRequest nodes)
**n8n Node IDs:** `download-char-image`, `upload-talking-photo`
**Purpose:** Handle talking photo upload if needed

**Python Implementation:**
```python
# backend/services/avatar.py
async def upload_talking_photo(self, image_url: str) -> str:
    """Download and upload talking photo to HeyGen."""
    # Download image
    img_response = await self._request(method="GET", url=image_url)

    # Upload to HeyGen
    response = await self._request(
        method="POST",
        url="https://upload.heygen.com/v1/talking_photo",
        headers={
            "X-Api-Key": settings.HEYGEN_API_KEY,
            "Content-Type": "image/png"
        },
        data=img_response.content
    )

    return response.json()["data"]["talking_photo_id"]
```

---

#### 6.7a Check Video Exists / Video Exists? (code + if)
**n8n Node IDs:** `check-video-exists`, `if-video-exists`
**Purpose:** Check if avatar video file already exists (after HeyGen generation)

**Note:** This is different from "Avatar Already Exists?" (6.1) which checks BEFORE starting HeyGen.
This node checks after Prepare HeyGen Data to see if we can skip to FFmpeg.

**Python Implementation:**
```python
# Already covered in get_or_create_avatar() method - the file existence check
# happens at multiple stages to allow skipping expensive operations.
# The logic flow in Python handles this naturally without separate nodes.
```

---

#### 6.8 Prepare HeyGen Data (code)
**n8n Node ID:** `prepare-heygen-data`
**Purpose:** Prepare request payload for HeyGen API

**Python Implementation:**
```python
# backend/services/avatar.py
def prepare_heygen_request(
    self,
    audio_url: str,
    config: dict,
    settings: dict,
    talking_photo_id: Optional[str] = None
) -> dict:
    """Prepare HeyGen video generation request."""

    video_settings = settings.get("video", {})
    greenscreen = video_settings.get("greenscreen_enabled", True)

    if talking_photo_id:
        return {
            "video_inputs": [{
                "character": {
                    "type": "talking_photo",
                    "talking_photo_id": talking_photo_id
                },
                "voice": {
                    "type": "audio",
                    "audio_url": audio_url
                },
                "background": {"type": "color", "value": "#00FF00"} if greenscreen else None
            }],
            "dimension": {"width": 1080, "height": 1920}
        }
    else:
        return {
            "video_inputs": [{
                "character": {
                    "type": "avatar",
                    "avatar_id": config.get("avatar_id"),
                    "avatar_style": "normal"
                },
                "voice": {
                    "type": "audio",
                    "audio_url": audio_url
                },
                "background": {"type": "color", "value": "#00FF00"} if greenscreen else None
            }],
            "dimension": {"width": 1080, "height": 1920}
        }
```

---

#### 6.9 Create HeyGen Video (httpRequest)
**n8n Node ID:** `create-heygen-video`
**URL:** `POST https://api.heygen.com/v2/video/generate`

**Python Implementation:**
```python
# backend/services/avatar.py
async def create_heygen_video(self, request_data: dict) -> str:
    """Create video generation job on HeyGen."""
    response = await self._request(
        method="POST",
        url="https://api.heygen.com/v2/video/generate",
        headers={
            "X-Api-Key": settings.HEYGEN_API_KEY,
            "Content-Type": "application/json"
        },
        json=request_data
    )

    result = response.json()
    return result["data"]["video_id"]
```

---

#### 6.10 Extract Video ID (code)
**n8n Node ID:** `extract-heygen-id`
**Purpose:** Extract video ID and save to database

**Python Implementation:**
```python
# backend/services/avatar.py
def save_heygen_id(self, asset_id: int, video_id: str):
    """Save HeyGen video ID to asset record."""
    with get_db() as db:
        asset = db.query(Asset).filter(Asset.id == asset_id).first()
        asset.heygen_video_id = video_id
        asset.status = "avatar_generating"
        db.commit()
```

---

#### 6.11 Wait 30s / Wait & Retry (wait nodes)
**n8n Node IDs:** `wait-heygen`, `wait-retry`
**Purpose:** Polling delays

**Python Implementation:**
```python
# backend/services/avatar.py
import asyncio

async def poll_and_download(
    self,
    video_id: str,
    script_id: int,
    max_retries: int = 60,
    poll_interval: int = 30
) -> Path:
    """Poll HeyGen status and download when complete."""

    for attempt in range(max_retries):
        status = await self.check_heygen_status(video_id)
        state = status.get("data", {}).get("status")

        if state == "completed":
            video_url = status["data"]["video_url"]
            return await self.download_avatar(video_url, script_id)

        if state == "failed":
            raise Exception(f"HeyGen video failed: {status}")

        logger.info("heygen_polling", attempt=attempt, status=state)
        await asyncio.sleep(poll_interval)

    raise Exception(f"HeyGen timeout after {max_retries} attempts")
```

---

#### 6.12 Check Status (httpRequest)
**n8n Node ID:** `check-heygen-status`
**URL:** `GET https://api.heygen.com/v1/video_status.get?video_id={id}`

**Python Implementation:** Already included in `poll_and_download()` above

---

#### 6.13 Route Status (code)
**n8n Node ID:** `route-status-switch`
**Purpose:** Route based on HeyGen status (completed/processing/failed)

**Python Implementation:** Already handled in `poll_and_download()` with if/elif

---

#### 6.14 Increment Retry (code)
**n8n Node ID:** `increment-retry`
**Purpose:** Track retry count

**Python Implementation:** Already handled by for loop in `poll_and_download()`

---

#### 6.15 Download Avatar (httpRequest)
**n8n Node ID:** `download-avatar`
**URL:** Dynamic from Check Status response

**Python Implementation:**
```python
# backend/services/avatar.py
async def download_avatar(self, video_url: str, script_id: int) -> Path:
    """Download completed avatar video from HeyGen."""
    response = await self._request(method="GET", url=video_url)

    output_path = Path(f"/app/assets/avatar/{script_id}_avatar.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)

    return output_path
```

---

#### 6.16 Save Avatar (code)
**n8n Node ID:** `save-avatar-file`
**Purpose:** Save avatar and update database

**Python Implementation:**
```python
# backend/services/avatar.py
def update_asset_avatar(self, asset_id: int, avatar_path: Path):
    """Update asset record with avatar path."""
    with get_db() as db:
        asset = db.query(Asset).filter(Asset.id == asset_id).first()
        asset.avatar_path = str(avatar_path)
        asset.status = "avatar_ready"
        db.commit()
```

---

#### 6.17 Stop on Failure (stopAndError)
**n8n Node ID:** `stop-heygen-error`
**Purpose:** Stop pipeline on HeyGen failure

**Python Implementation:**
```python
# Handled by raising exceptions in poll_and_download()
# Celery will mark task as failed and log error
```

---

### SECTION 7: VIDEO COMPOSITION (6 nodes)

#### 7.1 Build FFmpeg Command (code)
**n8n Node ID:** `build-ffmpeg-cmd`
**Purpose:** Build complex FFmpeg filter command

**Python Implementation:**
```python
# backend/services/video.py
class VideoService(BaseService):
    def build_ffmpeg_command(
        self,
        script_id: int,
        avatar_path: Path,
        source_path: Path,
        voice_path: Path,
        settings: dict,
        music_path: Optional[Path] = None
    ) -> List[str]:
        """Build FFmpeg command for video composition."""

        audio_settings = settings.get("audio", {})
        video_settings = settings.get("video", {})

        orig_vol = audio_settings.get("original_volume", 0.7)
        avatar_vol = audio_settings.get("avatar_volume", 1.0)
        music_vol = audio_settings.get("music_volume", 0.3)
        delay_ms = int(audio_settings.get("avatar_delay_seconds", 3) * 1000)

        output_path = f"/app/assets/output/{script_id}_combined.mp4"

        # Build filter complex
        filter_parts = [
            # Scale and crop background
            f"[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1[bg]",
            # Scale and chromakey avatar
            f"[1:v]scale=1188:-1,chromakey=0x00FF00:0.1:0.2[avatar_keyed]",
            # Overlay avatar on background
            f"[bg][avatar_keyed]overlay=-300:H-h+600:shortest=1[outv]",
            # Process audio
            f"[2:a]adelay={delay_ms}|{delay_ms},volume={avatar_vol}[voice]",
            f"[0:a]volume={orig_vol}[orig]",
        ]

        if music_path:
            filter_parts.extend([
                f"[3:a]volume={music_vol}[music]",
                f"[orig][voice][music]amix=inputs=3:duration=first:dropout_transition=0,volume=2[outa]"
            ])
            inputs = ["-i", str(source_path), "-i", str(avatar_path), "-i", str(voice_path), "-i", str(music_path)]
        else:
            filter_parts.append(
                f"[orig][voice]amix=inputs=2:duration=first:dropout_transition=0,volume=2[outa]"
            )
            inputs = ["-i", str(source_path), "-i", str(avatar_path), "-i", str(voice_path)]

        filter_complex = ";".join(filter_parts)

        return [
            "ffmpeg", "-y",
            "-stream_loop", "-1",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-s", "1080x1920",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart", "-pix_fmt", "yuv420p",
            "-t", "62",  # Max duration
            output_path
        ]
```

---

#### 7.2 Run FFmpeg (code)
**n8n Node ID:** `run-ffmpeg-combine`
**Purpose:** Execute FFmpeg command

**Python Implementation:**
```python
# backend/services/video.py
async def compose_video(
    self,
    script_id: int,
    avatar_path: Path,
    source_path: Path,
    voice_path: Path,
    settings: dict
) -> Path:
    """Compose final video with avatar overlay."""

    # Find active music file
    music_path = self.find_active_music()

    # Build command
    cmd = self.build_ffmpeg_command(
        script_id, avatar_path, source_path, voice_path, settings, music_path
    )

    # Execute
    logger.info("ffmpeg_starting", script_id=script_id)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("ffmpeg_failed", stderr=result.stderr)
        raise Exception(f"FFmpeg failed: {result.stderr}")

    output_path = Path(f"/app/assets/output/{script_id}_combined.mp4")
    logger.info("ffmpeg_completed", output=str(output_path))

    return output_path
```

---

#### 7.3 Success? (if)
**n8n Node ID:** `if-ffmpeg-success`
**Purpose:** Check if FFmpeg succeeded

**Python Implementation:** Handled by exception raising in `compose_video()`

---

#### 7.4 Handle Error (code)
**n8n Node ID:** `handle-ffmpeg-error`
**Purpose:** Log and handle FFmpeg errors

**Python Implementation:**
```python
# Celery task error handling
@celery_app.task(bind=True, max_retries=3)
def run_pipeline(self, content_idea_id: Optional[int] = None):
    try:
        # ... pipeline logic ...
    except Exception as e:
        logger.error("pipeline_failed", error=str(e), idea_id=content_idea_id)
        update_idea_status(content_idea_id, "failed")
        raise self.retry(exc=e, countdown=60)
```

---

### SECTION 8: CAPTION GENERATION (9 nodes)

#### 8.1 Prepare Whisper (code)
**n8n Node ID:** `prepare-whisper`
**Purpose:** Prepare audio for Whisper transcription

**Python Implementation:**
```python
# backend/services/captions.py
class CaptionService(BaseService):
    def prepare_whisper(self, voice_path: Path, script_id: int) -> dict:
        """Prepare data for Whisper transcription."""
        return {
            "audio_path": str(voice_path),
            "script_id": script_id,
            "output_path": f"/app/assets/captions/{script_id}_captions.srt"
        }
```

---

#### 8.2 Read Audio for Whisper (code)
**n8n Node ID:** `read-audio-for-whisper`
**Purpose:** Read audio file as binary

**Python Implementation:**
```python
# backend/services/captions.py
def read_audio(self, path: Path) -> bytes:
    """Read audio file for API submission."""
    return path.read_bytes()
```

---

#### 8.3 Whisper Transcribe (httpRequest)
**n8n Node ID:** `whisper-transcribe`
**URL:** `POST https://api.openai.com/v1/audio/transcriptions`

**Python Implementation:**
```python
# backend/services/captions.py
async def transcribe(self, audio_path: Path) -> str:
    """Transcribe audio using OpenAI Whisper API."""

    audio_data = audio_path.read_bytes()

    # Use multipart form data
    files = {
        "file": (audio_path.name, audio_data, "audio/mpeg"),
        "model": (None, "whisper-1"),
        "response_format": (None, "srt"),
        "language": (None, "en")
    }

    response = await self.client.post(
        "https://api.openai.com/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
        files=files
    )
    response.raise_for_status()

    return response.text
```

---

#### 8.4 Parse Whisper Response (code)
**n8n Node ID:** `parse-whisper-response`
**Purpose:** Extract SRT content from Whisper response

**Python Implementation:**
```python
# backend/services/captions.py
def parse_srt(self, srt_content: str) -> List[dict]:
    """Parse SRT content into structured data."""
    entries = []
    current = {}

    for line in srt_content.strip().split('\n'):
        line = line.strip()
        if line.isdigit():
            if current:
                entries.append(current)
            current = {"index": int(line)}
        elif '-->' in line:
            start, end = line.split(' --> ')
            current["start"] = start
            current["end"] = end
        elif line:
            current["text"] = current.get("text", "") + " " + line

    if current:
        entries.append(current)

    return entries
```

---

#### 8.5 Save SRT (code)
**n8n Node ID:** `save-srt`
**Purpose:** Save SRT file to disk

**Python Implementation:**
```python
# backend/services/captions.py
def save_srt(self, srt_content: str, script_id: int) -> Path:
    """Save SRT file to disk."""
    output_path = Path(f"/app/assets/captions/{script_id}_captions.srt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(srt_content)
    return output_path
```

---

#### 8.6 Build Caption Cmd (code)
**n8n Node ID:** `build-caption-cmd`
**Purpose:** Build FFmpeg command to burn captions

**Python Implementation:**
```python
# backend/services/captions.py
def build_caption_command(
    self,
    video_path: Path,
    srt_path: Path,
    output_path: Path,
    style: dict = None
) -> List[str]:
    """Build FFmpeg command to burn captions."""

    style = style or {
        "font": "Arial",
        "size": 48,
        "color": "white",
        "outline": "black",
        "outline_width": 2
    }

    subtitle_filter = (
        f"subtitles={srt_path}:force_style='"
        f"FontName={style['font']},"
        f"FontSize={style['size']},"
        f"PrimaryColour=&Hffffff,"
        f"OutlineColour=&H000000,"
        f"Outline={style['outline_width']},"
        f"Alignment=2'"
    )

    return [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", subtitle_filter,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "copy",
        str(output_path)
    ]
```

---

#### 8.7 Burn Captions (code)
**n8n Node ID:** `run-ffmpeg-caption`
**Purpose:** Execute FFmpeg caption burn

**Python Implementation:**
```python
# backend/services/captions.py
async def burn_captions(
    self,
    video_path: Path,
    srt_path: Path,
    script_id: int
) -> Path:
    """Burn captions onto video."""

    output_path = Path(f"/app/assets/output/{script_id}_final.mp4")

    cmd = self.build_caption_command(video_path, srt_path, output_path)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise Exception(f"Caption burn failed: {result.stderr}")

    return output_path
```

---

#### 8.8 Caption Success? (if)
**n8n Node ID:** `if-caption-success`
**Purpose:** Check if caption burn succeeded

**Python Implementation:** Handled by exception in `burn_captions()`

---

#### 8.9 Handle Caption Error (code)
**n8n Node ID:** `handle-caption-error`
**Purpose:** Handle caption failures

**Python Implementation:** Handled by Celery task error handling

---

### SECTION 9: PUBLISHING (11 nodes)

#### 9.1 Prepare Publish Data (code)
**n8n Node ID:** `prepare-publish`
**Purpose:** Prepare data for publishing

**Python Implementation:**
```python
# backend/services/publisher.py
class PublisherService(BaseService):
    def prepare_publish_data(
        self,
        script: Script,
        video_path: Path,
        idea: ContentIdea
    ) -> dict:
        """Prepare data for publishing."""
        return {
            "script_id": script.id,
            "content_idea_id": idea.id,
            "video_path": str(video_path),
            "caption": f"{script.hook}\n\n{script.cta}",
            "hashtags": script.hashtags
        }
```

---

#### 9.2 Read Final Video (readBinaryFile)
**n8n Node ID:** `read-final-video`
**Purpose:** Read final video file

**Python Implementation:**
```python
# backend/services/publisher.py
def read_video(self, path: Path) -> bytes:
    """Read video file content."""
    return path.read_bytes()
```

---

#### 9.3 Upload to Dropbox (code)
**n8n Node ID:** `dropbox-upload-code`
**Purpose:** Upload video to Dropbox

**Python Implementation:**
```python
# backend/services/storage.py
class StorageService(BaseService):
    async def upload_to_dropbox(self, file_path: Path, dest_path: str) -> str:
        """Upload file to Dropbox."""
        file_data = file_path.read_bytes()

        # Refresh token if needed
        access_token = await self.refresh_dropbox_token()

        response = await self._request(
            method="POST",
            url="https://content.dropboxapi.com/2/files/upload",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Dropbox-API-Arg": json.dumps({
                    "path": dest_path,
                    "mode": "overwrite"
                }),
                "Content-Type": "application/octet-stream"
            },
            data=file_data
        )

        return response.json()["path_display"]
```

---

#### 9.4 Create Share Link (code)
**n8n Node ID:** `dropbox-share-code`
**Purpose:** Create Dropbox share link

**Python Implementation:**
```python
# backend/services/storage.py
async def create_share_link(self, path: str) -> str:
    """Create shared link for Dropbox file."""
    access_token = await self.refresh_dropbox_token()

    response = await self._request(
        method="POST",
        url="https://api.dropboxapi.com/2/sharing/create_shared_link_with_settings",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        },
        json={
            "path": path,
            "settings": {"requested_visibility": "public"}
        }
    )

    # Convert to direct download URL
    url = response.json()["url"]
    return url.replace("?dl=0", "?dl=1")
```

---

#### 9.5 Parse Upload Response (code)
**n8n Node ID:** `format-gcs-url`
**Purpose:** Format upload URL for Blotato

**Python Implementation:**
```python
# backend/services/publisher.py
def format_media_url(self, dropbox_url: str) -> str:
    """Format Dropbox URL for Blotato media upload."""
    return dropbox_url.replace("www.dropbox.com", "dl.dropboxusercontent.com")
```

---

#### 9.6 Blotato Media Upload (httpRequest)
**n8n Node ID:** `blotato-media-upload`
**URL:** `POST https://backend.blotato.com/v2/media`

**Python Implementation:**
```python
# backend/services/publisher.py
async def upload_media(self, video_url: str) -> str:
    """Upload media to Blotato via URL."""
    response = await self._request(
        method="POST",
        url="https://backend.blotato.com/v2/media",
        headers={
            "Authorization": f"Bearer {settings.BLOTATO_API_KEY}",
            "Content-Type": "application/json"
        },
        json={"url": video_url}
    )

    return response.json()["media_id"]
```

---

#### 9.7 Publish (Blotato) (httpRequest)
**n8n Node ID:** `publish-blotato`
**URL:** `POST https://backend.blotato.com/v2/posts/create`

**Python Implementation:**
```python
# backend/services/publisher.py
async def create_post(
    self,
    media_id: str,
    caption: str,
    hashtags: List[str]
) -> dict:
    """Create post on Blotato for multi-platform publishing."""

    full_caption = f"{caption}\n\n{' '.join(f'#{h}' for h in hashtags)}"

    response = await self._request(
        method="POST",
        url="https://backend.blotato.com/v2/posts/create",
        headers={
            "Authorization": f"Bearer {settings.BLOTATO_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "media_ids": [media_id],
            "caption": full_caption,
            "platforms": ["tiktok", "instagram", "youtube"]
        }
    )

    return response.json()
```

---

#### 9.8 Parse Response (code)
**n8n Node ID:** `parse-publish-response`
**Purpose:** Parse Blotato response

**Python Implementation:**
```python
# backend/services/publisher.py
def parse_response(self, response: dict, content_idea_id: int, script_id: int) -> dict:
    """Parse Blotato response for database storage."""
    return {
        "content_idea_id": content_idea_id,
        "script_id": script_id,
        "blotato_post_id": response.get("post_id"),
        "platforms": response.get("platforms", []),
        "status": "published"
    }
```

---

#### 9.9 Save Publish Record (httpRequest)
**n8n Node ID:** `save-publish-record`
**URL:** `POST http://backend:8000/api/published`

**Python Implementation:**
```python
# backend/services/publisher.py
def save_publish_record(self, data: dict) -> Published:
    """Save publish record to database."""
    with get_db() as db:
        record = Published(**data)
        db.add(record)
        db.commit()
        db.refresh(record)
        return record
```

---

#### 9.10 Update Status Published (httpRequest)
**n8n Node ID:** `update-final-status`
**URL:** `PATCH http://backend:8000/api/content-ideas/{id}`

**Python Implementation:**
```python
# backend/services/publisher.py
def update_final_status(self, idea_id: int):
    """Update content idea status to published."""
    with get_db() as db:
        idea = db.query(ContentIdea).filter(ContentIdea.id == idea_id).first()
        idea.status = "published"
        idea.published_at = datetime.utcnow()
        db.commit()
```

---

### SECTION 10: FILE SERVER (6 nodes)

#### 10.1 Serve Audio / Serve Video (webhook nodes)
**n8n Node IDs:** `webhook-serve-audio`, `webhook-serve-video`
**Paths:** `/webhook/serve-audio/:script_id`, `/webhook/serve-video/:script_id`
**Purpose:** Serve media files to external services (HeyGen, Blotato)

**Python Implementation:**
```python
# backend/main.py
from fastapi.responses import FileResponse

@app.get("/webhook/serve-audio/{script_id}")
async def serve_audio(script_id: int):
    """Serve audio file for external API access."""
    file_path = Path(f"/app/assets/audio/{script_id}_voice.mp3")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(file_path, media_type="audio/mpeg")

@app.get("/webhook/serve-video/{script_id}")
async def serve_video(script_id: int):
    """Serve video file for external API access."""
    file_path = Path(f"/app/assets/output/{script_id}_final.mp4")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(file_path, media_type="video/mp4")
```

---

#### 10.2 Read Audio / Read Video / Return Audio / Return Video (code + respond nodes)
**n8n Node IDs:** `read-audio-code-server`, `read-video-file`, `respond-audio`, `respond-video`
**Purpose:** Read and return file content

**Python Implementation:** Already handled by FileResponse in routes above

---

## Celery Task Definitions

### Main Pipeline Task

```python
# backend/tasks/pipeline.py
from celery import Celery
from services import (
    ScriptGenerator, VoiceService, AvatarService,
    VideoService, CaptionService, PublisherService, StorageService
)

celery_app = Celery('tasks', broker='redis://redis:6379/0')

@celery_app.task(bind=True, max_retries=3)
def run_pipeline(self, content_idea_id: Optional[int] = None):
    """
    Main video production pipeline.

    Stages:
    1. Get content idea (specific or next approved)
    2. Generate script (if needed)
    3. Generate voice
    4. Create avatar
    5. Compose video
    6. Generate and burn captions
    7. Upload and publish
    """
    try:
        # Initialize services
        script_gen = ScriptGenerator()
        voice_svc = VoiceService()
        avatar_svc = AvatarService()
        video_svc = VideoService()
        caption_svc = CaptionService()
        publisher = PublisherService()
        storage = StorageService()

        # Stage 1: Get content idea
        if content_idea_id:
            idea = get_specific_idea(content_idea_id)
        else:
            idea = get_next_approved_idea()

        if not idea:
            return {"success": False, "message": "No content to process"}

        logger.info("pipeline_started", idea_id=idea.id)

        # Get settings and config
        settings = fetch_settings()
        config = get_character_config()

        # Stage 2: Script generation
        if idea.script_id:
            script = get_script(idea.script_id)
        else:
            update_idea_status(idea.id, "script_generating")
            raw_script = await script_gen.generate(idea, config)
            script = script_gen.parse_response(raw_script)
            script = await script_gen.save_script(script, idea.id)

        # Create asset record
        asset = create_asset_record(script.id)

        # Stage 3: Download source video
        source_path = await video_svc.download_source(idea.source_url, script.id)
        video_svc.verify_download(source_path)

        # Stage 4: Voice generation
        voice_path = await voice_svc.get_or_generate_voice(script, config, asset.id)
        duration = voice_svc.get_duration(voice_path)
        voice_svc.update_asset_voice(asset.id, voice_path, duration)

        # Stage 5: Avatar generation
        avatar_path = await avatar_svc.get_or_create_avatar(
            script.id, voice_path, config, asset, settings
        )
        avatar_svc.update_asset_avatar(asset.id, avatar_path)

        # Stage 6: Video composition
        combined_path = await video_svc.compose_video(
            script.id, avatar_path, source_path, voice_path, settings
        )

        # Stage 7: Caption generation and burn
        srt_content = await caption_svc.transcribe(voice_path)
        srt_path = caption_svc.save_srt(srt_content, script.id)
        final_path = await caption_svc.burn_captions(combined_path, srt_path, script.id)

        # Stage 8: Upload and publish
        dropbox_path = await storage.upload_to_dropbox(
            final_path, f"/n8n-outputs/{script.id}_final.mp4"
        )
        share_url = await storage.create_share_link(dropbox_path)
        media_url = publisher.format_media_url(share_url)
        media_id = await publisher.upload_media(media_url)

        publish_response = await publisher.create_post(
            media_id,
            f"{script.hook}\n\n{script.cta}",
            script.hashtags
        )

        publish_data = publisher.parse_response(publish_response, idea.id, script.id)
        publisher.save_publish_record(publish_data)
        publisher.update_final_status(idea.id)

        logger.info("pipeline_completed", idea_id=idea.id, script_id=script.id)

        return {
            "success": True,
            "content_idea_id": idea.id,
            "script_id": script.id,
            "final_video": str(final_path)
        }

    except Exception as e:
        logger.error("pipeline_failed", error=str(e), idea_id=content_idea_id)
        if content_idea_id:
            update_idea_status(content_idea_id, "failed")
        raise self.retry(exc=e, countdown=60)
```

---

### Scrape Task

```python
# backend/tasks/scrape.py
from services import ScraperService

@celery_app.task
def run_scrape(params: dict):
    """Run trend scraping task."""
    scraper = ScraperService()

    # Parse parameters
    scrape_params = scraper.parse_params(params)

    # Scrape platforms
    results = await scraper.scrape_platforms(scrape_params)

    # Normalize results
    normalized = scraper.normalize_all_results(results)

    # Analyze with LLM
    analyzed = await scraper.analyze_trends(normalized, scrape_params.niche)

    # Save to database
    saved = await scraper.save_trends(analyzed)

    # Format response
    return scraper.format_response(analyzed, scrape_params)
```

---

## API Endpoint Changes

### New Endpoints (add to main.py)

```python
# Webhook replacements
@app.post("/webhook/trigger-pipeline")
@app.post("/webhook/scrape-trends")
@app.get("/webhook/serve-audio/{script_id}")
@app.get("/webhook/serve-video/{script_id}")

# Task status endpoints
@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get Celery task status."""
    result = AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": result.status,
        "result": result.result if result.ready() else None
    }
```

---

## Docker Compose Changes

### Updated docker-compose.yml

```yaml
services:
  # REMOVE n8n service entirely
  # n8n:
  #   ... (delete this whole section)

  # ADD Redis for Celery
  redis:
    image: redis:7-alpine
    container_name: redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  # UPDATE backend to include Celery
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: backend
    restart: unless-stopped
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/0
      # ... all other env vars ...
    ports:
      - "8000:8000"
    volumes:
      - ./assets:/app/assets
      - ./backend:/app
    depends_on:
      - postgres
      - redis

  # ADD Celery worker
  celery-worker:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: celery-worker
    restart: unless-stopped
    command: celery -A celery_app worker --loglevel=info
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/0
      # ... all API keys ...
    volumes:
      - ./assets:/app/assets
      - ./backend:/app
    depends_on:
      - postgres
      - redis
      - backend

  # ADD Celery beat (scheduler)
  celery-beat:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: celery-beat
    restart: unless-stopped
    command: celery -A celery_app beat --loglevel=info
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - CELERY_BROKER_URL=redis://redis:6379/0
    volumes:
      - ./backend:/app
    depends_on:
      - redis
      - celery-worker

  # KEEP these unchanged
  postgres:
    # ... (unchanged)

  frontend:
    # ... (unchanged)

  video-processor:
    # ... (unchanged)

volumes:
  postgres_data:
  redis_data:  # NEW
  # n8n_data:  # REMOVE
```

---

## Migration Checklist

### Phase 1: Setup (Day 1)
- [ ] Create `backend/config.py` with pydantic-settings
- [ ] Create `backend/celery_app.py`
- [ ] Add Redis to docker-compose.yml
- [ ] Create `backend/services/` directory structure
- [ ] Create `backend/services/base.py` with BaseService class
- [ ] Update `backend/requirements.txt`

### Phase 2: Services (Day 2-3)
- [ ] Implement `services/scraper.py` (nodes 1.1-1.14)
- [ ] Implement `services/script_generator.py` (nodes 3.1-3.5)
- [ ] Implement `services/voice.py` (nodes 5.1-5.9)
- [ ] Implement `services/avatar.py` (nodes 6.1-6.17)
- [ ] Implement `services/video.py` (nodes 4.1-4.2, 7.1-7.4)
- [ ] Implement `services/captions.py` (nodes 8.1-8.9)
- [ ] Implement `services/publisher.py` (nodes 9.1-9.10)
- [ ] Implement `services/storage.py` (Dropbox upload)

### Phase 3: Tasks (Day 3)
- [ ] Implement `tasks/pipeline.py`
- [ ] Implement `tasks/scrape.py`
- [ ] Implement `tasks/scheduled.py`

### Phase 4: API Updates (Day 4)
- [ ] Add webhook replacement endpoints to `main.py`
- [ ] Add task status endpoint
- [ ] Add file serving endpoints
- [ ] Test all endpoints with curl

### Phase 5: Integration Testing (Day 4-5)
- [ ] Test scrape task end-to-end
- [ ] Test pipeline task end-to-end
- [ ] Test scheduled tasks
- [ ] Test error handling and retries
- [ ] Test file serving to external APIs

### Phase 6: Deployment (Day 5)
- [ ] Update docker-compose.yml
- [ ] Remove n8n container
- [ ] Add Redis container
- [ ] Add Celery worker container
- [ ] Add Celery beat container
- [ ] Deploy and monitor

### Phase 7: Cleanup (Day 6)
- [ ] Remove n8n Dockerfile
- [ ] Remove workflow JSON files (keep as reference)
- [ ] Update README.md
- [ ] Update CLAUDE.md
- [ ] Archive this MIGRATION.md

---

## Node Count Verification

### By Section
| Section | n8n Nodes | Python Equivalent |
|---------|-----------|-------------------|
| Scrape | 19 | ScraperService + scrape task |
| Trigger | 10 | Pipeline task + API endpoint |
| Script | 7 | ScriptGenerator service |
| Video Download | 3 | VideoService.download_source |
| Voice | 9 | VoiceService |
| Avatar | 19 | AvatarService |
| Compose | 6 | VideoService.compose_video |
| Caption | 9 | CaptionService |
| Publish | 11 | PublisherService + StorageService |
| File Server | 6 | FastAPI FileResponse routes |
| Sticky Notes | 10 | N/A (comments only) |
| NoOp (skip) | 4 | N/A (placeholders) |
| **TOTAL** | **107** | **All covered** |

### By Node Type
| Node Type | Count | Python Equivalent |
|-----------|-------|-------------------|
| code | 38 | Service methods |
| httpRequest | 25 | httpx async requests |
| if | 12 | Python if/else |
| stickyNote | 10 | N/A (comments) |
| webhook | 4 | FastAPI routes |
| respondToWebhook | 4 | FastAPI returns |
| noOp | 4 | N/A (placeholders) |
| wait | 2 | asyncio.sleep() |
| scheduleTrigger | 2 | Celery beat |
| readBinaryFile | 2 | Path.read_bytes() |
| merge | 2 | dict merge / asyncio.gather |
| writeBinaryFile | 1 | Path.write_bytes() |
| stopAndError | 1 | raise Exception |
| **TOTAL** | **107** | **93 functional, 14 non-functional** |

---

*Document created: January 12, 2026*
*Last verified against: AI_ContentGenerator_v2.9_ffmpeg_container_fix_20260109_152800.json*
