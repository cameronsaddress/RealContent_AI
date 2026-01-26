# CLAUDE.md

This file provides guidance to Claude Code when working with the SocialGen video pipeline repository.

---

## Permissions

Claude is **allowed and encouraged** to automatically use:

- **curl** - HTTP requests to APIs
- **docker** - Container management, logs, exec commands
- **bash** - Shell commands for file operations, git
- **jq** - JSON parsing

### REQUIRES USER PERMISSION (Cost-Incurring Services)

**ALWAYS ASK** before triggering:
- **Pipeline** - `curl /api/pipeline/trigger` uses HeyGen credits (~$1-3)
- **Apify** - Scraping operations (~$0.30-1.50)
- **HeyGen** - Video generation (~$1-3 per video)

### HeyGen MCP Server (READ-ONLY)

The HeyGen MCP server (`mcp__HeyGen__*`) is for **reading only**:
- Check credits, list voices/avatars
- **NEVER** use `generate_avatar_video` - all generation goes through Celery pipeline

---

## Critical Architecture Rules

### 1. Pipeline Stage Order - Source Video BEFORE Script

The pipeline MUST download and transcribe the source video BEFORE generating the script:

```
1. Get Content Idea
2. Download Source Video     <- BEFORE script!
3. Transcribe Source Video   <- Get what they SAID
4. Generate Script           <- Now has transcription context
5. Generate Voice
6. Generate Avatar
7. Compose Video
8. Burn Captions
9. Upload & Publish
```

This gives Grok the transcription of what was SAID in the viral video, not just the caption.

### 2. Asset Reuse - NEVER Skip Existence Checks

The pipeline checks for existing assets at each stage to avoid redundant API calls:

| Stage | Check Method | File Path |
|-------|--------------|-----------|
| Source Video | File exists | `videos/idea_{content_idea_id}_source.mp4` |
| Transcription | DB field | `content_ideas.source_transcription` |
| Script | DB query | `scripts.content_idea_id` |
| Voice | `voice_exists()` | `audio/{id}_voice.mp3` |
| Avatar | `avatar_exists()` | `avatar/{id}_avatar.mp4` |
| Combined | `combined_video_exists()` | `output/{id}_combined.mp4` |
| Final | `final_video_exists()` | `output/{id}_final.mp4` |

**To force regeneration:** Delete the file, don't modify the check logic.

### 3. Settings Tables - Auto-Synced

The API now automatically syncs both tables when you update settings:
- `video_settings` / `audio_settings` - Dedicated tables
- `system_settings` - JSONB key-value (pipeline reads from here)

```bash
# This updates BOTH tables automatically
curl -X PUT http://100.83.153.43:8000/api/settings/video \
  -H "Content-Type: application/json" \
  -d '{"avatar_offset_x": -250}'
```

To verify sync:
```sql
SELECT value FROM system_settings WHERE key = 'video_settings';
```

### 4. GPU Video Processing - Use video-processor Service

Video encoding runs on the GPU via the `video-processor` container:

```bash
# Compose (chromakey + overlay)
POST http://video-processor:8080/compose

# Caption burning
POST http://video-processor:8080/caption

# Viral clip rendering
POST http://video-processor:8080/render-viral-clip
```

**NEVER** run FFmpeg in `celery-worker` for video encoding. The pipeline calls the video-processor service via HTTP.

### 5. Volume Mounts in video-processor

The video-processor container has these mounts:
| Host Path | Container Path |
|-----------|----------------|
| `./assets/videos` | `/downloads` |
| `./assets/output` | `/outputs` |
| `./assets/audio` | `/audio` |
| `./assets/avatar` | `/avatar` |
| `./assets/captions` | `/captions` |
| `./assets/music` | `/music` |

When calling video-processor endpoints, use container paths:
```json
{
  "avatar_path": "/avatar/33_avatar.mp4",
  "background_path": "/downloads/33_source.mp4"
}
```

### 6. Karaoke Captions - ASS Format

Captions use ASS format with `\kf` (karaoke fill) effect:
- Primary: White (`&H00FFFFFF`)
- Secondary: Yellow (`&H0000FFFF`) - fills as word is spoken
- Alignment: 5 (center)
- MarginV: 960 (center of 1920px screen)

```ass
Dialogue: 0,0:00:00.00,0:00:01.51,Default,,0,0,0,,{\kf30}Hey {\kf30}neighbors
```

### 7. Avatar Positioning

Avatar position is controlled by:
- `avatar_scale`: 0.75 (75% of original size)
- `avatar_offset_x`: -250 (negative = left of center)
- `avatar_offset_y`: 600 (positive = down from top)

FFmpeg overlay formula:
```
X: 10 + avatar_offset_x
Y: (H-h-10) + avatar_offset_y
```

### 8. Location Branding

Use "North Idaho & Spokane area" - NOT "Coeur d'Alene". Files updated:
- `schemas.py`
- `models.py`
- `script_generator.py`
- `publisher.py`

### 9. TikTok Downloads Require Browser Headers

TikTok CDN returns 204 No Content without proper headers:
```python
headers = {
    "User-Agent": "Mozilla/5.0...",
    "Referer": "https://www.tiktok.com/",
    "Origin": "https://www.tiktok.com"
}
```

---

## Service URLs

| Service | URL |
|---------|-----|
| Frontend | http://100.83.153.43:3000 |
| Backend API | http://100.83.153.43:8000 |
| API Docs | http://100.83.153.43:8000/docs |
| Video Processor | http://100.83.153.43:8080 |

---

## Container Names

| Service | Container |
|---------|-----------|
| PostgreSQL | `SocialGen_postgres` |
| Redis | `SocialGen_redis` |
| Backend | `SocialGen_backend` |
| Celery Worker | `SocialGen_celery_worker` |
| Celery Beat | `SocialGen_celery_beat` |
| Frontend | `SocialGen_frontend` |
| Video Processor | `SocialGen_video_processor` |
| VPN (Gluetun) | `SocialGen_vpn` |

---

## Common Commands

### Restart Services After Code Changes
```bash
# Backend/Celery (Python changes)
docker compose restart celery-worker backend

# Video Processor (needs rebuild for code changes)
docker compose up -d --build video-processor
```

### View Logs
```bash
docker logs -f SocialGen_celery_worker  # Pipeline execution
docker logs -f SocialGen_backend        # API requests
docker logs -f SocialGen_video_processor # FFmpeg operations
```

### Database Access
```bash
docker exec -it SocialGen_postgres psql -U n8n -d content_pipeline
```

### Check GPU Status
```bash
docker exec SocialGen_video_processor nvidia-smi
curl http://localhost:8080/health | jq .gpu_encoder_available
```

---

## Pipeline Stages (Main RealEstate Pipeline)

1. **get_content** - Fetch approved content idea
2. **script_generation** - Grok 4.1 generates script
3. **voice_generation** - ElevenLabs TTS
4. **avatar_generation** - HeyGen video (polls until complete)
5. **source_video_download** - Apify downloads TikTok/Instagram
6. **video_composition** - GPU FFmpeg chromakey + overlay
7. **captioning** - Whisper transcription + ASS karaoke + GPU burn
8. **storage_upload** - Dropbox upload
9. **publishing** - Blotato multi-platform post

---

# VIRAL CLIP FACTORY - COMPLETE DEEP DIVE

The Viral Clip Factory is a sophisticated automated system for extracting viral moments from influencer videos, analyzing them with AI, and rendering them as stand-alone clips optimized for social media virality.

---

## Viral Factory Overview

### Purpose
- Monitor influencer channels (YouTube, Rumble)
- Automatically download and transcribe full-length videos
- Use Grok AI to identify viral-worthy segments (up to 20 per video)
- Render clips with TradWest-style effects: pulse zoom, color grading, karaoke captions, 9-grid outro
- Track and publish across platforms via Blotato

### System Flow
```
1. Add Influencer (YouTube/Rumble channel)
       |
       v
2. Fetch Latest Videos (yt-dlp metadata scrape)
       |
       v
3. Analyze Video (triggers Celery pipeline):
   a. Download Video (yt-dlp with VPN fallback)
   b. Transcribe (Whisper with word-level timing)
   c. Analyze (Grok 4.1 identifies 20 viral segments)
   d. Create ViralClip records
       |
       v
4. Manual Render (user selects clips to render)
       |
       v
5. GPU Rendering (video-processor service):
   - Extract clip segment
   - Apply pulse zoom effect on trigger words
   - Apply color grading (contrast +15%, saturation +25%)
   - Burn karaoke-style captions
   - Add 9-grid outro with channel handle
   - Mix background music with volume ramp
       |
       v
6. Download/Publish ready clips
```

---

## Viral Factory Database Schema

### ClipPersona Table
Defines editing style and AI prompting strategy for a clip brand.

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer | Primary key |
| `name` | String (unique) | Persona name (e.g., "Trad West Bot") |
| `description` | Text | Full description for Grok context |
| `prompt_template` | Text | Custom instructions for Grok analysis |
| `outro_style` | String | CapCut-style preset (e.g., "sunset_fade") |
| `font_style` | String | Font preset (e.g., "bold_impact") |
| `min_clip_duration` | Integer | Minimum clip length in seconds (default: 30) |
| `max_clip_duration` | Integer | Maximum clip length in seconds (default: 180) |
| `blotato_account_id` | String | Publishing account reference |

**Relationships:** One-to-many with Influencer

### Influencer Table
Represents a content creator channel to monitor.

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer | Primary key |
| `name` | String | Creator name |
| `platform` | Enum | `youtube` or `rumble` |
| `channel_id` | String | Platform-specific channel ID |
| `channel_url` | String | Full channel URL |
| `profile_image_url` | String | Avatar image URL |
| `persona_id` | Integer FK | Links to ClipPersona |

**Relationships:** Many-to-one with ClipPersona, One-to-many with InfluencerVideo

### InfluencerVideo Table
Represents a single video from an influencer's channel.

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer | Primary key |
| `influencer_id` | Integer FK | Links to Influencer |
| `platform_video_id` | String | External video ID (YouTube/Rumble) |
| `title` | String | Video title |
| `url` | String | Full video URL |
| `thumbnail_url` | String | Thumbnail image URL |
| `description` | Text | Video description |
| `publication_date` | DateTime | When video was published |
| `duration` | Integer | Video length in seconds |
| `view_count` | Integer | View count at fetch time |
| `local_path` | Text | Path to downloaded MP4 (`/downloads/video_{id}.mp4`) |
| `transcript_json` | JSONB | Whisper word-level transcript with timing |
| `analysis_json` | JSONB | Grok analysis results |
| `status` | String | Pipeline state (see below) |
| `error_message` | Text | Error details if failed |

**Status Values:**
- `pending` - Not yet processed
- `downloading` - Video download in progress
- `downloaded` - Download complete, awaiting transcription
- `transcribing (whisper)` - Transcription in progress
- `transcribed_raw` - Raw transcript saved (safety checkpoint)
- `transcribed` - Transcript processed and ready
- `analyzing (grok)` - Grok analysis in progress
- `analyzing (processing clips)` - Creating ViralClip records
- `analyzed` - Analysis complete, clips ready for rendering
- `error` - Pipeline failed

**Relationships:** Many-to-one with Influencer, One-to-many with ViralClip

### ViralClip Table
Represents a single viral segment identified by Grok.

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer | Primary key |
| `source_video_id` | Integer FK | Links to InfluencerVideo |
| `start_time` | Float | Clip start in seconds |
| `end_time` | Float | Clip end in seconds |
| `duration` | Float | Calculated `end_time - start_time` |
| `clip_type` | String | Category (see below) |
| `virality_explanation` | Text | Why Grok thinks it's viral |
| `title` | String | Generated clip title (e.g., "TOP G DESTROYS DEBATE OPPONENT") |
| `description` | Text | Generated caption for posting |
| `hashtags` | JSONB | Array of hashtags |
| `edited_video_path` | Text | Path to rendered output (`/outputs/clip_{id}_{video_id}.mp4`) |
| `status` | String | Render state (see below) |
| `error_message` | Text | Error details if failed |
| `render_metadata` | JSONB | Contains `trigger_words` array from Grok |
| `blotato_post_id` | String | Post ID after publishing |
| `published_at` | DateTime | Publication timestamp |

**Clip Types:**
- `antagonistic` - High conflict, confrontational moments
- `controversial` - Hot takes, "cancelled" moments
- `funny` - Memeable, out-of-context humor
- `inspirational` - Strong monologues, "based" takes

**Status Values:**
- `pending` - Not yet rendered
- `queued for rendering` - In Celery queue
- `rendering` - GPU processing in progress
- `ready` - Rendered and available for download/publish
- `published` - Posted via Blotato
- `error` - Rendering failed

---

## Viral Factory Celery Tasks

**File:** `backend/tasks/viral.py`

### Task Chain Architecture

```
process_viral_video_download(video_id)
        |
        | (on success)
        v
process_viral_video_transcribe(video_id)
        |
        | (on success)
        v
process_viral_video_analyze(video_id)
        |
        | Creates ViralClip records
        v
[Manual trigger via UI]
        |
        v
process_viral_clip_render(clip_id)
```

### Task 1: process_viral_video_download

**Purpose:** Download video from YouTube/Rumble using yt-dlp

**Shortcut Logic:**
1. If `status == "transcribed"` AND `transcript_json` exists AND file exists on disk -> Skip to Analysis
2. If `status == "downloaded"` AND `local_path` exists -> Skip to Transcription
3. Otherwise -> Download video

**Implementation Details:**
- Uses stable filename `video_{video_id}` for yt-dlp resume support
- Calls video-processor `/download` endpoint
- Path translation: Backend sees `/downloads/` paths from video-processor, must convert to `/app/assets/videos/` for local checks

**Error Handling:**
- Sets `status = "error"` and `error_message` on failure
- Does not retry automatically

### Task 2: process_viral_video_transcribe

**Purpose:** Transcribe video audio using Whisper (word-level timing)

**Shortcut Logic:**
- If `transcript_json` already exists -> Skip to Analysis

**Safety Checkpoint:**
- Saves raw transcript immediately after Whisper completes (before any processing)
- Sets `status = "transcribed_raw"` to prevent data loss if processing fails

**Intro Skipping Logic (Nicholas J Fuentes specific):**
1. Detects influencer by channel_id containing "nicholasjfuentes" or name containing "nicholas"
2. Searches for intro phrases:
   - Primary: "America First" + "good evening" or "watching"
   - Secondary: "We have a great show for you"
3. Finds LAST occurrence (handles multiple matches)
4. Looks ahead for name drops to extend cutoff
5. Removes ALL transcript segments before cutoff time
6. Fallback: If no intro found, treats video as starting at 0:00

**Timeout:** 1200 seconds (20 minutes) for Whisper processing

### Task 3: process_viral_video_analyze

**Purpose:** Send transcript to Grok and create ViralClip records

**Process:**
1. Calls `ClipAnalyzerService.analyze_video()`
2. Grok identifies up to 20 viral segments
3. Creates ViralClip records with:
   - Timing (start/end)
   - Type (antagonistic/controversial/funny/inspirational)
   - Title, description, hashtags
   - Trigger words for pulse effect (stored in `render_metadata`)

**Auto-Render:** DISABLED - User must manually click "Render" on each clip

### Task 4: process_viral_clip_render

**Purpose:** Render a single viral clip with effects

**Process:**
1. Filter full transcript for clip time range
2. Extract trigger words:
   - Priority 1: Grok-detected trigger words from `render_metadata`
   - Fallback: Search hardcoded `TRAD_TRIGGER_WORDS` list in transcript
3. Call video-processor `/render-viral-clip` endpoint with:
   - Video path, start/end times
   - Transcript segments for karaoke captions
   - Trigger words for pulse effect
   - Channel handle for outro
   - Status webhook URL for progress updates

**Trigger Words List:**
```python
TRAD_TRIGGER_WORDS = [
    "god", "jesus", "christ", "church", "faith", "pray", "win", "fight",
    "war", "strength", "power", "glory", "honor", "tradition", "men",
    "women", "family", "nation", "west", "save", "revive", "build",
    "create", "beauty", "truth", "justice", "freedom", "liberty",
    "order", "chaos", "evil", "good", "money", "rich", "wealth",
    "hustle", "grind", "success", "victory", "champion", "king",
    "lord", "spirit", "holy", "bible", "cross", "love", "hate",
    "fear", "death", "life", "soul", "mind", "heart", "body",
    "blood", "sweat", "tears", "pain", "gain", "gym", "train",
    "work", "hard", "focus", "discipline"
]
```

---

## Clip Analyzer Service

**File:** `backend/services/clip_analyzer.py`

### ClipAnalyzerService.analyze_video()

**Input:** video_id, db_session

**Process:**
1. Fetch video and transcript from DB
2. Get associated ClipPersona (or fallback to first available)
3. Check for `VIRAL_SYSTEM_PROMPT` override in LLMSettings
4. Build analysis prompt with:
   - Persona context
   - Video title and duration
   - Full transcript segments with timestamps
   - Clip type definitions
   - Duration constraints from persona
5. Call Grok via OpenRouter API
6. Parse JSON response
7. Create ViralClip records

### Grok Prompt Structure

```
You are an expert viral content editor acting as '{persona.name}'.
{persona.description}

Your goal is to identify the MOST viral segments from this video
transcript to repurpose for TikTok/Reels/Shorts.

Video Title: {video.title}
Duration: {video.duration}s

TRANSCRIPT SEGMENTS (with timestamps):
[{start, end, text}, ...]

TASK:
Identify up to 20 distinct clips (prioritizing quality and viral potential):
- OUTRAGEOUS / CONTROVERSIAL (Priority #1 - Hot Takes, "Cancelled" moments)
- Antagonistic / Conflict (High emotion battles)
- Funny / Memeable (Out of context or hilarious)
- Inspirational / "Based" (Strong monologues)

FOCUS ON THE MOST CONTROVERSIAL AND SHOCKING MOMENTS. DO NOT HOLD BACK.

Also identify 3-5 high-intensity SINGLE WORDS (triggers) within the clip
for visual impact (e.g., WAR, DIE, TRUMP, MONEY, LIAR).

Constraints:
- Clips should be between {min_duration} and {max_duration} seconds.
- Ensure start and end times cut cleanly (complete sentences).

Return ONLY valid JSON:
{
  "clips": [
    {
      "start": 10.5,
      "end": 45.2,
      "type": "antagonistic",
      "title": "TOP G DESTROYS DEBATE OPPONENT",
      "reason": "High conflict moment, very engaging",
      "caption": "Bro didn't stand a chance...",
      "hashtags": ["#viral", "#shorts", "#fyp"],
      "trigger_words": [
        {"word": "DESTROYED", "start": 30.5, "end": 31.0},
        {"word": "LIAR", "start": 40.2, "end": 40.8}
      ]
    }
  ]
}
```

### OpenRouter API Call

- **Endpoint:** `https://openrouter.ai/api/v1/chat/completions`
- **Model:** `x-ai/grok-4.1-fast`
- **Response Format:** JSON object
- **Temperature:** 0.7
- **Timeout:** 120 seconds

---

## B-Roll Category System

B-roll clips are matched to viral content using a semantic category system. Grok acts as a "movie director" and assigns B-roll insertions with specific categories.

### Valid B-Roll Categories (28 total)

Grok can ONLY use these category names. Any other category name will fail to match:

| Group | Categories |
|-------|------------|
| **Destruction & Conflict** | `war`, `chaos`, `explosions`, `storms`, `fire` |
| **Money & Success** | `money`, `luxury`, `wealth`, `city` |
| **Fitness & Strength** | `gym`, `sports`, `boxing`, `strength` |
| **Patriotic & Faith** | `patriotic`, `crowd`, `faith`, `cathedrals` |
| **Animals & Nature** | `lions`, `eagles`, `wolves`, `nature` |
| **Military Tech** | `jets`, `navy`, `helicopters` |
| **Vehicles** | `cars`, `racing` |
| **Other** | `history`, `people`, `victory`, `power` |

### B-Roll Sources

Each B-roll insertion can come from:
- **`source: "local"`** - Matches from `assets/broll/` using category + filename prefixes
- **`source: "youtube"`** - Grok provides search query, system fetches YouTube video and extracts face-free segments

### B-Roll Selection Flow

```
1. Grok returns broll_insertions: [{time, category, source, visual}]
2. For local sources:
   a. Check AI-tagged metadata.json for category matches
   b. Fallback: filename prefix matching (e.g., "warfare_*" for "war")
3. For YouTube sources:
   a. Fetch YouTube transcript for context
   b. Second Grok call picks timestamps from transcript
   c. Download and extract clips at selected timestamps
4. Clips inserted into render pipeline
```

### AI Tagging System

Local B-roll clips can be tagged using BLIP (captioning) + CLIP (classification):

```bash
# Run AI tagging on video-processor container
docker exec -it SocialGen_video_processor python /app/scripts/tag_broll.py

# Force re-tag all clips
docker exec -it SocialGen_video_processor python /app/scripts/tag_broll.py --force
```

Output: `assets/broll/metadata.json` with categories, captions, and confidence scores.

### Key Files

| File | Purpose |
|------|---------|
| `backend/services/clip_analyzer.py` | Grok prompt with B-roll category list |
| `backend/services/pexels.py` | `category_to_prefix` fallback mapping |
| `scripts/tag_broll.py` | AI tagging script (BLIP + CLIP) |
| `assets/broll/metadata.json` | AI-generated clip metadata |

---

## Video Processor Render Endpoint

**File:** `video-downloader/main.py`
**Endpoint:** `POST /render-viral-clip`

### Request Schema (RenderClipRequest)

```python
class RenderClipRequest(BaseModel):
    video_path: str              # Path to source video
    start_time: float            # Clip start in seconds
    end_time: float              # Clip end in seconds
    transcript_segments: list    # Word-level transcript for captions
    style_preset: str = "trad_west"  # Visual style
    font: str = "Arial"          # Caption font
    output_filename: str         # Output file name
    outro_path: str              # Path to outro video (optional)
    channel_handle: str          # Handle for outro text
    trigger_words: list          # Words to trigger pulse effect
    status_webhook_url: str      # Backend URL for status updates
```

### Rendering Pipeline (V7.0)

```
1. SETUP
   - Calculate duration
   - Select random background music from /music

2. KARAOKE ASS GENERATION
   - Generate ASS subtitle file with \k timing tags
   - Orange primary color (#FF6500)
   - Word-by-word highlight animation

3. MEGA-FILTER CHAIN (GPU + CPU Hybrid)
   [0:v] hwdownload,format=nv12,format=yuv420p
     -> scale=-2:1920 (fill height)
     -> scale=zoom_expr (pulse effect)
     -> crop=1080:1920 (center crop with pan)
     -> eq=contrast=1.15:brightness=0.03:saturation=1.25 (color grade)
     -> ass='{subtitle_path}' (karaoke captions)
     -> hwupload_cuda (back to GPU)
   [v] -> h264_nvenc encoder

4. PULSE EFFECT FORMULA
   zoom_base = min(1+0.001*t, 1.5)  # Slow growth
   heartbeat = 0.003*sin(2*PI*t*1.5)  # 1.5Hz pulse
   trigger_sum = SUM(0.20 * between(t, start, end))  # +20% on trigger words

   Final scale = zoom_base + heartbeat + trigger_sum

5. OUTRO GENERATION (if duration > 5s)
   - Extract last 2 seconds
   - Create 9-grid split effect (3x3 with zoom variations)
   - Overlay purple cross symbol
   - Add channel handle text
   - Fade out

6. CONCAT
   - Main clip (trimmed to remove last 2s)
   - + 9-grid outro

7. BGM MIXING
   - Loop background music to clip length
   - Volume: 0.3 normally, ramps to 0.8 in final 5 seconds
   - Mix with original audio
```

### Output
- **Path:** `/outputs/clip_{clip_id}_{video_id}.mp4`
- **Resolution:** 1080x1920 (9:16 vertical)
- **Codec:** H.264 (NVENC if available, libx264 fallback)
- **Audio:** AAC

---

## Viral Factory API Endpoints

**File:** `backend/routers/viral.py`
**Base URL:** `/api/viral`

### Influencer Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/influencers` | List all influencers |
| POST | `/influencers` | Create new influencer |
| POST | `/influencers/{id}/fetch` | Fetch latest videos from channel |

### Video Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/videos/{id}/details` | Get video metadata |
| POST | `/videos/{id}/analyze` | Trigger analysis pipeline (Download -> Transcribe -> Analyze) |
| GET | `/influencers/{id}/videos` | List videos for influencer (includes nested clips) |

### Clip Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/viral-clips` | List all clips (50 most recent) |
| POST | `/viral-clips/{id}/render` | Trigger clip rendering |
| PUT | `/viral-clips/{id}/status` | Update clip status (used by webhook) |
| GET | `/file/{filename}` | Download rendered clip file |

### Music Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/music` | List available background music |
| GET | `/music/{filename}` | Download music file |

---

## Viral Factory Frontend

**File:** `frontend/src/pages/ViralManager.js`

### Component Structure

```jsx
<ViralManager>
  <Header>
    <Tabs: Influencers | Videos | Clips>
  </Header>

  <Content>
    {activeTab === 'influencers' && <InfluencersPanel />}
    {activeTab === 'videos' && <VideosPanel />}
    {activeTab === 'clips' && <ClipsPanel />}
  </Content>

  <AddInfluencerModal />
  <VideoPlayerModal />
</ViralManager>
```

### Influencers Tab
- Display grid of influencer cards
- Click card to select and view videos
- "+ Add Influencer" button opens modal
- Form fields: Name, Platform (YouTube/Rumble), Channel URL, Persona ID

### Videos Tab
- "Fetch Latest Videos" button - calls video-processor to scrape channel
- Video list with:
  - Thumbnail (clickable to open in new tab)
  - Title, duration, status badge
  - "Auto-Generate Clips" button (triggers full pipeline)
  - "Re-Generate Clips" button (if already analyzed)
- Nested clips list for each video:
  - Clip type, title, status
  - "Play" button (opens video modal for ready clips)
  - "Render" button (for pending clips)
  - "Retry" button (for failed clips)

### Clips Tab
- Grid of all viral clips
- Shows: title, type, status
- "Render" button for pending clips
- "Download" button for ready clips

### Auto-Refresh Polling
- Detects active tasks (downloading, transcribing, analyzing, rendering)
- Refreshes video list every 3 seconds while tasks are running
- Stops polling when no active tasks

### Video Player Modal
- Opens when clicking "Play" on a ready clip
- Plays video from `{API_URL}/api/viral/file/{filename}`
- Close button in header

---

## Viral Factory API Functions (Frontend)

**File:** `frontend/src/api.js`

```javascript
// Influencer Management
getInfluencers()              // GET /api/viral/influencers
createInfluencer(data)        // POST /api/viral/influencers
fetchInfluencerVideos(id)     // POST /api/viral/influencers/{id}/fetch

// Video Management
getVideoDetails(id)           // GET /api/viral/videos/{id}/details
analyzeVideo(id)              // POST /api/viral/videos/{id}/analyze
getInfluencerVideos(id)       // GET /api/viral/influencers/{id}/videos

// Clip Management
getViralClips()               // GET /api/viral/viral-clips
renderClip(id)                // POST /api/viral/viral-clips/{id}/render

// Music
getViralMusic()               // GET /api/viral/music
```

---

## Viral Factory Path Mapping

### Container Path Translation

The backend and video-processor have different mount points for the same files:

| Host Path | Backend Container | Video-Processor Container |
|-----------|------------------|---------------------------|
| `./assets/videos/` | `/app/assets/videos/` | `/downloads/` |
| `./assets/output/` | `/app/assets/output/` | `/outputs/` |
| `./assets/audio/` | `/app/assets/audio/` | `/audio/` |
| `./assets/avatar/` | `/app/assets/avatar/` | `/avatar/` |
| `./assets/music/` | `/app/assets/music/` | `/music/` |

**Translation in tasks/viral.py:**
```python
# Video-processor returns paths like /downloads/video_123.mp4
# Backend must check /app/assets/videos/video_123.mp4
if path.startswith("/downloads/"):
    path = path.replace("/downloads/", "/app/assets/videos/")
```

### File Naming Conventions

| Asset Type | Pattern | Example |
|------------|---------|---------|
| Downloaded Video | `video_{video_id}.mp4` | `video_123.mp4` |
| Rendered Clip | `clip_{clip_id}_{video_id}.mp4` | `clip_456_123.mp4` |
| Temp Files | `temp_{uuid}.{ext}` | `temp_abc123.wav` |

---

## Viral Factory Configuration

### LLM Settings Override

Two settings can be stored in `llm_settings` table:

| Key | Purpose |
|-----|---------|
| `VIRAL_SYSTEM_PROMPT` | Override the default Grok system prompt |
| `VIRAL_CHANNEL_HANDLE` | Channel handle for outro text (default: "TheRealClipFactory") |

### ClipPersona Settings

Each persona can customize:
- `prompt_template` - Additional instructions for Grok
- `min_clip_duration` / `max_clip_duration` - Clip length constraints
- `outro_style` - Visual style preset
- `font_style` - Caption font preset

---

## Viral Factory Debugging

### Pipeline Not Progressing

```bash
# Check Celery worker status
docker logs -f SocialGen_celery_worker

# Check Redis queue length
docker exec SocialGen_redis redis-cli LLEN celery
```

### Video Stuck in Downloading

```bash
# Check if yt-dlp process is running
docker exec SocialGen_video_processor ps aux | grep yt-dlp

# Check video-processor logs
docker logs -f SocialGen_video_processor

# Verify VPN is connected (needed for some sources)
docker logs SocialGen_vpn | tail -20
```

### Transcription Failed

```bash
# Check Whisper GPU status
docker exec SocialGen_video_processor nvidia-smi

# Check if video file exists
docker exec SocialGen_video_processor ls -la /downloads/video_{id}.mp4
```

### Clip Not Rendering

```bash
# Check render logs
docker logs SocialGen_video_processor 2>&1 | grep -A 20 "render-viral-clip"

# Check clip status in DB
docker exec -it SocialGen_postgres psql -U n8n -d content_pipeline \
  -c "SELECT id, status, error_message FROM viral_clips WHERE id = {clip_id};"
```

### Manual Status Reset

```bash
# Reset video to pending
docker exec -it SocialGen_postgres psql -U n8n -d content_pipeline \
  -c "UPDATE influencer_videos SET status = 'pending', error_message = NULL WHERE id = {video_id};"

# Reset clip to pending
docker exec -it SocialGen_postgres psql -U n8n -d content_pipeline \
  -c "UPDATE viral_clips SET status = 'pending', error_message = NULL WHERE id = {clip_id};"
```

---

## Test Data

### Main Pipeline
- **Test Content Idea:** 141, 186
- **Test Script:** 22, 33
- **Test Voice:** `./assets/audio/22_voice.mp3`
- **Test Avatar:** `./assets/avatar/22_avatar.mp4`

### Viral Factory
- Query existing influencers: `SELECT * FROM influencers;`
- Query analyzed videos: `SELECT * FROM influencer_videos WHERE status = 'analyzed';`
- Query ready clips: `SELECT * FROM viral_clips WHERE status = 'ready';`

---

## File Locations

| Type | Path |
|------|------|
| **Main Pipeline** | |
| Pipeline task | `backend/tasks/pipeline.py` |
| Video service | `backend/services/video.py` |
| Caption service | `backend/services/captions.py` |
| Avatar service | `backend/services/avatar.py` |
| GPU processor | `video-downloader/main.py` |
| Asset paths | `backend/utils/paths.py` |
| Settings API | `backend/routers/settings.py` |
| **Viral Factory** | |
| Viral tasks | `backend/tasks/viral.py` |
| Clip analyzer | `backend/services/clip_analyzer.py` |
| Viral API | `backend/routers/viral.py` |
| Viral frontend | `frontend/src/pages/ViralManager.js` |
| Database models | `backend/models.py` (ClipPersona, Influencer, InfluencerVideo, ViralClip) |

---

## Cost Estimates

### Main Pipeline
| Service | Cost |
|---------|------|
| HeyGen | ~$1-3/video |
| ElevenLabs | ~$0.01-0.05/voice |
| Apify | ~$0.30-1.50/scrape |
| OpenRouter | ~$0.01/script |
| **Total per video** | **~$2-5** |

### Viral Factory
| Service | Cost |
|---------|------|
| OpenRouter (Grok) | ~$0.02-0.05/analysis |
| GPU Time | Free (local DGX) |
| Storage | Free (local) |
| **Total per video analyzed** | **~$0.02-0.05** |

---

## External Integrations

### Main Pipeline
- **HeyGen** - Avatar video generation
- **ElevenLabs** - Text-to-speech
- **Apify** - TikTok/Instagram scraping
- **Dropbox** - Video storage
- **Blotato** - Multi-platform publishing
- **Pexels** - Stock footage

### Viral Factory
- **OpenRouter (Grok 4.1)** - Viral segment detection
- **Whisper (local)** - Transcription
- **yt-dlp** - YouTube/Rumble downloading
- **FFmpeg (NVENC)** - GPU video rendering
- **NordVPN (Gluetun)** - VPN for blocked sources

---

*Last updated: January 26, 2026 - Added B-roll category system (28 categories, AI tagging)*
