# IMPLEMENATION PLAN: Viral Clip Factory

## Overview
This plan details the implementation of a new "Viral Clip Factory" pipeline. This pipeline is distinct from the existing Real Estate pipeline. Its purpose is to ingest long-form content from specific influencers (Rumble/YouTube), analyze it for viral moments (antagonistic, funny, controversial), and generate edited short clips with "CapCut-style" effects and a specific ("Trad West") aesthetic.

## 1. Database Schema Extensions (`backend/models.py`)

We will add the following tables to `backend/models.py`:

### `Influencer`
*   `id`: Integer, PK
*   `name`: String
*   `platform`: Enum (youtube, rumble)
*   `channel_id`: String (YouTube channel ID or Rumble user name)
*   `channel_url`: String
*   `persona_id`: FK to `ClipPersona`
*   `created_at`: DateTime

### `InfluencerVideo`
*   `id`: Integer, PK
*   `influencer_id`: FK to `Influencer`
*   `platform_video_id`: String
*   `title`: String
*   `url`: String
*   `thumbnail_url`: String
*   `publication_date`: DateTime
*   `duration`: Integer (seconds)
*   `local_path`: String (path to downloaded .mp4)
*   `transcript_json`: Text/JSONB (Whisper word-level output)
*   `analysis_json`: Text/JSONB (Grok analysis of viral moments)
*   `status`: Enum (pending, downloaded, transcribed, analyzed, error)

### `ViralClip`
*   `id`: Integer, PK
*   `source_video_id`: FK to `InfluencerVideo`
*   `start_time`: Float
*   `end_time`: Float
*   `clip_type`: Enum (antagonistic, funny, controversial, inspirational, etc.)
*   `virality_explanation`: Text (Grok's reasoning)
*   `suggested_caption`: Text
*   `edited_video_path`: String (path to final output)
*   `status`: Enum (pending, rendering, ready, published, error)
*   `blotato_post_id`: String

### `ClipPersona`
*   `id`: Integer, PK
*   `name`: String (e.g., "Trad West Bot")
*   `prompt_template`: Text (Instructions for Grok analysis)
*   `outro_style`: String (e.g., "sunset_fade")
*   `blotato_account_id`: String (Separate account for this pipeline)

## 2. Infrastructure Updates

### Video Processor (`video-downloader/Dockerfile`)
*   Add `openai-whisper` (or `faster-whisper` for GPU speed).
*   Add `moviepy` (optional, for complex effects if FFmpeg filters get too complex, but largely stick to FFmpeg).
*   **Action**: Update `Dockerfile` and rebuild.

### Video Processor API (`video-downloader/main.py`)
*   POST `/transcribe_precise`: Run Whisper with `word_timestamps=True`.
*   POST `/render_viral_clip`:
    *   Input: `video_path`, `start`, `end`, `overlay_text`, `outro_path`.
    *   Logic: MoviePy Base (Crop/Color/Zoom/Outro/Handle) -> FFmpeg Burn ASS Captions (Karaoke).
    *   **New**: `generate_karaoke_ass(segments)` function to build `.ass` file with word-level highlighting.
    *   **New**: MoviePy `clip.resize(lambda t: 1 + 0.02*t)` for slow zoom.
    *   Input: `video_path`, `start`, `end`, `overlay_text`, `outro_path`.
    *   Logic: FFmpeg trim -> Apply filters (color grading) -> Overlay text/images -> Burn Captions -> CONCAT Outro.
    *   **New**: Concatenation requires standardizing resolution/FPS of the outro to match the clip.

## 3. Backend Services (`backend/services/`)

### `services/influencer_mgr.py`
*   `fetch_channel_videos(url)`: Uses `yt-dlp` to list recent 10 videos.
*   `download_video(url)`: Uses `yt-dlp` to download to `assets/influencer_source/`.
*   `tasks/viral.py`: Update `_render_clip_async` to preserve `words` list in segments passed to render.

### `services/clip_analyzer.py` (Grok Integration)
*   `analyze_transcript(transcript_json, persona)`: Sends transcript to Grok.
*   **Prompt Strategy**: "Identify the 3 most viral segments. Criteria: most antagonistic, most vulgar, or most funny. Return start/end times and reasoning."

### `services/clip_editor.py`
*   Orchestrates calls to `video-processor`.
*   Manages asset paths.

## 4. Frontend Routes (`frontend/src/`)

*   `/influencers`: List/Add influencers.
*   `/influencers/:id`: View 10 recent videos + status.
*   `/influencer_videos/:id`: View video details, trigger "Analyze", view/generate clips.
*   `/clips`: List generated viral clips + Publish button.
*   `/settings`: Configure Viral Clips settings (Default Length, Outro Path).

## 5. Execution Steps

1.  **Models**: Update `backend/models.py` and run alembic/db migration (or `alembic revision --autogenerate`).
2.  **Tasks**: Update `tasks/viral.py` to forward word timestamps.
3.  **Renderer**: Implement `generate_karaoke_ass` and MoviePy Zoom.
4.  **Settings**: Seed `VIRAL_SYSTEM_PROMPT` in `backend/main.py` startup.
2.  **Docker**: Update `video-downloader/Dockerfile` to include Whisper.
3.  **Backend Logic**: Implement `yt-dlp` fetcher and `InfluencerService`.
4.  **Processor API**: Implement Whisper endpoint and basic Clip Rendering endpoint.
5.  **API Routes**: Add FastAPI routers for `/api/influencers`, `/api/viral-clips`.
6.  **Frontend**: Build the UI.

## 6. Verification
*   Test flow: Add Rumble channel -> List Videos -> Download -> Transcribe -> Analyze (Mock/Real) -> Render -> Verify Output.
