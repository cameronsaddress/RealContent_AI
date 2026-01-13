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

### 1. Asset Reuse - NEVER Skip Existence Checks

The pipeline checks for existing assets at each stage to avoid redundant API calls:

| Stage | Check Method | File Path |
|-------|--------------|-----------|
| Script | DB query | `scripts.content_idea_id` |
| Voice | `voice_exists()` | `audio/{id}_voice.mp3` |
| Avatar | `avatar_exists()` | `avatar/{id}_avatar.mp4` |
| Source | `source_video_exists()` | `videos/{id}_source.mp4` |
| Combined | `combined_video_exists()` | `output/{id}_combined.mp4` |
| Final | `final_video_exists()` | `output/{id}_final.mp4` |

**To force regeneration:** Delete the file, don't modify the check logic.

### 2. Settings Tables - Two Tables, Keep in Sync

There are TWO settings tables that MUST stay synchronized:
- `video_settings` - API endpoint reads/writes here
- `system_settings` - Pipeline reads from here

When updating settings via API:
```bash
# API updates video_settings
curl -X PUT http://100.83.153.43:8000/api/settings/video \
  -H "Content-Type: application/json" \
  -d '{"avatar_offset_x": -250}'
```

If settings aren't applying, check `system_settings`:
```sql
SELECT value FROM system_settings WHERE key = 'video_settings';
```

### 3. GPU Video Processing - Use video-processor Service

Video encoding runs on the GPU via the `video-processor` container:

```bash
# Compose (chromakey + overlay)
POST http://video-processor:8080/compose

# Caption burning
POST http://video-processor:8080/caption
```

**NEVER** run FFmpeg in `celery-worker` for video encoding. The pipeline calls the video-processor service via HTTP.

### 4. Volume Mounts in video-processor

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

### 5. Karaoke Captions - ASS Format

Captions use ASS format with `\kf` (karaoke fill) effect:
- Primary: White (`&H00FFFFFF`)
- Secondary: Yellow (`&H0000FFFF`) - fills as word is spoken
- Alignment: 5 (center)
- MarginV: 960 (center of 1920px screen)

```ass
Dialogue: 0,0:00:00.00,0:00:01.51,Default,,0,0,0,,{\kf30}Hey {\kf30}neighbors
```

### 6. Avatar Positioning

Avatar position is controlled by:
- `avatar_scale`: 0.75 (75% of original size)
- `avatar_offset_x`: -250 (negative = left of center)
- `avatar_offset_y`: 600 (positive = down from top)

FFmpeg overlay formula:
```
X: 10 + avatar_offset_x
Y: (H-h-10) + avatar_offset_y
```

### 7. Location Branding

Use "North Idaho & Spokane area" - NOT "Coeur d'Alene". Files updated:
- `schemas.py`
- `models.py`
- `script_generator.py`
- `publisher.py`

### 8. TikTok Downloads Require Browser Headers

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

## Pipeline Stages

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

## Test Data

- **Test Content Idea:** 141, 186
- **Test Script:** 22, 33
- **Test Voice:** `./assets/audio/22_voice.mp3`
- **Test Avatar:** `./assets/avatar/22_avatar.mp4`

---

## Debugging Checklist

### Pipeline Not Running
```bash
docker logs SocialGen_celery_worker 2>&1 | grep "ready"
docker exec SocialGen_redis redis-cli LLEN celery
```

### Settings Not Applied
1. Check `video_settings` table (API writes here)
2. Check `system_settings` table (pipeline reads here)
3. Restart celery-worker after manual DB updates

### Video Position Wrong
1. Delete `output/{id}_combined.mp4` and `output/{id}_final.mp4`
2. Update settings: `avatar_offset_x`, `avatar_offset_y`
3. Re-run pipeline

### GPU Not Working
1. Check `docker inspect SocialGen_video_processor | jq '.[0].HostConfig.DeviceRequests'`
2. Rebuild: `docker compose up -d --build video-processor`
3. Test: `curl http://localhost:8080/health`

---

## File Locations

| Type | Path |
|------|------|
| Pipeline task | `backend/tasks/pipeline.py` |
| Video service | `backend/services/video.py` |
| Caption service | `backend/services/captions.py` |
| Avatar service | `backend/services/avatar.py` |
| GPU processor | `video-downloader/main.py` |
| Asset paths | `backend/utils/paths.py` |
| Settings API | `backend/routers/settings.py` |

---

## Cost Estimates

| Service | Cost |
|---------|------|
| HeyGen | ~$1-3/video |
| ElevenLabs | ~$0.01-0.05/voice |
| Apify | ~$0.30-1.50/scrape |
| OpenRouter | ~$0.01/script |
| **Total per video** | **~$2-5** |

---

*Last updated: January 13, 2026 - GPU encoding, karaoke captions, video-processor service*
