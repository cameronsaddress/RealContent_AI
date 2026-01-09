# Implementation Plan V2: Video Pipeline Enhancements

**Created:** 2026-01-09
**Backup:** `workflows/COMPLETE_PIPELINE_V2_20260109_091834.json`

---

## Overview

This plan addresses major enhancements to the AI Video Content Pipeline:

1. **Video Compositing Fix** - Avatar overlay on source social video (no more B-roll)
2. **Audio Mixing** - Dual audio tracks with independent volume control + optional ducking
3. **Video Quality** - Force 1080x1920 output (currently outputting 480p)
4. **UI Enhancements** - Script editing, Settings page with LLM prompts, audio config, timing controls

---

## Current State Analysis

### What Works
- ElevenLabs voice generation (creates `{script_id}_voice.mp3`)
- Source video download from social platforms (creates `{script_id}_source.mp4`)
- Whisper transcription and SRT generation
- Dropbox upload for final video hosting
- Basic FFmpeg pipeline exists

### What's Broken/Missing
- **Avatar compositing**: Test avatar (78KB) has no green screen - chromakey fails silently
- **Source video not used**: `broll_paths` is empty, so FFmpeg uses solid color background
- **Audio OVERWRITTEN**: Original video audio completely replaced, not mixed
- **Video quality**: Output is 480p, needs to be 1080x1920
- **Captions**: Font size too large (FontSize=18 still huge on 1080x1920)
- **UI**: No script editing workflow, no settings page for LLM prompts or audio config

---

## Part 1: Video Compositing Architecture

### Target Output Structure
```
┌─────────────────────────────────────────┐
│                                         │
│     [CAPTIONS - Small, Top Center]      │  ← TikTok-style captions
│                                         │
│                                         │
│                                         │
│      ORIGINAL SOCIAL VIDEO              │  ← Downloaded from TikTok/IG/etc
│      (Full 1080x1920 background)        │     MUST be 1080x1920 output
│                                         │
│                                         │
│                                         │
│  ┌─────────────────────────────────────┐│
│  │                                     ││
│  │   AI AVATAR (Green Screen)         ││  ← HeyGen talking photo/video
│  │   Speaking ElevenLabs audio        ││     Chromakeyed, bottom 30-40%
│  │                                     ││
│  └─────────────────────────────────────┘│
└─────────────────────────────────────────┘

Resolution: 1080x1920 (9:16 vertical, 1080p quality)
```

### Audio Timeline (with Ducking Option)
```
Option A: Ducking ENABLED (default)
0s                    3s                                          End
├─────────────────────┼───────────────────────────────────────────┤
│ Original: 100%      │ Original: 50% (ducked)                    │
│ Avatar: 0%          │ Avatar: 100%                              │
│ (Avatar not visible)│ (Avatar appears and speaks)               │
└─────────────────────┴───────────────────────────────────────────┘

Option B: Ducking DISABLED (both play at configured volumes)
0s                                                                End
├─────────────────────────────────────────────────────────────────┤
│ Original: [configured volume, e.g. 30%]                         │
│ Avatar: [configured volume, e.g. 100%]                          │
│ (Both audio tracks play simultaneously at set levels)           │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow (Fixed)

```
Download Source Video
        │
        ▼
   source_video_path: /assets/videos/{script_id}_source.mp4
        │
        ├──────────────────────────────────────────────────┐
        ▼                                                  │
   ElevenLabs TTS                                          │
        │                                                  │
        ▼                                                  │
   voice_path: /assets/audio/{script_id}_voice.mp3         │
        │                                                  │
        ▼                                                  │
   HeyGen (with voice audio)                               │
        │                                                  │
        ▼                                                  │
   avatar_path: /assets/avatar/{script_id}_avatar.mp4      │
        │                                                  │
        ▼                                                  │
   FFmpeg Compose ◄────────────────────────────────────────┘
        │
        │  Inputs:
        │    - source_video_path (background)
        │    - avatar_path (overlay, chromakey green)
        │    - Audio config from Settings API:
        │      - original_volume (0-100%)
        │      - avatar_volume (0-100%)
        │      - ducking_enabled (boolean)
        │      - avatar_delay_seconds (default 3)
        │      - duck_to_percent (default 50%)
        │
        ▼
   combined_path: /assets/output/{script_id}_combined.mp4
        │                (1080x1920 enforced)
        ▼
   Whisper Transcribe
        │
        ▼
   FFmpeg Burn Captions
        │
        ▼
   final_path: /assets/output/{script_id}_final.mp4
        │                (1080x1920 enforced)
```

---

## Part 2: Settings Page Design

### Settings Page Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  SETTINGS                                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ AUDIO SETTINGS                                          │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                         │   │
│  │  Original Video Audio Volume                            │   │
│  │  [━━━━━━━━━━━━━━━━●━━━━━] 70%                           │   │
│  │                                                         │   │
│  │  Avatar/Voiceover Volume                                │   │
│  │  [━━━━━━━━━━━━━━━━━━━━●] 100%                           │   │
│  │                                                         │   │
│  │  ☑ Enable Audio Ducking                                 │   │
│  │    (Lower original audio when avatar speaks)            │   │
│  │                                                         │   │
│  │  ┌─ Ducking Settings (when enabled) ─────────────────┐  │   │
│  │  │                                                   │  │   │
│  │  │  Avatar Appears After: [  3  ] seconds            │  │   │
│  │  │                                                   │  │   │
│  │  │  Duck Original Audio To: [━━━━━●━━━━] 50%         │  │   │
│  │  │                                                   │  │   │
│  │  └───────────────────────────────────────────────────┘  │   │
│  │                                                         │   │
│  │  [ Save Audio Settings ]                                │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ VIDEO SETTINGS                                          │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                         │   │
│  │  Output Resolution: ( ) 720p  (●) 1080p  ( ) 4K        │   │
│  │                                                         │   │
│  │  Output Format: [  MP4 (H.264)  ▼]                      │   │
│  │                                                         │   │
│  │  [ Save Video Settings ]                                │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ LLM PROMPT SETTINGS                                     │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                         │   │
│  │  Script Generation System Prompt                        │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │ You are a viral content scriptwriter for...     │   │   │
│  │  │ ...                                             │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │  [ Edit ]                                               │   │
│  │                                                         │   │
│  │  Trend Analysis Prompt                                  │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │ Analyze this content for virality potential...  │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │  [ Edit ]                                               │   │
│  │                                                         │   │
│  │  Hook Generation Prompt                                 │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │ Create a 3-second attention-grabbing hook...    │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │  [ Edit ]                                               │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 3: Implementation Tasks

### Phase 1: Fix Data Flow (n8n Workflow)

#### Task 1.1: Update "Use Existing Avatar" Node
**File:** n8n workflow (node id: `mock-heygen-data`)

Current code doesn't pass source video path. Fix:

```javascript
// Add source_video_path to output
const sourceVideoPath = `/home/node/.n8n-files/assets/videos/${scriptId}_source.mp4`;

return [{
  json: {
    asset_id: inputData.asset_id,
    script_id: scriptId,
    content_idea_id: inputData.content_idea_id,
    avatar_path: `/home/node/.n8n-files/assets/avatar/${scriptId}_avatar.mp4`,
    source_video_path: sourceVideoPath,  // NEW
    voice_path: inputData.voice_path || `/home/node/.n8n-files/assets/audio/${scriptId}_voice.mp3`,  // NEW
    heygen_video_id: 'test-mock-id',
    retry_count: 0,
    max_retries: 20
  }
}];
```

#### Task 1.2: Update "Build FFmpeg Command" Node
**File:** n8n workflow (node id: `build-ffmpeg-cmd`)

Complete rewrite with:
- 1080x1920 output enforced
- Dual audio track support
- Configurable ducking

```javascript
const fs = require('fs');
const https = require('https');

const items = $input.all();
if (!items || items.length === 0) {
  return [{ json: { error: 'No input data', success: false } }];
}

const sourceData = items[0].json;
const scriptId = sourceData.script_id;

// Fetch audio settings from API (or use defaults)
// In production, this would call: GET http://backend:8000/api/settings/audio
const audioSettings = sourceData.audio_settings || {
  original_volume: 0.7,           // 70% volume for original
  avatar_volume: 1.0,             // 100% volume for avatar
  ducking_enabled: true,          // Enable ducking by default
  avatar_delay_seconds: 3,        // Avatar appears after 3 seconds
  duck_to_percent: 0.5            // Duck original to 50% when avatar speaks
};

// Video settings
const videoSettings = sourceData.video_settings || {
  output_width: 1080,
  output_height: 1920,
  crf: 18,                        // Higher quality (lower = better, 18 is very good)
  preset: 'slow'                  // Slower = better quality
};

// --- FILE PATHS ---
const avatarPath = sourceData.avatar_path || `/home/node/.n8n-files/assets/avatar/${scriptId}_avatar.mp4`;
const sourcePath = sourceData.source_video_path || `/home/node/.n8n-files/assets/videos/${scriptId}_source.mp4`;
const outputPath = `/home/node/.n8n-files/assets/output/${scriptId}_combined.mp4`;

// Check files exist
const avatarExists = fs.existsSync(avatarPath);
const sourceExists = fs.existsSync(sourcePath);

if (!avatarExists) {
  return [{ json: { error: `Avatar not found: ${avatarPath}`, success: false } }];
}
if (!sourceExists) {
  return [{ json: { error: `Source video not found: ${sourcePath}`, success: false } }];
}

// --- BUILD INPUTS ---
const inputs = `-i "${sourcePath}" -i "${avatarPath}"`;

// --- BUILD FILTER COMPLEX ---
let fc = '';

// 1. VIDEO PROCESSING
// Scale source to exact 1080x1920 (force aspect ratio, crop if needed)
fc += `[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1[bg];`;

// Scale avatar to width 1080, maintain aspect ratio, then chromakey
fc += `[1:v]scale=1080:-1,chromakey=0x00FF00:0.1:0.2[avatar_keyed];`;

// Overlay avatar at bottom of frame (y = H - h means bottom edge touches bottom)
fc += `[bg][avatar_keyed]overlay=(W-w)/2:H-h:shortest=1[outv];`;

// 2. AUDIO PROCESSING
const origVol = audioSettings.original_volume;
const avatarVol = audioSettings.avatar_volume;
const duckEnabled = audioSettings.ducking_enabled;
const avatarDelay = audioSettings.avatar_delay_seconds;
const duckTo = audioSettings.duck_to_percent;

if (duckEnabled) {
  // Ducking mode: original at full volume, then ducks when avatar starts
  // Original audio: full volume for avatarDelay seconds, then fade to duck level
  fc += `[0:a]volume=${origVol}:enable='lt(t,${avatarDelay})',volume=${origVol * duckTo}:enable='gte(t,${avatarDelay})'[orig_ducked];`;

  // Avatar audio: delayed by avatarDelay seconds
  const delayMs = avatarDelay * 1000;
  fc += `[1:a]adelay=${delayMs}|${delayMs},volume=${avatarVol}[avatar_delayed];`;

  // Mix both tracks
  fc += `[orig_ducked][avatar_delayed]amix=inputs=2:duration=longest:dropout_transition=0,volume=2[outa]`;
} else {
  // No ducking: both tracks play at their configured volumes throughout
  fc += `[0:a]volume=${origVol}[orig_vol];`;
  fc += `[1:a]volume=${avatarVol}[avatar_vol];`;
  fc += `[orig_vol][avatar_vol]amix=inputs=2:duration=longest:dropout_transition=0,volume=2[outa]`;
}

// --- FINAL COMMAND ---
// Using high quality settings for 1080p output
const ffmpegCmd = `ffmpeg -y ${inputs} -filter_complex "${fc}" ` +
  `-map "[outv]" -map "[outa]" ` +
  `-c:v libx264 -preset ${videoSettings.preset} -crf ${videoSettings.crf} ` +
  `-s ${videoSettings.output_width}x${videoSettings.output_height} ` +
  `-c:a aac -b:a 192k -ar 48000 ` +
  `-movflags +faststart ` +
  `-pix_fmt yuv420p ` +
  `"${outputPath}"`;

return [{
  json: {
    asset_id: sourceData.asset_id,
    script_id: scriptId,
    content_idea_id: sourceData.content_idea_id,
    ffmpeg_command: ffmpegCmd,
    output_path: outputPath,
    avatar_path: avatarPath,
    source_video_path: sourcePath,
    audio_settings: audioSettings,
    video_settings: videoSettings
  }
}];
```

#### Task 1.3: Fix Caption Styling + Enforce Resolution
**File:** n8n workflow (node id: `build-caption-cmd`)

```javascript
// Build FFmpeg caption burn command
const items = $input.all();
if (!items || items.length === 0) {
  return [{ json: { error: 'No input', success: false } }];
}

const inputData = items[0].json;
const scriptId = inputData.script_id;

const combinedPath = inputData.combined_video_path || `/home/node/.n8n-files/assets/output/${scriptId}_combined.mp4`;
const srtPath = inputData.srt_path;
const finalPath = `/home/node/.n8n-files/assets/output/${scriptId}_final.mp4`;

// TikTok-style captions - properly sized for 1080x1920
// FontSize=12 is appropriate for 1080p vertical
// Alignment=8 = top center
// MarginV=80 = padding from top
const subtitleFilter = `subtitles=${srtPath}:force_style='FontName=Arial,FontSize=12,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,BackColour=&H40000000,Outline=2,Shadow=1,Bold=1,Alignment=8,MarginV=80,MarginL=60,MarginR=60'`;

// Enforce 1080x1920 output with high quality
const ffmpegCmd = `ffmpeg -y -i "${combinedPath}" ` +
  `-vf "${subtitleFilter}" ` +
  `-c:v libx264 -preset slow -crf 18 ` +
  `-s 1080x1920 ` +
  `-c:a copy ` +
  `-movflags +faststart ` +
  `-pix_fmt yuv420p ` +
  `"${finalPath}"`;

return [{
  json: {
    ...inputData,
    ffmpeg_caption_command: ffmpegCmd,
    final_path: finalPath,
    combined_path: combinedPath
  }
}];
```

---

### Phase 2: Database Schema Updates

#### Task 2.1: Create Settings Tables
**File:** `backend/migrations/002_add_settings_tables.sql`

```sql
-- ============================================
-- AUDIO SETTINGS (System-wide defaults)
-- ============================================
CREATE TABLE IF NOT EXISTS audio_settings (
    id SERIAL PRIMARY KEY,
    original_volume FLOAT DEFAULT 0.7,           -- 70% default
    avatar_volume FLOAT DEFAULT 1.0,             -- 100% default
    ducking_enabled BOOLEAN DEFAULT true,        -- Enable ducking by default
    avatar_delay_seconds FLOAT DEFAULT 3.0,      -- Avatar appears after 3s
    duck_to_percent FLOAT DEFAULT 0.5,           -- Duck to 50%
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Insert default row
INSERT INTO audio_settings (id) VALUES (1) ON CONFLICT DO NOTHING;

-- ============================================
-- VIDEO SETTINGS (System-wide defaults)
-- ============================================
CREATE TABLE IF NOT EXISTS video_settings (
    id SERIAL PRIMARY KEY,
    output_width INTEGER DEFAULT 1080,
    output_height INTEGER DEFAULT 1920,
    output_format VARCHAR DEFAULT 'mp4',
    codec VARCHAR DEFAULT 'libx264',
    crf INTEGER DEFAULT 18,                      -- High quality
    preset VARCHAR DEFAULT 'slow',               -- Better compression
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Insert default row
INSERT INTO video_settings (id) VALUES (1) ON CONFLICT DO NOTHING;

-- ============================================
-- LLM SETTINGS (Prompt templates)
-- ============================================
CREATE TABLE IF NOT EXISTS llm_settings (
    id SERIAL PRIMARY KEY,
    key VARCHAR UNIQUE NOT NULL,
    value TEXT NOT NULL,
    description VARCHAR,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Insert default prompts
INSERT INTO llm_settings (key, value, description) VALUES
('script_system_prompt',
 'You are a viral content scriptwriter specializing in short-form video content. Create engaging, hook-driven scripts that capture attention in the first 3 seconds and maintain viewer interest throughout. Focus on the specified content pillar and target audience.',
 'System prompt for script generation with Grok'),

('script_user_template',
 'Create a viral script for the following content idea:\n\nTopic: {topic}\nContent Pillar: {pillar}\nOriginal Hook: {original_hook}\nSource Platform: {platform}\n\nGenerate:\n1. A 3-second attention-grabbing hook\n2. Main body content (15-45 seconds)\n3. Clear call-to-action\n4. Platform-specific captions for: TikTok, Instagram, YouTube, LinkedIn, X/Twitter',
 'User prompt template for script generation'),

('trend_analysis_prompt',
 'Analyze this content for viral potential. Score from 1-10 on:\n- Hook strength\n- Emotional resonance\n- Shareability\n- Trend alignment\n- Target audience fit\n\nProvide specific recommendations for improvement.',
 'Prompt for analyzing scraped content virality'),

('hook_generation_prompt',
 'Create 5 alternative hooks for this content that will stop scrollers in the first 1-3 seconds. Use pattern interrupts, curiosity gaps, or controversial statements. Keep each under 10 words.',
 'Prompt for generating alternative hooks')

ON CONFLICT (key) DO NOTHING;

-- ============================================
-- Add settings reference to assets (optional per-asset overrides)
-- ============================================
ALTER TABLE assets
ADD COLUMN IF NOT EXISTS audio_settings_override JSONB DEFAULT NULL,
ADD COLUMN IF NOT EXISTS video_settings_override JSONB DEFAULT NULL;
```

#### Task 2.2: Add Models
**File:** `backend/models.py`

```python
# Add these new models

class AudioSettings(Base):
    __tablename__ = "audio_settings"

    id = Column(Integer, primary_key=True, default=1)
    original_volume = Column(Float, default=0.7)
    avatar_volume = Column(Float, default=1.0)
    ducking_enabled = Column(Boolean, default=True)
    avatar_delay_seconds = Column(Float, default=3.0)
    duck_to_percent = Column(Float, default=0.5)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class VideoSettings(Base):
    __tablename__ = "video_settings"

    id = Column(Integer, primary_key=True, default=1)
    output_width = Column(Integer, default=1080)
    output_height = Column(Integer, default=1920)
    output_format = Column(String, default='mp4')
    codec = Column(String, default='libx264')
    crf = Column(Integer, default=18)
    preset = Column(String, default='slow')
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class LLMSettings(Base):
    __tablename__ = "llm_settings"

    id = Column(Integer, primary_key=True)
    key = Column(String, unique=True, nullable=False)
    value = Column(Text, nullable=False)
    description = Column(String)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

#### Task 2.3: Add Schemas
**File:** `backend/schemas.py`

```python
# Add these new schemas

class AudioSettingsBase(BaseModel):
    original_volume: float = 0.7
    avatar_volume: float = 1.0
    ducking_enabled: bool = True
    avatar_delay_seconds: float = 3.0
    duck_to_percent: float = 0.5

class AudioSettingsUpdate(BaseModel):
    original_volume: Optional[float] = None
    avatar_volume: Optional[float] = None
    ducking_enabled: Optional[bool] = None
    avatar_delay_seconds: Optional[float] = None
    duck_to_percent: Optional[float] = None

class AudioSettingsResponse(AudioSettingsBase):
    id: int
    updated_at: datetime

    class Config:
        from_attributes = True


class VideoSettingsBase(BaseModel):
    output_width: int = 1080
    output_height: int = 1920
    output_format: str = 'mp4'
    codec: str = 'libx264'
    crf: int = 18
    preset: str = 'slow'

class VideoSettingsUpdate(BaseModel):
    output_width: Optional[int] = None
    output_height: Optional[int] = None
    output_format: Optional[str] = None
    codec: Optional[str] = None
    crf: Optional[int] = None
    preset: Optional[str] = None

class VideoSettingsResponse(VideoSettingsBase):
    id: int
    updated_at: datetime

    class Config:
        from_attributes = True


class LLMSettingBase(BaseModel):
    key: str
    value: str
    description: Optional[str] = None

class LLMSettingUpdate(BaseModel):
    value: Optional[str] = None
    description: Optional[str] = None

class LLMSettingResponse(LLMSettingBase):
    id: int
    updated_at: datetime

    class Config:
        from_attributes = True
```

---

### Phase 3: Backend API Updates

#### Task 3.1: Settings Endpoints
**File:** `backend/main.py`

```python
# ============================================
# AUDIO SETTINGS ENDPOINTS
# ============================================

@app.get("/api/settings/audio", response_model=AudioSettingsResponse)
async def get_audio_settings(db: Session = Depends(get_db)):
    """Get current audio settings"""
    settings = db.query(AudioSettings).filter(AudioSettings.id == 1).first()
    if not settings:
        # Create default settings if not exists
        settings = AudioSettings(id=1)
        db.add(settings)
        db.commit()
        db.refresh(settings)
    return settings


@app.put("/api/settings/audio", response_model=AudioSettingsResponse)
async def update_audio_settings(update: AudioSettingsUpdate, db: Session = Depends(get_db)):
    """Update audio settings"""
    settings = db.query(AudioSettings).filter(AudioSettings.id == 1).first()
    if not settings:
        settings = AudioSettings(id=1)
        db.add(settings)

    for field, value in update.dict(exclude_unset=True).items():
        setattr(settings, field, value)

    db.commit()
    db.refresh(settings)
    return settings


# ============================================
# VIDEO SETTINGS ENDPOINTS
# ============================================

@app.get("/api/settings/video", response_model=VideoSettingsResponse)
async def get_video_settings(db: Session = Depends(get_db)):
    """Get current video settings"""
    settings = db.query(VideoSettings).filter(VideoSettings.id == 1).first()
    if not settings:
        settings = VideoSettings(id=1)
        db.add(settings)
        db.commit()
        db.refresh(settings)
    return settings


@app.put("/api/settings/video", response_model=VideoSettingsResponse)
async def update_video_settings(update: VideoSettingsUpdate, db: Session = Depends(get_db)):
    """Update video settings"""
    settings = db.query(VideoSettings).filter(VideoSettings.id == 1).first()
    if not settings:
        settings = VideoSettings(id=1)
        db.add(settings)

    for field, value in update.dict(exclude_unset=True).items():
        setattr(settings, field, value)

    db.commit()
    db.refresh(settings)
    return settings


# ============================================
# LLM SETTINGS ENDPOINTS
# ============================================

@app.get("/api/settings/llm")
async def get_llm_settings(db: Session = Depends(get_db)):
    """Get all LLM prompt settings"""
    settings = db.query(LLMSettings).all()
    return [{"key": s.key, "value": s.value, "description": s.description, "updated_at": s.updated_at} for s in settings]


@app.get("/api/settings/llm/{key}")
async def get_llm_setting(key: str, db: Session = Depends(get_db)):
    """Get a specific LLM prompt by key"""
    setting = db.query(LLMSettings).filter(LLMSettings.key == key).first()
    if not setting:
        raise HTTPException(404, f"LLM setting '{key}' not found")
    return {"key": setting.key, "value": setting.value, "description": setting.description}


@app.put("/api/settings/llm/{key}")
async def update_llm_setting(key: str, update: LLMSettingUpdate, db: Session = Depends(get_db)):
    """Update a specific LLM prompt"""
    setting = db.query(LLMSettings).filter(LLMSettings.key == key).first()
    if not setting:
        # Create new setting if doesn't exist
        setting = LLMSettings(key=key, value=update.value or '', description=update.description)
        db.add(setting)
    else:
        if update.value is not None:
            setting.value = update.value
        if update.description is not None:
            setting.description = update.description

    db.commit()
    db.refresh(setting)
    return {"key": setting.key, "value": setting.value, "description": setting.description}


# ============================================
# COMBINED SETTINGS ENDPOINT (for n8n)
# ============================================

@app.get("/api/settings/all")
async def get_all_settings(db: Session = Depends(get_db)):
    """Get all settings in one call (for n8n workflow)"""
    audio = db.query(AudioSettings).filter(AudioSettings.id == 1).first()
    video = db.query(VideoSettings).filter(VideoSettings.id == 1).first()
    llm = db.query(LLMSettings).all()

    return {
        "audio": {
            "original_volume": audio.original_volume if audio else 0.7,
            "avatar_volume": audio.avatar_volume if audio else 1.0,
            "ducking_enabled": audio.ducking_enabled if audio else True,
            "avatar_delay_seconds": audio.avatar_delay_seconds if audio else 3.0,
            "duck_to_percent": audio.duck_to_percent if audio else 0.5
        },
        "video": {
            "output_width": video.output_width if video else 1080,
            "output_height": video.output_height if video else 1920,
            "crf": video.crf if video else 18,
            "preset": video.preset if video else 'slow'
        },
        "llm": {s.key: s.value for s in llm}
    }
```

---

### Phase 4: Frontend UI Updates

#### Task 4.1: Settings Page
**File:** `frontend/src/pages/Settings.js` (NEW FILE)

```jsx
import React, { useState, useEffect } from 'react';
import { getAudioSettings, updateAudioSettings, getVideoSettings, updateVideoSettings, getLLMSettings, updateLLMSetting } from '../api';

export default function Settings() {
  // Audio settings state
  const [audio, setAudio] = useState({
    original_volume: 0.7,
    avatar_volume: 1.0,
    ducking_enabled: true,
    avatar_delay_seconds: 3.0,
    duck_to_percent: 0.5
  });

  // Video settings state
  const [video, setVideo] = useState({
    output_width: 1080,
    output_height: 1920,
    crf: 18,
    preset: 'slow'
  });

  // LLM settings state
  const [llmPrompts, setLlmPrompts] = useState([]);
  const [editingPrompt, setEditingPrompt] = useState(null);
  const [editValue, setEditValue] = useState('');

  // Loading states
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState('');

  useEffect(() => {
    loadAllSettings();
  }, []);

  const loadAllSettings = async () => {
    try {
      const [audioData, videoData, llmData] = await Promise.all([
        getAudioSettings(),
        getVideoSettings(),
        getLLMSettings()
      ]);
      setAudio(audioData);
      setVideo(videoData);
      setLlmPrompts(llmData);
    } catch (err) {
      console.error('Failed to load settings:', err);
    }
  };

  const saveAudioSettings = async () => {
    setSaving(true);
    try {
      await updateAudioSettings(audio);
      setMessage('Audio settings saved!');
      setTimeout(() => setMessage(''), 3000);
    } catch (err) {
      setMessage('Error saving audio settings');
    }
    setSaving(false);
  };

  const saveVideoSettings = async () => {
    setSaving(true);
    try {
      await updateVideoSettings(video);
      setMessage('Video settings saved!');
      setTimeout(() => setMessage(''), 3000);
    } catch (err) {
      setMessage('Error saving video settings');
    }
    setSaving(false);
  };

  const saveLLMPrompt = async (key) => {
    setSaving(true);
    try {
      await updateLLMSetting(key, { value: editValue });
      setEditingPrompt(null);
      loadAllSettings();
      setMessage('Prompt saved!');
      setTimeout(() => setMessage(''), 3000);
    } catch (err) {
      setMessage('Error saving prompt');
    }
    setSaving(false);
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-8">Settings</h1>

      {message && (
        <div className="mb-4 p-3 bg-green-100 text-green-800 rounded">
          {message}
        </div>
      )}

      {/* AUDIO SETTINGS */}
      <div className="mb-8 p-6 bg-white rounded-lg shadow">
        <h2 className="text-xl font-semibold mb-4">Audio Settings</h2>

        <div className="space-y-4">
          {/* Original Volume */}
          <div>
            <label className="block text-sm font-medium mb-1">
              Original Video Audio Volume: {Math.round(audio.original_volume * 100)}%
            </label>
            <input
              type="range"
              min="0"
              max="100"
              value={audio.original_volume * 100}
              onChange={(e) => setAudio({...audio, original_volume: e.target.value / 100})}
              className="w-full"
            />
          </div>

          {/* Avatar Volume */}
          <div>
            <label className="block text-sm font-medium mb-1">
              Avatar/Voiceover Volume: {Math.round(audio.avatar_volume * 100)}%
            </label>
            <input
              type="range"
              min="0"
              max="100"
              value={audio.avatar_volume * 100}
              onChange={(e) => setAudio({...audio, avatar_volume: e.target.value / 100})}
              className="w-full"
            />
          </div>

          {/* Ducking Toggle */}
          <div className="flex items-center">
            <input
              type="checkbox"
              id="ducking"
              checked={audio.ducking_enabled}
              onChange={(e) => setAudio({...audio, ducking_enabled: e.target.checked})}
              className="mr-2 h-4 w-4"
            />
            <label htmlFor="ducking" className="text-sm font-medium">
              Enable Audio Ducking (lower original audio when avatar speaks)
            </label>
          </div>

          {/* Ducking Settings (shown when enabled) */}
          {audio.ducking_enabled && (
            <div className="ml-6 p-4 bg-gray-50 rounded border">
              <div className="mb-3">
                <label className="block text-sm font-medium mb-1">
                  Avatar Appears After (seconds)
                </label>
                <input
                  type="number"
                  min="0"
                  max="30"
                  step="0.5"
                  value={audio.avatar_delay_seconds}
                  onChange={(e) => setAudio({...audio, avatar_delay_seconds: parseFloat(e.target.value)})}
                  className="w-24 p-2 border rounded"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">
                  Duck Original Audio To: {Math.round(audio.duck_to_percent * 100)}%
                </label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={audio.duck_to_percent * 100}
                  onChange={(e) => setAudio({...audio, duck_to_percent: e.target.value / 100})}
                  className="w-full"
                />
              </div>
            </div>
          )}

          <button
            onClick={saveAudioSettings}
            disabled={saving}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
          >
            {saving ? 'Saving...' : 'Save Audio Settings'}
          </button>
        </div>
      </div>

      {/* VIDEO SETTINGS */}
      <div className="mb-8 p-6 bg-white rounded-lg shadow">
        <h2 className="text-xl font-semibold mb-4">Video Settings</h2>

        <div className="space-y-4">
          {/* Resolution */}
          <div>
            <label className="block text-sm font-medium mb-2">Output Resolution</label>
            <div className="flex space-x-4">
              <label className="flex items-center">
                <input
                  type="radio"
                  name="resolution"
                  checked={video.output_height === 1280}
                  onChange={() => setVideo({...video, output_width: 720, output_height: 1280})}
                  className="mr-2"
                />
                720p (720x1280)
              </label>
              <label className="flex items-center">
                <input
                  type="radio"
                  name="resolution"
                  checked={video.output_height === 1920}
                  onChange={() => setVideo({...video, output_width: 1080, output_height: 1920})}
                  className="mr-2"
                />
                1080p (1080x1920)
              </label>
              <label className="flex items-center">
                <input
                  type="radio"
                  name="resolution"
                  checked={video.output_height === 3840}
                  onChange={() => setVideo({...video, output_width: 2160, output_height: 3840})}
                  className="mr-2"
                />
                4K (2160x3840)
              </label>
            </div>
          </div>

          {/* Quality */}
          <div>
            <label className="block text-sm font-medium mb-1">
              Quality (CRF): {video.crf} ({video.crf <= 18 ? 'High' : video.crf <= 23 ? 'Medium' : 'Low'})
            </label>
            <input
              type="range"
              min="15"
              max="28"
              value={video.crf}
              onChange={(e) => setVideo({...video, crf: parseInt(e.target.value)})}
              className="w-full"
            />
            <p className="text-xs text-gray-500">Lower = better quality, larger file size</p>
          </div>

          <button
            onClick={saveVideoSettings}
            disabled={saving}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
          >
            {saving ? 'Saving...' : 'Save Video Settings'}
          </button>
        </div>
      </div>

      {/* LLM PROMPT SETTINGS */}
      <div className="mb-8 p-6 bg-white rounded-lg shadow">
        <h2 className="text-xl font-semibold mb-4">LLM Prompt Settings</h2>

        <div className="space-y-6">
          {llmPrompts.map(prompt => (
            <div key={prompt.key} className="border-b pb-4">
              <h3 className="font-medium text-gray-900">{prompt.key.replace(/_/g, ' ').toUpperCase()}</h3>
              <p className="text-sm text-gray-500 mb-2">{prompt.description}</p>

              {editingPrompt === prompt.key ? (
                <div>
                  <textarea
                    className="w-full h-40 p-3 border rounded font-mono text-sm"
                    value={editValue}
                    onChange={(e) => setEditValue(e.target.value)}
                  />
                  <div className="mt-2 flex space-x-2">
                    <button
                      onClick={() => saveLLMPrompt(prompt.key)}
                      disabled={saving}
                      className="px-3 py-1 bg-green-600 text-white rounded hover:bg-green-700"
                    >
                      Save
                    </button>
                    <button
                      onClick={() => setEditingPrompt(null)}
                      className="px-3 py-1 bg-gray-300 rounded hover:bg-gray-400"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              ) : (
                <div>
                  <pre className="bg-gray-100 p-3 rounded text-sm overflow-auto max-h-32 whitespace-pre-wrap">
                    {prompt.value}
                  </pre>
                  <button
                    onClick={() => { setEditingPrompt(prompt.key); setEditValue(prompt.value); }}
                    className="mt-2 px-3 py-1 bg-gray-200 rounded hover:bg-gray-300"
                  >
                    Edit
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
```

#### Task 4.2: API Client Updates
**File:** `frontend/src/api.js`

```javascript
// Add these new API functions

// Audio Settings
export async function getAudioSettings() {
  const response = await fetch(`${API_BASE}/api/settings/audio`);
  return response.json();
}

export async function updateAudioSettings(settings) {
  const response = await fetch(`${API_BASE}/api/settings/audio`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(settings)
  });
  return response.json();
}

// Video Settings
export async function getVideoSettings() {
  const response = await fetch(`${API_BASE}/api/settings/video`);
  return response.json();
}

export async function updateVideoSettings(settings) {
  const response = await fetch(`${API_BASE}/api/settings/video`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(settings)
  });
  return response.json();
}

// LLM Settings
export async function getLLMSettings() {
  const response = await fetch(`${API_BASE}/api/settings/llm`);
  return response.json();
}

export async function updateLLMSetting(key, update) {
  const response = await fetch(`${API_BASE}/api/settings/llm/${key}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(update)
  });
  return response.json();
}

// All Settings (for preloading)
export async function getAllSettings() {
  const response = await fetch(`${API_BASE}/api/settings/all`);
  return response.json();
}
```

#### Task 4.3: Add Settings Route
**File:** `frontend/src/App.js`

```jsx
// Add import
import Settings from './pages/Settings';

// Add route (in the Routes section)
<Route path="/settings" element={<Settings />} />

// Add navigation link (in the nav section)
<NavLink to="/settings" className={navLinkClass}>
  Settings
</NavLink>
```

---

### Phase 5: n8n Workflow - Fetch Settings from API

#### Task 5.1: Add "Fetch Settings" Node
**Position:** Early in pipeline, after Webhook/Trigger

```javascript
// Fetch all settings from backend API
const http = require('http');

return new Promise((resolve, reject) => {
  const options = {
    hostname: 'backend',
    port: 8000,
    path: '/api/settings/all',
    method: 'GET',
    timeout: 10000
  };

  const req = http.request(options, (res) => {
    let data = '';
    res.on('data', (chunk) => data += chunk);
    res.on('end', () => {
      try {
        const settings = JSON.parse(data);
        resolve([{ json: { settings, ...($input.first()?.json || {}) } }]);
      } catch (e) {
        // Use defaults if API fails
        resolve([{ json: {
          settings: {
            audio: { original_volume: 0.7, avatar_volume: 1.0, ducking_enabled: true, avatar_delay_seconds: 3, duck_to_percent: 0.5 },
            video: { output_width: 1080, output_height: 1920, crf: 18, preset: 'slow' }
          },
          ...($input.first()?.json || {})
        }}]);
      }
    });
  });

  req.on('error', () => {
    // Use defaults if API fails
    resolve([{ json: {
      settings: {
        audio: { original_volume: 0.7, avatar_volume: 1.0, ducking_enabled: true, avatar_delay_seconds: 3, duck_to_percent: 0.5 },
        video: { output_width: 1080, output_height: 1920, crf: 18, preset: 'slow' }
      },
      ...($input.first()?.json || {})
    }}]);
  });

  req.end();
});
```

#### Task 5.2: Update All Downstream Nodes
Pass `settings` through each node and use in Build FFmpeg Command.

---

## Part 4: Testing Plan

### Test 1: FFmpeg Compositing with Dual Audio
```bash
# Manual test with existing files - dual audio, no ducking
docker exec n8n ffmpeg -y \
  -i /home/node/.n8n-files/assets/videos/22_source.mp4 \
  -i /home/node/.n8n-files/assets/avatar/22_avatar.mp4 \
  -filter_complex "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920[bg];[1:v]scale=1080:-1,chromakey=0x00FF00:0.1:0.2[avatar];[bg][avatar]overlay=(W-w)/2:H-h:shortest=1[outv];[0:a]volume=0.7[orig];[1:a]volume=1.0[avatar_a];[orig][avatar_a]amix=inputs=2:duration=longest[outa]" \
  -map "[outv]" -map "[outa]" \
  -c:v libx264 -preset slow -crf 18 \
  -s 1080x1920 \
  -c:a aac -b:a 192k \
  /home/node/.n8n-files/assets/output/test_dual_audio.mp4
```

### Test 2: Audio Ducking
```bash
# Test with ducking - original fades after 3 seconds
docker exec n8n ffmpeg -y \
  -i /home/node/.n8n-files/assets/videos/22_source.mp4 \
  -i /home/node/.n8n-files/assets/audio/22_voice.mp3 \
  -filter_complex "[0:a]volume=0.7:enable='lt(t,3)',volume=0.35:enable='gte(t,3)'[orig];[1:a]adelay=3000|3000,volume=1.0[voice];[orig][voice]amix=inputs=2:duration=longest,volume=2[outa]" \
  -map "0:v" -map "[outa]" \
  -s 1080x1920 \
  /home/node/.n8n-files/assets/output/test_ducking.mp4
```

### Test 3: Verify 1080p Output
```bash
# Check output resolution
docker exec n8n ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 /home/node/.n8n-files/assets/output/test_dual_audio.mp4
# Should output: 1080,1920
```

### Test 4: Full Pipeline
```bash
curl -X POST "http://100.83.153.43:5678/webhook/trigger-pipeline" \
  -H "Content-Type: application/json" \
  -d '{"content_idea_id": 141}'
```

---

## Part 5: Implementation Order

### Day 1: Core FFmpeg + Database
1. [ ] Run database migration (002_add_settings_tables.sql)
2. [ ] Add models to backend/models.py
3. [ ] Add schemas to backend/schemas.py
4. [ ] Add API endpoints to backend/main.py
5. [ ] Test API endpoints with curl

### Day 2: n8n Workflow Updates
1. [ ] Update "Use Existing Avatar" node to pass source_video_path
2. [ ] Add "Fetch Settings" node after trigger
3. [ ] Rewrite "Build FFmpeg Command" with settings support
4. [ ] Update "Build Caption Cmd" with 1080p enforcement
5. [ ] Test with manual FFmpeg commands
6. [ ] Test full pipeline

### Day 3: Frontend
1. [ ] Create Settings.js page
2. [ ] Add API functions to api.js
3. [ ] Add Settings route to App.js
4. [ ] Add Settings link to navigation
5. [ ] Test all UI flows

### Day 4: Integration & Polish
1. [ ] End-to-end testing with different audio settings
2. [ ] Verify 1080p output quality
3. [ ] Test script editing workflow
4. [ ] Fix any issues
5. [ ] Create backup of working workflow

---

## Files to Modify

| File | Changes |
|------|---------|
| **n8n workflow** | Update 4 nodes + add 1 new node (Fetch Settings) |
| `backend/migrations/002_*.sql` | NEW - Settings tables migration |
| `backend/models.py` | Add AudioSettings, VideoSettings, LLMSettings models |
| `backend/schemas.py` | Add settings schemas |
| `backend/main.py` | Add 8 new API endpoints |
| `frontend/src/App.js` | Add Settings route + nav link |
| `frontend/src/api.js` | Add 6 new API functions |
| `frontend/src/pages/Settings.js` | NEW - Full settings page |

---

## Settings Summary

| Setting | Location | Default | Configurable |
|---------|----------|---------|--------------|
| Original Audio Volume | Settings Page | 70% | Yes (slider 0-100%) |
| Avatar Audio Volume | Settings Page | 100% | Yes (slider 0-100%) |
| Audio Ducking Enabled | Settings Page | Yes | Yes (checkbox) |
| Avatar Delay (seconds) | Settings Page | 3.0 | Yes (number input) |
| Duck To Percent | Settings Page | 50% | Yes (slider 0-100%) |
| Output Resolution | Settings Page | 1080x1920 | Yes (720p/1080p/4K) |
| Video Quality (CRF) | Settings Page | 18 | Yes (slider 15-28) |
| Script System Prompt | Settings Page | [default] | Yes (textarea) |
| Script User Template | Settings Page | [default] | Yes (textarea) |
| Trend Analysis Prompt | Settings Page | [default] | Yes (textarea) |
| Hook Generation Prompt | Settings Page | [default] | Yes (textarea) |

---

## Risk Mitigation

1. **Backup before changes**: `workflows/COMPLETE_PIPELINE_V2_20260109_091834.json`
2. **Test FFmpeg commands manually** before updating workflow
3. **Database migrations are additive** - no data loss
4. **Frontend changes are isolated** - won't break existing functionality
5. **Keep HeyGen bypass** until HeyGen account upgraded
6. **API has fallback defaults** - if settings API fails, use hardcoded defaults

---

## Decisions Made

| Question | Decision |
|----------|----------|
| Avatar appear timing | Configurable via Settings page, default 3 seconds |
| Script approval flow | Edit + Save is sufficient (no separate "Approve" button) |
| Audio tracks | Both play simultaneously with independent volume controls |
| Audio ducking | Optional checkbox, when enabled original ducks after avatar delay |
| Video quality | Enforce 1080x1920 output, CRF 18 for high quality |
