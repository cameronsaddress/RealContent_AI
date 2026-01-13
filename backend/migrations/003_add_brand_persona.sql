-- Migration: 003_add_brand_persona.sql
-- Purpose: Add brand persona table for script generation context + source video transcription fields
-- Created: 2026-01-12

-- ============================================
-- BRAND PERSONA TABLE
-- Stores the creator's identity, tone, values, and content boundaries
-- Used by script generator to create on-brand responses to viral content
-- ============================================
CREATE TABLE IF NOT EXISTS brand_persona (
    id SERIAL PRIMARY KEY,

    -- Identity
    name VARCHAR DEFAULT 'Beth Anderson',
    title VARCHAR DEFAULT 'Real Estate Expert',
    location VARCHAR DEFAULT 'Coeur d''Alene, Idaho & Liberty Lake, Washington',

    -- Bio/Background (shown to LLM for context)
    bio TEXT DEFAULT 'A knowledgeable and approachable real estate professional who helps first-time homebuyers and families find their dream homes in the beautiful Pacific Northwest.',

    -- Tone & Voice
    tone VARCHAR DEFAULT 'professional',           -- professional, casual, energetic, warm, authoritative
    energy_level VARCHAR DEFAULT 'warm',           -- calm, warm, energetic, high-energy
    humor_style VARCHAR DEFAULT 'light',           -- none, light, playful, witty

    -- Values & Approach (what we stand for) - JSONB array
    core_values JSONB DEFAULT '[
        "Honesty and transparency",
        "Client-first approach",
        "Education over sales",
        "Community connection",
        "Professional excellence"
    ]'::jsonb,

    -- Content Boundaries (what we WON'T do) - JSONB array
    content_boundaries JSONB DEFAULT '[
        "No dancing or twerking - we maintain professional dignity",
        "No clickbait or misleading claims",
        "No putting down other realtors or competitors",
        "No political or controversial topics",
        "No inappropriate language or innuendo"
    ]'::jsonb,

    -- How we respond to different content types
    response_style TEXT DEFAULT 'When reviewing viral content:
- If the content is professional and educational: Praise it, add our own insights, and connect it to our local market
- If the content is entertaining but unprofessional: Acknowledge the entertainment value, then pivot to show a more professional approach
- If the content has misinformation: Gently correct it while being respectful to the creator
- If the content is cringe or inappropriate: Focus on the underlying topic, not the presentation style
- Always provide genuine value - actionable tips, local insights, or helpful information',

    -- Signature phrases/CTAs
    signature_intro VARCHAR DEFAULT 'Hey neighbors!',
    signature_cta VARCHAR DEFAULT 'DM me to chat about your home journey in {location}',
    hashtags JSONB DEFAULT '["CDAhomes", "LibertyLake", "NorthIdahoRealEstate", "PNWliving"]'::jsonb,

    updated_at TIMESTAMP DEFAULT NOW()
);

-- Insert default persona row
INSERT INTO brand_persona (id) VALUES (1) ON CONFLICT DO NOTHING;

-- ============================================
-- ADD SOURCE TRANSCRIPTION FIELDS TO CONTENT_IDEAS
-- These store what was actually SAID in viral videos for smarter script generation
-- ============================================
DO $$
BEGIN
    -- Source transcription (Whisper output - what was said in the video)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'content_ideas' AND column_name = 'source_transcription') THEN
        ALTER TABLE content_ideas ADD COLUMN source_transcription TEXT;
        COMMENT ON COLUMN content_ideas.source_transcription IS 'Whisper transcription of what was said in the source video';
    END IF;

    -- Source video path (local path to downloaded video)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'content_ideas' AND column_name = 'source_video_path') THEN
        ALTER TABLE content_ideas ADD COLUMN source_video_path TEXT;
        COMMENT ON COLUMN content_ideas.source_video_path IS 'Local path to the downloaded source video file';
    END IF;

    -- Why viral (LLM analysis of why content went viral)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'content_ideas' AND column_name = 'why_viral') THEN
        ALTER TABLE content_ideas ADD COLUMN why_viral TEXT;
        COMMENT ON COLUMN content_ideas.why_viral IS 'LLM analysis of why this content went viral';
    END IF;
END $$;

-- ============================================
-- UPDATE VIDEO_SETTINGS WITH LATEST COLUMNS
-- Ensure all new video settings columns exist
-- ============================================
DO $$
BEGIN
    -- Greenscreen settings
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'video_settings' AND column_name = 'greenscreen_enabled') THEN
        ALTER TABLE video_settings ADD COLUMN greenscreen_enabled BOOLEAN DEFAULT true;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'video_settings' AND column_name = 'greenscreen_color') THEN
        ALTER TABLE video_settings ADD COLUMN greenscreen_color VARCHAR DEFAULT '#00FF00';
    END IF;

    -- Avatar position settings
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'video_settings' AND column_name = 'avatar_position') THEN
        ALTER TABLE video_settings ADD COLUMN avatar_position VARCHAR DEFAULT 'bottom-left';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'video_settings' AND column_name = 'avatar_scale') THEN
        ALTER TABLE video_settings ADD COLUMN avatar_scale FLOAT DEFAULT 0.8;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'video_settings' AND column_name = 'avatar_offset_x') THEN
        ALTER TABLE video_settings ADD COLUMN avatar_offset_x INTEGER DEFAULT -200;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'video_settings' AND column_name = 'avatar_offset_y') THEN
        ALTER TABLE video_settings ADD COLUMN avatar_offset_y INTEGER DEFAULT 500;
    END IF;

    -- Caption settings
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'video_settings' AND column_name = 'caption_style') THEN
        ALTER TABLE video_settings ADD COLUMN caption_style VARCHAR DEFAULT 'karaoke';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'video_settings' AND column_name = 'caption_font_size') THEN
        ALTER TABLE video_settings ADD COLUMN caption_font_size INTEGER DEFAULT 96;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'video_settings' AND column_name = 'caption_font') THEN
        ALTER TABLE video_settings ADD COLUMN caption_font VARCHAR DEFAULT 'Arial';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'video_settings' AND column_name = 'caption_color') THEN
        ALTER TABLE video_settings ADD COLUMN caption_color VARCHAR DEFAULT '#FFFFFF';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'video_settings' AND column_name = 'caption_highlight_color') THEN
        ALTER TABLE video_settings ADD COLUMN caption_highlight_color VARCHAR DEFAULT '#FFFF00';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'video_settings' AND column_name = 'caption_outline_color') THEN
        ALTER TABLE video_settings ADD COLUMN caption_outline_color VARCHAR DEFAULT '#000000';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'video_settings' AND column_name = 'caption_outline_width') THEN
        ALTER TABLE video_settings ADD COLUMN caption_outline_width INTEGER DEFAULT 5;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'video_settings' AND column_name = 'caption_position_y') THEN
        ALTER TABLE video_settings ADD COLUMN caption_position_y INTEGER DEFAULT 850;
    END IF;
END $$;

-- ============================================
-- ADD MUSIC SETTINGS TO AUDIO_SETTINGS
-- ============================================
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'audio_settings' AND column_name = 'music_volume') THEN
        ALTER TABLE audio_settings ADD COLUMN music_volume FLOAT DEFAULT 0.3;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'audio_settings' AND column_name = 'music_autoduck') THEN
        ALTER TABLE audio_settings ADD COLUMN music_autoduck BOOLEAN DEFAULT true;
    END IF;
END $$;
