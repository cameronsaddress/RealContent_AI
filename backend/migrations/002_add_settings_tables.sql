-- Migration: 002_add_settings_tables.sql
-- Purpose: Add audio, video, and LLM settings tables for pipeline configuration
-- Created: 2026-01-09

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
 'Create a viral script for the following content idea:

Topic: {topic}
Content Pillar: {pillar}
Original Hook: {original_hook}
Source Platform: {platform}

Generate:
1. A 3-second attention-grabbing hook
2. Main body content (15-45 seconds)
3. Clear call-to-action
4. Platform-specific captions for: TikTok, Instagram, YouTube, LinkedIn, X/Twitter',
 'User prompt template for script generation'),

('trend_analysis_prompt',
 'Analyze this content for viral potential. Score from 1-10 on:
- Hook strength
- Emotional resonance
- Shareability
- Trend alignment
- Target audience fit

Provide specific recommendations for improvement.',
 'Prompt for analyzing scraped content virality'),

('hook_generation_prompt',
 'Create 5 alternative hooks for this content that will stop scrollers in the first 1-3 seconds. Use pattern interrupts, curiosity gaps, or controversial statements. Keep each under 10 words.',
 'Prompt for generating alternative hooks')

ON CONFLICT (key) DO NOTHING;

-- ============================================
-- Add settings override columns to assets (optional per-asset overrides)
-- ============================================
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'assets' AND column_name = 'audio_settings_override') THEN
        ALTER TABLE assets ADD COLUMN audio_settings_override JSONB DEFAULT NULL;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'assets' AND column_name = 'video_settings_override') THEN
        ALTER TABLE assets ADD COLUMN video_settings_override JSONB DEFAULT NULL;
    END IF;
END $$;
