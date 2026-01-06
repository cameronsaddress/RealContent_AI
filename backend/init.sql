-- Content Pipeline Database Schema

-- Enum types
CREATE TYPE content_status AS ENUM (
    'pending',
    'approved',
    'rejected',
    'script_generating',
    'script_ready',
    'voice_generating',
    'voice_ready',
    'avatar_generating',
    'avatar_ready',
    'assembling',
    'captioning',
    'ready_to_publish',
    'publishing',
    'published',
    'error'
);

CREATE TYPE pillar_type AS ENUM (
    'market_intelligence',
    'educational_tips',
    'lifestyle_local',
    'brand_humanization'
);

CREATE TYPE platform_type AS ENUM (
    'tiktok',
    'reddit',
    'youtube',
    'twitter',
    'instagram',
    'linkedin',
    'facebook',
    'threads',
    'pinterest'
);

-- Content Ideas table (scraped content)
CREATE TABLE content_ideas (
    id SERIAL PRIMARY KEY,
    source_url TEXT,
    source_platform platform_type,
    original_text TEXT,
    pillar pillar_type,
    viral_score INTEGER CHECK (viral_score >= 1 AND viral_score <= 10),
    suggested_hook TEXT,
    status content_status DEFAULT 'pending',
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Scripts table
CREATE TABLE scripts (
    id SERIAL PRIMARY KEY,
    content_idea_id INTEGER REFERENCES content_ideas(id) ON DELETE CASCADE,
    hook TEXT,
    body TEXT,
    cta TEXT,
    full_script TEXT,
    duration_estimate INTEGER, -- seconds
    tiktok_caption TEXT,
    ig_caption TEXT,
    yt_title TEXT,
    yt_description TEXT,
    linkedin_text TEXT,
    x_text TEXT,
    facebook_text TEXT,
    threads_text TEXT,
    status content_status DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Assets table (files and media)
CREATE TABLE assets (
    id SERIAL PRIMARY KEY,
    script_id INTEGER REFERENCES scripts(id) ON DELETE CASCADE,
    voiceover_path TEXT,
    voiceover_duration FLOAT,
    srt_path TEXT,
    ass_path TEXT,
    avatar_video_path TEXT,
    background_video_path TEXT,
    combined_video_path TEXT,
    final_video_path TEXT,
    status content_status DEFAULT 'pending',
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Published table (tracking posts)
CREATE TABLE published (
    id SERIAL PRIMARY KEY,
    asset_id INTEGER REFERENCES assets(id) ON DELETE CASCADE,
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
    published_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Analytics table
CREATE TABLE analytics (
    id SERIAL PRIMARY KEY,
    published_id INTEGER REFERENCES published(id) ON DELETE CASCADE,
    platform platform_type,
    views INTEGER DEFAULT 0,
    likes INTEGER DEFAULT 0,
    comments INTEGER DEFAULT 0,
    shares INTEGER DEFAULT 0,
    engagement_rate FLOAT,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Pipeline runs table (for tracking workflow executions)
CREATE TABLE pipeline_runs (
    id SERIAL PRIMARY KEY,
    content_idea_id INTEGER REFERENCES content_ideas(id),
    workflow_name TEXT,
    status TEXT,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    metadata JSONB
);

-- Create indexes for common queries
CREATE INDEX idx_content_ideas_status ON content_ideas(status);
CREATE INDEX idx_content_ideas_pillar ON content_ideas(pillar);
CREATE INDEX idx_content_ideas_created ON content_ideas(created_at DESC);
CREATE INDEX idx_scripts_status ON scripts(status);
CREATE INDEX idx_assets_status ON assets(status);
CREATE INDEX idx_published_date ON published(published_at DESC);

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_content_ideas_updated_at
    BEFORE UPDATE ON content_ideas
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_scripts_updated_at
    BEFORE UPDATE ON scripts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_assets_updated_at
    BEFORE UPDATE ON assets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- View for full pipeline status
CREATE VIEW pipeline_overview AS
SELECT
    ci.id as content_id,
    ci.source_platform,
    ci.pillar,
    ci.viral_score,
    ci.status as content_status,
    s.id as script_id,
    s.status as script_status,
    s.duration_estimate,
    a.id as asset_id,
    a.status as asset_status,
    p.id as published_id,
    p.published_at,
    ci.created_at
FROM content_ideas ci
LEFT JOIN scripts s ON s.content_idea_id = ci.id
LEFT JOIN assets a ON a.script_id = s.id
LEFT JOIN published p ON p.asset_id = a.id
ORDER BY ci.created_at DESC;
