-- Migration: Add scrape runs and niche presets tables
-- Run this if your database already exists

-- Scrape run status enum
DO $$ BEGIN
    CREATE TYPE scrape_run_status AS ENUM (
        'pending',
        'running',
        'completed',
        'failed'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Scrape runs table (for tracking trend scrapes)
CREATE TABLE IF NOT EXISTS scrape_runs (
    id SERIAL PRIMARY KEY,
    niche TEXT,
    hashtags JSONB,
    platforms JSONB,
    status scrape_run_status DEFAULT 'pending',
    results_count INTEGER DEFAULT 0,
    results_data JSONB,
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Niche presets table (saved niche configurations)
CREATE TABLE IF NOT EXISTS niche_presets (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    keywords JSONB,
    hashtags JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_scrape_runs_status ON scrape_runs(status);
CREATE INDEX IF NOT EXISTS idx_scrape_runs_started ON scrape_runs(started_at DESC);

-- Insert default niche presets (only if table is empty)
INSERT INTO niche_presets (name, keywords, hashtags)
SELECT 'Real Estate', '["real estate", "realtor", "home buying", "selling home"]'::jsonb, '["realestate", "realtor", "homebuying", "firsttimehomebuyer", "housingmarket"]'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM niche_presets WHERE name = 'Real Estate');

INSERT INTO niche_presets (name, keywords, hashtags)
SELECT 'Mortgage', '["mortgage", "home loan", "refinance", "interest rates"]'::jsonb, '["mortgage", "homeloan", "refinance", "mortgagetips", "homebuyer"]'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM niche_presets WHERE name = 'Mortgage');

INSERT INTO niche_presets (name, keywords, hashtags)
SELECT 'Investment', '["real estate investing", "rental property", "property investment"]'::jsonb, '["realestateinvesting", "rentalproperty", "passiveincome", "investingtips"]'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM niche_presets WHERE name = 'Investment');

INSERT INTO niche_presets (name, keywords, hashtags)
SELECT 'Luxury Homes', '["luxury real estate", "million dollar homes", "luxury living"]'::jsonb, '["luxuryrealestate", "luxuryhomes", "milliondollarlisting", "dreamhome"]'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM niche_presets WHERE name = 'Luxury Homes');
