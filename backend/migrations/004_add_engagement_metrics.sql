-- Add engagement metrics to content_ideas table
-- These fields capture view counts, likes, shares, etc. from scraped viral content

ALTER TABLE content_ideas ADD COLUMN IF NOT EXISTS views INTEGER DEFAULT 0;
ALTER TABLE content_ideas ADD COLUMN IF NOT EXISTS likes INTEGER DEFAULT 0;
ALTER TABLE content_ideas ADD COLUMN IF NOT EXISTS shares INTEGER DEFAULT 0;
ALTER TABLE content_ideas ADD COLUMN IF NOT EXISTS comments INTEGER DEFAULT 0;
ALTER TABLE content_ideas ADD COLUMN IF NOT EXISTS author TEXT;
ALTER TABLE content_ideas ADD COLUMN IF NOT EXISTS author_followers INTEGER DEFAULT 0;
