from sqlalchemy import (
    Column, Integer, String, Text, Float, DateTime, ForeignKey, Boolean,
    Enum as SQLEnum, CheckConstraint, create_engine
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import enum
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://n8n:n8n_password@postgres:5432/content_pipeline")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class ContentStatus(str, enum.Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"
    script_generating = "script_generating"
    script_ready = "script_ready"
    voice_generating = "voice_generating"
    voice_ready = "voice_ready"
    avatar_generating = "avatar_generating"
    avatar_ready = "avatar_ready"
    assembling = "assembling"
    captioning = "captioning"
    ready_to_publish = "ready_to_publish"
    publishing = "publishing"
    published = "published"
    error = "error"


class PillarType(str, enum.Enum):
    market_intelligence = "market_intelligence"
    educational_tips = "educational_tips"
    lifestyle_local = "lifestyle_local"
    brand_humanization = "brand_humanization"


class PlatformType(str, enum.Enum):
    tiktok = "tiktok"
    reddit = "reddit"
    youtube = "youtube"
    twitter = "twitter"
    instagram = "instagram"
    linkedin = "linkedin"
    unknown = "unknown"
    facebook = "facebook"
    threads = "threads"
    pinterest = "pinterest"
    rumble = "rumble"


class ContentIdea(Base):
    __tablename__ = "content_ideas"

    id = Column(Integer, primary_key=True, index=True)
    source_url = Column(Text)
    source_platform = Column(SQLEnum(PlatformType, name="platform_type", create_type=False))
    original_text = Column(Text)  # Caption/description from the post
    source_transcription = Column(Text)  # What was actually SAID in the video (Whisper)
    source_video_path = Column(Text)  # Local path to downloaded source video
    pillar = Column(SQLEnum(PillarType, name="pillar_type", create_type=False))
    viral_score = Column(Integer)
    suggested_hook = Column(Text)
    why_viral = Column(Text)  # LLM analysis of why it went viral
    status = Column(SQLEnum(ContentStatus, name="content_status", create_type=False), default=ContentStatus.pending)
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    # Engagement metrics from scraping
    views = Column(Integer, default=0)
    likes = Column(Integer, default=0)
    shares = Column(Integer, default=0)
    comments = Column(Integer, default=0)
    author = Column(Text)
    author_followers = Column(Integer, default=0)

    __table_args__ = (
        CheckConstraint('viral_score >= 1 AND viral_score <= 10', name='check_viral_score'),
    )

    scripts = relationship("Script", back_populates="content_idea", cascade="all, delete-orphan")


class Script(Base):
    __tablename__ = "scripts"

    id = Column(Integer, primary_key=True, index=True)
    content_idea_id = Column(Integer, ForeignKey("content_ideas.id", ondelete="CASCADE"))
    hook = Column(Text)
    body = Column(Text)
    cta = Column(Text)
    full_script = Column(Text)
    duration_estimate = Column(Integer)  # seconds
    tiktok_caption = Column(Text)
    ig_caption = Column(Text)
    yt_title = Column(Text)
    yt_description = Column(Text)
    linkedin_text = Column(Text)
    x_text = Column(Text)
    facebook_text = Column(Text)
    threads_text = Column(Text)
    status = Column(SQLEnum(ContentStatus, name="content_status", create_type=False), default=ContentStatus.pending)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    content_idea = relationship("ContentIdea", back_populates="scripts")
    assets = relationship("Asset", back_populates="script", cascade="all, delete-orphan")


class Asset(Base):
    __tablename__ = "assets"

    id = Column(Integer, primary_key=True, index=True)
    script_id = Column(Integer, ForeignKey("scripts.id", ondelete="CASCADE"))
    voiceover_path = Column(Text)
    voiceover_duration = Column(Float)
    srt_path = Column(Text)
    ass_path = Column(Text)
    avatar_video_path = Column(Text)
    heygen_video_id = Column(Text)  # Store HeyGen video ID for resume capability
    background_video_path = Column(Text)
    combined_video_path = Column(Text)
    final_video_path = Column(Text)
    status = Column(SQLEnum(ContentStatus, name="content_status", create_type=False), default=ContentStatus.pending)
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    script = relationship("Script", back_populates="assets")
    published = relationship("Published", back_populates="asset", cascade="all, delete-orphan")


class Published(Base):
    __tablename__ = "published"

    id = Column(Integer, primary_key=True, index=True)
    asset_id = Column(Integer, ForeignKey("assets.id", ondelete="CASCADE"))
    tiktok_url = Column(Text)
    tiktok_id = Column(Text)
    ig_url = Column(Text)
    ig_id = Column(Text)
    yt_url = Column(Text)
    yt_id = Column(Text)
    linkedin_url = Column(Text)
    linkedin_id = Column(Text)
    x_url = Column(Text)
    x_id = Column(Text)
    facebook_url = Column(Text)
    facebook_id = Column(Text)
    threads_url = Column(Text)
    threads_id = Column(Text)
    pinterest_url = Column(Text)
    pinterest_id = Column(Text)
    published_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    asset = relationship("Asset", back_populates="published")
    analytics = relationship("Analytics", back_populates="published", cascade="all, delete-orphan")


class Analytics(Base):
    __tablename__ = "analytics"

    id = Column(Integer, primary_key=True, index=True)
    published_id = Column(Integer, ForeignKey("published.id", ondelete="CASCADE"))
    platform = Column(SQLEnum(PlatformType, name="platform_type", create_type=False))
    views = Column(Integer, default=0)
    likes = Column(Integer, default=0)
    comments = Column(Integer, default=0)
    shares = Column(Integer, default=0)
    engagement_rate = Column(Float)
    recorded_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    published = relationship("Published", back_populates="analytics")


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id = Column(Integer, primary_key=True, index=True)
    content_idea_id = Column(Integer, ForeignKey("content_ideas.id"))
    workflow_name = Column(Text)
    status = Column(Text)
    started_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    completed_at = Column(DateTime(timezone=True))
    error_message = Column(Text)
    run_metadata = Column(JSONB)


class ScrapeRunStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class ScrapeRun(Base):
    __tablename__ = "scrape_runs"

    id = Column(Integer, primary_key=True, index=True)
    niche = Column(Text)
    hashtags = Column(JSONB)
    platforms = Column(JSONB)
    status = Column(SQLEnum(ScrapeRunStatus, name="scrape_run_status", create_type=False), default=ScrapeRunStatus.pending)
    results_count = Column(Integer, default=0)
    results_data = Column(JSONB)
    error_message = Column(Text)
    started_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    completed_at = Column(DateTime(timezone=True))


class NichePreset(Base):
    __tablename__ = "niche_presets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(Text, unique=True)
    keywords = Column(JSONB)
    hashtags = Column(JSONB)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class SystemSettings(Base):
    __tablename__ = "system_settings"

    key = Column(String(50), primary_key=True)
    value = Column(JSONB)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)


# ==================== PIPELINE SETTINGS ====================

class AudioSettings(Base):
    __tablename__ = "audio_settings"

    id = Column(Integer, primary_key=True, default=1)
    original_volume = Column(Float, default=0.7)
    avatar_volume = Column(Float, default=1.0)
    music_volume = Column(Float, default=0.3)
    ducking_enabled = Column(Boolean, default=True)
    avatar_delay_seconds = Column(Float, default=3.0)
    duck_to_percent = Column(Float, default=0.5)
    music_autoduck = Column(Boolean, default=True)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)


class VideoSettings(Base):
    __tablename__ = "video_settings"

    id = Column(Integer, primary_key=True, default=1)
    output_width = Column(Integer, default=1080)
    output_height = Column(Integer, default=1920)
    output_format = Column(String, default='mp4')
    codec = Column(String, default='libx264')
    crf = Column(Integer, default=18)
    preset = Column(String, default='slow')
    greenscreen_enabled = Column(Boolean, default=True)
    greenscreen_color = Column(String, default='#00FF00')

    # Avatar composition settings
    avatar_position = Column(String, default='bottom-left')  # bottom-left, bottom-center, bottom-right
    avatar_scale = Column(Float, default=0.8)  # 0.3 to 1.0 (percentage of screen width)
    avatar_offset_x = Column(Integer, default=-200)  # horizontal offset (negative = left)
    avatar_offset_y = Column(Integer, default=500)  # vertical offset (positive = push down below frame)

    # Caption settings
    caption_style = Column(String, default='karaoke')  # karaoke, static, none
    caption_font_size = Column(Integer, default=96)  # font size in points
    caption_font = Column(String, default='Arial')
    caption_color = Column(String, default='#FFFFFF')  # primary text color
    caption_highlight_color = Column(String, default='#FFFF00')  # karaoke highlight color
    caption_outline_color = Column(String, default='#000000')
    caption_outline_width = Column(Integer, default=5)
    caption_position_y = Column(Integer, default=850)  # MarginV from bottom in ASS format

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)


class BrandPersona(Base):
    """
    Brand persona settings for script generation.
    Defines who the creator is, their tone, values, and content boundaries.
    """
    __tablename__ = "brand_persona"

    id = Column(Integer, primary_key=True, default=1)

    # Identity
    name = Column(String, default="Beth Anderson")
    title = Column(String, default="Real Estate Expert")
    location = Column(String, default="North Idaho & Spokane area")

    # Bio/Background (shown to LLM for context)
    bio = Column(Text, default="A knowledgeable and approachable real estate professional who helps first-time homebuyers and families find their dream homes in the beautiful Pacific Northwest.")

    # Tone & Voice
    tone = Column(String, default="professional")  # professional, casual, energetic, warm, authoritative
    energy_level = Column(String, default="warm")  # calm, warm, energetic, high-energy
    humor_style = Column(String, default="light")  # none, light, playful, witty

    # Values & Approach (what we stand for)
    core_values = Column(JSONB, default=lambda: [
        "Honesty and transparency",
        "Client-first approach",
        "Education over sales",
        "Community connection",
        "Professional excellence"
    ])

    # Content Boundaries (what we WON'T do)
    content_boundaries = Column(JSONB, default=lambda: [
        "No dancing or twerking - we maintain professional dignity",
        "No clickbait or misleading claims",
        "No putting down other realtors or competitors",
        "No political or controversial topics",
        "No inappropriate language or innuendo"
    ])

    # How we respond to different content types
    response_style = Column(Text, default="""When reviewing viral content:
- If the content is professional and educational: Praise it, add our own insights, and connect it to our local market
- If the content is entertaining but unprofessional: Acknowledge the entertainment value, then pivot to show a more professional approach
- If the content has misinformation: Gently correct it while being respectful to the creator
- If the content is cringe or inappropriate: Focus on the underlying topic, not the presentation style
- Always provide genuine value - actionable tips, local insights, or helpful information""")

    # Signature phrases/CTAs
    signature_intro = Column(String, default="Hey neighbors!")
    signature_cta = Column(String, default="DM me to chat about your home journey in {location}")
    hashtags = Column(JSONB, default=lambda: ["CDAhomes", "LibertyLake", "NorthIdahoRealEstate", "PNWliving"])

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)


class LLMSettings(Base):
    __tablename__ = "llm_settings"

    id = Column(Integer, primary_key=True)
    key = Column(String, unique=True, nullable=False)
    value = Column(Text, nullable=False)
    description = Column(String)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)


class InfluencerPlatformType(str, enum.Enum):
    youtube = "youtube"
    rumble = "rumble"


class ClipPersona(Base):
    """
    Persona settings for the Viral Clip Factory.
    Defines the editing style, Grok prompting strategy, and publishing targets.
    """
    __tablename__ = "clip_personas"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True)  # e.g. "Trad West Bot"
    description = Column(Text)

    # Analysis Configuration
    prompt_template = Column(Text, default="Identify the most viral moments...")

    # Editing Configuration
    outro_style = Column(String, default="sunset_fade") # CapCut style preset
    font_style = Column(String, default="bold_impact")
    min_clip_duration = Column(Integer, default=30)
    max_clip_duration = Column(Integer, default=180)

    # Publishing Configuration
    blotato_account_id = Column(String)  # ID for the specific Blotato account

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    influencers = relationship("Influencer", back_populates="persona")
    publishing_configs = relationship("PublishingConfig", back_populates="persona")


class Influencer(Base):
    __tablename__ = "influencers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    platform = Column(SQLEnum(InfluencerPlatformType, name="influencer_platform_type", create_type=True))
    channel_id = Column(String)  # YouTube channel ID or Rumble username
    channel_url = Column(String)
    profile_image_url = Column(String)

    persona_id = Column(Integer, ForeignKey("clip_personas.id"))

    # === AUTO-MODE SETTINGS ===
    auto_mode_enabled = Column(Boolean, default=False)
    auto_mode_enabled_at = Column(DateTime(timezone=True))  # CRITICAL: Only process videos published AFTER this timestamp
    fetch_frequency_hours = Column(Integer, default=24)  # How often to check for new videos
    last_fetch_at = Column(DateTime(timezone=True))  # Last successful fetch timestamp
    max_videos_per_fetch = Column(Integer, default=5)  # Limit new videos per fetch cycle
    auto_analyze_enabled = Column(Boolean, default=True)  # Auto-start analysis on new videos
    auto_render_enabled = Column(Boolean, default=False)  # Auto-render all clips after analysis
    auto_publish_enabled = Column(Boolean, default=False)  # Auto-publish ready clips

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    persona = relationship("ClipPersona", back_populates="influencers")
    videos = relationship("InfluencerVideo", back_populates="influencer", cascade="all, delete-orphan")
    publishing_configs = relationship("PublishingConfig", back_populates="influencer")


class InfluencerVideo(Base):
    __tablename__ = "influencer_videos"

    id = Column(Integer, primary_key=True, index=True)
    influencer_id = Column(Integer, ForeignKey("influencers.id", ondelete="CASCADE"))
    
    platform_video_id = Column(String)
    title = Column(String)
    url = Column(String)
    thumbnail_url = Column(String)
    description = Column(Text)
    
    publication_date = Column(DateTime(timezone=True))
    duration = Column(Integer)  # seconds
    view_count = Column(Integer, default=0)
    
    # Processing State
    local_path = Column(Text)  # Path to downloaded .mp4
    transcript_json = Column(JSONB)  # Whisper word-level output
    analysis_json = Column(JSONB)  # Grok analysis results

    status = Column(String, default="pending")  # pending, downloaded, transcribed, analyzed, error
    error_message = Column(Text)
    processing_started_at = Column(DateTime(timezone=True))  # For progress tracking
    
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    influencer = relationship("Influencer", back_populates="videos")
    clips = relationship("ViralClip", back_populates="source_video", cascade="all, delete-orphan")


class ViralClip(Base):
    __tablename__ = "viral_clips"

    id = Column(Integer, primary_key=True, index=True)
    source_video_id = Column(Integer, ForeignKey("influencer_videos.id", ondelete="CASCADE"))

    # Timing
    start_time = Column(Float)
    end_time = Column(Float)
    duration = Column(Float)
    climax_time = Column(Float)  # Peak intensity moment for B-roll montage trigger

    # Content
    clip_type = Column(String)  # antagonistic, funny, controversial, inspirational
    virality_explanation = Column(Text)
    title = Column(String)
    description = Column(Text)
    hashtags = Column(JSONB)
    virality_score = Column(Float, default=0.5)  # Grok-assigned score (0-1) for publishing prioritization

    # Production
    edited_video_path = Column(Text)
    status = Column(String, default="pending")  # pending, rendering, ready, published, error
    error_message = Column(Text)
    render_metadata = Column(JSONB)
    template_id = Column(Integer, ForeignKey("render_templates.id", ondelete="SET NULL"), nullable=True)
    recommended_template_id = Column(Integer, nullable=True)  # Grok's recommendation (before user override)

    # Publishing
    blotato_post_id = Column(String)
    published_at = Column(DateTime(timezone=True))
    publishing_status = Column(String, default="unpublished")  # unpublished, queued, approved, published, skipped
    skip_reason = Column(Text)  # Why clip was skipped for publishing

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    source_video = relationship("InfluencerVideo", back_populates="clips")
    publishing_queue_items = relationship("PublishingQueueItem", back_populates="clip")


class RenderTemplate(Base):
    """
    Render templates for viral clips - defines visual style, B-roll categories, and effects.
    Grok can recommend templates based on clip content, or user can manually select.
    """
    __tablename__ = "render_templates"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)  # e.g., "Crusader", "Military", "Gym Bro"
    description = Column(Text)  # Description for UI and Grok context

    # B-roll configuration
    broll_categories = Column(JSONB, default=list)  # ["crusades", "warfare", "fighter_jets"]
    broll_enabled = Column(Boolean, default=True)

    # Visual effects settings
    effect_settings = Column(JSONB, default=dict)  # {"pulse_intensity": 0.5, "grain_intensity": 15, etc.}

    # Keywords for Grok matching (helps Grok choose the right template)
    keywords = Column(JSONB, default=list)  # ["christian", "faith", "crusade", "holy war"]

    # Template ordering/priority
    is_default = Column(Boolean, default=False)  # One template should be default
    sort_order = Column(Integer, default=0)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)


class BRollClip(Base):
    """
    Cached B-roll footage from Pexels for reuse across viral clips.
    """
    __tablename__ = "broll_clips"

    id = Column(Integer, primary_key=True, index=True)
    pexels_video_id = Column(String, unique=True, index=True)

    # Search metadata
    search_query = Column(String)
    category = Column(String, index=True)  # warfare, crusades, fighter_jets, etc.

    # File info
    local_path = Column(Text)  # /app/assets/broll/{id}.mp4
    duration = Column(Float)
    width = Column(Integer)
    height = Column(Integer)

    # Usage tracking
    use_count = Column(Integer, default=0)
    last_used_at = Column(DateTime(timezone=True))

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


# ==================== AUTO-MODE PUBLISHING ====================

class PublishingConfig(Base):
    """
    Publishing schedule and platform configuration per influencer/persona.
    Controls when and how clips are automatically published to social platforms.
    """
    __tablename__ = "publishing_configs"

    id = Column(Integer, primary_key=True, index=True)

    # Ownership - either influencer-specific OR persona-wide default
    influencer_id = Column(Integer, ForeignKey("influencers.id", ondelete="CASCADE"), nullable=True)
    persona_id = Column(Integer, ForeignKey("clip_personas.id", ondelete="CASCADE"), nullable=True)

    # Blotato Account Selection
    blotato_account_id = Column(String(100), nullable=False)
    blotato_account_name = Column(String(200))  # Display name for UI

    # Platform Selection (which platforms to post to)
    platforms = Column(JSONB, default=lambda: ["tiktok", "instagram_reels", "youtube_shorts"])

    # Posting Schedule
    enabled = Column(Boolean, default=False)
    posts_per_day = Column(Integer, default=3)  # Max clips to publish per day
    posting_hours = Column(JSONB, default=lambda: [9, 13, 18])  # Hours (UTC) to post at
    posting_days = Column(JSONB, default=lambda: [0, 1, 2, 3, 4, 5, 6])  # Days of week (0=Monday)

    # Content Rules
    min_virality_score = Column(Float, default=0.0)  # Skip clips below threshold
    clip_types_allowed = Column(JSONB, default=lambda: ["antagonistic", "controversial", "funny", "inspirational"])
    require_manual_approval = Column(Boolean, default=True)  # Queue for approval vs auto-post

    # Rate Limiting (best practices)
    min_hours_between_posts = Column(Integer, default=4)  # Avoid spam detection
    max_posts_per_platform_per_day = Column(Integer, default=2)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    influencer = relationship("Influencer", back_populates="publishing_configs")
    persona = relationship("ClipPersona", back_populates="publishing_configs")
    queue_items = relationship("PublishingQueueItem", back_populates="config")


class PublishingQueueItem(Base):
    """
    Queue of clips ready/scheduled for publishing.
    Supports approval workflow and scheduled posting.
    """
    __tablename__ = "publishing_queue"

    id = Column(Integer, primary_key=True, index=True)
    clip_id = Column(Integer, ForeignKey("viral_clips.id", ondelete="CASCADE"), nullable=False)
    config_id = Column(Integer, ForeignKey("publishing_configs.id", ondelete="CASCADE"), nullable=False)

    # Scheduling
    scheduled_time = Column(DateTime(timezone=True))  # When to publish (null = ASAP)
    priority = Column(Integer, default=50)  # Higher = sooner (for manual bumps)

    # Status
    status = Column(String(50), default="pending")  # pending, approved, publishing, published, failed, skipped

    # Platform targeting (subset of config.platforms for this specific post)
    target_platforms = Column(JSONB, default=list)

    # Results
    blotato_post_ids = Column(JSONB, default=dict)  # {platform: post_id}
    published_at = Column(DateTime(timezone=True))
    error_message = Column(Text)

    # Approval workflow
    requires_approval = Column(Boolean, default=True)
    approved_by = Column(String(100))  # User who approved
    approved_at = Column(DateTime(timezone=True))

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    clip = relationship("ViralClip", back_populates="publishing_queue_items")
    config = relationship("PublishingConfig", back_populates="queue_items")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
