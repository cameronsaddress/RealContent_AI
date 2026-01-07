from sqlalchemy import (
    Column, Integer, String, Text, Float, DateTime, ForeignKey,
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


class ContentIdea(Base):
    __tablename__ = "content_ideas"

    id = Column(Integer, primary_key=True, index=True)
    source_url = Column(Text)
    source_platform = Column(SQLEnum(PlatformType, name="platform_type", create_type=False))
    original_text = Column(Text)
    pillar = Column(SQLEnum(PillarType, name="pillar_type", create_type=False))
    viral_score = Column(Integer)
    suggested_hook = Column(Text)
    status = Column(SQLEnum(ContentStatus, name="content_status", create_type=False), default=ContentStatus.pending)
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

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


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
