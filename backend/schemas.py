from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ContentStatus(str, Enum):
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


class PillarType(str, Enum):
    market_intelligence = "market_intelligence"
    educational_tips = "educational_tips"
    lifestyle_local = "lifestyle_local"
    brand_humanization = "brand_humanization"


class PlatformType(str, Enum):
    tiktok = "tiktok"
    reddit = "reddit"
    youtube = "youtube"
    twitter = "twitter"
    instagram = "instagram"
    linkedin = "linkedin"
    facebook = "facebook"
    threads = "threads"
    pinterest = "pinterest"
    unknown = "unknown"


# Content Ideas Schemas
class ContentIdeaBase(BaseModel):
    source_url: Optional[str] = None
    source_platform: Optional[PlatformType] = None
    original_text: Optional[str] = None
    pillar: Optional[PillarType] = None
    viral_score: Optional[int] = Field(None, ge=1, le=10)
    suggested_hook: Optional[str] = None
    status: ContentStatus = ContentStatus.pending
    error_message: Optional[str] = None


class ContentIdeaCreate(ContentIdeaBase):
    pass


class ContentIdeaUpdate(BaseModel):
    source_url: Optional[str] = None
    source_platform: Optional[PlatformType] = None
    original_text: Optional[str] = None
    pillar: Optional[PillarType] = None
    viral_score: Optional[int] = Field(None, ge=1, le=10)
    suggested_hook: Optional[str] = None
    status: Optional[ContentStatus] = None
    error_message: Optional[str] = None


class ContentIdea(ContentIdeaBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Script Schemas
class ScriptBase(BaseModel):
    content_idea_id: int
    hook: Optional[str] = None
    body: Optional[str] = None
    cta: Optional[str] = None
    full_script: Optional[str] = None
    duration_estimate: Optional[int] = None
    tiktok_caption: Optional[str] = None
    ig_caption: Optional[str] = None
    yt_title: Optional[str] = None
    yt_description: Optional[str] = None
    linkedin_text: Optional[str] = None
    x_text: Optional[str] = None
    facebook_text: Optional[str] = None
    threads_text: Optional[str] = None
    status: ContentStatus = ContentStatus.pending


class ScriptCreate(ScriptBase):
    pass


class ScriptUpdate(BaseModel):
    hook: Optional[str] = None
    body: Optional[str] = None
    cta: Optional[str] = None
    full_script: Optional[str] = None
    duration_estimate: Optional[int] = None
    tiktok_caption: Optional[str] = None
    ig_caption: Optional[str] = None
    yt_title: Optional[str] = None
    yt_description: Optional[str] = None
    linkedin_text: Optional[str] = None
    x_text: Optional[str] = None
    facebook_text: Optional[str] = None
    threads_text: Optional[str] = None
    status: Optional[ContentStatus] = None


class Script(ScriptBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Asset Schemas
class AssetBase(BaseModel):
    script_id: int
    voiceover_path: Optional[str] = None
    voiceover_duration: Optional[float] = None
    srt_path: Optional[str] = None
    ass_path: Optional[str] = None
    avatar_video_path: Optional[str] = None
    background_video_path: Optional[str] = None
    combined_video_path: Optional[str] = None
    final_video_path: Optional[str] = None
    status: ContentStatus = ContentStatus.pending
    error_message: Optional[str] = None


class AssetCreate(AssetBase):
    pass


class AssetUpdate(BaseModel):
    voiceover_path: Optional[str] = None
    voiceover_duration: Optional[float] = None
    srt_path: Optional[str] = None
    ass_path: Optional[str] = None
    avatar_video_path: Optional[str] = None
    background_video_path: Optional[str] = None
    combined_video_path: Optional[str] = None
    final_video_path: Optional[str] = None
    status: Optional[ContentStatus] = None
    error_message: Optional[str] = None


class Asset(AssetBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Published Schemas
class PublishedBase(BaseModel):
    asset_id: int
    tiktok_url: Optional[str] = None
    tiktok_id: Optional[str] = None
    ig_url: Optional[str] = None
    ig_id: Optional[str] = None
    yt_url: Optional[str] = None
    yt_id: Optional[str] = None
    linkedin_url: Optional[str] = None
    linkedin_id: Optional[str] = None
    x_url: Optional[str] = None
    x_id: Optional[str] = None
    facebook_url: Optional[str] = None
    facebook_id: Optional[str] = None
    threads_url: Optional[str] = None
    threads_id: Optional[str] = None
    pinterest_url: Optional[str] = None
    pinterest_id: Optional[str] = None


class PublishedCreate(PublishedBase):
    pass


class PublishedUpdate(BaseModel):
    tiktok_url: Optional[str] = None
    tiktok_id: Optional[str] = None
    ig_url: Optional[str] = None
    ig_id: Optional[str] = None
    yt_url: Optional[str] = None
    yt_id: Optional[str] = None
    linkedin_url: Optional[str] = None
    linkedin_id: Optional[str] = None
    x_url: Optional[str] = None
    x_id: Optional[str] = None
    facebook_url: Optional[str] = None
    facebook_id: Optional[str] = None
    threads_url: Optional[str] = None
    threads_id: Optional[str] = None
    pinterest_url: Optional[str] = None
    pinterest_id: Optional[str] = None


class Published(PublishedBase):
    id: int
    published_at: datetime
    created_at: datetime

    class Config:
        from_attributes = True


# Analytics Schemas
class AnalyticsBase(BaseModel):
    published_id: int
    platform: PlatformType
    views: int = 0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    engagement_rate: Optional[float] = None


class AnalyticsCreate(AnalyticsBase):
    pass


class AnalyticsUpdate(BaseModel):
    views: Optional[int] = None
    likes: Optional[int] = None
    comments: Optional[int] = None
    shares: Optional[int] = None
    engagement_rate: Optional[float] = None


class Analytics(AnalyticsBase):
    id: int
    recorded_at: datetime

    class Config:
        from_attributes = True


# Pipeline Overview Schema (for the view)
class PipelineOverview(BaseModel):
    content_id: int
    source_platform: Optional[PlatformType]
    pillar: Optional[PillarType]
    viral_score: Optional[int]
    content_status: ContentStatus
    script_id: Optional[int]
    script_status: Optional[ContentStatus]
    duration_estimate: Optional[int]
    asset_id: Optional[int]
    asset_status: Optional[ContentStatus]
    published_id: Optional[int]
    published_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


# Stats Schema
class PipelineStats(BaseModel):
    total_ideas: int
    pending_ideas: int
    approved_ideas: int
    scripts_ready: int
    assets_ready: int
    published: int
    errors: int
    by_pillar: dict
    by_platform: dict


# Scrape Run Schemas
class ScrapeRunStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class ScrapeRunBase(BaseModel):
    niche: str
    hashtags: Optional[List[str]] = None
    platforms: Optional[List[str]] = ["tiktok", "instagram"]


class ScrapeRunCreate(ScrapeRunBase):
    pass


class ScrapeRun(ScrapeRunBase):
    id: int
    status: ScrapeRunStatus
    results_count: int
    results_data: Optional[dict] = None
    error_message: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class TrendItem(BaseModel):
    url: str
    platform: str
    title: str
    views: int
    likes: int
    viral_score: int
    pillar: str
    suggested_hook: str
    why_viral: Optional[str] = None
    adaptable: bool = True


class ScrapeResponse(BaseModel):
    success: bool
    niche: str
    scrapedAt: str
    totalScraped: int
    analyzedCount: int
    trends: Optional[List[TrendItem]] = None
    error: Optional[str] = None


# Niche Preset Schemas
class NichePresetBase(BaseModel):
    name: str
    keywords: List[str]
    hashtags: List[str]


class NichePresetCreate(NichePresetBase):
    pass


class NichePreset(NichePresetBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


# System Settings & Character Config
class CharacterConfig(BaseModel):
    voice_id: str
    avatar_id: str
    avatar_type: str = "pretrained"  # pretrained, static, generated
    image_url: Optional[str] = None
    voice_name: Optional[str] = None
    avatar_name: Optional[str] = None


class SystemSettingsBase(BaseModel):
    key: str
    value: Dict[str, Any]


class SystemSettingsCreate(SystemSettingsBase):
    pass


class SystemSettings(SystemSettingsBase):
    updated_at: datetime

    class Config:
        from_attributes = True


class AvatarGenerationRequest(BaseModel):
    prompt_enhancements: Optional[str] = None


class AvatarGenerationResponse(BaseModel):
    image_url: str
