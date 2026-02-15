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


# ==================== ASSET SCHEMAS ====================
class AssetBase(BaseModel):
    script_id: Optional[int] = None
    voiceover_path: Optional[str] = None
    voiceover_duration: Optional[float] = None
    srt_path: Optional[str] = None
    ass_path: Optional[str] = None
    avatar_video_path: Optional[str] = None
    heygen_video_id: Optional[str] = None
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
    heygen_video_id: Optional[str] = None
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


# ==================== SCRIPT SCHEMAS ====================
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
    assets: List[Asset] = []

    class Config:
        from_attributes = True


# ==================== CONTENT IDEA SCHEMAS ====================
class ContentIdeaBase(BaseModel):
    source_url: Optional[str] = None
    source_platform: Optional[PlatformType] = None
    original_text: Optional[str] = None
    pillar: Optional[PillarType] = None
    viral_score: Optional[int] = Field(None, ge=1, le=10)
    suggested_hook: Optional[str] = None
    status: ContentStatus = ContentStatus.pending
    error_message: Optional[str] = None
    # Engagement metrics
    views: Optional[int] = 0
    likes: Optional[int] = 0
    shares: Optional[int] = 0
    comments: Optional[int] = 0
    author: Optional[str] = None
    author_followers: Optional[int] = 0


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
    views: Optional[int] = None
    likes: Optional[int] = None
    shares: Optional[int] = None
    comments: Optional[int] = None
    author: Optional[str] = None
    author_followers: Optional[int] = None


class ContentIdea(ContentIdeaBase):
    id: int
    created_at: datetime
    updated_at: datetime
    scripts: List[Script] = []

    class Config:
        from_attributes = True


# ==================== PUBLISHED SCHEMAS ====================
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


# ==================== ANALYTICS SCHEMAS ====================
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


# ==================== PIPELINE OVERVIEW ====================
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


# ==================== STATS ====================
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


# ==================== SCRAPE RUN SCHEMAS ====================
class ScrapeRunStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class ScrapeRunBase(BaseModel):
    niche: str
    hashtags: Optional[List[str]] = None
    platforms: Optional[List[str]] = ["tiktok", "instagram"]
    # Enable 2-phase discovery: first find trending hashtags, then scrape
    discover_hashtags: bool = False
    # Seed keyword for hashtag discovery (e.g., "realtor")
    seed_keyword: Optional[str] = None


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
    message: Optional[str] = None
    niche: Optional[str] = None
    scrapedAt: Optional[str] = None
    totalScraped: Optional[int] = None
    analyzedCount: Optional[int] = None
    ideas_created: Optional[int] = None
    analyzed_count: Optional[int] = None
    trends: Optional[List[TrendItem]] = None
    error: Optional[str] = None
    task_id: Optional[str] = None
    scrape_run_id: Optional[int] = None


# ==================== NICHE PRESET SCHEMAS ====================
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


# ==================== SYSTEM SETTINGS ====================
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


# ==================== AUDIO SETTINGS SCHEMAS ====================

class AudioSettingsBase(BaseModel):
    original_volume: float = 0.7
    avatar_volume: float = 1.0
    music_volume: float = 0.3
    ducking_enabled: bool = True
    avatar_delay_seconds: float = 3.0
    duck_to_percent: float = 0.5
    music_autoduck: bool = True


class AudioSettingsUpdate(BaseModel):
    original_volume: Optional[float] = None
    avatar_volume: Optional[float] = None
    music_volume: Optional[float] = None
    ducking_enabled: Optional[bool] = None
    avatar_delay_seconds: Optional[float] = None
    duck_to_percent: Optional[float] = None
    music_autoduck: Optional[bool] = None


class AudioSettingsResponse(AudioSettingsBase):
    id: int
    updated_at: datetime

    class Config:
        from_attributes = True


# ==================== VIDEO SETTINGS SCHEMAS ====================

class VideoSettingsBase(BaseModel):
    output_width: int = 1080
    output_height: int = 1920
    output_format: str = 'mp4'
    codec: str = 'libx264'
    crf: int = 18
    preset: str = 'slow'
    greenscreen_enabled: bool = True
    greenscreen_color: str = '#00FF00'
    # Avatar composition
    avatar_position: str = 'bottom-left'
    avatar_scale: float = 0.8
    avatar_offset_x: int = -200
    avatar_offset_y: int = 500
    # Caption settings
    caption_style: str = 'karaoke'
    caption_font_size: int = 96
    caption_font: str = 'Arial'
    caption_color: str = '#FFFFFF'
    caption_highlight_color: str = '#FFFF00'
    caption_outline_color: str = '#000000'
    caption_outline_width: int = 5
    caption_position_y: int = 850


class VideoSettingsUpdate(BaseModel):
    output_width: Optional[int] = None
    output_height: Optional[int] = None
    output_format: Optional[str] = None
    codec: Optional[str] = None
    crf: Optional[int] = None
    preset: Optional[str] = None
    greenscreen_enabled: Optional[bool] = None
    greenscreen_color: Optional[str] = None
    # Avatar composition
    avatar_position: Optional[str] = None
    avatar_scale: Optional[float] = None
    avatar_offset_x: Optional[int] = None
    avatar_offset_y: Optional[int] = None
    # Caption settings
    caption_style: Optional[str] = None
    caption_font_size: Optional[int] = None
    caption_font: Optional[str] = None
    caption_color: Optional[str] = None
    caption_highlight_color: Optional[str] = None
    caption_outline_color: Optional[str] = None
    caption_outline_width: Optional[int] = None
    caption_position_y: Optional[int] = None


class VideoSettingsResponse(VideoSettingsBase):
    id: int
    updated_at: datetime

    class Config:
        from_attributes = True


# ==================== BRAND PERSONA SCHEMAS ====================

class BrandPersonaBase(BaseModel):
    # Identity
    name: str = "Sarah Mitchell"
    title: str = "Real Estate Expert"
    location: str = "North Idaho & Spokane area"
    bio: str = "A knowledgeable and approachable real estate professional who helps first-time homebuyers and families find their dream homes in the beautiful Pacific Northwest."

    # Tone & Voice
    tone: str = "professional"  # professional, casual, energetic, warm, authoritative
    energy_level: str = "warm"  # calm, warm, energetic, high-energy
    humor_style: str = "light"  # none, light, playful, witty

    # Values & Approach
    core_values: List[str] = [
        "Honesty and transparency",
        "Client-first approach",
        "Education over sales",
        "Community connection",
        "Professional excellence"
    ]

    # Content Boundaries
    content_boundaries: List[str] = [
        "No dancing or twerking - we maintain professional dignity",
        "No clickbait or misleading claims",
        "No putting down other realtors or competitors",
        "No political or controversial topics",
        "No inappropriate language or innuendo"
    ]

    # How we respond to different content
    response_style: str = """When reviewing viral content:
- If the content is professional and educational: Praise it, add our own insights, and connect it to our local market
- If the content is entertaining but unprofessional: Acknowledge the entertainment value, then pivot to show a more professional approach
- If the content has misinformation: Gently correct it while being respectful to the creator
- If the content is cringe or inappropriate: Focus on the underlying topic, not the presentation style
- Always provide genuine value - actionable tips, local insights, or helpful information"""

    # Signature phrases
    signature_intro: str = "Hey neighbors!"
    signature_cta: str = "DM me to chat about your home journey in {location}"
    hashtags: List[str] = ["CDAhomes", "LibertyLake", "NorthIdahoRealEstate", "PNWliving"]


class BrandPersonaUpdate(BaseModel):
    name: Optional[str] = None
    title: Optional[str] = None
    location: Optional[str] = None
    bio: Optional[str] = None
    tone: Optional[str] = None
    energy_level: Optional[str] = None
    humor_style: Optional[str] = None
    core_values: Optional[List[str]] = None
    content_boundaries: Optional[List[str]] = None
    response_style: Optional[str] = None
    signature_intro: Optional[str] = None
    signature_cta: Optional[str] = None
    hashtags: Optional[List[str]] = None


class BrandPersonaResponse(BrandPersonaBase):
    id: int
    updated_at: datetime

    class Config:
        from_attributes = True


# ==================== LLM SETTINGS SCHEMAS ====================

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


# ==================== COMBINED SETTINGS ====================

class AllSettingsResponse(BaseModel):
    audio: Dict[str, Any]
    video: Dict[str, Any]
    llm: Dict[str, str]
