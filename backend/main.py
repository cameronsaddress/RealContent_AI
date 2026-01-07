from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session, joinedload

from sqlalchemy import func, text
from typing import List, Optional
import os
import base64
import uuid
from datetime import datetime

import httpx
from models import (
    get_db, ContentIdea as ContentIdeaModel, Script as ScriptModel,
    Asset as AssetModel, Published as PublishedModel, Analytics as AnalyticsModel,
    PipelineRun as PipelineRunModel, ContentStatus as DBContentStatus,
    PillarType as DBPillarType, PlatformType as DBPlatformType,
    ScrapeRun as ScrapeRunModel, NichePreset as NichePresetModel,
    SystemSettings as SystemSettingsModel,
    ScrapeRunStatus as DBScrapeRunStatus, Base, engine
)
from schemas import (
    ContentIdea, ContentIdeaCreate, ContentIdeaUpdate,
    Script, ScriptCreate, ScriptUpdate,
    Asset, AssetCreate, AssetUpdate,
    Published, PublishedCreate, PublishedUpdate,
    Analytics, AnalyticsCreate, AnalyticsUpdate,
    PipelineOverview, PipelineStats, ContentStatus, PillarType, PlatformType,
    ScrapeRun, ScrapeRunCreate, ScrapeResponse, ScrapeRunStatus,
    NichePreset, NichePresetCreate,
    SystemSettings, SystemSettingsCreate, CharacterConfig,
    AvatarGenerationRequest, AvatarGenerationResponse
)

app = FastAPI(
    title="Content Pipeline API",
    description="API for managing automated AI video content pipeline",
    version="1.0.0"
)

# Create tables
Base.metadata.create_all(bind=engine)

# CORS configuration
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return {"message": "Content Pipeline API", "version": "1.0.0"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# ==================== Content Ideas ====================

@app.get("/api/content-ideas", response_model=List[ContentIdea])
def list_content_ideas(
    status: Optional[ContentStatus] = None,
    pillar: Optional[PillarType] = None,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
    db: Session = Depends(get_db)
):
    query = db.query(ContentIdeaModel).options(
        joinedload(ContentIdeaModel.scripts).joinedload(ScriptModel.assets)
    )
    if status:
        query = query.filter(ContentIdeaModel.status == status.value)
    if pillar:
        query = query.filter(ContentIdeaModel.pillar == pillar.value)
    return query.order_by(ContentIdeaModel.created_at.desc()).offset(offset).limit(limit).all()


@app.get("/api/content-ideas/{idea_id}", response_model=ContentIdea)
def get_content_idea(idea_id: int, db: Session = Depends(get_db)):
    idea = db.query(ContentIdeaModel).filter(ContentIdeaModel.id == idea_id).first()
    if not idea:
        raise HTTPException(status_code=404, detail="Content idea not found")
    return idea


@app.post("/api/content-ideas", response_model=ContentIdea)
def create_content_idea(idea: ContentIdeaCreate, db: Session = Depends(get_db)):
    db_idea = ContentIdeaModel(**idea.model_dump())
    db.add(db_idea)
    db.commit()
    db.refresh(db_idea)
    return db_idea


@app.patch("/api/content-ideas/{idea_id}", response_model=ContentIdea)
async def update_content_idea(idea_id: int, idea: ContentIdeaUpdate, db: Session = Depends(get_db)):
    db_idea = db.query(ContentIdeaModel).filter(ContentIdeaModel.id == idea_id).first()
    if not db_idea:
        raise HTTPException(status_code=404, detail="Content idea not found")

    update_data = idea.model_dump(exclude_unset=True)
    print(f"DEBUG: PATCH received for idea {idea_id}. Data: {update_data}", flush=True)
    status_changed_to_approved = False
    
    for key, value in update_data.items():
        setattr(db_idea, key, value)
        # Check for status change to approved (handle both string and Enum)
        if key == "status" and str(value) == "approved":
            status_changed_to_approved = True
            print(f"DEBUG: Status change detected: {value}", flush=True)

    db.commit()
    db.refresh(db_idea)
    
    # If not detected in loop, check final state as fallback
    if not status_changed_to_approved and db_idea.status == DBContentStatus.approved and "status" in update_data:
        print(f"DEBUG: Fallback status check passed. Status is approved.", flush=True)
        status_changed_to_approved = True

    if status_changed_to_approved:
        # Trigger n8n pipeline
        N8N_PIPELINE_WEBHOOK_URL = os.getenv("N8N_PIPELINE_WEBHOOK_URL", "http://n8n:5678/webhook/trigger-pipeline")
        print(f"DEBUG: Status changed to approved. Triggering webhook at {N8N_PIPELINE_WEBHOOK_URL}")
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    N8N_PIPELINE_WEBHOOK_URL,
                    json={"content_idea_id": idea_id},
                    timeout=5.0
                )
                print(f"DEBUG: Webhook response: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"ERROR: Failed to trigger pipeline for idea {idea_id}: {e}")

    return db_idea


@app.delete("/api/content-ideas/{idea_id}")
def delete_content_idea(idea_id: int, db: Session = Depends(get_db)):
    db_idea = db.query(ContentIdeaModel).filter(ContentIdeaModel.id == idea_id).first()
    if not db_idea:
        raise HTTPException(status_code=404, detail="Content idea not found")
    db.delete(db_idea)
    db.commit()
    return {"message": "Content idea deleted"}


# ==================== Scripts ====================

@app.get("/api/scripts", response_model=List[Script])
def list_scripts(
    status: Optional[ContentStatus] = None,
    content_idea_id: Optional[int] = None,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
    db: Session = Depends(get_db)
):
    query = db.query(ScriptModel)
    if status:
        query = query.filter(ScriptModel.status == status.value)
    if content_idea_id:
        query = query.filter(ScriptModel.content_idea_id == content_idea_id)
    return query.order_by(ScriptModel.created_at.desc()).offset(offset).limit(limit).all()


@app.get("/api/scripts/{script_id}", response_model=Script)
def get_script(script_id: int, db: Session = Depends(get_db)):
    script = db.query(ScriptModel).filter(ScriptModel.id == script_id).first()
    if not script:
        raise HTTPException(status_code=404, detail="Script not found")
    return script


@app.post("/api/scripts", response_model=Script)
def create_script(script: ScriptCreate, db: Session = Depends(get_db)):
    # Check for existing script to support idempotency/retries
    existing_script = db.query(ScriptModel).filter(
        ScriptModel.content_idea_id == script.content_idea_id
    ).order_by(ScriptModel.created_at.desc()).first()
    
    if existing_script:
        print(f"DEBUG: Returning existing script {existing_script.id} for idea {script.content_idea_id}", flush=True)
        # Ensure parent status is updated even on retry
        content_idea = db.query(ContentIdeaModel).filter(ContentIdeaModel.id == script.content_idea_id).first()
        if content_idea and content_idea.status != DBContentStatus.script_ready:
            print(f"DEBUG: Updating ContentIdea {script.content_idea_id} status to script_ready (retry)", flush=True)
            content_idea.status = DBContentStatus.script_ready
            db.add(content_idea)
            db.commit()
            
        return existing_script

    db_script = ScriptModel(**script.model_dump())
    db.add(db_script)
    
    # Update parent status
    content_idea = db.query(ContentIdeaModel).filter(ContentIdeaModel.id == script.content_idea_id).first()
    if content_idea:
        print(f"DEBUG: Updating ContentIdea {script.content_idea_id} status to script_ready", flush=True)
        content_idea.status = DBContentStatus.script_ready
        db.add(content_idea)
        
    db.commit()
    db.refresh(db_script)
    return db_script


@app.patch("/api/scripts/{script_id}", response_model=Script)
def update_script(script_id: int, script: ScriptUpdate, db: Session = Depends(get_db)):
    db_script = db.query(ScriptModel).filter(ScriptModel.id == script_id).first()
    if not db_script:
        raise HTTPException(status_code=404, detail="Script not found")

    update_data = script.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_script, key, value)

    db.commit()
    db.refresh(db_script)
    return db_script


@app.delete("/api/scripts/{script_id}")
def delete_script(script_id: int, db: Session = Depends(get_db)):
    db_script = db.query(ScriptModel).filter(ScriptModel.id == script_id).first()
    if not db_script:
        raise HTTPException(status_code=404, detail="Script not found")
    db.delete(db_script)
    db.commit()
    return {"message": "Script deleted"}


# ==================== Assets ====================

@app.get("/api/assets", response_model=List[Asset])
def list_assets(
    status: Optional[ContentStatus] = None,
    script_id: Optional[int] = None,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
    db: Session = Depends(get_db)
):
    query = db.query(AssetModel)
    if status:
        query = query.filter(AssetModel.status == status.value)
    if script_id:
        query = query.filter(AssetModel.script_id == script_id)
    return query.order_by(AssetModel.created_at.desc()).offset(offset).limit(limit).all()


@app.get("/api/assets/{asset_id}", response_model=Asset)
def get_asset(asset_id: int, db: Session = Depends(get_db)):
    asset = db.query(AssetModel).filter(AssetModel.id == asset_id).first()
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    return asset


@app.post("/api/assets", response_model=Asset)
def create_asset(asset: AssetCreate, db: Session = Depends(get_db)):
    # Check for existing asset to support idempotency/retries
    existing_asset = db.query(AssetModel).filter(
        AssetModel.script_id == asset.script_id
    ).order_by(AssetModel.created_at.desc()).first()
    
    if existing_asset:
        print(f"DEBUG: Returning existing asset {existing_asset.id} for script {asset.script_id}", flush=True)
        return existing_asset

    db_asset = AssetModel(**asset.model_dump())
    db.add(db_asset)
    db.commit()
    db.refresh(db_asset)
    return db_asset


@app.patch("/api/assets/{asset_id}", response_model=Asset)
def update_asset(asset_id: int, asset: AssetUpdate, db: Session = Depends(get_db)):
    db_asset = db.query(AssetModel).filter(AssetModel.id == asset_id).first()
    if not db_asset:
        raise HTTPException(status_code=404, detail="Asset not found")

    update_data = asset.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_asset, key, value)

    db.commit()
    db.refresh(db_asset)
    return db_asset


@app.delete("/api/assets/{asset_id}")
def delete_asset(asset_id: int, db: Session = Depends(get_db)):
    db_asset = db.query(AssetModel).filter(AssetModel.id == asset_id).first()
    if not db_asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    db.delete(db_asset)
    db.commit()
    return {"message": "Asset deleted"}


# ==================== Published ====================

@app.get("/api/published", response_model=List[Published])
def list_published(
    asset_id: Optional[int] = None,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
    db: Session = Depends(get_db)
):
    query = db.query(PublishedModel)
    if asset_id:
        query = query.filter(PublishedModel.asset_id == asset_id)
    return query.order_by(PublishedModel.published_at.desc()).offset(offset).limit(limit).all()


@app.get("/api/published/{published_id}", response_model=Published)
def get_published(published_id: int, db: Session = Depends(get_db)):
    published = db.query(PublishedModel).filter(PublishedModel.id == published_id).first()
    if not published:
        raise HTTPException(status_code=404, detail="Published record not found")
    return published


@app.post("/api/published", response_model=Published)
def create_published(published: PublishedCreate, db: Session = Depends(get_db)):
    db_published = PublishedModel(**published.model_dump())
    db.add(db_published)
    db.commit()
    db.refresh(db_published)
    return db_published


@app.patch("/api/published/{published_id}", response_model=Published)
def update_published(published_id: int, published: PublishedUpdate, db: Session = Depends(get_db)):
    db_published = db.query(PublishedModel).filter(PublishedModel.id == published_id).first()
    if not db_published:
        raise HTTPException(status_code=404, detail="Published record not found")

    update_data = published.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_published, key, value)

    db.commit()
    db.refresh(db_published)
    return db_published


# ==================== Analytics ====================

@app.get("/api/analytics", response_model=List[Analytics])
def list_analytics(
    published_id: Optional[int] = None,
    platform: Optional[PlatformType] = None,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
    db: Session = Depends(get_db)
):
    query = db.query(AnalyticsModel)
    if published_id:
        query = query.filter(AnalyticsModel.published_id == published_id)
    if platform:
        query = query.filter(AnalyticsModel.platform == platform.value)
    return query.order_by(AnalyticsModel.recorded_at.desc()).offset(offset).limit(limit).all()


@app.post("/api/analytics", response_model=Analytics)
def create_analytics(analytics: AnalyticsCreate, db: Session = Depends(get_db)):
    db_analytics = AnalyticsModel(**analytics.model_dump())
    db.add(db_analytics)
    db.commit()
    db.refresh(db_analytics)
    return db_analytics


@app.patch("/api/analytics/{analytics_id}", response_model=Analytics)
def update_analytics(analytics_id: int, analytics: AnalyticsUpdate, db: Session = Depends(get_db)):
    db_analytics = db.query(AnalyticsModel).filter(AnalyticsModel.id == analytics_id).first()
    if not db_analytics:
        raise HTTPException(status_code=404, detail="Analytics record not found")

    update_data = analytics.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_analytics, key, value)

    db.commit()
    db.refresh(db_analytics)
    return db_analytics


# ==================== Pipeline Overview & Stats ====================

@app.get("/api/pipeline/overview", response_model=List[PipelineOverview])
def get_pipeline_overview(
    limit: int = Query(default=50, le=200),
    offset: int = 0,
    db: Session = Depends(get_db)
):
    result = db.execute(text("""
        SELECT * FROM pipeline_overview
        LIMIT :limit OFFSET :offset
    """), {"limit": limit, "offset": offset})

    rows = result.fetchall()
    return [PipelineOverview(
        content_id=row.content_id,
        source_platform=row.source_platform,
        pillar=row.pillar,
        viral_score=row.viral_score,
        content_status=row.content_status,
        script_id=row.script_id,
        script_status=row.script_status,
        duration_estimate=row.duration_estimate,
        asset_id=row.asset_id,
        asset_status=row.asset_status,
        published_id=row.published_id,
        published_at=row.published_at,
        created_at=row.created_at
    ) for row in rows]


@app.get("/api/pipeline/stats", response_model=PipelineStats)
def get_pipeline_stats(db: Session = Depends(get_db)):
    total = db.query(ContentIdeaModel).count()
    pending = db.query(ContentIdeaModel).filter(ContentIdeaModel.status == "pending").count()
    approved = db.query(ContentIdeaModel).filter(ContentIdeaModel.status == "approved").count()
    scripts_ready = db.query(ScriptModel).filter(ScriptModel.status == "script_ready").count()
    assets_ready = db.query(AssetModel).filter(AssetModel.status == "ready_to_publish").count()
    published = db.query(PublishedModel).count()
    errors = db.query(ContentIdeaModel).filter(ContentIdeaModel.status == "error").count()

    # By pillar
    pillar_counts = db.query(
        ContentIdeaModel.pillar,
        func.count(ContentIdeaModel.id)
    ).group_by(ContentIdeaModel.pillar).all()
    by_pillar = {str(p[0].value) if p[0] else "unknown": p[1] for p in pillar_counts}

    # By platform
    platform_counts = db.query(
        ContentIdeaModel.source_platform,
        func.count(ContentIdeaModel.id)
    ).group_by(ContentIdeaModel.source_platform).all()
    by_platform = {str(p[0].value) if p[0] else "unknown": p[1] for p in platform_counts}

    return PipelineStats(
        total_ideas=total,
        pending_ideas=pending,
        approved_ideas=approved,
        scripts_ready=scripts_ready,
        assets_ready=assets_ready,
        published=published,
        errors=errors,
        by_pillar=by_pillar,
        by_platform=by_platform
    )


# ==================== Bulk Operations ====================

@app.post("/api/content-ideas/bulk-approve")
async def bulk_approve_ideas(idea_ids: List[int], db: Session = Depends(get_db)):
    updated = db.query(ContentIdeaModel).filter(
        ContentIdeaModel.id.in_(idea_ids),
        ContentIdeaModel.status == "pending"
    ).update({"status": "approved"}, synchronize_session=False)
    db.commit()

    # Trigger n8n pipeline for each approved idea
    N8N_PIPELINE_WEBHOOK_URL = os.getenv("N8N_PIPELINE_WEBHOOK_URL", "http://n8n:5678/webhook/trigger-pipeline")
    
    async with httpx.AsyncClient() as client:
        for idea_id in idea_ids:
            try:
                # We fire the webhook to start script generation
                await client.post(
                    N8N_PIPELINE_WEBHOOK_URL,
                    json={"content_idea_id": idea_id},
                    timeout=5.0
                )
            except Exception as e:
                print(f"Failed to trigger pipeline for idea {idea_id}: {e}")

    return {"message": f"Approved {updated} ideas"}


@app.post("/api/content-ideas/bulk-reject")
def bulk_reject_ideas(idea_ids: List[int], db: Session = Depends(get_db)):
    updated = db.query(ContentIdeaModel).filter(
        ContentIdeaModel.id.in_(idea_ids),
        ContentIdeaModel.status == "pending"
    ).update({"status": "rejected"}, synchronize_session=False)
    db.commit()
    return {"message": f"Rejected {updated} ideas"}


# ==================== Trend Scraping ====================

N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "http://n8n:5678/webhook/scrape-trends")


@app.post("/api/scrape/run", response_model=ScrapeResponse)
async def run_scrape(scrape_request: ScrapeRunCreate, db: Session = Depends(get_db)):
    """Trigger a trend scrape with niche targeting"""
    # Create scrape run record
    db_run = ScrapeRunModel(
        niche=scrape_request.niche,
        hashtags=scrape_request.hashtags,
        platforms=scrape_request.platforms,
        status=DBScrapeRunStatus.running
    )
    db.add(db_run)
    db.commit()
    db.refresh(db_run)

    # Build hashtags from niche if not provided
    hashtags = scrape_request.hashtags
    if not hashtags:
        # Generate hashtags from niche keywords
        niche_words = scrape_request.niche.lower().replace(',', ' ').split()
        hashtags = [w.strip() for w in niche_words if len(w.strip()) > 2]

    # Call n8n webhook
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                N8N_WEBHOOK_URL,
                json={
                    "niche": scrape_request.niche,
                    "hashtags": hashtags,
                    "platforms": scrape_request.platforms or ["tiktok", "instagram"],
                    "results_per_platform": 20
                }
            )
            result = response.json()

        # Update scrape run with results
        db_run.status = DBScrapeRunStatus.completed if result.get("success") else DBScrapeRunStatus.failed
        db_run.results_count = result.get("analyzedCount", 0)
        db_run.results_data = result
        db_run.completed_at = func.now()
        if not result.get("success"):
            db_run.error_message = result.get("error", "Unknown error")
        db.commit()

        return ScrapeResponse(**result)

    except Exception as e:
        db_run.status = DBScrapeRunStatus.failed
        db_run.error_message = str(e)
        db_run.completed_at = func.now()
        db.commit()
        raise HTTPException(status_code=500, detail=f"Scrape failed: {str(e)}")


@app.get("/api/scrape/runs", response_model=List[ScrapeRun])
def list_scrape_runs(
    limit: int = Query(default=20, le=100),
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """List recent scrape runs"""
    return db.query(ScrapeRunModel).order_by(
        ScrapeRunModel.started_at.desc()
    ).offset(offset).limit(limit).all()


@app.get("/api/scrape/runs/{run_id}", response_model=ScrapeRun)
def get_scrape_run(run_id: int, db: Session = Depends(get_db)):
    """Get a specific scrape run with results"""
    run = db.query(ScrapeRunModel).filter(ScrapeRunModel.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Scrape run not found")
    return run


@app.post("/api/scrape/use-trend")
def use_trend_as_idea(trend: dict, db: Session = Depends(get_db)):
    """Convert a trend into a content idea for the pipeline"""
    # Map platform string to enum
    platform_map = {
        "tiktok": DBPlatformType.tiktok,
        "instagram": DBPlatformType.instagram,
        "youtube": DBPlatformType.youtube
    }
    # Map pillar string to enum
    pillar_map = {
        "market_intelligence": DBPillarType.market_intelligence,
        "educational_tips": DBPillarType.educational_tips,
        "lifestyle_local": DBPillarType.lifestyle_local,
        "brand_humanization": DBPillarType.brand_humanization
    }

    db_idea = ContentIdeaModel(
        source_url=trend.get("url"),
        source_platform=platform_map.get(trend.get("platform")),
        original_text=trend.get("title", ""),
        pillar=pillar_map.get(trend.get("pillar")),
        viral_score=trend.get("viral_score", 7),
        suggested_hook=trend.get("suggested_hook", ""),
        status=DBContentStatus.pending
    )
    db.add(db_idea)
    db.commit()
    db.refresh(db_idea)
    return {"message": "Trend added to pipeline", "content_idea_id": db_idea.id}


# ==================== Niche Presets ====================

@app.get("/api/niche-presets", response_model=List[NichePreset])
def list_niche_presets(db: Session = Depends(get_db)):
    """List all saved niche presets"""
    return db.query(NichePresetModel).order_by(NichePresetModel.name).all()


@app.post("/api/niche-presets", response_model=NichePreset)
def create_niche_preset(preset: NichePresetCreate, db: Session = Depends(get_db)):
    """Create a new niche preset"""
    db_preset = NichePresetModel(
        name=preset.name,
        keywords=preset.keywords,
        hashtags=preset.hashtags
    )
    db.add(db_preset)
    db.commit()
    db.refresh(db_preset)
    return db_preset


@app.delete("/api/niche-presets/{preset_id}")
def delete_niche_preset(preset_id: int, db: Session = Depends(get_db)):
    """Delete a niche preset"""
    preset = db.query(NichePresetModel).filter(NichePresetModel.id == preset_id).first()
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")
    db.delete(preset)
    db.commit()
    return {"message": "Preset deleted"}


# ==================== Character & Settings ====================

@app.get("/api/settings/voices")
async def get_elevenlabs_voices():
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ELEVENLABS_API_KEY not configured")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                "https://api.elevenlabs.io/v1/voices",
                headers={"xi-api-key": api_key}
            )
            response.raise_for_status()
            data = response.json()
            voices = []
            for v in data.get("voices", []):
                voices.append({
                    "voice_id": v["voice_id"],
                    "name": v["name"],
                    "category": v.get("category"),
                    "preview_url": v.get("preview_url"),
                    "labels": v.get("labels", {})
                })
            return {"voices": voices}
        except Exception as e:
            # Fallback mock for development if API fails or quota exceeded
            print(f"ElevenLabs API Error: {e}")
            return {"voices": [
                {"voice_id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel", "preview_url": "", "category": "premade"},
                {"voice_id": "AZnzlk1XvdvUeBnXmlld", "name": "Domi", "preview_url": "", "category": "premade"},
                {"voice_id": "EXAVITQu4vr4xnSDxMaL", "name": "Bella", "preview_url": "", "category": "premade"},
            ]}


@app.get("/api/settings/avatars")
async def get_heygen_avatars():
    api_key = os.getenv("HEYGEN_API_KEY")
    if not api_key:
        print("HEYGEN_API_KEY not found, returning mock data.")
        return {"avatars": [
                {"avatar_id": "Angela-in-T-shirt-20220819", "name": "Angela (Mock)", "preview_image_url": "https://files.heygen.ai/avatar/v3/Angela-in-T-shirt-20220819/full/preview_target.webp"},
                {"avatar_id": "Anna_public_3_20240108", "name": "Anna (Mock)", "preview_image_url": "https://files.heygen.ai/avatar/v3/Anna_public_3_20240108/full/preview_target.webp"},
        ]}
        
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                "https://api.heygen.com/v2/avatars",
                headers={
                    "X-Api-Key": api_key,
                    "Content-Type": "application/json"
                }
            )
            response.raise_for_status()
            data = response.json()
            return {"avatars": data.get("data", {}).get("avatars", [])}
        except Exception as e:
             # Fallback mock
            print(f"HeyGen API Error: {e}")
            return {"avatars": [
                {"avatar_id": "Angela-in-T-shirt-20220819", "name": "Angela", "preview_image_url": "https://files.heygen.ai/avatar/v3/Angela-in-T-shirt-20220819/full/preview_target.webp"},
                {"avatar_id": "Anna_public_3_20240108", "name": "Anna", "preview_image_url": "https://files.heygen.ai/avatar/v3/Anna_public_3_20240108/full/preview_target.webp"},
            ]}


@app.get("/api/config/character", response_model=CharacterConfig)
def get_character_config(db: Session = Depends(get_db)):
    setting = db.query(SystemSettingsModel).filter(SystemSettingsModel.key == "active_character").first()
    if not setting or not setting.value:
        return CharacterConfig(
            voice_id="21m00Tcm4TlvDq8ikWAM",
            avatar_id="Angela-in-T-shirt-20220819",
            avatar_type="pretrained",
            voice_name="Rachel",
            avatar_name="Angela"
        )
    return CharacterConfig(**setting.value)


@app.post("/api/config/character", response_model=CharacterConfig)
def save_character_config(config: CharacterConfig, db: Session = Depends(get_db)):
    setting = db.query(SystemSettingsModel).filter(SystemSettingsModel.key == "active_character").first()
    if not setting:
        setting = SystemSettingsModel(key="active_character", value=config.model_dump())
        db.add(setting)
    else:
        setting.value = config.model_dump()
    
    db.commit()
    db.refresh(setting)
    return CharacterConfig(**setting.value)


@app.post("/api/generate-avatar-image", response_model=AvatarGenerationResponse)
async def generate_avatar_image(req: AvatarGenerationRequest, db: Session = Depends(get_db)):
    # Using OpenAI DALL-E 3 as a reliable fallback for 'AI Image Generation'
    # requesting b64_json to save locally as per requirement.
    api_key = os.getenv("OPENAI_API_KEY") 
    if not api_key:
        return JSONResponse(status_code=500, content={"detail": "OPENAI_API_KEY not configured"})
    
    # "Nano Banana" style realism prompt as requested
    # "Nano Banana" style realism prompt as requested
    base_prompt = (
        "Raw unedited photo of a real human woman, news anchor, looking at camera. "
        "Shot on Sony A7R IV. Hyper-realistic, 8k, pore-level skin detail, slight imperfections. "
        "Green screen background. Natural studio lighting. "
        "NOT an illustration, NOT 3D render. 100% photograph."
    )
    final_prompt = req.prompt_enhancements if req.prompt_enhancements else base_prompt

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.openai.com/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "dall-e-3",
                    "prompt": final_prompt,
                    "n": 1,
                    "size": "1024x1024",
                    "response_format": "b64_json"
                },
                timeout=60.0
            )
            response.raise_for_status()
            data = response.json()
            
            # Handle Base64
            b64_data = data["data"][0]["b64_json"]
            image_data = base64.b64decode(b64_data)
            
            # Save to static directory
            filename = f"{uuid.uuid4()}.png"
            AVATAR_DIR = os.path.join("static", "avatars")
            os.makedirs(AVATAR_DIR, exist_ok=True)
            
            file_path = os.path.join(AVATAR_DIR, filename)
            
            with open(file_path, "wb") as f:
                f.write(image_data)
                
            local_url = f"http://localhost:8000/static/avatars/{filename}"
            
            # Return relative path so frontend can construct the full URL
            # The frontend should prepend the API host, or we can use a relative protocol if served from same origin.
            # But here they are on different ports.
            # Returning relative path starting with /static
            relative_url = f"/static/avatars/{filename}"
            
            # Create Asset Record in DB for "Assets" page visibility
            # We use script_id=None to indicate a global/library asset
            new_asset = AssetModel(
                script_id=None,
                avatar_video_path=relative_url, # Store relative path or full? relative is safer for migrations.
                status=DBContentStatus.avatar_ready,
                created_at=datetime.utcnow()
            )
            db.add(new_asset)
            db.commit()

            return AvatarGenerationResponse(image_url=relative_url)
        except Exception as e:
            print(f"Image Gen Error: {e}")
            raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")
