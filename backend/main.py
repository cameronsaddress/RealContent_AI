from fastapi import FastAPI, Depends, HTTPException, Query, File, UploadFile, Request
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
import shutil
import json
from pathlib import Path

# Import the new pipeline router (n8n replacement)
from routers.pipeline import router as pipeline_router
from routers.viral import router as viral_router
from routers.publishing import router as publishing_router

# Import Celery tasks for direct triggering (replacing n8n webhooks)
from tasks.pipeline import run_pipeline
from tasks.scrape import run_scrape as run_scrape_task

from models import (
    get_db, ContentIdea as ContentIdeaModel, Script as ScriptModel,
    Asset as AssetModel, Published as PublishedModel, Analytics as AnalyticsModel,
    PipelineRun as PipelineRunModel, ContentStatus as DBContentStatus,
    PillarType as DBPillarType, PlatformType as DBPlatformType,
    ScrapeRun as ScrapeRunModel, NichePreset as NichePresetModel,
    SystemSettings as SystemSettingsModel,
    ScrapeRunStatus as DBScrapeRunStatus, Base, engine,
    AudioSettings as AudioSettingsModel,
    VideoSettings as VideoSettingsModel,
    LLMSettings as LLMSettingsModel
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
    AvatarGenerationRequest, AvatarGenerationResponse,
    AudioSettingsResponse, AudioSettingsUpdate,
    VideoSettingsResponse, VideoSettingsUpdate,
    LLMSettingResponse, LLMSettingUpdate, AllSettingsResponse
)

app = FastAPI(
    title="Content Pipeline API",
    description="API for managing automated AI video content pipeline",
    version="1.0.0"
)

# Create tables
Base.metadata.create_all(bind=engine)

def seed_settings():
    """Seed default settings if not present"""
    db = Session(bind=engine)
    try:
        # Seed Viral Prompt
        key = "VIRAL_SYSTEM_PROMPT"
        if not db.query(LLMSettingsModel).filter(LLMSettingsModel.key == key).first():
            default_prompt = """You are a viral content expert. Your goal is to find the most controversial, vulgar, intelligent, exciting, and viral moments. 
Focus on high-conflict debates, shock value, deep insights, or high-energy moments. 
Ignore boring or mundane filler. 
Prioritize clips that trigger strong emotional reactions (anger, laughter, awe).
Ensure clips have a clear hook and payoff."""
            db.add(LLMSettingsModel(key=key, value=default_prompt, description="Strategy for detecting viral clips"))
            db.commit()
            print("Seeded VIRAL_SYSTEM_PROMPT")
    except Exception as e:
        print(f"Error seeding settings: {e}")
    finally:
        db.close()

seed_settings()

# CORS configuration
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the pipeline router (replaces n8n webhooks)
app.include_router(pipeline_router)
app.include_router(viral_router)
app.include_router(publishing_router)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
# Mount assets for media playback (Music, Videos)
app.mount("/assets", StaticFiles(directory="/app/assets"), name="assets")

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
        # Analyze with LLM if not already analyzed (viral_score=0 means unanalyzed)
        if db_idea.viral_score == 0 or not db_idea.pillar:
            print(f"DEBUG: Analyzing idea {idea_id} with LLM before pipeline...")
            try:
                from services.scraper import ScraperService
                scraper = ScraperService()
                analysis = await scraper.analyze_single_idea(
                    idea_id=idea_id,
                    title=db_idea.original_text or "",
                    url=db_idea.source_url or "",
                    platform=db_idea.source_platform or "unknown",
                    views=db_idea.views or 0,
                    likes=db_idea.likes or 0
                )
                # Update idea with analysis
                db_idea.viral_score = analysis.get("viral_score", 7)
                db_idea.pillar = analysis.get("pillar", "educational_tips")
                db_idea.suggested_hook = analysis.get("suggested_hook", "")
                db_idea.why_viral = analysis.get("why_viral", "")
                db.commit()
                db.refresh(db_idea)
                print(f"DEBUG: Idea {idea_id} analyzed: score={db_idea.viral_score}, pillar={db_idea.pillar}")
            except Exception as e:
                print(f"ERROR: Failed to analyze idea {idea_id}: {e}")

        # Trigger pipeline via Celery task (replaces n8n webhook)
        print(f"DEBUG: Status changed to approved. Triggering Celery pipeline task for idea {idea_id}")
        try:
            task = run_pipeline.delay(idea_id)
            print(f"DEBUG: Pipeline task queued with ID: {task.id}")
        except Exception as e:
            print(f"ERROR: Failed to queue pipeline task for idea {idea_id}: {e}")

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
def bulk_approve_ideas(idea_ids: List[int], db: Session = Depends(get_db)):
    updated = db.query(ContentIdeaModel).filter(
        ContentIdeaModel.id.in_(idea_ids),
        ContentIdeaModel.status == "pending"
    ).update({"status": "approved"}, synchronize_session=False)
    db.commit()

    # Trigger pipeline via Celery tasks (replaces n8n webhooks)
    task_ids = []
    for idea_id in idea_ids:
        try:
            task = run_pipeline.delay(idea_id)
            task_ids.append({"idea_id": idea_id, "task_id": task.id})
        except Exception as e:
            print(f"Failed to queue pipeline task for idea {idea_id}: {e}")

    return {"message": f"Approved {updated} ideas", "tasks": task_ids}


@app.post("/api/content-ideas/bulk-reject")
def bulk_reject_ideas(idea_ids: List[int], db: Session = Depends(get_db)):
    updated = db.query(ContentIdeaModel).filter(
        ContentIdeaModel.id.in_(idea_ids),
        ContentIdeaModel.status == "pending"
    ).update({"status": "rejected"}, synchronize_session=False)
    db.commit()
    return {"message": f"Rejected {updated} ideas"}


# ==================== Trend Scraping ====================

@app.post("/api/scrape/run", response_model=ScrapeResponse)
def run_scrape_endpoint(scrape_request: ScrapeRunCreate, db: Session = Depends(get_db)):
    """Trigger a trend scrape with niche targeting via Celery task"""
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

    # Queue Celery scrape task (replaces n8n webhook)
    try:
        task_params = {
            "niche": scrape_request.niche,
            "hashtags": hashtags,
            "platforms": scrape_request.platforms or ["tiktok", "instagram"],
            "results_per_platform": 20,
            "scrape_run_id": db_run.id
        }

        # Add discovery mode parameters if enabled
        if scrape_request.discover_hashtags and scrape_request.seed_keyword:
            task_params["discover_hashtags"] = True
            task_params["seed_keyword"] = scrape_request.seed_keyword

        task = run_scrape_task.delay(task_params)

        return ScrapeResponse(
            success=True,
            message=f"Scrape task queued with ID: {task.id}",
            ideas_created=0,
            analyzed_count=0,
            task_id=task.id,
            scrape_run_id=db_run.id
        )

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


# Cache for HeyGen avatars (refresh every 10 minutes)
_heygen_avatar_cache = {
    "video_avatars": [],
    "talking_photos": [],
    "last_fetched": None
}
HEYGEN_CACHE_TTL = 600  # 10 minutes


async def _fetch_heygen_avatars():
    """Fetch all avatars from HeyGen API and cache them."""
    import time

    # Check cache validity
    if (_heygen_avatar_cache["last_fetched"] and
        time.time() - _heygen_avatar_cache["last_fetched"] < HEYGEN_CACHE_TTL and
        _heygen_avatar_cache["video_avatars"]):
        return _heygen_avatar_cache

    api_key = os.getenv("HEYGEN_API_KEY")
    if not api_key:
        return {
            "video_avatars": [
                {"avatar_id": "Angela-in-T-shirt-20220819", "avatar_name": "Angela", "preview_image_url": "https://files.heygen.ai/avatar/v3/Angela-in-T-shirt-20220819/full/preview_target.webp", "avatar_type": "video"},
                {"avatar_id": "Anna_public_3_20240108", "avatar_name": "Anna", "preview_image_url": "https://files.heygen.ai/avatar/v3/Anna_public_3_20240108/full/preview_target.webp", "avatar_type": "video"},
            ],
            "talking_photos": [],
            "last_fetched": time.time()
        }

    video_avatars = []
    talking_photos = []

    async with httpx.AsyncClient(timeout=60.0) as client:
        headers = {"X-Api-Key": api_key, "Accept": "application/json"}

        # Fetch all avatars from HeyGen v2 API
        try:
            avatar_response = await client.get(
                "https://api.heygen.com/v2/avatars",
                headers=headers
            )
            if avatar_response.status_code == 200:
                avatar_data = avatar_response.json()
                data = avatar_data.get("data", {})

                # Process video avatars from the "avatars" array
                avatars = data.get("avatars", [])
                for avatar in avatars:
                    video_avatars.append({
                        "avatar_id": avatar.get("avatar_id"),
                        "avatar_name": avatar.get("avatar_name", "Unknown"),
                        "preview_image_url": avatar.get("preview_image_url", ""),
                        "preview_video_url": avatar.get("preview_video_url", ""),
                        "gender": avatar.get("gender", ""),
                        "avatar_type": "video",
                        "premium": avatar.get("premium", False),
                    })

                # Process talking photos from the "talking_photos" array
                photos = data.get("talking_photos", [])
                for photo in photos:
                    talking_photos.append({
                        "avatar_id": photo.get("talking_photo_id"),
                        "avatar_name": photo.get("talking_photo_name", "Unknown"),
                        "preview_image_url": photo.get("preview_image_url", ""),
                        "avatar_type": "talking_photo",
                    })

                print(f"HeyGen v2 API: Cached {len(video_avatars)} video avatars, {len(talking_photos)} talking photos")
            else:
                print(f"HeyGen v2/avatars failed with status {avatar_response.status_code}: {avatar_response.text}")
        except Exception as e:
            import traceback
            print(f"HeyGen v2/avatars error: {e}")
            traceback.print_exc()

        # Fetch talking photos from v1 endpoint only if v2 didn't return any
        if not talking_photos:
            try:
                tp_response = await client.get(
                    "https://api.heygen.com/v1/talking_photo.list",
                    headers={"X-Api-Key": api_key, "Content-Type": "application/json"}
                )
                if tp_response.status_code == 200:
                    tp_data = tp_response.json()
                    photos = tp_data.get("data", [])

                    for photo in photos:
                        photo_id = photo.get("id")
                        if photo_id:
                            talking_photos.append({
                                "avatar_id": photo_id,
                                "avatar_name": photo.get("name") or f"Photo {photo_id[:8]}...",
                                "preview_image_url": photo.get("image_url", ""),
                                "avatar_type": "talking_photo",
                                "is_preset": photo.get("is_preset", False)
                            })
                    print(f"HeyGen v1 talking_photo.list fallback: Added {len(photos)} photos")
            except Exception as e:
                print(f"HeyGen talking_photo.list error: {e}")

    # If no video avatars from API, fall back to known public avatars
    if not video_avatars:
        print("No video avatars from API, using fallback list")
        video_avatars = [
            {"avatar_id": "Angela-in-T-shirt-20220819", "avatar_name": "Angela", "preview_image_url": "https://files.heygen.ai/avatar/v3/Angela-in-T-shirt-20220819/full/preview_target.webp", "avatar_type": "video"},
            {"avatar_id": "Anna_public_3_20240108", "avatar_name": "Anna", "preview_image_url": "https://files.heygen.ai/avatar/v3/Anna_public_3_20240108/full/preview_target.webp", "avatar_type": "video"},
            {"avatar_id": "Kristin_pubblic_3_20240108", "avatar_name": "Kristin", "preview_image_url": "https://files.heygen.ai/avatar/v3/Kristin_pubblic_3_20240108/full/preview_target.webp", "avatar_type": "video"},
            {"avatar_id": "josh_lite3_20230714", "avatar_name": "Josh", "preview_image_url": "https://files.heygen.ai/avatar/v3/josh_lite3_20230714/full/preview_target.webp", "avatar_type": "video"},
            {"avatar_id": "Kayla-incasualsuit-20220818", "avatar_name": "Kayla", "preview_image_url": "https://files.heygen.ai/avatar/v3/Kayla-incasualsuit-20220818/full/preview_target.webp", "avatar_type": "video"},
            {"avatar_id": "Briana_expressive_public_20240426", "avatar_name": "Briana", "preview_image_url": "https://files.heygen.ai/avatar/v3/Briana_expressive_public_20240426/full/preview_target.webp", "avatar_type": "video"},
        ]

    # Update cache
    _heygen_avatar_cache["video_avatars"] = video_avatars
    _heygen_avatar_cache["talking_photos"] = talking_photos
    _heygen_avatar_cache["last_fetched"] = time.time()

    return _heygen_avatar_cache


@app.get("/api/settings/heygen-avatars")
async def get_heygen_avatars_categorized(
    avatar_type: str = "video",
    page: int = 1,
    limit: int = 24,
    search: str = None
):
    """Fetch avatars from HeyGen API with pagination.

    Args:
        avatar_type: "video" or "talking_photo"
        page: Page number (1-indexed)
        limit: Items per page (default 24)
        search: Optional search term to filter by name
    """
    cache = await _fetch_heygen_avatars()

    # Select the right list based on type
    if avatar_type == "talking_photo":
        all_items = cache["talking_photos"]
    else:
        all_items = cache["video_avatars"]

    # Apply search filter if provided
    if search:
        search_lower = search.lower()
        all_items = [a for a in all_items if search_lower in a.get("avatar_name", "").lower()]

    # Calculate pagination
    total = len(all_items)
    total_pages = (total + limit - 1) // limit
    start = (page - 1) * limit
    end = start + limit

    items = all_items[start:end]

    return {
        "items": items,
        "page": page,
        "limit": limit,
        "total": total,
        "total_pages": total_pages,
        "has_more": page < total_pages,
        # Include totals for tab counts
        "video_avatars_total": len(cache["video_avatars"]),
        "talking_photos_total": len(cache["talking_photos"])
    }


@app.get("/api/settings/elevenlabs-voices")
async def get_elevenlabs_voices_categorized():
    """Fetch voices from ElevenLabs API, separating cloned voices from library voices."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        return {
            "cloned_voices": [],
            "library_voices": [
                {"voice_id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel", "category": "premade", "preview_url": "", "labels": {"gender": "female", "accent": "american"}},
                {"voice_id": "AZnzlk1XvdvUeBnXmlld", "name": "Domi", "category": "premade", "preview_url": "", "labels": {"gender": "female", "accent": "american"}},
                {"voice_id": "EXAVITQu4vr4xnSDxMaL", "name": "Bella", "category": "premade", "preview_url": "", "labels": {"gender": "female", "accent": "american"}},
            ]
        }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                "https://api.elevenlabs.io/v1/voices",
                headers={"xi-api-key": api_key}
            )
            response.raise_for_status()
            data = response.json()

            cloned_voices = []
            library_voices = []

            for v in data.get("voices", []):
                voice_data = {
                    "voice_id": v["voice_id"],
                    "name": v["name"],
                    "category": v.get("category", "premade"),
                    "preview_url": v.get("preview_url", ""),
                    "labels": v.get("labels", {}),
                    "description": v.get("description", ""),
                    "sharing": v.get("sharing")
                }

                # Cloned voices have category "cloned" or are user-created
                if v.get("category") == "cloned" or v.get("category") == "professional":
                    cloned_voices.append(voice_data)
                else:
                    library_voices.append(voice_data)

            return {
                "cloned_voices": cloned_voices,
                "library_voices": library_voices
            }
        except Exception as e:
            print(f"ElevenLabs API Error: {e}")
            return {
                "cloned_voices": [],
                "library_voices": [
                    {"voice_id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel", "category": "premade", "preview_url": "", "labels": {"gender": "female", "accent": "american"}},
                ]
            }


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
                
            # Construct URL
            # N8N_HOST env var is 100.83.153.43 (for example), port 8000
            # Ideally clients use relative or construct full URL
            image_url = f"/static/avatars/{filename}"
            
            return AvatarGenerationResponse(image_url=image_url)
            
        except httpx.HTTPError as e:
            msg = f"OpenAI API Error: {str(e)}"
            if e.response:
                msg += f" - Response: {e.response.text}"
            print(msg)
            return JSONResponse(status_code=500, content={"detail": msg})

@app.post("/api/upload/avatar")
async def upload_avatar_image(file: UploadFile = File(...)):
    try:
        AVATAR_DIR = os.path.join("static", "avatars")
        os.makedirs(AVATAR_DIR, exist_ok=True)
        
        # Generate unique filename
        ext = os.path.splitext(file.filename)[1] or ".png"
        filename = f"{uuid.uuid4()}{ext}"
        file_path = os.path.join(AVATAR_DIR, filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {"image_url": f"/static/avatars/{filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/avatar-images")
def list_avatar_images():
    AVATAR_DIR = os.path.join("static", "avatars")
    if not os.path.exists(AVATAR_DIR):
        return {"images": []}
        
    try:
        # Sort by creation time (newest first)
        files = []
        with os.scandir(AVATAR_DIR) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    files.append({
                        "name": entry.name,
                        "url": f"/static/avatars/{entry.name}",
                        "created_at": entry.stat().st_mtime
                    })
        
        files.sort(key=lambda x: x["created_at"], reverse=True)
        return {"images": files}
    except Exception as e:
        print(f"Error listing avatars: {e}")
        return {"images": []}


# ==================== Music Management ====================

MUSIC_DIR = Path("/app/assets/music")
MUSIC_STATE_FILE = MUSIC_DIR / "custom_state.json"
ACTIVE_MUSIC_FILE = MUSIC_DIR / "background_music.mp3"

def get_active_music_filename():
    if not MUSIC_STATE_FILE.exists():
        return None
    try:
        with open(MUSIC_STATE_FILE, 'r') as f:
            data = json.load(f)
            return data.get('active_filename')
    except:
        return None

def set_active_music_filename(filename):
    MUSIC_DIR.mkdir(parents=True, exist_ok=True)
    with open(MUSIC_STATE_FILE, 'w') as f:
        json.dump({'active_filename': filename}, f)

@app.post("/api/music/upload")
async def upload_music(file: UploadFile = File(...)):
    """Upload background music, save to library, and set as active."""
    ALLOWED_TYPES = ["audio/mpeg", "audio/mp3"]
    # Check mime or extension
    if file.content_type not in ALLOWED_TYPES and not file.filename.endswith(".mp3"):
        pass # Allow loose checking
        
    if not file.filename.lower().endswith(".mp3"):
         raise HTTPException(status_code=400, detail="Only MP3 files are allowed")

    MUSIC_DIR.mkdir(parents=True, exist_ok=True)
    
    # Sanitize filename (basic)
    safe_filename = "".join([c for c in file.filename if c.isalpha() or c.isdigit() or c in (' ', '.', '_', '-')]).strip()
    if not safe_filename:
        safe_filename = f"track_{uuid.uuid4().hex[:8]}.mp3"
        
    destination = MUSIC_DIR / safe_filename
    
    # Save to library
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
        
    # Activate (copy to background_music.mp3)
    try:
        shutil.copy2(destination, ACTIVE_MUSIC_FILE)
        set_active_music_filename(safe_filename)
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Failed to activate track: {str(e)}")

    return {
        "filename": safe_filename,
        "status": "uploaded_and_activated",
        "path": str(destination)
    }

@app.get("/api/music/files")
def list_music_files():
    """List all MP3 files in library"""
    if not MUSIC_DIR.exists():
        return {"files": []}
        
    files = []
    active_filename = get_active_music_filename()
    
    # Legacy Migration: If no state but background_music.mp3 exists, loop might return empty.
    # We want to "import" the existing background_music.mp3 into the library.
    if not active_filename and (MUSIC_DIR / "background_music.mp3").exists():
        # Check if we have any OTHER mp3s (if so, maybe we just lost state, don't blindly import)
        # But if ONLY background_music.mp3 exists, it's definitely a legacy state.
        has_library_files = False
        with os.scandir(MUSIC_DIR) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith('.mp3') and entry.name != "background_music.mp3":
                    has_library_files = True
                    break
        
        if not has_library_files:
            # Import legacy file
            legacy_name = f"legacy_track_{int(datetime.utcnow().timestamp())}.mp3"
            try:
                shutil.copy2(MUSIC_DIR / "background_music.mp3", MUSIC_DIR / legacy_name)
                set_active_music_filename(legacy_name)
                active_filename = legacy_name # Update local var for the list loop below
            except Exception as e:
                print(f"Failed to migrate legacy track: {e}")

    
    # If no state file, but background_music.mp3 exists, we don't know source. 
    # But we can still list files.
    
    try:
        with os.scandir(MUSIC_DIR) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith('.mp3'):
                    # Skip the actual active file pointer if it shows up (it's named background_music.mp3)
                    if entry.name == "background_music.mp3":
                        continue
                        
                    files.append({
                        "filename": entry.name,
                        "url": f"/assets/music/{entry.name}",
                        "is_active": (entry.name == active_filename),
                        "size": entry.stat().st_size,
                        "created_at": entry.stat().st_mtime
                    })
        
        files.sort(key=lambda x: x["created_at"], reverse=True)
        return {"files": files}
    except Exception as e:
        print(f"Error listing music: {e}")
        return {"files": []}

@app.post("/api/music/activate")
async def activate_music(request: Request):
    """Set a specific library file as the active background track"""
    try:
        body = await request.json()
        filename = body.get("filename")
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
        
    if not filename:
        raise HTTPException(status_code=400, detail="Filename required")
        
    source_path = MUSIC_DIR / filename
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
        
    try:
        shutil.copy2(source_path, ACTIVE_MUSIC_FILE)
        set_active_music_filename(filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to activate: {str(e)}")
        
    return {"status": "activated", "active_filename": filename}

@app.get("/api/music/info")
def get_music_info():
    """Get status of the active background music file"""
    if not ACTIVE_MUSIC_FILE.exists():
        return {"exists": False}

    active_filename = get_active_music_filename() or "Unknown Source"

    stat = ACTIVE_MUSIC_FILE.stat()
    return {
        "exists": True,
        "filename": active_filename, # Return the source filename name
        "size_bytes": stat.st_size,
        "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        # URL always points to the active file for playback
        "url": "/assets/music/background_music.mp3"
    }

@app.delete("/api/music/{filename}")
async def delete_music(filename: str):
    """Delete a music file from the library"""
    # Prevent deleting the active background file directly via name (though it's a copy)
    if filename == "background_music.mp3":
         raise HTTPException(status_code=400, detail="Cannot delete system file")
         
    target = MUSIC_DIR / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")
        
    try:
        # Check if it was the active one
        active_name = get_active_music_filename()
        if active_name == filename:
            # We allow deleting it, but we should probably unset active state or warn
            # For now, just delete the source. The active copy (background_music.mp3) remains until changed.
            pass
            
        target.unlink()
        return {"success": True, "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {str(e)}")


# ==================== PIPELINE SETTINGS ====================

@app.get("/api/settings/audio", response_model=AudioSettingsResponse)
def get_audio_settings(db: Session = Depends(get_db)):
    """Get current audio settings"""
    settings = db.query(AudioSettingsModel).filter(AudioSettingsModel.id == 1).first()
    if not settings:
        # Create default settings if not exists
        settings = AudioSettingsModel(id=1)
        db.add(settings)
        db.commit()
        db.refresh(settings)
    return settings


@app.put("/api/settings/audio", response_model=AudioSettingsResponse)
def update_audio_settings(update: AudioSettingsUpdate, db: Session = Depends(get_db)):
    """Update audio settings - syncs to both audio_settings table and system_settings for pipeline"""
    settings = db.query(AudioSettingsModel).filter(AudioSettingsModel.id == 1).first()
    if not settings:
        settings = AudioSettingsModel(id=1)
        db.add(settings)

    for field, value in update.model_dump(exclude_unset=True).items():
        setattr(settings, field, value)

    # ALSO sync to system_settings table (pipeline reads from here)
    sys_setting = db.query(SystemSettingsModel).filter(SystemSettingsModel.key == "audio_settings").first()
    if not sys_setting:
        sys_setting = SystemSettingsModel(key="audio_settings", value={})
        db.add(sys_setting)

    # Merge update into existing system_settings value
    current_value = dict(sys_setting.value) if sys_setting.value else {}
    current_value.update(update.model_dump(exclude_unset=True))
    sys_setting.value = current_value

    db.commit()
    db.refresh(settings)
    return settings


@app.get("/api/settings/video", response_model=VideoSettingsResponse)
def get_video_settings(db: Session = Depends(get_db)):
    """Get current video settings"""
    settings = db.query(VideoSettingsModel).filter(VideoSettingsModel.id == 1).first()
    if not settings:
        settings = VideoSettingsModel(id=1)
        db.add(settings)
        db.commit()
        db.refresh(settings)
    return settings


@app.put("/api/settings/video", response_model=VideoSettingsResponse)
def update_video_settings(update: VideoSettingsUpdate, db: Session = Depends(get_db)):
    """Update video settings - syncs to both video_settings table and system_settings for pipeline"""
    settings = db.query(VideoSettingsModel).filter(VideoSettingsModel.id == 1).first()
    if not settings:
        settings = VideoSettingsModel(id=1)
        db.add(settings)

    for field, value in update.model_dump(exclude_unset=True).items():
        setattr(settings, field, value)

    # ALSO sync to system_settings table (pipeline reads from here)
    sys_setting = db.query(SystemSettingsModel).filter(SystemSettingsModel.key == "video_settings").first()
    if not sys_setting:
        sys_setting = SystemSettingsModel(key="video_settings", value={})
        db.add(sys_setting)

    # Merge update into existing system_settings value
    current_value = dict(sys_setting.value) if sys_setting.value else {}
    current_value.update(update.model_dump(exclude_unset=True))
    sys_setting.value = current_value

    db.commit()
    db.refresh(settings)
    return settings


@app.get("/api/settings/llm")
def get_llm_settings(db: Session = Depends(get_db)):
    """Get all LLM prompt settings"""
    settings = db.query(LLMSettingsModel).all()
    return [{"key": s.key, "value": s.value, "description": s.description, "updated_at": s.updated_at} for s in settings]


@app.get("/api/settings/llm/{key}")
def get_llm_setting(key: str, db: Session = Depends(get_db)):
    """Get a specific LLM prompt by key"""
    setting = db.query(LLMSettingsModel).filter(LLMSettingsModel.key == key).first()
    if not setting:
        raise HTTPException(404, f"LLM setting '{key}' not found")
    return {"key": setting.key, "value": setting.value, "description": setting.description}


@app.put("/api/settings/llm/{key}")
def update_llm_setting(key: str, update: LLMSettingUpdate, db: Session = Depends(get_db)):
    """Update a specific LLM prompt"""
    setting = db.query(LLMSettingsModel).filter(LLMSettingsModel.key == key).first()
    if not setting:
        # Create new setting if doesn't exist
        setting = LLMSettingsModel(key=key, value=update.value or '', description=update.description)
        db.add(setting)
    else:
        if update.value is not None:
            setting.value = update.value
        if update.description is not None:
            setting.description = update.description

    db.commit()
    db.refresh(setting)
    return {"key": setting.key, "value": setting.value, "description": setting.description}


@app.get("/api/settings/all")
def get_all_settings(db: Session = Depends(get_db)):
    """Get all settings in one call (for n8n workflow)"""
    audio = db.query(AudioSettingsModel).filter(AudioSettingsModel.id == 1).first()
    video = db.query(VideoSettingsModel).filter(VideoSettingsModel.id == 1).first()
    llm = db.query(LLMSettingsModel).all()

    return {
        "audio": {
            "original_volume": audio.original_volume if audio else 0.7,
            "avatar_volume": audio.avatar_volume if audio else 1.0,
            "music_volume": getattr(audio, 'music_volume', 0.3) if audio else 0.3,
            "ducking_enabled": audio.ducking_enabled if audio else True,
            "avatar_delay_seconds": audio.avatar_delay_seconds if audio else 3.0,
            "duck_to_percent": audio.duck_to_percent if audio else 0.5,
            "music_autoduck": getattr(audio, 'music_autoduck', True) if audio else True
        },
        "video": {
            "output_width": video.output_width if video else 1080,
            "output_height": video.output_height if video else 1920,
            "crf": video.crf if video else 18,
            "preset": video.preset if video else 'slow',
            "greenscreen_enabled": video.greenscreen_enabled if video else True,
            "greenscreen_color": video.greenscreen_color if video else '#00FF00'
        },
        "llm": {s.key: s.value for s in llm}
    }


# ==================== Brand Persona Settings ====================

from models import BrandPersona as BrandPersonaModel
from schemas import BrandPersonaBase, BrandPersonaUpdate, BrandPersonaResponse

@app.get("/api/settings/persona", response_model=BrandPersonaResponse)
def get_brand_persona(db: Session = Depends(get_db)):
    """Get brand persona settings"""
    persona = db.query(BrandPersonaModel).filter(BrandPersonaModel.id == 1).first()
    if not persona:
        # Create default persona
        persona = BrandPersonaModel(id=1)
        db.add(persona)
        db.commit()
        db.refresh(persona)
    return persona


@app.put("/api/settings/persona", response_model=BrandPersonaResponse)
def update_brand_persona(update: BrandPersonaUpdate, db: Session = Depends(get_db)):
    """Update brand persona settings"""
    persona = db.query(BrandPersonaModel).filter(BrandPersonaModel.id == 1).first()
    if not persona:
        persona = BrandPersonaModel(id=1)
        db.add(persona)

    # Update only provided fields
    update_data = update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        if value is not None:
            setattr(persona, field, value)

    db.commit()
    db.refresh(persona)
    return persona


# ==================== API Credits / Usage ====================

@app.get("/api/credits")
async def get_api_credits():
    """Fetch remaining credits/usage from all external API services."""
    results = {}

    async with httpx.AsyncClient(timeout=15.0) as client:
        # ElevenLabs - GET /v1/user/subscription
        elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
        if elevenlabs_key:
            try:
                resp = await client.get(
                    "https://api.elevenlabs.io/v1/user/subscription",
                    headers={"xi-api-key": elevenlabs_key}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    results["elevenlabs"] = {
                        "status": "ok",
                        "character_count": data.get("character_count", 0),
                        "character_limit": data.get("character_limit", 0),
                        "remaining": data.get("character_limit", 0) - data.get("character_count", 0),
                        "tier": data.get("tier", "unknown"),
                        "next_reset": data.get("next_character_count_reset_unix")
                    }
                else:
                    results["elevenlabs"] = {"status": "error", "message": f"HTTP {resp.status_code}"}
            except Exception as e:
                results["elevenlabs"] = {"status": "error", "message": str(e)}
        else:
            results["elevenlabs"] = {"status": "not_configured"}

        # HeyGen - GET /v2/user/remaining_quota
        heygen_key = os.getenv("HEYGEN_API_KEY")
        if heygen_key:
            try:
                resp = await client.get(
                    "https://api.heygen.com/v2/user/remaining_quota",
                    headers={"X-Api-Key": heygen_key}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    quota = data.get("data", {}).get("remaining_quota", 0)
                    # HeyGen quota is in seconds, divide by 60 for credits
                    results["heygen"] = {
                        "status": "ok",
                        "remaining_quota_seconds": quota,
                        "remaining_credits": round(quota / 60, 1) if quota else 0,
                        "details": data.get("data", {}).get("details", {})
                    }
                else:
                    results["heygen"] = {"status": "error", "message": f"HTTP {resp.status_code}"}
            except Exception as e:
                results["heygen"] = {"status": "error", "message": str(e)}
        else:
            results["heygen"] = {"status": "not_configured"}

        # OpenRouter - GET /api/v1/auth/key (returns credits info)
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key:
            try:
                resp = await client.get(
                    "https://openrouter.ai/api/v1/auth/key",
                    headers={"Authorization": f"Bearer {openrouter_key}"}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    # OpenRouter returns usage and limit in USD
                    key_data = data.get("data", {})
                    results["openrouter"] = {
                        "status": "ok",
                        "usage_usd": key_data.get("usage", 0),
                        "limit_usd": key_data.get("limit"),
                        "remaining_usd": (key_data.get("limit") or 0) - (key_data.get("usage") or 0) if key_data.get("limit") else None,
                        "is_free_tier": key_data.get("is_free_tier", False),
                        "rate_limit": key_data.get("rate_limit", {})
                    }
                else:
                    results["openrouter"] = {"status": "error", "message": f"HTTP {resp.status_code}"}
            except Exception as e:
                results["openrouter"] = {"status": "error", "message": str(e)}
        else:
            results["openrouter"] = {"status": "not_configured"}

        # Blotato - GET /v2/users/me (no credits endpoint available)
        blotato_key = os.getenv("BLOTATO_API_KEY")
        if blotato_key:
            try:
                resp = await client.get(
                    "https://backend.blotato.com/v2/users/me",
                    headers={"blotato-api-key": blotato_key}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    results["blotato"] = {
                        "status": "ok",
                        "subscription_status": data.get("subscriptionStatus", "unknown"),
                        "message": "Credits visible in Blotato dashboard only"
                    }
                else:
                    results["blotato"] = {"status": "error", "message": f"HTTP {resp.status_code}"}
            except Exception as e:
                results["blotato"] = {"status": "error", "message": str(e)}
        else:
            results["blotato"] = {"status": "not_configured"}

        # OpenAI - No direct credits endpoint, but we can check billing via dashboard API
        # Using the /v1/dashboard/billing/credit_grants endpoint (unofficial but commonly used)
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                # Try the organization billing endpoint
                resp = await client.get(
                    "https://api.openai.com/v1/organization/usage",
                    headers={"Authorization": f"Bearer {openai_key}"},
                    params={"date": datetime.now().strftime("%Y-%m-%d")}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    results["openai"] = {
                        "status": "ok",
                        "message": "Usage tracking available via dashboard",
                        "data": data
                    }
                else:
                    # Fallback - just indicate it's configured
                    results["openai"] = {
                        "status": "ok",
                        "message": "API key configured. Check usage at platform.openai.com/usage",
                        "configured": True
                    }
            except Exception as e:
                results["openai"] = {
                    "status": "ok",
                    "message": "API key configured. Check usage at platform.openai.com/usage",
                    "configured": True
                }
        else:
            results["openai"] = {"status": "not_configured"}

    return {"credits": results, "fetched_at": datetime.now().isoformat()}
