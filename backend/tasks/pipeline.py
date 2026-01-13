"""
Pipeline Celery tasks - full video production pipeline.
Replaces the main n8n workflow orchestration.

Pipeline Stages (in order):
1. Get Content Idea - Fetch approved content from database
2. Download Source Video - Get the viral video we're reacting to
3. Transcribe Source Video - Use Whisper to get what they SAID (for script context)
4. Generate Script - LLM creates script with transcription + persona context
5. Generate Voice - ElevenLabs TTS from script
6. Generate Avatar - HeyGen video with our voice
7. Compose Video - Chromakey avatar over source video (GPU)
8. Burn Captions - Karaoke-style captions on final video (GPU)
9. Upload & Publish - Dropbox + Blotato multi-platform
"""

from celery import shared_task
from celery.utils.log import get_task_logger
from typing import Optional, Dict, Any

from config import settings
from utils.paths import asset_paths

logger = get_task_logger(__name__)


@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def run_pipeline(
    self,
    content_idea_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run the full video production pipeline.

    This is the main orchestration task that replaces the n8n workflow.
    It can process a specific content idea or pick the next approved one.

    Args:
        content_idea_id: Optional specific content idea to process

    Returns:
        Pipeline result summary
    """
    import asyncio
    return asyncio.get_event_loop().run_until_complete(
        _run_pipeline_async(content_idea_id)
    )


async def _run_pipeline_async(
    content_idea_id: Optional[int] = None
) -> Dict[str, Any]:
    """Async implementation of the pipeline."""
    from services import (
        ScriptGenerator,
        VoiceService,
        AvatarService,
        VideoService,
        CaptionService,
        PublisherService,
        StorageService,
        VideoDownloaderService
    )
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker

    # Initialize services
    script_gen = ScriptGenerator()
    voice_svc = VoiceService()
    avatar_svc = AvatarService()
    video_svc = VideoService()
    caption_svc = CaptionService()
    publisher_svc = PublisherService()
    storage_svc = StorageService()
    video_dl_svc = VideoDownloaderService()

    # Create database session
    engine = create_async_engine(settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"))
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    result = {
        "content_idea_id": content_idea_id,
        "script_id": None,
        "success": False,
        "stage": "init",
        "error": None
    }

    async with async_session() as session:
        try:
            # ================================================================
            # STAGE 1: Get Content Idea
            # ================================================================
            result["stage"] = "get_content"
            content_idea = await _get_content_idea(session, content_idea_id)

            if not content_idea:
                result["error"] = "No approved content ideas found"
                return result

            content_idea_id = content_idea["id"]
            result["content_idea_id"] = content_idea_id
            source_url = content_idea.get("source_url")
            logger.info(f"Processing content idea {content_idea_id}: {source_url}")

            # ================================================================
            # STAGE 2: Download Source Video (BEFORE script generation)
            # We need this to transcribe what they SAID in the video
            # ================================================================
            result["stage"] = "source_video_download"

            source_video_path = None
            # Use content_idea_id for source video (not script_id, which doesn't exist yet)
            source_video_key = f"idea_{content_idea_id}"

            if source_url:
                # Check if already downloaded (using content_idea_id as key)
                existing_source = asset_paths.base_path / "videos" / f"{source_video_key}_source.mp4"
                if existing_source.exists() and existing_source.stat().st_size > 1000:
                    source_video_path = str(existing_source)
                    logger.info(f"Using existing source video: {source_video_path}")
                else:
                    # Download via Apify (TikTok, Instagram, YouTube) or fallback to yt-dlp
                    try:
                        # Pass content_idea_id as the key for now
                        dl_result = await video_dl_svc.download_video(source_url, content_idea_id)
                        if dl_result.success:
                            # Rename to use idea_ prefix
                            import shutil
                            downloaded_path = asset_paths.source_video_path(content_idea_id)
                            if downloaded_path.exists():
                                shutil.move(str(downloaded_path), str(existing_source))
                                source_video_path = str(existing_source)
                            else:
                                source_video_path = dl_result.video_path
                            logger.info(f"Downloaded source video: {source_video_path}")
                        else:
                            logger.warning(f"Source video download failed: {dl_result.error}")
                    except Exception as e:
                        logger.warning(f"Source video download error: {e}")
                        # Continue without source video - avatar will be full screen

            # ================================================================
            # STAGE 3: Transcribe Source Video (get what they SAID)
            # This is CRITICAL for script generation context
            # ================================================================
            result["stage"] = "source_transcription"

            source_transcription = content_idea.get("source_transcription") or ""

            if source_video_path and not source_transcription:
                logger.info(f"Transcribing source video for context...")
                try:
                    # Extract audio from source video for transcription
                    import subprocess
                    import tempfile

                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_audio:
                        tmp_audio_path = tmp_audio.name

                    # Extract audio using ffmpeg
                    extract_cmd = [
                        "ffmpeg", "-y",
                        "-i", source_video_path,
                        "-vn",  # No video
                        "-acodec", "libmp3lame",
                        "-ar", "16000",  # 16kHz for Whisper
                        "-ac", "1",  # Mono
                        tmp_audio_path
                    ]
                    subprocess.run(extract_cmd, capture_output=True, check=True, timeout=60)

                    # Transcribe using Whisper API (same as caption service)
                    import httpx
                    with open(tmp_audio_path, "rb") as f:
                        audio_data = f.read()

                    async with httpx.AsyncClient(timeout=120.0) as client:
                        response = await client.post(
                            "https://api.openai.com/v1/audio/transcriptions",
                            headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
                            files={"file": ("audio.mp3", audio_data, "audio/mpeg")},
                            data={"model": "whisper-1", "response_format": "text"}
                        )
                        source_transcription = response.text.strip()

                    # Clean up temp file
                    import os
                    os.unlink(tmp_audio_path)

                    # Save transcription back to content_idea for future use
                    from models import ContentIdea
                    await session.execute(
                        ContentIdea.__table__.update()
                        .where(ContentIdea.id == content_idea_id)
                        .values(source_transcription=source_transcription)
                    )
                    await session.commit()

                    logger.info(f"Transcribed source video ({len(source_transcription)} chars): {source_transcription[:100]}...")

                except Exception as e:
                    logger.warning(f"Source transcription failed (continuing without): {e}")
                    source_transcription = ""

            # ================================================================
            # STAGE 4: Generate Script (NOW with transcription context!)
            # ================================================================
            result["stage"] = "script_generation"
            from models import Script as ScriptModel

            # Check for existing script first
            existing_script = await session.execute(
                ScriptModel.__table__.select()
                .where(ScriptModel.content_idea_id == content_idea_id)
                .order_by(ScriptModel.created_at.desc())
                .limit(1)
            )
            existing_row = existing_script.fetchone()

            if existing_row:
                script_id = existing_row.id
                # Build script object from existing data
                from services.script_generator import GeneratedScript
                script = GeneratedScript(
                    content_idea_id=content_idea_id,
                    hook=existing_row.hook or "",
                    body=existing_row.body or "",
                    cta=existing_row.cta or "",
                    full_text=existing_row.full_script or "",
                    duration_estimate=existing_row.duration_estimate or 60,
                    pillar=content_idea.get("pillar", "educational_tips"),
                    tiktok_caption=existing_row.tiktok_caption or "",
                    ig_caption=existing_row.ig_caption or "",
                    yt_title=existing_row.yt_title or "",
                    yt_description=existing_row.yt_description or "",
                    linkedin_text=existing_row.linkedin_text or "",
                    x_text=existing_row.x_text or "",
                    facebook_text=existing_row.facebook_text or "",
                    threads_text=existing_row.threads_text or ""
                )
                logger.info(f"Using existing script {script_id}")
            else:
                from services.script_generator import ScriptRequest

                # NOW we have transcription to pass to the LLM!
                script_request = ScriptRequest(
                    content_idea_id=content_idea_id,
                    pillar=content_idea.get("pillar") or "educational_tips",
                    source_url=source_url or "",
                    original_text=content_idea.get("original_text") or "",
                    source_transcription=source_transcription,  # <-- THE KEY CHANGE
                    suggested_hook=content_idea.get("suggested_hook"),
                    why_viral=content_idea.get("why_viral")
                )

                logger.info(f"Generating script with {len(source_transcription)} chars of transcription context")
                script = await script_gen.generate_script(script_request)
                script_id = await script_gen.save_script(script, session)
                logger.info(f"Generated NEW script {script_id}")

            result["script_id"] = script_id

            # Now copy source video to script_id naming convention for later stages
            if source_video_path:
                script_source_path = asset_paths.source_video_path(script_id)
                if not script_source_path.exists():
                    import shutil
                    shutil.copy(source_video_path, script_source_path)
                    logger.info(f"Copied source video to {script_source_path}")
                source_video_path = str(script_source_path)

            # ================================================================
            # STAGE 5: Generate Voice
            # ================================================================
            result["stage"] = "voice_generation"

            if not voice_svc.voice_exists(script_id):
                from services.voice import VoiceRequest

                voice_request = VoiceRequest(
                    script_id=script_id,
                    text=script.full_text
                )
                voice_result = await voice_svc.generate_voice(voice_request)
                await voice_svc.update_asset_status(
                    script_id, session,
                    status="voice_ready",
                    duration=voice_result.duration_seconds
                )
            logger.info(f"Voice ready for script {script_id}")

            # ================================================================
            # STAGE 6: Generate Avatar (HeyGen)
            # ================================================================
            result["stage"] = "avatar_generation"

            if not avatar_svc.avatar_exists(script_id):
                # Check if we have a pending HeyGen video job to resume
                from models import Asset
                asset_result = await session.execute(
                    Asset.__table__.select().where(Asset.script_id == script_id)
                )
                asset_row = asset_result.fetchone()
                existing_heygen_id = asset_row.heygen_video_id if asset_row else None

                if existing_heygen_id:
                    # Resume polling for existing video job
                    logger.info(f"Resuming HeyGen video {existing_heygen_id} for script {script_id}")
                    status = await avatar_svc.poll_until_complete(existing_heygen_id)

                    if not status.video_url:
                        raise ValueError(f"Completed video has no URL: {existing_heygen_id}")

                    # Download video
                    avatar_path = await avatar_svc.download_video(status.video_url, script_id)

                    # Get duration
                    import subprocess
                    duration_result = subprocess.run(
                        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                         "-of", "default=noprint_wrappers=1:nokey=1", str(avatar_path)],
                        capture_output=True, text=True
                    )
                    duration = float(duration_result.stdout.strip()) if duration_result.returncode == 0 else 0.0

                    await avatar_svc.update_asset_status(
                        script_id, session,
                        video_id=existing_heygen_id,
                        video_path=str(avatar_path)
                    )
                else:
                    # Create new HeyGen video job
                    # Get character config from database
                    from models import SystemSettings as SystemSettingsModel
                    char_setting = await session.execute(
                        SystemSettingsModel.__table__.select().where(
                            SystemSettingsModel.key == "active_character"
                        )
                    )
                    char_row = char_setting.fetchone()
                    char_config = char_row.value if char_row and char_row.value else {}

                    # Upload audio to HeyGen
                    voice_path = asset_paths.voice_path(script_id)
                    audio_asset_id = await avatar_svc.upload_audio(voice_path)

                    from services.avatar import AvatarRequest, AvatarType

                    # Map avatar_type from config to enum
                    avatar_type_str = char_config.get("avatar_type", "video_avatar")
                    if avatar_type_str in ("video", "video_avatar"):
                        avatar_type = AvatarType.VIDEO_AVATAR
                    elif avatar_type_str in ("talking_photo", "photo"):
                        avatar_type = AvatarType.TALKING_PHOTO
                    else:
                        avatar_type = AvatarType.VIDEO_AVATAR

                    avatar_request = AvatarRequest(
                        script_id=script_id,
                        audio_url=audio_asset_id,
                        avatar_id=char_config.get("avatar_id"),
                        avatar_type=avatar_type
                    )

                    # Create video and store the HeyGen ID immediately
                    video_id = await avatar_svc.create_video(avatar_request)

                    # Store heygen_video_id so we can resume if interrupted
                    await session.execute(
                        Asset.__table__.update()
                        .where(Asset.script_id == script_id)
                        .values(heygen_video_id=video_id)
                    )
                    await session.commit()
                    logger.info(f"Stored HeyGen video ID {video_id} for script {script_id}")

                    # Poll until complete
                    status = await avatar_svc.poll_until_complete(video_id)

                    if not status.video_url:
                        raise ValueError(f"Completed video has no URL: {video_id}")

                    # Download video
                    avatar_path = await avatar_svc.download_video(status.video_url, script_id)

                    # Get duration
                    import subprocess
                    duration_result = subprocess.run(
                        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                         "-of", "default=noprint_wrappers=1:nokey=1", str(avatar_path)],
                        capture_output=True, text=True
                    )
                    duration = float(duration_result.stdout.strip()) if duration_result.returncode == 0 else 0.0

                    await avatar_svc.update_asset_status(
                        script_id, session,
                        video_id=video_id,
                        video_path=str(avatar_path)
                    )
            logger.info(f"Avatar ready for script {script_id}")

            # ================================================================
            # STAGE 7: Compose Video (GPU video-processor service)
            # ================================================================
            result["stage"] = "video_composition"

            if not video_svc.combined_video_exists(script_id):
                import httpx
                from models import SystemSettings as SystemSettingsModel

                # Get ALL settings from database
                video_setting = await session.execute(
                    SystemSettingsModel.__table__.select().where(
                        SystemSettingsModel.key == "video_settings"
                    )
                )
                video_row = video_setting.fetchone()
                video_config = video_row.value if video_row and video_row.value else {}

                # Audio settings
                audio_setting = await session.execute(
                    SystemSettingsModel.__table__.select().where(
                        SystemSettingsModel.key == "audio_settings"
                    )
                )
                audio_row = audio_setting.fetchone()
                audio_config = audio_row.value if audio_row and audio_row.value else {}

                # Get active music if any
                music_path = asset_paths.get_active_music()

                # Convert greenscreen color from #RRGGBB to 0xRRGGBB for FFmpeg
                gs_color = video_config.get("greenscreen_color", "#00FF00")
                if gs_color.startswith("#"):
                    gs_color = "0x" + gs_color[1:]

                # Call GPU video-processor service
                compose_payload = {
                    "script_id": str(script_id),
                    "avatar_path": f"/avatar/{script_id}_avatar.mp4",
                    "background_path": f"/downloads/{script_id}_source.mp4" if source_video_path else f"/avatar/{script_id}_avatar.mp4",
                    "audio_path": f"/audio/{script_id}_voice.mp3",
                    "output_filename": f"{script_id}_combined.mp4",
                    "use_gpu": True,
                    "avatar_scale": video_config.get("avatar_scale", 0.75),
                    "avatar_offset_x": video_config.get("avatar_offset_x", -250),
                    "avatar_offset_y": video_config.get("avatar_offset_y", 600),
                    "greenscreen_color": gs_color,
                    "original_volume": audio_config.get("original_volume", 0.7),
                    "avatar_volume": audio_config.get("avatar_volume", 1.0),
                    "music_volume": audio_config.get("music_volume", 0.3),
                }

                if music_path:
                    compose_payload["music_path"] = f"/music/{music_path.name}"

                async with httpx.AsyncClient(timeout=600.0) as client:
                    response = await client.post(
                        "http://video-processor:8080/compose",
                        json=compose_payload
                    )

                    if response.status_code != 200:
                        result["error"] = f"Video composition failed: {response.text}"
                        return result

                    compose_result = response.json()
                    if not compose_result.get("success"):
                        result["error"] = f"Video composition failed: {compose_result}"
                        return result

                    logger.info(f"GPU composed video: {compose_result.get('output_path')} using {compose_result.get('encoder_used')}")
            else:
                # Need video_config for caption stage even if we skip composition
                from models import SystemSettings as SystemSettingsModel
                video_setting = await session.execute(
                    SystemSettingsModel.__table__.select().where(
                        SystemSettingsModel.key == "video_settings"
                    )
                )
                video_row = video_setting.fetchone()
                video_config = video_row.value if video_row and video_row.value else {}

            logger.info(f"Video composed for script {script_id}")

            # ================================================================
            # STAGE 8: Burn Captions (GPU video-processor service)
            # Transcribes our VOICE audio for karaoke timing
            # ================================================================
            result["stage"] = "captioning"

            if not caption_svc.final_video_exists(script_id):
                import httpx
                # Get caption settings from video_config (they're stored together)
                caption_style = video_config.get("caption_style", "karaoke")

                if caption_style == "none":
                    # No captions - just copy combined to final
                    import shutil
                    shutil.copy(asset_paths.combined_path(script_id), asset_paths.final_path(script_id))
                    logger.info(f"Skipped captions (style=none) for script {script_id}")
                else:
                    # Transcribe OUR voice audio (not source video) for caption timing
                    transcription = await caption_svc.transcribe_audio(script_id)

                    if caption_style == "karaoke":
                        # Generate ASS file with karaoke timing using ALL caption settings
                        ass_path = caption_svc.generate_ass(
                            transcription,
                            font_name=video_config.get("caption_font", "Arial"),
                            font_size=video_config.get("caption_font_size", 96),
                            margin_v=video_config.get("caption_position_y", 850),
                            font_color=video_config.get("caption_color", "#FFFFFF"),
                            highlight_color=video_config.get("caption_highlight_color", "#FFFF00"),
                            outline_color=video_config.get("caption_outline_color", "#000000"),
                            outline_width=video_config.get("caption_outline_width", 5)
                        )

                        # Call GPU video-processor service for caption burning
                        caption_payload = {
                            "script_id": str(script_id),
                            "video_path": f"/outputs/{script_id}_combined.mp4",
                            "ass_path": f"/captions/{script_id}_captions.ass",
                            "output_filename": f"{script_id}_final.mp4",
                            "use_gpu": True
                        }

                        async with httpx.AsyncClient(timeout=600.0) as client:
                            response = await client.post(
                                "http://video-processor:8080/caption",
                                json=caption_payload
                            )

                            if response.status_code != 200:
                                result["error"] = f"Caption burning failed: {response.text}"
                                return result

                            caption_result = response.json()
                            if not caption_result.get("success"):
                                result["error"] = f"Caption burning failed: {caption_result}"
                                return result

                            logger.info(f"GPU burned captions: {caption_result.get('output_path')}")

                        srt_path = ass_path  # For record keeping
                    else:
                        # Static SRT captions via GPU service
                        srt_path = caption_svc.generate_srt(transcription)

                        caption_payload = {
                            "script_id": str(script_id),
                            "video_path": f"/outputs/{script_id}_combined.mp4",
                            "srt_path": f"/captions/{script_id}_captions.srt",
                            "output_filename": f"{script_id}_final.mp4",
                            "font_size": video_config.get("caption_font_size", 48),
                            "use_gpu": True
                        }

                        async with httpx.AsyncClient(timeout=600.0) as client:
                            response = await client.post(
                                "http://video-processor:8080/caption",
                                json=caption_payload
                            )

                            if response.status_code != 200:
                                result["error"] = f"Caption burning failed: {response.text}"
                                return result

                            caption_result = response.json()
                            if not caption_result.get("success"):
                                result["error"] = f"Caption burning failed: {caption_result}"
                                return result

                    await caption_svc.update_asset_status(
                        script_id, session,
                        srt_path=str(srt_path),
                        final_path=str(asset_paths.final_path(script_id))
                    )
            logger.info(f"Captions processed for script {script_id}")

            # ================================================================
            # STAGE 9: Upload & Publish
            # ================================================================
            result["stage"] = "storage_upload"

            final_path = asset_paths.final_path(script_id)

            # Upload to Dropbox for Blotato to access
            upload_result = await storage_svc.upload_to_dropbox(
                script_id=script_id,
                file_path=final_path
            )
            video_url = upload_result.public_url
            logger.info(f"Uploaded to Dropbox: {video_url}")

            # Publish to platforms
            result["stage"] = "publishing"

            publish_request = publisher_svc.prepare_publish_data(
                script_id=script_id,
                video_url=video_url,
                script_text=script.cta,
                pillar=script.pillar
            )

            publish_results = await publisher_svc.publish(publish_request)
            await publisher_svc.save_publish_record(publish_results, session)
            await publisher_svc.update_content_status(script_id, session)

            successful_platforms = [r.platform for r in publish_results if r.success]
            logger.info(f"Published to {successful_platforms}")

            result["success"] = True
            result["stage"] = "completed"
            result["published_platforms"] = successful_platforms

            return result

        except Exception as e:
            logger.error(f"Pipeline failed at {result['stage']}: {e}")
            result["error"] = str(e)

            # Update content idea status to error
            if content_idea_id:
                try:
                    from models import ContentIdea
                    await session.execute(
                        ContentIdea.__table__.update()
                        .where(ContentIdea.id == content_idea_id)
                        .values(status="error")
                    )
                    await session.commit()
                except Exception as update_err:
                    logger.error(f"Failed to update status to error: {update_err}")

            return result


async def _get_content_idea(session, content_idea_id: Optional[int] = None) -> Optional[Dict]:
    """Get content idea from database."""
    from models import ContentIdea

    if content_idea_id:
        result = await session.execute(
            ContentIdea.__table__.select().where(ContentIdea.id == content_idea_id)
        )
    else:
        # Get next approved idea
        result = await session.execute(
            ContentIdea.__table__.select()
            .where(ContentIdea.status == "approved")
            .order_by(ContentIdea.created_at.asc())
            .limit(1)
        )

    row = result.fetchone()
    if row:
        return dict(row._mapping)
    return None


@shared_task(bind=True, max_retries=2)
def process_specific_idea(self, content_idea_id: int) -> Dict[str, Any]:
    """
    Process a specific content idea through the pipeline.

    Args:
        content_idea_id: Content idea ID to process

    Returns:
        Pipeline result
    """
    return run_pipeline(content_idea_id)


@shared_task
def check_pending_ideas() -> Dict[str, Any]:
    """
    Check for pending approved ideas and queue them.

    This runs every 15 minutes to pick up newly approved content.

    Returns:
        Summary of queued items
    """
    import asyncio
    return asyncio.get_event_loop().run_until_complete(_check_pending_async())


async def _check_pending_async() -> Dict[str, Any]:
    """Async implementation of pending check."""
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from models import ContentIdea

    engine = create_async_engine(settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"))
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        result = await session.execute(
            ContentIdea.__table__.select()
            .where(ContentIdea.status == "approved")
            .order_by(ContentIdea.created_at.asc())
        )
        ideas = result.fetchall()

        queued = []
        for idea in ideas:
            idea_dict = dict(idea._mapping)
            # Queue each idea
            run_pipeline.delay(idea_dict["id"])
            queued.append(idea_dict["id"])

        logger.info(f"Queued {len(queued)} content ideas for processing")

        return {
            "queued_count": len(queued),
            "idea_ids": queued
        }
