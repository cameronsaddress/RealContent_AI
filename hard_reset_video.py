
import sys
import os
sys.path.append(os.getcwd())
try:
    from models import InfluencerVideo, ViralClip, SessionLocal
except ImportError:
    # Fallback to local import if run inside container app root
    try:
        from app.models import InfluencerVideo, ViralClip, SessionLocal
    except:
        # If in /app, models is likely top level
        sys.path.append("/app")
        from models import InfluencerVideo, ViralClip, SessionLocal

db = SessionLocal()
try:
    video_id = 2
    video = db.query(InfluencerVideo).filter(InfluencerVideo.id == video_id).first()
    
    # 1. Delete Clips
    deleted_clips = db.query(ViralClip).filter(ViralClip.source_video_id == video_id).delete()
    
    # 2. Reset Video Fields
    video.transcript_json = None
    video.local_path = None # Force it to forget the file existed
    video.duration = None
    video.error_message = None
    video.status = "pending" # Fresh start
    
    db.commit()
    print(f"HARD RESET Complete for Video {video_id}.")
    print(f"Deleted {deleted_clips} clips.")
    print("Cleared transcript, local_path, and duration.")
    print("Status set to: pending")
    
finally:
    db.close()
