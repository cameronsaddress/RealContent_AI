import sys
import os
from sqlalchemy.orm import Session

# Add current directory to path so we can import backend
sys.path.append("/app")

# Correct import paths for running inside container
try:
    from models import SessionLocal, InfluencerVideo, ViralClip
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), "backend"))
    from backend.models import SessionLocal, InfluencerVideo, ViralClip

def reset_video_state(video_id: int):
    print(f"Resetting state for Video ID {video_id}...")
    db = SessionLocal()
    try:
        video = db.query(InfluencerVideo).filter(InfluencerVideo.id == video_id).first()
        if not video:
            print(f"Video {video_id} not found.")
            return

        # Delete existing clips
        clips = db.query(ViralClip).filter(ViralClip.source_video_id == video_id).all()
        print(f"Deleting {len(clips)} existing clips...")
        for clip in clips:
            db.delete(clip)
        
        # Reset Status
        video.status = "transcribed"
        # DO NOT CLEAR transcript_json - we want to keep the medium model result!
        
        db.commit()
        print(f"Video {video_id} reset to 'transcribed'. Logic update applied via worker restart.")
        print("Ready for 'Auto-Generate Clips' trigger.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    reset_video_state(2) # Default to Video 2
