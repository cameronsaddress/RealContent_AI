import sys
import os
from sqlalchemy.orm import Session

# Add current directory to path so we can import backend
sys.path.append("/app")

try:
    from models import SessionLocal, InfluencerVideo
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), "backend"))
    from backend.models import SessionLocal, InfluencerVideo

def force_transcribed(video_id: int):
    print(f"Forcing status to 'transcribed' for Video ID {video_id}...")
    db = SessionLocal()
    try:
        video = db.query(InfluencerVideo).filter(InfluencerVideo.id == video_id).first()
        if not video:
            print(f"Video {video_id} not found.")
            return

        print(f"Old Status: {video.status}")
        video.status = "transcribed"
        # video.error_message = None # Clear error if any
        db.commit()
        print(f"New Status: {video.status}")
        print("Ready for retry.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    force_transcribed(2)
