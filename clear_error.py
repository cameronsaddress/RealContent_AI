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

def clear_error(video_id: int):
    print(f"Clearing error for Video ID {video_id}...")
    db = SessionLocal()
    try:
        video = db.query(InfluencerVideo).filter(InfluencerVideo.id == video_id).first()
        if not video:
            print(f"Video {video_id} not found.")
            return

        print(f"Current Status: {video.status}")
        print(f"Current Error: {video.error_message}")
        
        # Clear error
        video.error_message = None
        video.status = "transcribed" 
        
        db.commit()
        print("Error message cleared. Status ensure 'transcribed'.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    clear_error(2)
