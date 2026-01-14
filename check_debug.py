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

def check_status(video_id: int):
    db = SessionLocal()
    try:
        video = db.query(InfluencerVideo).filter(InfluencerVideo.id == video_id).first()
        if not video:
            print(f"Video {video_id} not found.")
            return

        print(f"ID: {video.id}")
        print(f"Status: {video.status}")
        print(f"Error Message: {video.error_message}")
        print(f"Transcript JSON present: {bool(video.transcript_json)}")
        if video.local_path:
            import os
            try:
                size = os.path.getsize(video.local_path)
                print(f"Local Path: {video.local_path}")
                print(f"File Size: {size / (1024*1024):.2f} MB")
            except OSError:
                 print(f"Local Path: {video.local_path} (File not found)")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    check_status(2)
