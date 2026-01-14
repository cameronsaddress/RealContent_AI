import sys
import os

# Add current directory to path so we can import backend
sys.path.append(os.getcwd())

try:
    from backend.models import SessionLocal, InfluencerVideo
except ImportError:
    from models import SessionLocal, InfluencerVideo

def check_url():
    db = SessionLocal()
    try:
        video = db.query(InfluencerVideo).filter(InfluencerVideo.id == 2).first()
        if not video:
            print("Video ID 2 not found.")
            return
            
        print(f"Video ID 2 Title: {video.title}")
        print(f"Video ID 2 URL: {video.url}")
        
    finally:
        db.close()

if __name__ == "__main__":
    check_url()
