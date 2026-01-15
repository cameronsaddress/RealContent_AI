
import sys
import os
sys.path.append(os.getcwd())
try:
    from models import InfluencerVideo, SessionLocal
except ImportError:
    from app.models import InfluencerVideo, SessionLocal

db = SessionLocal()
try:
    video_id = 2
    video = db.query(InfluencerVideo).filter(InfluencerVideo.id == video_id).first()
    
    print(f"Video ID: {video.id}")
    print(f"Title: {video.title}")
    print(f"URL: {video.url}")
    print(f"Platform: {video.platform}")
    
finally:
    db.close()
