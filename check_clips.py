
import sys
import os

# Ensure we can import from the app's directories
sys.path.append(os.getcwd())

from models import ViralClip, SessionLocal

db = SessionLocal()
try:
    video_id = 2
    clips = db.query(ViralClip).filter(ViralClip.source_video_id == video_id).all()

    print(f"Clips for Video {video_id}:")
    for clip in clips:
        print(f"ID: {clip.id} | Status: {clip.status} | Error: {clip.error_message}")
finally:
    db.close()
