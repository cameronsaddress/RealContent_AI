
import sys
import os
sys.path.append(os.getcwd())
from models import InfluencerVideo, ViralClip, SessionLocal

db = SessionLocal()
try:
    video_id = 2
    video = db.query(InfluencerVideo).filter(InfluencerVideo.id == video_id).first()
    
    # Clear clips
    db.query(ViralClip).filter(ViralClip.source_video_id == video_id).delete()
    
    # Clear bad transcript
    video.transcript_json = None
    
    # Set status to 'downloaded' so it triggers transcription next
    video.status = "downloaded"
    video.error_message = None
    
    db.commit()
    print(f"Video {video_id} reset. Transcript CLEARED. Status: downloaded.")
    
finally:
    db.close()
