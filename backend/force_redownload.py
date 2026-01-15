
import sys
import os
sys.path.append(os.getcwd())
from models import InfluencerVideo, SessionLocal

db = SessionLocal()
try:
    video_id = 2
    video = db.query(InfluencerVideo).filter(InfluencerVideo.id == video_id).first()
    
    # CRITICAL FIX: Set status to "pending" NOT "downloaded".
    # "downloaded" triggers the skip-logic. "pending" forces the download code to run.
    video.status = "pending"
    video.error_message = None
    
    # We leave video.local_path alone or clear it? 
    # yt-dlp uses the URL to determine filename, so it will find the .part file on disk regardless of DB.
    # But let's verify if the code relies on local_path being None to call download?
    # No, the code checks: if video.status == "downloaded" ... skip.
    # Since status is now "pending", it will NOT skip.
    
    db.commit()
    print(f"Video {video_id} reset to 'pending'. This will FORCE the download step to run (and resume).")
    
finally:
    db.close()
