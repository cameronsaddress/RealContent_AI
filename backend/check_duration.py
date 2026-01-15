
import sys
import os
sys.path.append(os.getcwd())
from models import InfluencerVideo, SessionLocal

db = SessionLocal()
try:
    video_id = 2
    video = db.query(InfluencerVideo).filter(InfluencerVideo.id == video_id).first()
    
    print(f"Video ID: {video.id}")
    print(f"DB Duration: {video.duration}")
    
    if video.local_path and os.path.exists(video.local_path):
        import shutil
        # quick ffprobe or just trust the error msg for now. 
        # Actually MoviePy is installed in backend
        try:
             from moviepy.editor import VideoFileClip
             clip = VideoFileClip(video.local_path)
             print(f"Actual File Duration: {clip.duration}")
             clip.close()
        except Exception as e:
            print(f"Could not read file duration: {e}")
            
    if video.transcript_json:
        segments = video.transcript_json.get("segments", [])
        if segments:
            last_seg = segments[-1]
            print(f"Last Transcript Segment End: {last_seg.get('end')}")
            print(f"Total Segments: {len(segments)}")
        else:
            print("Transcript has no segments")
            
finally:
    db.close()
