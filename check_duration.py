import sys
import os
import subprocess

# Add current directory to path so we can import backend
sys.path.append(os.getcwd())

try:
    from backend.models import SessionLocal, InfluencerVideo
except ImportError:
    from models import SessionLocal, InfluencerVideo

def check_duration():
    db = SessionLocal()
    try:
        video = db.query(InfluencerVideo).filter(InfluencerVideo.id == 2).first()
        if not video:
            print("Video ID 2 not found.")
            return
            
        print(f"Video ID 2: {video.title}")
        if video.duration:
            print(f"DB Duration: {video.duration} seconds ({video.duration/60:.2f} minutes)")
        else:
            print("DB Duration: None")
        print(f"Local Path: {video.local_path}")
        
        # Check actual file duration using ffprobe if available, or just file size
        if video.local_path and os.path.exists(video.local_path):
            size = os.path.getsize(video.local_path)
            print(f"File Size: {size / (1024*1024):.2f} MB")
            
            # Simple duration check via ffmpeg if possible?
            # Creating a quick subprocess call
            try:
                cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video.local_path]
                result = subprocess.run(cmd, capture_output=True, text=True)
                print(f"FFprobe Duration: {result.stdout.strip()} seconds")
            except Exception as e:
                print(f"FFprobe failed: {e}")
                
        else:
            print("File not found on disk.")
            
    finally:
        db.close()

if __name__ == "__main__":
    check_duration()
