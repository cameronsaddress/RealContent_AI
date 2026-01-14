import sys
import os
import asyncio
from sqlalchemy.orm import Session

# Add current directory to path so we can import backend
sys.path.append("/app")

# Correct import paths for running inside container
try:
    from models import SessionLocal, InfluencerVideo
    from tasks.viral import _transcribe_video_async
except ImportError:
    # Fallback to backend module syntax if running from outside (not ideal for this script)
    sys.path.append(os.path.join(os.getcwd(), "backend"))
    from backend.models import SessionLocal, InfluencerVideo
    from backend.tasks.viral import _transcribe_video_async

def force_retranscribe(video_id: int):
    print(f"Force re-transcribing Video ID {video_id}...")
    db = SessionLocal()
    try:
        video = db.query(InfluencerVideo).filter(InfluencerVideo.id == video_id).first()
        if not video:
            print(f"Video {video_id} not found.")
            return

        print(f"Current Status: {video.status}")
        # Reset transcript
        video.transcript_json = None
        # Reset status to 'downloaded' so transcribe task accepts it
        video.status = "downloaded" 
        db.commit()
        print("Transcript cleared. Status reset to 'downloaded'.")
        
        # Run async task synchronously
        print("Triggering transcription task (this may take a while)...")
        loop = asyncio.get_event_loop()
        loop.run_until_complete(_transcribe_video_async(video_id))
        
        # Reload to check result
        db.refresh(video)
        print(f"New Status: {video.status}")
        if video.transcript_json:
            segments = video.transcript_json.get("segments", [])
            print(f"New Transcript has {len(segments)} segments.")
        else:
            print("No transcript JSON found after run.")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        vid = int(sys.argv[1])
    else:
        vid = 2
    force_retranscribe(vid)
