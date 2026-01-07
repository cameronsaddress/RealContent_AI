import os
import sys
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
try:
    from backend.models import ContentIdea, Script, Asset, ContentStatus, Base
except ImportError:
    # Inside container where /app is root
    from models import ContentIdea, Script, Asset, ContentStatus, Base

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://n8n:n8n_password@postgres:5432/content_pipeline")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

def analyze_duplicates():
    print("--- Analyzing Active/Stuck Ideas ---")
    
    # Filter for active or stuck statuses
    target_statuses = [
        ContentStatus.approved,
        ContentStatus.script_generating,
        ContentStatus.script_ready,
        ContentStatus.voice_generating,
        ContentStatus.voice_ready,
        ContentStatus.avatar_generating,
        ContentStatus.avatar_ready,
        ContentStatus.assembling,
        ContentStatus.captioning,
        ContentStatus.ready_to_publish,
        ContentStatus.publishing,
        ContentStatus.error
    ]
    
    ideas = db.query(ContentIdea).filter(ContentIdea.status.in_(target_statuses)).all()
    print(f"Found {len(ideas)} active/stuck ideas.")

    for idea in ideas:
        scripts = db.query(Script).filter(Script.content_idea_id == idea.id).order_by(Script.created_at.asc()).all()
        
        status_str = f"[{idea.status}]"
        if not scripts:
            print(f"Idea {idea.id} {status_str}: No scripts.")
            continue
        
        print(f"Idea {idea.id} {status_str}: {len(scripts)} scripts.")
        
        best_script = None
        best_score = -1
        
        for script in scripts:
            assets = db.query(Asset).filter(Asset.script_id == script.id).all()
            
            # Score based on progress
            score = 0
            has_audio = False
            has_video = False
            
            for asset in assets:
                if asset.voiceover_path: 
                    score += 10
                    has_audio = True
                if asset.avatar_video_path: 
                    score += 20
                    has_video = True
                if asset.final_video_path: score += 50
            
            print(f"  - Script {script.id} (Created: {script.created_at}): {len(assets)} assets. Score: {score}")
            
            if score > best_score:
                best_score = score
                best_script = script
        
        if len(scripts) > 1:
            print(f"  *** DUPLICATE DETECTED: Found {len(scripts)} scripts. Arbitrating... ***")
            
            # File verification
            for script in scripts:
                script.real_score = 0
                # Check Audio
                audio_path = f"/app/assets/audio/{script.id}_voice.mp3"
                if os.path.exists(audio_path):
                    script.real_score += 10
                    print(f"    - Script {script.id}: Found AUDIO on disk.")
                
                # Check Avatar
                avatar_path = f"/app/assets/avatar/{script.id}_avatar.mp4"
                if os.path.exists(avatar_path):
                    script.real_score += 20
                    print(f"    - Script {script.id}: Found AVATAR on disk.")

            # Sort by Score (Desc), then Created At (Desc)
            # We want the one with most files. If tie, latest one.
            sorted_scripts = sorted(scripts, key=lambda s: (getattr(s, 'real_score', 0), s.created_at), reverse=True)
            winner = sorted_scripts[0]
            losers = sorted_scripts[1:]
            
            print(f"  >>> WINNER: Script {winner.id} (Score: {getattr(winner, 'real_score', 0)})")
            
            for loser in losers:
                print(f"  >>> DELETING Loser Script {loser.id} (Score: {getattr(loser, 'real_score', 0)})")
                db.delete(loser)
            
            db.commit()
            print("  *** Cleanup Complete ***")


if __name__ == "__main__":
    analyze_duplicates()
