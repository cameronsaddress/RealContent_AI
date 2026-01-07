import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

try:
    from backend.models import ContentIdea, Script, ContentStatus, Base
except ImportError:
    from models import ContentIdea, Script, ContentStatus, Base

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://n8n:n8n_password@postgres:5432/content_pipeline")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

def fix_stuck_statuses():
    print("--- Fixing Stuck Statuses ---")
    
    # Find ideas stuck in script_generating
    stuck_ideas = db.query(ContentIdea).filter(ContentIdea.status == ContentStatus.script_generating).all()
    print(f"Found {len(stuck_ideas)} ideas in 'script_generating'.")
    
    fixed_count = 0
    
    for idea in stuck_ideas:
        # Check if they actually have a script
        script = db.query(Script).filter(Script.content_idea_id == idea.id).first()
        
        if script:
            print(f"Fixing Idea {idea.id}: Has script {script.id}. Updating status to 'script_ready'.")
            idea.status = ContentStatus.script_ready
            fixed_count += 1
        else:
            print(f"Skipping Idea {idea.id}: No script found.")
            
    db.commit()
    print(f"--- Complete. Fixed {fixed_count} ideas. ---")

if __name__ == "__main__":
    fix_stuck_statuses()
