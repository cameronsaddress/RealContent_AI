import sys
import os
import json

# Add current directory to path so we can import backend
sys.path.append(os.getcwd())

try:
    from backend.models import SessionLocal, InfluencerVideo, Influencer
except ImportError:
    from models import SessionLocal, InfluencerVideo, Influencer

def check_intro():
    db = SessionLocal()
    try:
        print("Checking videos for 'Nicholas J Fuentes' intro phrase...", flush=True)
        
        # Get videos that might be relevant (or just all for now to see text)
        videos = db.query(InfluencerVideo).join(Influencer).filter(
            (Influencer.channel_id.ilike("%nicholas%")) | 
            (Influencer.name.ilike("%nicholas%"))
        ).all()
        
        if not videos:
            print("No Nicholas J Fuentes videos found in DB.", flush=True)
            videos = db.query(InfluencerVideo).all()
        
        for v in videos:
            print(f"\nScanning Video ID {v.id}: {v.title} (Influencer: {v.influencer.name})", flush=True)
            if not v.transcript_json:
                print("  - No transcript found.", flush=True)
                continue
                
            segments = v.transcript_json.get("segments", [])
            print(f"  - Transcript has {len(segments)} segments.", flush=True)
            if segments:
                last_seg = segments[-1]
                print(f"  - Transcript Ends at: {last_seg['end']} seconds ({last_seg['end']/60:.2f} minutes = {last_seg['end']/3600:.2f} hours)", flush=True)
            
            # Run the logic
            matches = []
            music_segments = []
            
            for i, seg in enumerate(segments):
                text = seg.get("text", "").lower()
                
                # Check for Music
                if "music plays" in text or "music playing" in text:
                    music_segments.append(seg)
                
                # Search for specific user phrase (Fuzzy)
                if "collapse" in text and "currency" in text:
                    print(f"  🎯 FUZZY PHRASE FOUND at segment {i}:", flush=True)
                    print(f"     Time: {seg['start']} -> {seg['end']}", flush=True)
                    print(f"     Text: '{seg['text']}'", flush=True)
                
                # Check for Phrase
                match_sequence = False
                if i > 0:
                     prev_text = segments[i-1].get("text", "").lower()
                     if "good evening" in prev_text and "america" in text:
                         match_sequence = True
                
                match_current = "america first" in text and ("good evening" in text or "watching" in text)
                
                if match_current or match_sequence:
                    start_time = seg["start"]
                    end_time = seg["end"]
                    
                    found_extended = False
                    if i + 1 < len(segments):
                        next_seg = segments[i+1]
                        if "nicholas" in next_seg.get("text", "").lower():
                            end_time = next_seg["end"]
                            found_extended = True
                    
                    matches.append({
                        "segment_index": i,
                        "time": start_time,
                        "cutoff": end_time,
                        "text": text,
                        "extended": found_extended
                    })

            print(f"\nMusic Analysis (First 3 Hours / 10800s):", flush=True)
            last_music_time = 0
            for i, seg in enumerate(segments):
                text = seg.get("text", "").lower()
                if ("music plays" in text or "music playing" in text) and seg["end"] < 10800:
                    music_segments.append(seg)
                    last_music_time = seg["end"]
            
            print(f"  Last 'Music plays' detected at: {last_music_time:.2f}s", flush=True)

            # Print segments around 1:24:18 (5058s)
            print("  - RAW Transcript Segments (5000s - 5200s):", flush=True)
            count = 0
            for s in segments:
                if 5000 < s['start'] < 5200:
                    print(f"    [{s['start']:.2f}-{s['end']:.2f}] {s['text']}", flush=True)
                    count += 1
            if count == 0:
                print("    (No segments found in this range)", flush=True)

            print(f"\nAnalysis Results:", flush=True)
            print(f"Total Phrase Matches: {len(matches)}", flush=True)
            for m in matches:
                 print(f"  Match at {m['time']:.2f}s -> Cutoff: {m['cutoff']:.2f}s (Extended: {m['extended']})", flush=True)

    except Exception as e:
        print(f"Error: {e}", flush=True)
    finally:
        db.close()

if __name__ == "__main__":
    check_intro()
