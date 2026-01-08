
import json
import sys

file_path = '/home/canderson/n8n/workflows/COMPLETE_PIPELINE.json'

try:
    with open(file_path, 'r') as f:
        data = json.load(f)
except Exception as e:
    print(f"Error reading file: {e}")
    sys.exit(1)

nodes_list = data.get('nodes', [])
found = False

# The ROBUST JavaScript Code for FFmpeg Mixing with Safety Check
new_js_code = r"""
const fs = require('fs');
const extractNode = $('Extract Video ID').first();
// Robust fallback if extraction hasn't happened yet (idempotency)
const sourceData = extractNode ? extractNode.json : $('Prepare HeyGen Data').first().json;
const scriptId = sourceData.script_id;

// --- FILE PATHS ---
// Ensure these match your Docker volume mapping
const avatarPath = `/home/node/.n8n-files/assets/avatar/${scriptId}_avatar.mp4`;
const outputPath = `/home/node/.n8n-files/assets/output/${scriptId}_combined.mp4`;
const musicPath = `/home/node/.n8n-files/assets/music/background_music.mp3`; 
const brollPaths = sourceData.broll_paths || [];

// --- BUILD INPUTS ---
let inputs = `-i ${avatarPath}`; // Input 0: The Avatar Video
let inputCount = 1;

// SAFETY CHECK: Does the music file actually exist?
let musicExists = false;
try {
    if (fs.existsSync(musicPath)) {
        musicExists = true;
    }
} catch (e) {
    console.log("Music check failed, proceeding without music.");
}

let musicIndex = -1;

if (musicExists) {
  // -stream_loop -1 makes the music repeat if the video is longer than the song
  inputs += ` -stream_loop -1 -i ${musicPath}`;
  musicIndex = inputCount;
  inputCount++;
}

// Add B-Roll Inputs
brollPaths.forEach(path => {
  inputs += ` -i ${path}`;
});
const brollStartIndex = inputCount;

// --- BUILD FILTER COMPLEX ---
let fc = '';

// 1. VIDEO COMPOSITING
if (brollPaths.length > 0) {
  // Create a chain of B-Roll clips
  const brollInputs = brollPaths.map((_, i) => `[${brollStartIndex + i}:v]`).join('');
  
  // Concatenate B-Roll
  fc += `${brollInputs}concat=n=${brollPaths.length}:v=1:a=0[broll_raw];`;
  
  // Scale and Crop B-Roll to 9:16 (1080x1920) to match Avatar
  fc += `[broll_raw]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920[broll];`;
  
  // Chroma Key the Avatar (Green Screen Removal)
  fc += `[0:v]chromakey=0x00FF00:0.1:0.2[avatar_keyed];`;
  
  // Overlay Avatar on top of B-Roll
  fc += `[broll][avatar_keyed]overlay=(W-w)/2:H-h-100:shortest=1[outv];`;
} else {
  // Fallback: If no B-Roll, put Avatar over a solid background
  fc += `[0:v]chromakey=0x00FF00:0.1:0.2[avatar_keyed];`;
  fc += `color=c=#1a1a2e:s=1080x1920:d=120[bg];`;
  fc += `[bg][avatar_keyed]overlay=(W-w)/2:H-h-100:shortest=1[outv];`;
}

// 2. AUDIO MIXING
if (musicExists) {
  // [0:a] is the Voice (Avatar)
  // [musicIndex:a] is the Background Music
  // volume=0.1 drops music to 10% so voice is clear
  // amix mixes them. duration=first cuts audio when video ends.
  fc += `[${musicIndex}:a]volume=0.1[music_low];[0:a][music_low]amix=inputs=2:duration=first[outa]`;
} else {
  // If no music, just pass the voice through as stereo
  fc += `[0:a]aformat=channel_layouts=stereo[outa]`;
}

// --- FINAL COMMAND ---
const ffmpegCmd = `ffmpeg -y ${inputs} -filter_complex "${fc}" -map "[outv]" -map "[outa]" -c:v libx264 -preset fast -crf 23 -c:a aac -b:a 128k -movflags +faststart ${outputPath}`;

return [{
  json: {
    asset_id: sourceData.asset_id,
    script_id: scriptId,
    content_idea_id: sourceData.content_idea_id,
    ffmpeg_command: ffmpegCmd,
    output_path: outputPath
  }
}];
"""

for node in nodes_list:
    if node.get('name') == "Build FFmpeg Command":
        print(f"Found 'Build FFmpeg Command' (ID: {node.get('id')})")
        params = node.get('parameters', {})
        
        # Update JS Code
        params['jsCode'] = new_js_code.strip()
        
        node['parameters'] = params
        found = True
        print(" -> Updated jsCode with ROBUST Music Check.")
        break

if found:
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    print("SUCCESS: Updated workflow file.")
else:
    print("Error: 'Build FFmpeg Command' node not found.")
    sys.exit(1)
