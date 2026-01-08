
import json
import sys

file_path = '/home/canderson/n8n/workflows/COMPLETE_PIPELINE.json'

try:
    with open(file_path, 'r') as f:
        data = json.load(f)
except Exception as e:
    print(f"Error reading file: {e}")
    sys.exit(1)

nodes = data.get('nodes', [])
connections = data.get('connections', {})

def get_node_index(name):
    for i, n in enumerate(nodes):
        if n.get('name') == name:
            return i
    return -1

# 1. REMOVE 'Create Talking Photo Entity'
nodes = [n for n in nodes if n.get('name') != 'Create Talking Photo Entity']

# 2. UPDATE 'Upload Talking Photo'
idx_upload = get_node_index("Upload Talking Photo")
if idx_upload != -1:
    nodes[idx_upload]['parameters']['url'] = "https://upload.heygen.com/v1/asset"
    # Ensure binary mode is set (it should be, but let's be safe)
    nodes[idx_upload]['parameters']['sendBody'] = True
    nodes[idx_upload]['parameters']['contentType'] = "binaryData"
    nodes[idx_upload]['parameters']['binaryPropertyName'] = "data"
    print("Updated 'Upload Talking Photo' to v1/asset and binary mode.")

# 3. UPDATE 'Prepare HeyGen Data' JS Code
# Logic: Use the ID from 'Upload Talking Photo' directly.
prepare_js_code_revert = r"""
// Parse inputs
const uploadResult = $('Upload HeyGen Audio').first().json;
const durationResult = $('Get Duration').first().json;
const ttsData = $('Prepare TTS Text').first().json;

// Check for Dynamic Photo Upload
let dynamicPhotoId = null;
try {
  // Reverted: Use the ID directly from the Upload node (v1/asset ID works as Talking Photo ID)
  const photoUpload = $('Upload Talking Photo').first();
  if (photoUpload && photoUpload.json && photoUpload.json.data) {
      dynamicPhotoId = photoUpload.json.data.id;
  }
} catch(e) {}

let duration = 45;
try {
  duration = parseFloat(durationResult.stdout.trim()) || 45;
} catch (e) {}

// Avatar Logic
let avatarId = ttsData.avatar_id;
let avatarType = ttsData.avatar_type;

// If static_image, we MUST have a dynamic ID now
if (avatarId === 'static_image') {
    if (dynamicPhotoId) {
        avatarType = 'talking_photo';
        avatarId = dynamicPhotoId;
    } else {
        // Fallback
        console.log('Missing dynamic photo upload, falling back to default avatar');
        avatarId = 'Anna_public_3_20240108';
        avatarType = 'avatar';
    }
}

return [{
  json: {
    asset_id: ttsData.asset_id,
    script_id: ttsData.script_id,
    content_idea_id: ttsData.content_idea_id,
    voice_path: `/home/node/.n8n-files/assets/audio/${ttsData.script_id}_voice.mp3`,
    duration_seconds: Math.round(duration * 10) / 10,
    broll_paths: ttsData.broll_paths,
    avatar_id: avatarId,
    avatar_type: avatarType,
    audio_asset_id: uploadResult.data.id
  }
}];
"""

idx_prepare = get_node_index("Prepare HeyGen Data")
if idx_prepare != -1:
    nodes[idx_prepare]['parameters']['jsCode'] = prepare_js_code_revert.strip()
    print("Updated 'Prepare HeyGen Data' JS to use upload ID directly.")

# 4. REWIRE CONNECTIONS
# Remove 'Create Talking Photo Entity' from connections mapping
if "Create Talking Photo Entity" in connections:
    del connections["Create Talking Photo Entity"]

# Connect 'Upload Talking Photo' -> 'Prepare HeyGen Data'
if "Upload Talking Photo" in connections:
    connections["Upload Talking Photo"]["main"] = [
        [
            {
                "node": "Prepare HeyGen Data",
                "type": "main",
                "index": 0
            }
        ]
    ]

data['nodes'] = nodes
data['connections'] = connections

print("Connections rewired: Upload -> Prepare.")

with open(file_path, 'w') as f:
    json.dump(data, f, indent=2)

print("SUCCESS: Reverted to Asset Upload pipeline.")
