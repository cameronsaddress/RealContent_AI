
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

def get_node_index(name):
    for i, n in enumerate(nodes):
        if n.get('name') == name:
            return i
    return -1

# 1. UPDATE 'Upload Talking Photo' Node
idx_upload = get_node_index("Upload Talking Photo")
if idx_upload != -1:
    nodes[idx_upload]['parameters']['url'] = "https://upload.heygen.com/v1/talking_photo?name=avatar.jpg"
    # Ensure binary mode settings per user request
    nodes[idx_upload]['parameters']['sendBody'] = True
    nodes[idx_upload]['parameters']['contentType'] = "binaryData"
    nodes[idx_upload]['parameters']['binaryPropertyName'] = "data"
    nodes[idx_upload]['parameters']['inputDataFieldName'] = "data"
    nodes[idx_upload]['parameters']['method'] = "POST"
    print("Updated 'Upload Talking Photo' URL to v1/talking_photo.")

# 2. UPDATE 'Prepare HeyGen Data' JS Code
prepare_js_code_final = r"""
// Parse inputs
const uploadResult = $('Upload HeyGen Audio').first().json;
const durationResult = $('Get Duration').first().json;
const ttsData = $('Prepare TTS Text').first().json;

// Check for Dynamic Photo Upload
let dynamicPhotoId = null;
try {
  const photoUpload = $('Upload Talking Photo').first();
  if (photoUpload && photoUpload.json && photoUpload.json.data) {
      // v1/talking_photo returns 'talking_photo_id'. v1/asset returns 'id'.
      // We prioritize talking_photo_id.
      dynamicPhotoId = photoUpload.json.data.talking_photo_id || photoUpload.json.data.id;
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
    nodes[idx_prepare]['parameters']['jsCode'] = prepare_js_code_final.strip()
    print("Updated 'Prepare HeyGen Data' JS to handle talking_photo_id.")

with open(file_path, 'w') as f:
    json.dump(data, f, indent=2)

print("SUCCESS: Updated Upload Endpoint to v1/talking_photo.")
