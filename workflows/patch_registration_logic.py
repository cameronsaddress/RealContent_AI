
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

# --- 1. NEW NODE: Create Talking Photo Entity ---
create_tp_node = {
  "parameters": {
    "method": "POST",
    "url": "https://api.heygen.com/v2/talking_photo",
    "authentication": "genericCredentialType",
    "genericAuthType": "httpHeaderAuth",
    "sendHeaders": True,
    "headerParameters": {
      "parameters": [
        {
          "name": "Content-Type",
          "value": "application/json"
        }
      ]
    },
    "sendBody": True,
    "specifyBody": "json",
    "jsonBody": "={\n  \"image_asset_id\": \"{{ $('Upload Talking Photo').first().json.data.id }}\"\n}",
    "options": {}
  },
  "id": "create-talking-photo-entity",
  "name": "Create Talking Photo Entity",
  "type": "n8n-nodes-base.httpRequest",
  "typeVersion": 4.2,
  "position": [ 2350, 1300 ],
  "credentials": {
    "httpHeaderAuth": {
      "id": "heygen",
      "name": "heygen"
    }
  }
}

# Fix credentials: copy from 'Upload Talking Photo' if exists
idx_upload = get_node_index("Upload Talking Photo")
if idx_upload != -1:
    creds = nodes[idx_upload].get('credentials', {})
    if creds:
        create_tp_node['credentials'] = creds
        print(f"Copied credentials from 'Upload Talking Photo': {creds}")

nodes.append(create_tp_node)

# --- 2. UPDATE: Prepare HeyGen Data JS ---
prepare_js_code_new = r"""
// Parse inputs
const uploadResult = $('Upload HeyGen Audio').first().json;
const durationResult = $('Get Duration').first().json;
const ttsData = $('Prepare TTS Text').first().json;

// Check for Dynamic Photo Upload
let dynamicPhotoId = null;
try {
  // CHANGED: We now look for the REGISTERED Talking Photo ID, not the raw upload ID
  const registrationNode = $('Create Talking Photo Entity').first();
  
  if (registrationNode && registrationNode.json && registrationNode.json.data) {
      // The API returns data.talking_photo_id
      dynamicPhotoId = registrationNode.json.data.talking_photo_id;
  }
} catch(e) {
  // Fallback to check if we just have a raw upload (in case the registration node was skipped)
  try {
     const photoUpload = $('Upload Talking Photo').first();
     if (photoUpload && photoUpload.json && photoUpload.json.data) {
         dynamicPhotoId = photoUpload.json.data.id;
     }
  } catch(err) {}
}

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
        // We pass the dynamic ID in a specific field to keep things clean
        avatarId = dynamicPhotoId;
    } else {
        // Fallback if upload failed (safety net)
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
    // This avatar_id is now either a Stock ID OR a Talking Photo Asset ID
    avatar_id: avatarId,
    avatar_type: avatarType,
    audio_asset_id: uploadResult.data.id
  }
}];
"""

idx_prepare = get_node_index("Prepare HeyGen Data")
if idx_prepare != -1:
    nodes[idx_prepare]['parameters']['jsCode'] = prepare_js_code_new.strip()
    print("Updated 'Prepare HeyGen Data' JS Code.")


# --- 3. REWIRE CONNECTIONS ---
# Goal: Upload -> Create Entity -> Prepare

# A. Disconnect 'Upload Talking Photo' from 'Prepare HeyGen Data'
if "Upload Talking Photo" in connections:
     # Remove existing connection to Prepare (or any)
     if "main" in connections["Upload Talking Photo"]:
         connections["Upload Talking Photo"]["main"][0] = [] # Clear output
         
     # Connect to 'Create Talking Photo Entity'
     connections["Upload Talking Photo"]["main"] = [
         [
             {
                 "node": "Create Talking Photo Entity",
                 "type": "main",
                 "index": 0
             }
         ]
     ]

# B. Connect 'Create Talking Photo Entity' to 'Prepare HeyGen Data'
connections["Create Talking Photo Entity"] = {
    "main": [
        [
            {
                "node": "Prepare HeyGen Data",
                "type": "main",
                "index": 0
            }
        ]
    ]
}

print("Connections rewired: Upload -> Registration -> Prepare.")

with open(file_path, 'w') as f:
    json.dump(data, f, indent=2)

print("SUCCESS: Talking Photo Registration logic applied.")
