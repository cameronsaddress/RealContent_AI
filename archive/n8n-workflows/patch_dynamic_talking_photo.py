
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

# --- HELPER: Find Node Index by Name ---
def get_node_index(name):
    for i, n in enumerate(nodes):
        if n.get('name') == name:
            return i
    return -1

# --- 1. DEFINE NEW NODES ---

check_is_photo_node = {
  "parameters": {
    "conditions": {
      "options": {
        "caseSensitive": True,
        "leftValue": "",
        "typeValidation": "strict"
      },
      "conditions": [
        {
          "id": "check-is-photo",
          "leftValue": "={{ $('Prepare TTS Text').first().json.avatar_id }}",
          "rightValue": "static_image",
          "operator": {
            "type": "string",
            "operation": "equals"
          }
        }
      ],
      "combinator": "and"
    },
    "options": {}
  },
  "id": "check-is-photo",
  "name": "Is Talking Photo?",
  "type": "n8n-nodes-base.if",
  "typeVersion": 2,
  "position": [ 1800, 1400 ]
}

download_char_image_node = {
  "parameters": {
    "method": "GET",
    "url": "={{ $('Prepare TTS Text').first().json.config_image_url }}",
    "options": {
      "response": {
        "response": {
          "responseFormat": "file"
        }
      }
    }
  },
  "id": "download-char-image",
  "name": "Download Character Image",
  "type": "n8n-nodes-base.httpRequest",
  "typeVersion": 4.2,
  "position": [ 2000, 1300 ]
}

upload_talking_photo_node = {
  "parameters": {
    "method": "POST",
    "url": "https://upload.heygen.com/v1/asset",
    "authentication": "genericCredentialType",
    "genericAuthType": "httpHeaderAuth",
    "sendBody": True,
    "contentType": "binaryData",
    "binaryPropertyName": "data",
    "options": {},
    "inputDataFieldName": "data"
  },
  "id": "upload-talking-photo",
  "name": "Upload Talking Photo",
  "type": "n8n-nodes-base.httpRequest",
  "typeVersion": 4.2,
  "position": [ 2200, 1300 ],
  # Note: Credentials usually require ID reference. 
  # Assuming 'heygen' credential exists with ID 'heygen' or similar. 
  # If checking existing nodes, 'Upload HeyGen Audio' probably has correct credential ID.
  "credentials": {
    "httpHeaderAuth": {
      "id": "heygen", # Use lowercase as generic guess, or copy from existing
      "name": "heygen"
    }
  }
}

# --- 2. UPDATE EXISTING NODES ---

# Update 'Prepare HeyGen Data' JS Code
prepare_js_code = r"""
// Parse inputs
const uploadResult = $('Upload HeyGen Audio').first().json;
const durationResult = $('Get Duration').first().json;
const ttsData = $('Prepare TTS Text').first().json;

// Check for Dynamic Photo Upload
let dynamicPhotoId = null;
try {
  // Try to find the upload result from the 'True' branch
  // We use .first() which might be tricky if not executed, but previous nodes 
  // provide context. If strictly unconnected, might throw. 
  // n8n requires functional style usually, but $() selector works globally if node finished.
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

# Update 'Create HeyGen Video' JSON Body
create_json_body = r"""={{
{
  "video_inputs": [
    {
      "character": $json.avatar_type === 'talking_photo'
        ? {
            "type": "talking_photo",
            "talking_photo_id": $json.avatar_id
          }
        : {
            "type": "avatar",
            "avatar_id": $json.avatar_id,
            "avatar_style": "normal"
          },
      "voice": {
        "type": "audio",
        "audio_asset_id": $json.audio_asset_id
      },
      "background": {
        "type": "color",
        "value": "#00FF00"
      }
    }
  ],
  "dimension": {
    "width": 1080,
    "height": 1920
  },
  "aspect_ratio": "9:16"
}
}}"""


# --- EXECUTE MODIFICATIONS ---

# Check credential ID from 'Upload HeyGen Audio' to match
upload_audio_idx = get_node_index("Upload HeyGen Audio")
if upload_audio_idx != -1:
    creds = nodes[upload_audio_idx].get('credentials', {})
    if 'httpHeaderAuth' in creds:
        upload_talking_photo_node['credentials'] = creds
        print(f"Copied credentials from 'Upload HeyGen Audio': {creds}")

# Append new nodes
nodes.append(check_is_photo_node)
nodes.append(download_char_image_node)
nodes.append(upload_talking_photo_node)

# Apply updates to existing nodes
idx_prepare = get_node_index("Prepare HeyGen Data")
if idx_prepare != -1:
    nodes[idx_prepare]['parameters']['jsCode'] = prepare_js_code.strip()
    print("Updated 'Prepare HeyGen Data' JS Code.")

idx_create = get_node_index("Create HeyGen Video")
if idx_create != -1:
    nodes[idx_create]['parameters']['jsonBody'] = create_json_body.strip()
    print("Updated 'Create HeyGen Video' JSON Body.")

# --- 3. REWIRE CONNECTIONS ---
# Current Flow: Upload HeyGen Audio -> Prepare HeyGen Data

# 1. Disconnect 'Upload HeyGen Audio' from 'Prepare HeyGen Data'
if "Upload HeyGen Audio" in connections:
    if "main" in connections["Upload HeyGen Audio"]:
         # Remove connection to Prepare HeyGen Data
         connections["Upload HeyGen Audio"]["main"][0] = [
             conn for conn in connections["Upload HeyGen Audio"]["main"][0] 
             if conn["node"] != "Prepare HeyGen Data"
         ]
         # Add connection to 'Is Talking Photo?'
         connections["Upload HeyGen Audio"]["main"][0].append({
             "node": "Is Talking Photo?",
             "type": "main",
             "index": 0
         })

# 2. Wire 'Is Talking Photo?'
connections["Is Talking Photo?"] = {
    "main": [
        [ # True: Download Character Image
            {
                "node": "Download Character Image",
                "type": "main",
                "index": 0
            }
        ],
        [ # False: Prepare HeyGen Data (Stock Avatar)
            {
                "node": "Prepare HeyGen Data",
                "type": "main",
                "index": 0
            }
        ]
    ]
}

# 3. Wire 'Download Character Image' -> 'Upload Talking Photo'
connections["Download Character Image"] = {
    "main": [
        [
            {
                "node": "Upload Talking Photo",
                "type": "main",
                "index": 0
            }
        ]
    ]
}

# 4. Wire 'Upload Talking Photo' -> 'Prepare HeyGen Data'
connections["Upload Talking Photo"] = {
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

print("Connections rewired successfully.")

with open(file_path, 'w') as f:
    json.dump(data, f, indent=2)

print("SUCCESS: Dynamic Talking Photo logic applied.")
