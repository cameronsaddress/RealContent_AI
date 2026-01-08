
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
heygen_found = False
wait_found = False

# 1. EXPANDED N8N EXPRESSION for HeyGen
# We use a ternary operator to handle the logic inside the expression {{ ... }}
heygen_expression = """={{
{
  "video_inputs": [
    {
      "character": $json.avatar_id === 'static_image' 
        ? {
            "type": "talking_photo",
            "talking_photo_id": $env.HEYGEN_TALKING_PHOTO_ID
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

for node in nodes_list:
    # ---------------------------------------------------------
    # PATCH 1: Create HeyGen Video (Dynamic Payload)
    # ---------------------------------------------------------
    if node.get('name') == "Create HeyGen Video":
        print(f"Found 'Create HeyGen Video' (ID: {node.get('id')})")
        params = node.get('parameters', {})
        
        # Set Body to JSON Expression
        params['specifyBody'] = 'json'
        params['jsonBody'] = heygen_expression
        
        node['parameters'] = params
        heygen_found = True
        print(" -> Updated jsonBody to Dynamic Expression.")

    # ---------------------------------------------------------
    # PATCH 2: Wait & Retry (Increase Timeout)
    # ---------------------------------------------------------
    elif node.get('id') == "wait-retry" or node.get('name') == "Wait & Retry":
        print(f"Found 'Wait & Retry' (ID: {node.get('id')})")
        params = node.get('parameters', {})
        
        # Change amount from 15 to 60
        old_amount = params.get('amount')
        params['amount'] = 60
        
        node['parameters'] = params
        wait_found = True
        print(f" -> Updated amount: {old_amount} -> 60")

if not heygen_found:
    print("Error: 'Create HeyGen Video' node not found.")
if not wait_found:
    print("Error: 'Wait & Retry' node not found.")

if heygen_found and wait_found:
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    print("SUCCESS: Updated workflow file.")
else:
    print("Partial failure. Check node names.")
    sys.exit(1)
