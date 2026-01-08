
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
found = False

for node in nodes:
    if node.get('name') == "Create HeyGen Video":
        print(f"Found 'Create HeyGen Video' (ID: {node.get('id')})")
        
        # Get current jsonBody
        json_body = node['parameters'].get('jsonBody', '')
        
        # Replace dimensions
        if '"width": 1080' in json_body and '"height": 1920' in json_body:
            new_body = json_body.replace('"width": 1080', '"width": 720').replace('"height": 1920', '"height": 1280')
            node['parameters']['jsonBody'] = new_body
            found = True
            print(" -> Updated dimension to 720x1280.")
        else:
            print("WARNING: Could not find exact 1080x1920 string in jsonBody. Maybe already updated?")
            print(f"Current body snippet: {json_body[0:200]}...")
            # Attempt generic replacement just in case of formatting
            # But let's stick to simple replace execution
            
        break

if found:
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    print("SUCCESS: Updated resolution.")
else:
    print("Error: 'Create HeyGen Video' node not found or pattern mismatch.")
    sys.exit(1)
