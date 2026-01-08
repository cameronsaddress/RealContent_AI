
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

# Correct Endpoint found via Curl Testing
correct_url = "https://upload.heygen.com/v1/talking_photo"

for node in nodes:
    if node.get('name') == "Create Talking Photo Entity":
        print(f"Found 'Create Talking Photo Entity' (ID: {node.get('id')})")
        
        # Update URL
        node['parameters']['url'] = correct_url
        found = True
        print(f" -> Updated URL to '{correct_url}'.")
        break

if found:
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    print("SUCCESS: Updated workflow endpoint.")
else:
    print("Error: 'Create Talking Photo Entity' node not found.")
    sys.exit(1)
