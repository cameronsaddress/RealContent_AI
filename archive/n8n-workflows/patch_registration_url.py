
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

# New JSON Body using URL from upload node
new_json_body = "={\n  \"url\": \"{{ $('Upload Talking Photo').first().json.data.url }}\"\n}"

for node in nodes:
    if node.get('name') == "Create Talking Photo Entity":
        print(f"Found 'Create Talking Photo Entity' (ID: {node.get('id')})")
        
        # Update jsonBody
        node['parameters']['jsonBody'] = new_json_body
        found = True
        print(" -> Updated jsonBody to use 'url'.")
        break

if found:
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    print("SUCCESS: Updated workflow file.")
else:
    print("Error: 'Create Talking Photo Entity' node not found.")
    sys.exit(1)
