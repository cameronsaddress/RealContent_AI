
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

# 1. DELETE NODE
initial_count = len(nodes)
nodes = [n for n in nodes if n.get('id') != 'create-talking-photo-entity' and n.get('name') != 'Create Talking Photo Entity']
final_count = len(nodes)

if initial_count > final_count:
    print(f"Removed {initial_count - final_count} node(s) matching 'Create Talking Photo Entity'.")
else:
    print("No 'Create Talking Photo Entity' nodes found (Clean).")

# 2. FORCE REWIRE 'Upload Talking Photo' -> 'Prepare HeyGen Data'
# We do this regardless of existing connections to ensure it points to the right place.
if "Upload Talking Photo" in connections:
    print("Rewiring 'Upload Talking Photo' to 'Prepare HeyGen Data'...")
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
else:
    print("WARNING: 'Upload Talking Photo' not found in connections map!")

# 3. CLEANUP ORPHANED CONNECTIONS
if "Create Talking Photo Entity" in connections:
    del connections["Create Talking Photo Entity"]
    print("Removed 'Create Talking Photo Entity' from connections.")

data['nodes'] = nodes
data['connections'] = connections

with open(file_path, 'w') as f:
    json.dump(data, f, indent=2)

print("SUCCESS: Forced cleanup completed.")
