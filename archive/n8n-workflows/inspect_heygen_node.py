
import json

file_path = '/home/canderson/n8n/workflows/COMPLETE_PIPELINE.json'

with open(file_path, 'r') as f:
    data = json.load(f)

nodes = data.get('nodes', [])
found = False
for i, node in enumerate(nodes):
    if node.get('name') == "Upload HeyGen Audio":
        print(f"--- Node Found at Index {i} ---")
        print(json.dumps(node, indent=2))
        found = True

if not found:
    print("Node not found.")
