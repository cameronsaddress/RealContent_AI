
import json

file_path = '/home/canderson/n8n/workflows/COMPLETE_PIPELINE.json'

with open(file_path, 'r') as f:
    data = json.load(f)

# Helper to find node
def find_node_by_name(nodes, name):
    for node in nodes:
        if node.get('name') == name:
            return node
    return None

# Find the node
nodes_list = data.get('nodes', [])
node = find_node_by_name(nodes_list, "Upload HeyGen Audio")

if node:
    print(f"Found '{node['name']}' ({node['type']})")
    
    # Check if it is the new HTTP Node
    if node['type'] == 'n8n-nodes-base.httpRequest':
        # Add 'binaryPropertyName': 'data' to parameters
        node['parameters']['binaryPropertyName'] = 'data'
        print("Updated binaryPropertyName to 'data'.")
    else:
        print(f"Warning: Found node but type is {node['type']}, expected httpRequest.")
        # Attempt to set it anyway if it makes sense (N8N naming is strict)
        # Verify it's not the old Code node?
        # The refactor script renamed the old code node to "Read Audio File"
        # So "Upload HeyGen Audio" SHOULD be the HTTP node.
        pass

else:
    print("Node 'Upload HeyGen Audio' not found!")
    exit(1)

with open(file_path, 'w') as f:
    json.dump(data, f, indent=2)

print("Successfully wrote updated workflow to file.")
