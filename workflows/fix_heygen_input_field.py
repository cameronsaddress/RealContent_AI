
import json
import sys

file_path = '/home/canderson/n8n/workflows/COMPLETE_PIPELINE.json'

try:
    with open(file_path, 'r') as f:
        data = json.load(f)
except Exception as e:
    print(f"Error reading file: {e}")
    sys.exit(1)

# Find the node
nodes_list = data.get('nodes', [])
target_node = None

for node in nodes_list:
    if node.get('name') == "Upload HeyGen Audio":
        target_node = node
        break

if target_node:
    print(f"Found Node: {target_node['name']} (Type: {target_node['type']})")
    
    # Ensure it's the right type
    if target_node['type'] == 'n8n-nodes-base.httpRequest':
        params = target_node.get('parameters', {})
        
        # SHOTGUN APPROACH: Set all potential keys for "Input Data Field Name"
        
        # 1. Standard (what we had)
        params['binaryPropertyName'] = "data"
        
        # 2. Potential V4.2 variant (matches UI Label "Input Data Field Name" snake_cased?)
        params['inputDataFieldName'] = "data"
        
        # 3. ReadBinary variant (just in case)
        params['dataPropertyName'] = "data"
        
        # 4. Another common one
        # params['propertyName'] = "data" 
        
        target_node['parameters'] = params
        
        print("SUCCESS: Updated 'inputDataFieldName' and others to 'data'.")
        print("Current Parameters:", json.dumps(params, indent=2))
        
        # Save file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print("File saved.")
        
    else:
        print(f"ERROR: Node found but wrong type? Expected n8n-nodes-base.httpRequest, got {target_node['type']}")
        sys.exit(1)
        
else:
    print("ERROR: Node 'Upload HeyGen Audio' not found!")
    sys.exit(1)
