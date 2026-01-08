
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
    if node.get('name') == "Save Voice":
        target_node = node
        break

if target_node:
    print(f"Found Node: {target_node['name']} (Old Type: {target_node['type']})")
    
    # CONVERT TO NATIVE WRITE BINARY FILE
    target_node['type'] = 'n8n-nodes-base.writeBinaryFile'
    target_node['typeVersion'] = 1
    
    # Parameters
    # Note: Expression requires '=' prefix
    file_path_expr = "=/home/node/.n8n-files/assets/audio/{{ $('Prepare TTS Text').first().json.script_id }}_voice.mp3"
    
    target_node['parameters'] = {
        "operation": "write",
        "fileName": file_path_expr,
        "binaryPropertyName": "data",
        "options": {}
    }
    
    # Ensure ID remains same to preserve connections?
    # Yes, ID 'save-voice-file' is fine.
    
    print("Converted node to 'n8n-nodes-base.writeBinaryFile'.")
    print("New Parameters:", json.dumps(target_node['parameters'], indent=2))
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    print("File saved.")

else:
    print("Error: Node 'Save Voice' not found.")
    sys.exit(1)
