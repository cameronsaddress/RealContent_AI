
import json

file_path = '/home/canderson/n8n/workflows/COMPLETE_PIPELINE.json'

with open(file_path, 'r') as f:
    data = json.load(f)

# Helper to find node index
def find_node_index(nodes, name):
    for i, node in enumerate(nodes):
        if node.get('name') == name:
            return i
    return -1

nodes = data.get('nodes', [])
connections = data.get('connections', {})

# 1. FIND existing "Upload HeyGen Audio" (Code Node)
idx = find_node_index(nodes, "Upload HeyGen Audio")
if idx == -1:
    print("Error: 'Upload HeyGen Audio' node not found.")
    exit(1)

code_node = nodes[idx]
print("Found 'Upload HeyGen Audio' Code Node.")

# 2. CONVERT Code Node to "Read Audio File" (ReadBinaryFile)
# We repurpose the existing node object to keep its ID/position/connections (partially)
read_node = code_node
read_node['name'] = "Read Audio File"
read_node['id'] = "read-audio-file" # Changing ID requires updating connections!
read_node['type'] = "n8n-nodes-base.readBinaryFile"
read_node['typeVersion'] = 1
read_node['parameters'] = {
    "mode": "filePath",
    "filePath": "=/home/node/assets/audio/{{ $json.script_id }}_voice.mp3",
    "dataPropertyName": "data"
}
# Keep position roughly same
# Verify connections to this node?
# N8N connections usage "NodeName".main...
# Check connections pointing TO "Upload HeyGen Audio" -> Change to "Read Audio File"

# 3. CREATE new "Upload HeyGen Audio" (HTTP Request)
http_node = {
    "name": "Upload HeyGen Audio",
    "id": "upload-heygen-audio-http",
    "type": "n8n-nodes-base.httpRequest",
    "typeVersion": 4.2,
    "position": [
        read_node['position'][0] + 200, # Move it right suitable
        read_node['position'][1]
    ],
    "credentials": {
        "httpHeaderAuth": {
            "id": "heygen",
            "name": "heygen"
        }
    },
    "parameters": {
        "method": "POST",
        "url": "=https://upload.heygen.com/v1/asset?name={{ $('Read Audio File').first().json.script_id }}_voice.mp3",
        "authentication": "genericCredentialType",
        "genericAuthType": "httpHeaderAuth",
        "sendBody": True,
        "contentType": "binaryData",
        "binaryPropertyName": "data",
        "options": {}
    }
}
nodes.append(http_node)

# 4. FIX CONNECTIONS
# A. Inputs into "Upload HeyGen Audio" (Old Name) -> Point to "Read Audio File"
# B. "Read Audio File" -> "Upload HeyGen Audio" (New HTTP)
# C. "Upload HeyGen Audio" (New HTTP) -> "Prepare HeyGen Data"

# We must scan 'connections' object.
# Structure: { "SourceNode": { "main": [ [ { "node": "DestNode", ... } ] ] } }

new_connections = {}

for src_node, outputs in connections.items():
    new_outputs = {}
    for output_name, channels in outputs.items(): # usually "main"
        new_channels = []
        for channel in channels:
            new_channel = []
            for connection in channel:
                target = connection['node']
                
                # Case A: If target was old Code Node, point to Read Node
                if target == "Upload HeyGen Audio":
                    connection['node'] = "Read Audio File"
                    new_channel.append(connection)
                
                # Case Output of Old Node (now Read Node):
                # We need to handle the output OF the Read Node separately
                # Wait, we are iterating Key=Source.
                
                else:
                    new_channel.append(connection)
            new_channels.append(new_channel)
        new_outputs[output_name] = new_channels
    
    # Store with potentially NEW key if the source node was renamed
    final_src_name = src_node
    if src_node == "Upload HeyGen Audio":
        final_src_name = "Read Audio File"
        
        # This was the output of the old node. 
        # It used to point to "Prepare HeyGen Data".
        # Now "Read Audio File" should point to "Upload HeyGen Audio" (HTTP).
        # And "Upload HeyGen Audio" (HTTP) should point to "Prepare HeyGen Data".
        
        # Override the outputs for "Read Audio File"
        new_outputs = {
            "main": [
                [
                    {
                        "node": "Upload HeyGen Audio",
                        "type": "main",
                        "index": 0
                    }
                ]
            ]
        }
    
    new_connections[final_src_name] = new_outputs

# Add connections for the NEW HTTP Node
# It should point to whatever the old node pointed to, which is "Prepare HeyGen Data"
# But we just overwrote "Read Audio File" outputs.
# We need to find what "Upload HeyGen Audio" pointed to originally.
# In original 'connections', "Upload HeyGen Audio" -> "Prepare HeyGen Data".
# So set "Upload HeyGen Audio" -> "Prepare HeyGen Data"

new_connections["Upload HeyGen Audio"] = {
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

data['connections'] = new_connections

# Save
with open(file_path, 'w') as f:
    json.dump(data, f, indent=2)

print("Successfully refactored HeyGen Upload nodes.")
