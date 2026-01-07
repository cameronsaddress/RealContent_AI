
import json

file_path = '/home/canderson/n8n/workflows/COMPLETE_PIPELINE.json'

with open(file_path, 'r') as f:
    data = json.load(f)

# Allow function to recursively find nodes
def find_node_by_name(nodes, name):
    for node in nodes:
        if node.get('name') == name:
            return node
    return None

# Find the node
nodes_list = data.get('nodes', [])
node = find_node_by_name(nodes_list, "Generate Voice (ElevenLabs)")

if node:
    print("Found 'Generate Voice (ElevenLabs)' node.")
    params = node.get('parameters', {})
    
    # 1. FIX AUTHENTICATION
    # Switch to Generic Credential Type -> Http Header Auth
    params['authentication'] = 'genericCredentialType'
    params['genericAuthType'] = 'httpHeaderAuth'
    
    # Set the Credential ID (found in credentials.json as "elevenlabs")
    # Note: Node object usually has a separate "credentials" field for mapping
    # OR params['credentialId'] in older versions?
    # In V2+ genericCredentialType, we need to map it in the 'credentials' property of the node object.
    
    if 'credentials' not in node:
        node['credentials'] = {}
        
    # Map 'httpHeaderAuth' to the credential ID 'elevenlabs'
    node['credentials']['httpHeaderAuth'] = {
        "id": "elevenlabs",
        "name": "elevenlabs"
    }
    
    # Remove manual header if present
    header_params = params.get('headerParameters', {}).get('parameters', [])
    new_header_params = [p for p in header_params if p['name'] != 'xi-api-key']
    
    if 'headerParameters' in params:
        params['headerParameters']['parameters'] = new_header_params

    # 2. FIX RESPONSE FORMAT (FILE/BINARY)
    # The user noted "9 byte file" -> corrupted text response saved as binary
    # We must explicitly set response format to file.
    
    # Correct structure for httpRequest v4.2 usually involves:
    # options: { response: { format: 'file' } }
    
    options = params.get('options', {})
    if 'response' in options:
        # Detected potentially bad nesting: "response": { "response": { ... } }
        # Reset it to clean structure
        options['response'] = {
            "format": "file"
        }
    else:
        options['response'] = {
            "format": "file"
        }
        
    params['options'] = options
    
    # Also ensure binary property name is set?
    # For "file" format, defaults to 'data', but some versions have a param.
    # In httpRequest V4, it is often implicit or 'binaryPropertyName'
    # Let's add 'binaryPropertyName': 'data' to top-level params just in case functionality supports it
    # checking docs... V4 usually puts content in 'data' binary property by default.
    
    # Update node
    node['parameters'] = params
    
    print("Updated ElevenLabs node: Enable Credential Auth, Response Format = File.")

else:
    print("Node not found!")
    exit(1)

with open(file_path, 'w') as f:
    json.dump(data, f, indent=2)

print("Successfully wrote updated workflow to file.")
