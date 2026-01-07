
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

# Find the ElevenLabs node
if 'nodes' in data:
    node = find_node_by_name(data['nodes'], "Generate Voice (ElevenLabs)")
else:
    # Handle case where it might be a list (rare for full export)
    node = find_node_by_name(data, "Generate Voice (ElevenLabs)")

if node:
    print("Found 'Generate Voice (ElevenLabs)' node.")
    # The current invalid value is likely causing JSON parse issues for N8N, 
    # but strictly speaking it IS a string in the JSON file, so python loads it fine.
    
    # We want to replace it with a valid N8N object expression.
    # Pattern: ={{ { key: value, ... } }}
    # N8N handles the serialization of the object to JSON body automatically.
    
    new_json_body = "={{ { \"text\": $json.text_for_tts, \"model_id\": \"eleven_multilingual_v2\", \"voice_settings\": { \"stability\": 0.5, \"similarity_boost\": 0.75, \"style\": 0.5, \"use_speaker_boost\": true } } }}"
    
    node['parameters']['jsonBody'] = new_json_body
    print("Updated 'jsonBody' to valid Object Expression.")

else:
    print("Node not found!")
    exit(1)

with open(file_path, 'w') as f:
    json.dump(data, f, indent=2)

print("Successfully wrote updated workflow to file.")
