
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
node = find_node_by_name(nodes_list, "Check Voice Exists")

if node:
    print("Found 'Check Voice Exists' node.")
    
    # New code that merges source data
    new_code = r"""const fs = require('fs');
try {
  const sourceNode = $('Prepare TTS Text').first();
  const sourceData = sourceNode.json;
  const scriptId = sourceData.script_id;
  const filePath = `/home/node/assets/audio/${scriptId}_voice.mp3`;
  const exists = fs.existsSync(filePath);
  
  // MERGE source data so voice_id/text_for_tts persist to the Generator
  return [{ 
      json: { 
          ...sourceData,
          exists, 
          filePath, 
          script_id: scriptId,
          // Explicitly ensure voice_id is passed if it exists in source
          voice_id: sourceData.voice_id 
      } 
  }];
} catch (e) {
  return [{ json: { exists: false, error: e.message } }];
}"""

    node['parameters']['jsCode'] = new_code
    print("Updated 'Check Voice Exists' logic to preserve data.")

else:
    print("Node not found!")
    exit(1)

with open(file_path, 'w') as f:
    json.dump(data, f, indent=2)

print("Successfully wrote updated workflow to file.")
