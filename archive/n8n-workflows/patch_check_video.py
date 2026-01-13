
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
node = find_node_by_name(nodes_list, "Check Video Exists")

if node:
    print("Found 'Check Video Exists' node.")
    
    # New code that merges source data
    new_code = r"""const fs = require('fs');
try {
  const sourceNode = $('Prepare HeyGen Data').first();
  const sourceData = sourceNode.json;
  const scriptId = sourceData.script_id;
  const filePath = `/home/node/assets/avatar/${scriptId}_avatar.mp4`;
  const exists = fs.existsSync(filePath);
  
  // MERGE source data so audio_asset_id / avatar_id persist to the Generator
  return [{ 
      json: { 
          ...sourceData,
          exists, 
          filePath, 
          script_id: scriptId
      } 
  }];
} catch (e) {
  return [{ json: { exists: false, error: e.message } }];
}"""

    node['parameters']['jsCode'] = new_code
    print("Updated 'Check Video Exists' logic to preserve data.")

else:
    print("Node not found!")
    exit(1)

with open(file_path, 'w') as f:
    json.dump(data, f, indent=2)

print("Successfully wrote updated workflow to file.")
