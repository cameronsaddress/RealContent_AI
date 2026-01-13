
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

# Find the Prepare TTS Text node
nodes_list = data.get('nodes', [])
node = find_node_by_name(nodes_list, "Prepare TTS Text")

if node:
    print("Found 'Prepare TTS Text' node.")
    original_code = node['parameters']['jsCode']
    
    # New code with robust ID validation
    # If voice_id is missing, short, or invalid, use Rachel ID.
    new_code = r"""// Aggregate B-roll results and prepare for voice generation
const brollItems = $('Merge B-Roll').all();
const charConfig = $input.first().json || {};

const firstItem = $('Select Videos').first().json;

// Get script and asset data
const scriptData = $('Save Script').first().json;
const assetData = $('Create Asset Record').first().json;
const parseData = $('Parse Script').first().json;

// Clean script for TTS
let text = parseData.script || '';
text = text.replace(/\[PAUSE\]/gi, '...');
text = text.replace(/\*(.*?)\*/g, '$1');
text = text.replace(/\s+/g, ' ').trim();

if (parseData.cta) {
  text = text + ' ... ' + parseData.cta;
}

// Robust Voice ID Selection
let voiceId = charConfig.voice_id;
if (!voiceId || typeof voiceId !== 'string' || voiceId.trim().length < 5) {
    console.log(`Invalid or missing voice_id '${voiceId}', falling back to Rachel.`);
    voiceId = '21m00Tcm4TlvDq8ikWAM'; // Rachel
} else {
    voiceId = voiceId.trim();
}

return [{
  json: {
    asset_id: assetData.id,
    script_id: scriptData.id,
    content_idea_id: parseData.content_idea_id,
    text_for_tts: text,
    voice_id: voiceId,
    avatar_id: charConfig.avatar_id || 'Kristin_pubblic_3_20240108',
    avatar_type: charConfig.avatar_type || 'avatar',
    config_image_url: charConfig.image_url,
    broll_paths: brollItems.filter(i => i.json.fileName).map((i, idx) => 
      `/home/node/assets/videos/${scriptData.id}_scene${idx}.mp4`
    )
  }
}];"""

    node['parameters']['jsCode'] = new_code
    print("Updated 'Prepare TTS Text' logic.")

else:
    print("Node not found!")
    exit(1)

with open(file_path, 'w') as f:
    json.dump(data, f, indent=2)

print("Successfully wrote updated workflow to file.")
