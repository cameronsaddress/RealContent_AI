
import json

# Define the new JS Code for the Upload Node
JS_CODE = r"""const { execSync } = require('child_process');
const fs = require('fs');

const ttsData = $('Prepare TTS Text').first().json;
const scriptId = ttsData.script_id;
const filePath = `/home/node/assets/audio/${scriptId}_voice.mp3`;
const apiKey = $env.HEYGEN_API_KEY;

if (!apiKey) throw new Error('HEYGEN_API_KEY environment variable is missing');

if (!fs.existsSync(filePath)) {
    throw new Error(`File not found: ${filePath}`);
}

try {
    // Universal HTTP Fallback: Use WGET (Present in Alpine/BusyBox) 
    // to bypass Node/N8N header quirks and missing Curl.
    // -O - : Output to stdout
    // --post-file : Raw binary upload
    // --header : Explicit headers
    const command = `wget -O - \
      --header="X-Api-Key: ${apiKey}" \
      --header="Content-Type: audio/mpeg" \
      --post-file="${filePath}" \
      "https://upload.heygen.com/v1/asset"`;

    const stdout = execSync(command).toString();
    
    // Parse Response
    try {
        const responseJson = JSON.parse(stdout);
        
        if (responseJson.error) {
             throw new Error(`API Error: ${JSON.stringify(responseJson.error)}`);
        }
        
        if (!responseJson.data || !responseJson.data.id) {
             throw new Error(`Invalid Response: ${stdout}`);
        }

        return [{
            json: {
                ...ttsData,
                heygen_response: responseJson,
                data: responseJson.data, 
                audio_asset_id: responseJson.data.id
            }
        }];

    } catch (e) {
        throw new Error(`Failed to parse WGET response: ${e.message} | Raw: ${stdout}`);
    }

} catch (error) {
    throw new Error(`WGET Upload Failed: ${error.message}`);
}"""

file_path = '/home/canderson/n8n/workflows/COMPLETE_PIPELINE.json'

with open(file_path, 'r') as f:
    workflow = json.load(f)

nodes = workflow['nodes']
connections = workflow['connections']

# 1. REMOVE 'Read Audio for Upload'
# 2. MODIFY 'Upload HeyGen Audio' to be a Code Node

new_nodes = []
for node in nodes:
    if node['name'] == 'Read Audio for Upload':
        continue # Remove it
    
    if node['name'] == 'Upload HeyGen Audio':
        # Convert to Code Node
        node['type'] = 'n8n-nodes-base.code'
        node['typeVersion'] = 2
        # Clean obsolete parameters
        node['parameters'] = {
            "jsCode": JS_CODE
        }
        # Remove credentials link if present (Code node uses env)
        if 'credentials' in node:
            del node['credentials']
            
    new_nodes.append(node)

workflow['nodes'] = new_nodes

# 3. FIX CONNECTIONS
# - Remove connections FROM 'Read Audio for Upload'
# - Redirect 'Update Asset Voice' (which went to Read Audio) to 'Upload HeyGen Audio'

params_to_remove = []

if 'Read Audio for Upload' in connections:
     del connections['Read Audio for Upload']

# Redirect Update Asset Voice
if 'Update Asset Voice' in connections:
    # It has main -> [ [ {node: 'Read Audio for Upload', ...} ] ]
    outputs = connections['Update Asset Voice']['main']
    for output_channel in outputs:
        for conn in output_channel:
            if conn['node'] == 'Read Audio for Upload':
                conn['node'] = 'Upload HeyGen Audio'

workflow['connections'] = connections

with open(file_path, 'w') as f:
    json.dump(workflow, f, indent=2)

print("Refactoring complete.")
