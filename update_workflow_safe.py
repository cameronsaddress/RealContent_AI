from n8n_api_client import N8NClient
import json

# Configuration
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhMDc3M2M4MS1hZTg4LTQ5ZDAtYjQ5MC02YmQyN2U4Yzc0MDUiLCJpc3MiOiJuOG4iLCJhdWQiOiJwdWJsaWMtYXBpIiwiaWF0IjoxNzY3ODU3MjQyfQ.dCgFdvQhCf9dvHuIuRGGzibqVYzet8pb1fBQNvcZ3bo"
WORKFLOW_ID = "LNpBI12tR5p3NX3g"

def apply_fixes(workflow):
    """Modify the workflow object in memory."""
    nodes = workflow['nodes']
    connections = workflow['connections']
    
    print("Applying fixes...")

    # 1. FIX: "Normalize Idea" -> Run Once for All Items
    for node in nodes:
        if node['name'] == 'Normalize Idea':
            print("  - Updating 'Normalize Idea' to runOnceForAllItems")
            if 'parameters' not in node: node['parameters'] = {}
            node['parameters']['mode'] = 'runOnceForAllItems'

    # 2. FIX: Delete Obsolete Nodes
    nodes_to_delete = [
        "Prepare B-Roll Searches", 
        "Search Pexels", 
        "Select Videos", 
        "Video Found?", 
        "Skip B-Roll", 
        "Merge B-Roll",
        "Download B-Roll",
        "Download Source Video" 
    ]
    
    print(f"  - Deleting nodes: {nodes_to_delete}")
    workflow['nodes'] = [n for n in nodes if n['name'] not in nodes_to_delete]
    
    # Clean up connections for deleted nodes
    new_connections = {}
    for source_node, outputs in connections.items():
        if source_node not in nodes_to_delete:
            new_outputs = {}
            for output_name, connection_multiplex in outputs.items():
                new_multiplex = []
                for connection_path in connection_multiplex:
                    # connection_path is a list of connection objects
                    new_path = [
                        conn for conn in connection_path
                        if conn['node'] not in nodes_to_delete
                    ]
                    if new_path:
                        new_multiplex.append(new_path)
                
                if new_multiplex:
                    new_outputs[output_name] = new_multiplex
            
            if new_outputs:
                new_connections[source_node] = new_outputs
    
    workflow['connections'] = new_connections

    # 3. FIX: Add "Download Universal Video" (Execute Command / yt-dlp)
    # Find position of 'Create Asset Record' to place new node nearby
    ref_node = next((n for n in workflow['nodes'] if n['name'] == 'Create Asset Record'), None)
    pos = ref_node['position'] if ref_node else [1000, 1000]
    
    new_node = {
        "parameters": {
            "command": "yt-dlp -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best' -o '/home/node/.n8n-files/assets/videos/{{$('Save Script').first().json.id}}_source.%(ext)s' --no-playlist {{$('Normalize Idea').first().json.source_url}}"
        },
        "id": "download-universal-video",
        "name": "Download Universal Video",
        "type": "n8n-nodes-base.executeCommand",
        "typeVersion": 1,
        "position": [pos[0] + 200, pos[1]] 
    }
    workflow['nodes'].append(new_node)
    print("  - Added 'Download Universal Video' node")

    # 4. FIX: Update "Save B-Roll" Logic
    save_broll_code = """
const fs = require('fs');
const scriptId = $('Save Script').first().json.id;
const filePath = `/home/node/.n8n-files/assets/videos/${scriptId}_source.mp4`;

// Simple verification that file exists
if (fs.existsSync(filePath)) {
    return [{
        json: {
            saved_path: filePath,
            script_id: scriptId
        }
    }];
} else {
    // If exact match failed, check if yt-dlp saved it with a different extension or name
    // (Optional enhancement: list dir and find matching file)
    throw new Error(`Video download failed. File not found at ${filePath}`);
}
"""
    # Ensure Save B-Roll exists
    save_broll_node = next((n for n in workflow['nodes'] if n['name'] == 'Save B-Roll'), None)
    
    if save_broll_node:
        print("  - Updating 'Save B-Roll' code")
        save_broll_node['parameters']['jsCode'] = save_broll_code
        save_broll_node['parameters']['mode'] = 'runOnceForEachItem' 
        
        # Connect Download -> Save B-Roll
        if 'Download Universal Video' not in workflow['connections']:
            workflow['connections']['Download Universal Video'] = {}
        workflow['connections']['Download Universal Video']['main'] = [
            [{"node": "Save B-Roll", "type": "main", "index": 0}]
        ]
        
        # Connect Create Asset Record -> Download Universal Video
        # First ensure Create Asset Record exists
        if 'Create Asset Record' in workflow['connections']:
             workflow['connections']['Create Asset Record']['main'] = [
                [{"node": "Download Universal Video", "type": "main", "index": 0}]
            ]
        
    # 5. FIX: Update "Prepare TTS Text" to usage single file path
    prepare_tts_node = next((n for n in workflow['nodes'] if n['name'] == 'Prepare TTS Text'), None)
    if prepare_tts_node:
        print("  - Updating 'Prepare TTS Text' code")
        new_tts_code = """
// Get character configuration from the previous node (HTTP Request)
const charConfig = $input.first().json || {};

// Get script, asset, and parse data from upstream
const scriptData = $('Save Script').first().json;
const assetData = $('Create Asset Record').first().json;
const parseData = $('Parse Script').first().json;

// --- NEW LOGIC START ---
// Get the single source video path we just saved
const brollNode = $('Save B-Roll').first();
// Default to an empty array if something went wrong, but wrap the path in an array
const brollPath = brollNode && brollNode.json.saved_path ? [ brollNode.json.saved_path ] : [];
// --- NEW LOGIC END ---

// Clean script for TTS (remove pauses, scene directions, extra spaces)
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
    broll_paths: brollPath 
  }
}];
"""
        prepare_tts_node['parameters']['jsCode'] = new_tts_code
        
        # Connect Save B-Roll -> Get Character Config (Filling the gap from deleted Merge B-Roll)
        # Note: Prepare TTS Text is seemingly connected to Get Character Config in existing flow, 
        # but Save B-Roll was connected to Merge B-Roll which was connected to Get Character Config.
        # So we need Connect Save B-Roll -> Get Character Config
        
        if 'Save B-Roll' not in workflow['connections']:
            workflow['connections']['Save B-Roll'] = {}
        
        workflow['connections']['Save B-Roll']['main'] = [
             [{"node": "Get Character Config", "type": "main", "index": 0}]
        ]

    return workflow

if __name__ == "__main__":
    client = N8NClient(api_key=API_KEY)
    
    print(f"Fetching workflow {WORKFLOW_ID}...")
    try:
        wf = client.get_workflow(WORKFLOW_ID)
        
        # apply fixes
        fixed_wf = apply_fixes(wf)
        
        # update
        print("Pushing updates...")
        client.update_workflow(WORKFLOW_ID, fixed_wf)
        print("Success! Workflow updated.")
        
    except Exception as e:
        print(f"Failed: {e}")
