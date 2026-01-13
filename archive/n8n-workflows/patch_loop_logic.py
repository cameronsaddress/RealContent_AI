
import json
import sys

file_path = '/home/canderson/n8n/workflows/COMPLETE_PIPELINE.json'

try:
    with open(file_path, 'r') as f:
        data = json.load(f)
except Exception as e:
    print(f"Error reading file: {e}")
    sys.exit(1)

nodes = data.get('nodes', [])
connections = data.get('connections', {})

# 1. FIND 'Completed?' NODE
completed_node_id = 'if-heygen-complete'
print(f"Looking for node ID: {completed_node_id}")

retry_node_name = None
retry_connection_obj = None

# Check connections to find what follows 'Completed?' on False path (index 1)
if 'Completed?' in connections and 'main' in connections['Completed?']:
    outputs = connections['Completed?']['main']
    if len(outputs) > 1:
        retry_connection_obj = outputs[1][0] # Assuming single connection
        retry_node_name = retry_connection_obj['node']
        print(f"Found Retry Node Name: {retry_node_name}")

if not retry_node_name:
    print("Error: Could not determine retry node from 'Completed?' connections.")
    # Fallback to visually known 'Wait & Retry' or 'Increment' if simpler logic failed?
    # But relying on connections is safer.
    sys.exit(1)

# 2. CREATE NEW NODE 'Failed?'
failed_node_id = 'check-heygen-failed'
failed_node = {
    "parameters": {
        "conditions": {
            "string": [
                {
                    "value1": "={{ $json.data.status }}",
                    "operation": "equal",
                    "value2": "failed"
                }
            ]
        }
    },
    "id": failed_node_id,
    "name": "Failed?",
    "type": "n8n-nodes-base.if",
    "typeVersion": 2,
    "position": [
        1250, # Rough position between Completed (1160) and Retry (unknown)
        1900 
    ]
}

# 3. CREATE NEW NODE 'Stop Error'
stop_node_id = 'stop-heygen-error'
stop_node = {
    "parameters": {
        "errorMessage": "HeyGen Generation Failed",
        "errorDescription": "={{ $json.data.error || $json.data.detail || 'Unknown Error' }}"
    },
    "id": stop_node_id,
    "name": "Stop on Failure",
    "type": "n8n-nodes-base.stopAndError", # Or just stop. StopAndError is clearer.
    "typeVersion": 1,
    "position": [
        1450,
        1900
    ]
}

# Add nodes
nodes.append(failed_node)
nodes.append(stop_node)

# 4. REWIRE CONNECTIONS
# A. 'Completed?' (False) -> 'Failed?'
connections['Completed?']['main'][1] = [
    {
        "node": "Failed?",
        "type": "main",
        "index": 0
    }
]

# B. 'Failed?' (True) -> 'Stop on Failure'
connections['Failed?'] = {
    "main": [
        [
            {
                "node": "Stop on Failure",
                "type": "main",
                "index": 0
            }
        ],
        # C. 'Failed?' (False) -> Original Retry Node
        [
            {
                "node": retry_node_name,
                "type": "main",
                "index": 0
            }
        ]
    ]
}

print("Rewired connections successfully.")

# Save
with open(file_path, 'w') as f:
    json.dump(data, f, indent=2)

print("SUCCESS: Patch applied.")
