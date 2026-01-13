
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

# 1. REMOVE OLD NODES
ids_to_remove = ["if-heygen-complete", "check-heygen-failed"]
nodes = [n for n in nodes if n.get('id') not in ids_to_remove]
# Clean up their connections section if they exist as keys
for uid in ids_to_remove:
    # Need to find the name to delete from connections
    # This is tricky because connection keys are names, but we filtered by ID.
    # However, we know their names from previous greps: "Completed?" and "Failed?" (or similar).
    # Let's clean up by Name as well just to be safe.
    pass

# We can also filter connections by ensuring the Source Name still exists in the nodes list.
current_node_names = [n['name'] for n in nodes]
new_connections = {k: v for k, v in connections.items() if k in current_node_names}

# 2. ADD NEW SWITCH NODE
switch_node = {
  "parameters": {
    "dataType": "string",
    "value1": "={{ $json.data.status }}",
    "rules": {
      "rules": [
        {
          "value2": "completed",
          "output": 0
        },
        {
          "value2": "processing",
          "output": 1
        },
        {
          "value2": "pending",
          "output": 1
        }
      ]
    },
    "fallbackOutput": 2
  },
  "id": "route-status-switch",
  "name": "Route Status",
  "type": "n8n-nodes-base.switch",
  "typeVersion": 3,
  "position": [ 1160, 1760 ]
}
nodes.append(switch_node)

# 3. UPDATE CONNECTIONS
# Check Status -> Route Status
if "Check Status" in new_connections:
    new_connections["Check Status"] = {
        "main": [
            [
                {
                    "node": "Route Status",
                    "type": "main",
                    "index": 0
                }
            ]
        ]
    }
else:
    print("WARNING: 'Check Status' node connection block not found/updated.")

# Route Status -> [Download, Increment, Stop]
# We assume "Download Avatar", "Increment Retry", "Stop on Failure" exist.
new_connections["Route Status"] = {
  "main": [
    [
      {
        "node": "Download Avatar",
        "type": "main",
        "index": 0
      }
    ],
    [
      {
        "node": "Increment Retry",
        "type": "main",
        "index": 0
      }
    ],
    [
      {
        "node": "Stop on Failure",
        "type": "main",
        "index": 0
      }
    ]
  ]
}

data['nodes'] = nodes
data['connections'] = new_connections

with open(file_path, 'w') as f:
    json.dump(data, f, indent=2)

print("SUCCESS: Status logic refactored to Switch node.")
