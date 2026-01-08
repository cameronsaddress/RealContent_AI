
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

found = False
for node in nodes:
    if node.get('name') == "Failed?" or node.get('id') == "check-heygen-failed":
        print("Found 'Failed?' node. Updating condition to 'Contains'...")
        # Update condition to String Contains 'fail'
        node['parameters']['conditions']['string'][0]['operation'] = 'contains'
        node['parameters']['conditions']['string'][0]['value2'] = 'fail'
        found = True
        break

if found:
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    print("SUCCESS: Updated 'Failed?' node to robust condition.")
else:
    print("Error: 'Failed?' node not found. Did the previous patch run?")
    sys.exit(1)
