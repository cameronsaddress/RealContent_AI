
import json

def fix_workflow_nodes(params):
    input_file = params.get('input_file')
    output_file = params.get('output_file')

    with open(input_file, 'r') as f:
        nodes = json.load(f)

    fixed_count = 0
    for node in nodes:
        # Fix Sticky Note Colors (must be 1-7)
        if node.get('type') == 'n8n-nodes-base.stickyNote':
            params = node.get('parameters', {})
            if 'color' in params:
                try:
                    color_val = int(params['color'])
                    if color_val < 1 or color_val > 7:
                        print(f"Fixing invalid color {color_val} in node {node.get('name')}")
                        node['parameters']['color'] = 1
                        fixed_count += 1
                except:
                    node['parameters']['color'] = 1
                    fixed_count += 1

        # Fix Code Node Mode (v2 usually requires mode)
        if node.get('type') == 'n8n-nodes-base.code' and node.get('typeVersion') == 2:
            if 'mode' not in node.get('parameters', {}):
                 print(f"Adding default mode to Code node {node.get('name')}")
                 # Default is usually 'runOnceForAllItems' in some versions, but 'runOnceForEachItem' is safer?
                 # Actually, let's not touch unless we are sure.
                 # Apply default mode
                 print(f"Adding default mode to Code node {node.get('name')}")
                 if 'parameters' not in node:
                     node['parameters'] = {}
                 node['parameters']['mode'] = 'runOnceForEachItem'
                 fixed_count += 1

    # Restore "Route Status" node as Code Node Router
    for node in nodes:
        if node.get('name') == 'Route Status':
            print("Converting Route Status node to Code Node Router")
            node['type'] = 'n8n-nodes-base.code'
            node['typeVersion'] = 2
            
            # Logic to route to 3 outputs: 0=Completed, 1=Wait, 2=Fail
            js_code = """
const items = $input.all();
const completed = [];
const waiting = [];
const error = [];

for (const item of items) {
    const status = item.json.data?.status;
    if (status === 'completed') {
        completed.push(item);
    } else if (status === 'processing' || status === 'pending') {
        waiting.push(item);
    } else {
        error.push(item);
    }
}

return [completed, waiting, error];
"""
            
            node['parameters'] = {
                "jsCode": js_code.strip(),
                "mode": "runOnceForAllItems"
            }

    # Fix "Download Character Image" node options (double nesting)
    for node in nodes:
        if node.get('name') == 'Download Character Image':
            options = node.get('parameters', {}).get('options', {})
            # Check for double nesting: options.response.response
            if 'response' in options and 'response' in options['response']:
                print("Fixing double-nested response options in Download Character Image")
                # Flatten it
                inner_response = options['response']['response']
                node['parameters']['options']['response'] = inner_response
                fixed_count += 1


    # Generate SQL
    json_str = json.dumps(nodes).replace("'", "''")
    sql = f"UPDATE workflow_entity SET nodes = '{json_str}'::json WHERE id = 'LNpBI12tR5p3NX3g';"
    
    with open('/home/canderson/n8n/workflows/apply_fix.sql', 'w') as f:
        f.write(sql)
    
    print(f"Fixed {fixed_count} nodes. SQL generated in apply_fix.sql")

if __name__ == "__main__":
    fix_workflow_nodes({
        'input_file': '/home/canderson/n8n/workflows/entity_nodes_dump.json',
        'output_file': '/home/canderson/n8n/workflows/fixed_nodes.json'
    })
