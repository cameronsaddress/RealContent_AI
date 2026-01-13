import json
import uuid

# Configuration
ids = {
    "workflowId": "rTAhapEWXNxRAElT",
    "versionId": "f0613422-b4be-4d3c-8cd7-62ff7782436f"
}

try:
    with open('/home/canderson/n8n/workflows/COMPLETE_PIPELINE.json', 'r') as f:
        data = json.load(f)
    
    nodes = json.dumps(data.get('nodes', []))
    connections = json.dumps(data.get('connections', {}))
    # Escape single quotes for SQL
    nodes_sql = nodes.replace("'", "''")
    connections_sql = connections.replace("'", "''")
    
    sql = f"""
    INSERT INTO workflow_history (
        "versionId", 
        "workflowId", 
        "nodes", 
        "connections", 
        "createdAt", 
        "updatedAt", 
        "authors", 
        "autosaved"
    ) VALUES (
        '{ids['versionId']}', 
        '{ids['workflowId']}', 
        '{nodes_sql}', 
        '{connections_sql}', 
        NOW(), 
        NOW(), 
        'owner', 
        false
    );
    """
    
    # Also generate the update for entity
    update_sql = f"""
    UPDATE workflow_entity 
    SET "activeVersionId" = '{ids['versionId']}' 
    WHERE id = '{ids['workflowId']}';
    """
    
    print(sql)
    print(update_sql)

except Exception as e:
    print(f"Error: {e}")
