import json

# IDs retrieved from DB/Import
ids = {
    "workflowId": "LNpBI12tR5p3NX3g",
    "versionId": "3cb5b5d7-7b35-47a3-9df0-956fc93bd453"
}

try:
    # Use the temp_import because that matches what was imported into the Entity
    with open('/home/canderson/n8n/workflows/temp_import.json', 'r') as f:
        data = json.load(f)
    
    nodes = json.dumps(data.get('nodes', []))
    connections = json.dumps(data.get('connections', {}))
    # Escape single quotes for SQL
    nodes_sql = nodes.replace("'", "''")
    connections_sql = connections.replace("'", "''")
    
    # 1. Insert History if missing (ON CONFLICT DO NOTHING to avoid dupes)
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
    ) ON CONFLICT ("versionId") DO UPDATE 
      SET nodes = EXCLUDED.nodes, connections = EXCLUDED.connections;
    """
    
    # 2. Force Sync Workflow Entity
    update_sql = f"""
    UPDATE workflow_entity 
    SET "activeVersionId" = '{ids['versionId']}',
        "nodes" = '{nodes_sql}',
        "connections" = '{connections_sql}',
        "updatedAt" = NOW()
    WHERE id = '{ids['workflowId']}';
    """
    
    print(sql)
    print(update_sql)

except Exception as e:
    print(f"Error: {e}")
