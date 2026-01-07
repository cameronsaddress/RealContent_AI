#!/bin/bash
# Syncs the local COMPLETE_PIPELINE.json to the running n8n instance

echo "Syncing COMPLETE_PIPELINE.json to n8n..."
docker exec -u node n8n n8n import:workflow --input=/home/node/workflows/COMPLETE_PIPELINE.json

echo "---------------------------------------------------"
echo "✅ Sync Complete"
echo "👉 Please REFRESH your n8n browser tab to see changes."
