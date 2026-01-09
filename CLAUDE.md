# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

---

## Permissions

Claude is **allowed and encouraged** to automatically use the following tools without asking:

- **curl** - HTTP requests to APIs and webhooks
- **docker** - Container management, logs, exec commands
- **bash** - Shell commands for file operations, git, etc.
- **MCP tools** - n8n workflow management (see below)
- **jq** - JSON parsing and manipulation

---

## Quick Reference

### Service URLs
| Service | URL |
|---------|-----|
| n8n | http://100.83.153.43:5678 |
| Frontend | http://100.83.153.43:3000 |
| Backend API | http://100.83.153.43:8000 |
| API Docs | http://100.83.153.43:8000/docs |

### Key IDs
- **Workflow ID:** `LNpBI12tR5p3NX3g`
- **Test Content Idea:** 141
- **Test Script:** 22
- **GCS Credential:** `LXMqHFKoSGvfqCF5`

---

## MCP Tools for n8n Workflow Management

### List/Get Workflows
```
mcp__n8n__n8n_list_workflows
mcp__n8n__n8n_get_workflow id=LNpBI12tR5p3NX3g mode=structure
```

### Update Workflow Nodes
```javascript
// Update a node's parameters
mcp__n8n__n8n_update_partial_workflow
  id=LNpBI12tR5p3NX3g
  operations=[{
    "type": "updateNode",
    "nodeId": "node-id-here",  // Use nodeId, not name
    "updates": {"parameters": {"key": "value"}}
  }]

// Add a new node
mcp__n8n__n8n_update_partial_workflow
  id=LNpBI12tR5p3NX3g
  operations=[{
    "type": "addNode",
    "node": {
      "id": "unique-id",
      "name": "Node Name",
      "type": "n8n-nodes-base.code",
      "typeVersion": 2,
      "position": [100, 200],
      "parameters": {}
    }
  }]

// Add a connection
mcp__n8n__n8n_update_partial_workflow
  id=LNpBI12tR5p3NX3g
  operations=[{
    "type": "addConnection",
    "from": {"node": "Source Node", "output": 0},
    "to": {"node": "Target Node", "input": 0}
  }]

// Remove a node
mcp__n8n__n8n_update_partial_workflow
  id=LNpBI12tR5p3NX3g
  operations=[{
    "type": "removeNode",
    "name": "Node Name"
  }]
```

### Monitor Executions
```
// List recent executions
mcp__n8n__n8n_executions action=list workflowId=LNpBI12tR5p3NX3g limit=5

// Get execution details with error info
mcp__n8n__n8n_executions action=get id=<execution_id> mode=error
```

### Validate Workflow
```
mcp__n8n__n8n_validate_workflow id=LNpBI12tR5p3NX3g
```

---

## End-to-End Pipeline Testing

### Step 1: Trigger Pipeline
```bash
curl -X POST "http://100.83.153.43:5678/webhook/trigger-pipeline" \
  -H "Content-Type: application/json" \
  -d '{"content_idea_id": 141}'
```

### Step 2: Monitor Execution
```
mcp__n8n__n8n_executions action=list workflowId=LNpBI12tR5p3NX3g limit=5
```

### Step 3: Debug Errors
```
mcp__n8n__n8n_executions action=get id=<execution_id> mode=error
```

The error response includes:
- `primaryError`: Error message and failing node
- `upstreamContext`: Data from previous nodes
- `executionPath`: Which nodes ran successfully

### Step 4: Common Fixes

**Code node sandbox error ("Can't use .all()"):**
```javascript
mcp__n8n__n8n_update_partial_workflow
  id=LNpBI12tR5p3NX3g
  operations=[{
    "type": "updateNode",
    "name": "Node Name",
    "updates": {"parameters": {"mode": "runOnceForAllItems"}}
  }]
```

**Missing file errors:**
Check paths in `/home/node/.n8n-files/assets/`

**Webhook errors:**
Ensure `Respond Immediately` node is connected after `Webhook`

---

## Asset File Paths

All assets stored in `/home/node/.n8n-files/assets/`:

| Type | Path Pattern |
|------|--------------|
| Audio | `audio/{script_id}_voice.mp3` |
| Avatar | `avatar/{script_id}_avatar.mp4` |
| Combined | `combined/{script_id}_combined.mp4` |
| Captions | `captions/{script_id}.srt` |
| Final | `final/{script_id}_final.mp4` |
| Source Video | `videos/{script_id}_source.mp4` |

---

## Docker Commands

```bash
# View logs
docker compose logs -f n8n
docker compose logs -f backend
docker compose logs -f frontend

# Restart services
docker compose restart n8n
docker compose restart backend
docker compose restart frontend

# Rebuild and restart
docker compose up -d --build n8n

# Access container shell
docker exec -it n8n /bin/sh
docker exec -it backend /bin/bash

# Database access
docker exec -it n8n_postgres psql -U n8n -d content_pipeline
```

---

## Pipeline Flow (Current Nodes)

1. `Webhook` -> `Respond Immediately` -> `Specific ID?`
2. `Get Specific Idea` / `Get Approved Idea` -> `Merge` -> `Normalize Idea`
3. `Has Content?` -> `Update Status` -> `Generate Script (Grok)` -> `Parse Script` -> `Save Script`
4. `Create Asset Record` -> `Download Source Video` -> `Verify Download`
5. `Get Character Config` -> `Fetch Settings` -> `Prepare TTS Text` -> `Check Voice Exists` -> `Voice Exists?`
6. `Generate Voice (ElevenLabs)` -> `Save Voice` -> `Get Duration` -> `Update Asset Voice`
7. `Read Audio File` -> `Avatar Already Exists?` -> `Skip HeyGen?`
8. (Skip path) `Use Existing Avatar` -> `Build FFmpeg Command`
9. (Generate path) `Upload HeyGen Audio` -> `Is Talking Photo?` -> `Prepare HeyGen Data` -> `Create HeyGen Video` -> polling -> `Save Avatar`
10. `Build FFmpeg Command` -> `Run FFmpeg` -> `Success?`
11. `Prepare Whisper` -> `Read Audio for Whisper` -> `Whisper Transcribe` -> `Parse Whisper Response` -> `Save SRT`
12. `Build Caption Cmd` -> `Burn Captions` -> `Caption Success?`
13. `Prepare Publish Data` -> `Read Final Video` -> `GCS Upload` -> `Format GCS URL`
14. `Publish (Blotato)` -> `Parse Response` -> `Save Publish Record` -> `Update Status Published`

---

## Backend API Endpoints

### Content Management
```bash
# List content ideas
curl http://100.83.153.43:8000/api/content-ideas

# Create content idea (direct link submission)
curl -X POST http://100.83.153.43:8000/api/content-ideas \
  -H "Content-Type: application/json" \
  -d '{"source_url": "https://x.com/...", "source_platform": "twitter", "status": "approved"}'

# Update content idea status
curl -X PATCH http://100.83.153.43:8000/api/content-ideas/141 \
  -H "Content-Type: application/json" \
  -d '{"status": "approved"}'
```

### Settings
```bash
# Get all settings (video, audio, LLM)
curl http://100.83.153.43:8000/api/settings/all

# Update video settings (greenscreen, resolution, etc.)
curl -X PUT http://100.83.153.43:8000/api/settings/video \
  -H "Content-Type: application/json" \
  -d '{"greenscreen_enabled": true, "greenscreen_color": "#00FF00"}'

# Get character config
curl http://100.83.153.43:8000/api/settings/character
```

### Pipeline Stats
```bash
curl http://100.83.153.43:8000/api/pipeline/stats
curl http://100.83.153.43:8000/api/pipeline/overview
```

---

## Test Data

- **Content Idea 141** - Has script 22, voice file, test avatar
- **Voice file:** `/home/node/.n8n-files/assets/audio/22_voice.mp3`
- **Avatar video:** `/home/node/.n8n-files/assets/avatar/22_avatar.mp4`

Create test avatar for bypass testing:
```bash
docker exec -it n8n bash -c 'ffmpeg -f lavfi -i testsrc=duration=10:size=1920x1080:rate=30 \
  -f lavfi -i sine=frequency=440:duration=10 \
  -c:v libx264 -c:a aac -pix_fmt yuv420p \
  /home/node/.n8n-files/assets/avatar/22_avatar.mp4'
```

---

## Workflow Backup

```bash
# Create timestamped backup
cp workflows/COMPLETE_PIPELINE.json "workflows/COMPLETE_PIPELINE_BACKUP_$(date +%Y%m%d_%H%M%S).json"

# List backups
ls -la workflows/COMPLETE_PIPELINE_BACKUP_*.json

# Compare node counts
jq '.nodes | length' workflows/COMPLETE_PIPELINE.json
```

---

## External API Configuration

### OpenRouter (LLM)
```
Base URL: https://openrouter.ai/api/v1
Model: x-ai/grok-4.1-fast
```

### HeyGen Avatar Types
- **Talking Photo:** Uses `avatar_id` with `avatar_style: "normal"`
- **Video Avatar:** Uses `avatar_id` with `avatar_style: "normal"` (premade)
- **Greenscreen:** Controlled by `greenscreen_enabled` and `greenscreen_color` in settings

### ElevenLabs Voice
- Voice ID configured in character settings
- Cloned voices accessible via `/api/settings/character`

---

## Project Structure

```
/home/canderson/n8n/
├── docker-compose.yml     # All services config
├── Dockerfile             # n8n with FFmpeg + yt-dlp
├── backend/               # FastAPI (models, schemas, main)
├── frontend/              # React dashboard
├── workflows/             # n8n workflow JSON exports
└── assets/                # Fonts, logos (Docker volume for media)
```

---

*Last updated: January 9, 2026*
