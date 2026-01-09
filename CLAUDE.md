# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 🚀 START HERE - END-TO-END PIPELINE TEST PROCEDURE

**When starting a new session, follow these steps to test the pipeline:**

### Step 1: Trigger Test Pipeline
```bash
curl -X POST "http://100.83.153.43:5678/webhook/trigger-pipeline" \
  -H "Content-Type: application/json" \
  -d '{"content_idea_id": 141}'
```
This uses content idea 141 which has existing test assets (script 22).

### Step 2: Monitor Execution
```
Use MCP tool:
mcp__n8n__n8n_executions action=list workflowId=LNpBI12tR5p3NX3g limit=5
```
Look for the latest execution and check its status.

### Step 3: Debug Errors (if any)
```
Use MCP tool:
mcp__n8n__n8n_executions action=get id=<execution_id> mode=error
```
This shows:
- `primaryError`: The error message and which node failed
- `upstreamContext`: Data from previous nodes
- `executionPath`: Which nodes ran successfully

### Step 4: Fix Issues and Re-test
Common fixes:
- **Code node sandbox error** ("Can't use .all()"): Change node mode to `runOnceForAllItems`
- **Missing file errors**: Check asset paths in `/home/node/.n8n-files/assets/`
- **Webhook errors**: Ensure `Respond Immediately` node is connected after `Webhook`

### Test Data Available
- Content Idea ID: **141**
- Script ID: **22**
- Voice file: `/home/node/.n8n-files/assets/audio/22_voice.mp3`
- Avatar video: `/home/node/.n8n-files/assets/avatar/22_avatar.mp4`
- Workflow ID: **LNpBI12tR5p3NX3g**

---

## Project Goal: Automated AI Video Content Pipeline

This n8n instance automates end-to-end viral video content creation and distribution:

### Pipeline Stages

1. **Content Scraping** - Apify to pull trending/viral ideas from Reddit, TikTok, YouTube, X (Twitter), etc.
2. **Script Generation** - OpenRouter (Grok 4.1) to create unique scripts from scraped ideas
3. **Voiceover Creation** - ElevenLabs for AI voice cloning or custom recordings
4. **AI Avatar Videos** - HeyGen (or similar) for lip-synced avatar videos over background content
5. **Captions/Overlays** - Whisper AI for auto-generated subtitles and text overlays
6. **Editing & Management** - PostgreSQL + React UI for asset management, FFmpeg for video processing
7. **Multi-Platform Publishing** - Blotato to auto-post to X, YouTube, TikTok, Instagram, LinkedIn, Threads, Facebook, Pinterest

### Required Integrations

| Service | Purpose |
|---------|---------|
| Apify | Web scraping (Reddit, TikTok, YouTube, X) |
| OpenRouter | LLM gateway - using Grok 4.1 for script gen, FFmpeg commands |
| OpenAI | Whisper API for transcription only |
| ElevenLabs | AI voiceover/voice cloning |
| HeyGen | AI avatar video generation |
| FFmpeg | Video processing/editing (self-hosted in container) |
| PostgreSQL | Database for content pipeline management |
| FastAPI | Backend API for content pipeline |
| React | Frontend dashboard for content management |
| Blotato | Multi-platform social media posting |

### OpenRouter Configuration

```
Base URL: https://openrouter.ai/api/v1
Model: x-ai/grok-4.1-fast
```

**HTTP Request Node for LLM calls:**
```json
{
  "method": "POST",
  "url": "https://openrouter.ai/api/v1/chat/completions",
  "headers": {
    "Authorization": "Bearer {{$credentials.openRouter.apiKey}}",
    "HTTP-Referer": "http://100.83.153.43:5678",
    "X-Title": "n8n-video-pipeline"
  },
  "body": {
    "model": "x-ai/grok-4.1-fast",
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."}
    ]
  }
}
```

## n8n Setup

n8n is running via Docker on DGX Spark. Access the web UI at **http://100.83.153.43:5678**

### Docker Commands
```bash
# Start n8n
docker compose up -d

# Stop n8n
docker compose down

# View logs
docker compose logs -f n8n

# Restart n8n
docker compose restart

# Update to latest version
docker compose pull && docker compose up -d
```

## Workflow Management

### Workflow File Format
n8n workflows are JSON files with this structure:
```json
{
  "name": "Workflow Name",
  "nodes": [...],
  "connections": {...},
  "settings": {...}
}
```

### Import/Export Workflows
- **Export from UI**: Workflow menu → Download
- **Import from UI**: Workflows → Import from File
- **CLI export**: `docker exec n8n n8n export:workflow --all --output=/home/node/workflows/`
- **CLI import**: `docker exec n8n n8n import:workflow --input=/home/node/workflows/workflow.json`

### Working with Workflow JSON

**Node structure:**
```json
{
  "id": "uuid",
  "name": "Node Display Name",
  "type": "n8n-nodes-base.nodeName",
  "position": [x, y],
  "parameters": {...}
}
```

**Connection structure:**
```json
{
  "Source Node Name": {
    "main": [[{"node": "Target Node Name", "type": "main", "index": 0}]]
  }
}
```

### Common Node Types
- `n8n-nodes-base.manualTrigger` - Manual execution trigger
- `n8n-nodes-base.scheduleTrigger` - Cron-based scheduling
- `n8n-nodes-base.webhook` - HTTP webhook trigger
- `n8n-nodes-base.httpRequest` - Make HTTP requests
- `n8n-nodes-base.code` - Custom JavaScript/Python code
- `n8n-nodes-base.if` - Conditional branching
- `n8n-nodes-base.merge` - Combine data from multiple branches
- `n8n-nodes-base.set` - Set/modify data fields

### Expressions
n8n uses expressions with `{{ }}` syntax:
- `{{ $json.fieldName }}` - Access current item's field
- `{{ $('Node Name').item.json.field }}` - Access another node's output
- `{{ $input.all() }}` - Get all input items
- `{{ $now }}` - Current datetime
- `{{ $env.VAR_NAME }}` - Environment variable

## Directory Structure
```
/home/canderson/n8n/
├── docker-compose.yml    # Docker configuration (n8n, postgres, backend, frontend)
├── Dockerfile            # Custom n8n image with FFmpeg
├── workflows/            # Store exported workflow JSON files
├── assets/               # Fonts, logos, generated media
│   ├── fonts/
│   └── logos/
├── backend/              # FastAPI backend
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py           # FastAPI app with all routes
│   ├── models.py         # SQLAlchemy models
│   ├── schemas.py        # Pydantic schemas
│   └── init.sql          # PostgreSQL schema
├── frontend/             # React frontend
│   ├── Dockerfile
│   ├── package.json
│   └── src/
│       ├── App.js
│       ├── api.js        # API client
│       └── pages/        # Dashboard, ContentIdeas, Scripts, Assets, Published
├── CLAUDE.md
├── MASTERPLAN.md
└── readme.md
```

## Environment Variables

Set in docker-compose.yml under `environment:`:
- `N8N_HOST` - Hostname for n8n
- `WEBHOOK_URL` - Base URL for webhooks
- `GENERIC_TIMEZONE` - Timezone setting
- `N8N_ENCRYPTION_KEY` - For credential encryption (add for production)

## API Access

### n8n REST API
Available at `http://100.83.153.43:5678/api/v1/`
- Enable API access in Settings → API
- Generate API key for authentication

### Content Pipeline API (FastAPI)
Available at `http://100.83.153.43:8000`

**Endpoints:**
- `GET /api/content-ideas` - List content ideas
- `POST /api/content-ideas` - Create content idea
- `PATCH /api/content-ideas/{id}` - Update content idea
- `POST /api/content-ideas/bulk-approve` - Bulk approve ideas
- `GET /api/scripts` - List scripts
- `PATCH /api/scripts/{id}` - Update script
- `GET /api/assets` - List assets
- `GET /api/published` - List published content
- `GET /api/analytics` - Get analytics data
- `GET /api/pipeline/overview` - Full pipeline status
- `GET /api/pipeline/stats` - Pipeline statistics

### Frontend Dashboard
Available at `http://100.83.153.43:3000`

**Pages:**
- `/` - Dashboard with stats and recent activity
- `/ideas` - Content ideas management (approve/reject, bulk actions)
- `/scripts` - Script viewing and editing
- `/assets` - Asset tracking (voiceovers, videos, captions)
- `/published` - Published content with platform links

## Database Schema

PostgreSQL database with the following tables:
- `content_ideas` - Scraped content with pillar, viral score, status
- `scripts` - Generated scripts with platform-specific captions
- `assets` - File paths for voiceovers, videos, captions
- `published` - Platform URLs and IDs for published content
- `analytics` - Views, likes, comments, shares per platform
- `pipeline_runs` - Workflow execution tracking

**View:** `pipeline_overview` - Joins all tables for full status view

## Starting the Full Stack

```bash
# Build and start all services
docker compose up -d --build

# Check service status
docker compose ps

# View logs
docker compose logs -f backend
docker compose logs -f frontend

# Access services
# n8n: http://100.83.153.43:5678
# Backend API: http://100.83.153.43:8000
# Frontend: http://100.83.153.43:3000
```

---

## END-TO-END PIPELINE TESTING

### Main Workflow Info
- **Workflow ID:** `LNpBI12tR5p3NX3g`
- **Workflow Name:** "AI ContentGenerator"
- **Webhook Path:** `trigger-pipeline` (POST)

### MCP Tools for n8n
Use these MCP tools to manage the workflow:
- `mcp__n8n__n8n_get_workflow` - Get workflow details (modes: minimal, structure, full)
- `mcp__n8n__n8n_update_partial_workflow` - Update nodes/connections incrementally
- `mcp__n8n__n8n_executions` - List/get execution details (action: list, get, delete)
- `mcp__n8n__n8n_validate_workflow` - Validate workflow by ID

### Testing the Pipeline

**1. Trigger pipeline with a specific content idea:**
```bash
curl -X POST "http://100.83.153.43:5678/webhook/trigger-pipeline" \
  -H "Content-Type: application/json" \
  -d '{"content_idea_id": 141}'
```

**2. Check execution status:**
```bash
# Using MCP tool:
mcp__n8n__n8n_executions action=list workflowId=LNpBI12tR5p3NX3g limit=5

# Or get specific execution details with error info:
mcp__n8n__n8n_executions action=get id=<execution_id> mode=error
```

**3. Activate/deactivate workflow via API:**
```bash
# Activate
curl -X POST "http://100.83.153.43:5678/api/v1/workflows/LNpBI12tR5p3NX3g/activate" \
  -H "X-N8N-API-KEY: $N8N_API_KEY"

# Deactivate
curl -X POST "http://100.83.153.43:5678/api/v1/workflows/LNpBI12tR5p3NX3g/deactivate" \
  -H "X-N8N-API-KEY: $N8N_API_KEY"
```

### Common Issues & Fixes

**1. Code Node Sandbox Errors - "Can't use .all() here"**
- The node mode is `runOnceForEachItem` but code uses `$input.all()`
- Fix: Change mode to `runOnceForAllItems` or rewrite code to not use `.all()`
```javascript
// To update node mode via MCP:
mcp__n8n__n8n_update_partial_workflow id=LNpBI12tR5p3NX3g operations=[
  {"type": "updateNode", "name": "Node Name", "updates": {"parameters": {"mode": "runOnceForAllItems"}}}
]
```

**2. Webhook "No Respond to Webhook node found"**
- The webhook has `responseMode: "responseNode"` but no branch reaches a Respond node
- Fix: Ensure `Respond Immediately` or `Respond to Webhook` is in the execution path
- Current fix: Flow is `Webhook` → `Respond Immediately` → `Specific ID?` → ...

**3. Missing Nodes After Updates**
- Always backup before major changes
- Compare node lists: `jq '.nodes[].name' workflow.json`
- Backups are in: `/home/canderson/n8n/workflows/COMPLETE_PIPELINE_BACKUP_*.json`

### HeyGen Bypass (for testing without API credits)
The workflow has bypass nodes to skip HeyGen when avatar video exists:
- `Avatar Already Exists?` - Checks if `/home/node/.n8n-files/assets/avatar/{script_id}_avatar.mp4` exists
- `Skip HeyGen?` - IF node branches based on existence
- `Use Existing Avatar` - Mocks HeyGen response data for testing

To create a test avatar video for testing:
```bash
docker exec -it n8n bash -c 'ffmpeg -f lavfi -i testsrc=duration=10:size=1920x1080:rate=30 \
  -f lavfi -i sine=frequency=440:duration=10 \
  -c:v libx264 -c:a aac -pix_fmt yuv420p \
  /home/node/.n8n-files/assets/avatar/22_avatar.mp4'
```

### Pipeline Flow (nodes in order)
1. `Webhook` → `Respond Immediately` → `Specific ID?`
2. `Get Specific Idea` or `Get Approved Idea` → `Merge` → `Normalize Idea`
3. `Has Content?` → `Update Status` → `Generate Script (Grok)` → `Parse Script` → `Save Script`
4. `Create Asset Record` → `Download Source Video` → `Verify Download`
5. `Get Character Config` → `Prepare TTS Text` → `Check Voice Exists` → `Voice Exists?`
6. `Generate Voice (ElevenLabs)` → `Save Voice` → `Get Duration` → `Update Asset Voice`
7. `Read Audio File` → `Avatar Already Exists?` → `Skip HeyGen?`
8. (If skip) `Use Existing Avatar` → `Build FFmpeg Command`
9. (If not skip) `Upload HeyGen Audio` → `Is Talking Photo?` → `Prepare HeyGen Data` → `Create HeyGen Video` → polling loop → `Save Avatar`
10. `Build FFmpeg Command` → `Run FFmpeg` → `Success?`
11. `Prepare Whisper` → `Read Audio for Whisper` → `Whisper Transcribe` → `Parse Whisper Response` → `Save SRT`
12. `Build Caption Cmd` → `Burn Captions` → `Caption Success?`
13. `Prepare Publish Data` → `Read Final Video` → `GCS Upload` → `Format GCS URL`
14. `Publish (Blotato)` → `Parse Response` → `Save Publish Record` → `Update Status Published`

### Asset File Paths
All assets are stored in `/home/node/.n8n-files/assets/`:
- Audio: `/home/node/.n8n-files/assets/audio/{script_id}_voice.mp3`
- Avatar: `/home/node/.n8n-files/assets/avatar/{script_id}_avatar.mp4`
- Combined: `/home/node/.n8n-files/assets/combined/{script_id}_combined.mp4`
- Captions: `/home/node/.n8n-files/assets/captions/{script_id}.srt`
- Final: `/home/node/.n8n-files/assets/final/{script_id}_final.mp4`
- Videos (source): `/home/node/.n8n-files/assets/videos/{script_id}_source.mp4`

### GCS Upload Configuration
- **Credential ID:** `LXMqHFKoSGvfqCF5` (Google Service Account account 2)
- **Bucket:** `content-pipeline-assets`
- **ACL:** `publicRead`
- **Node type:** `n8n-nodes-base.googleCloudStorage`

### Debugging Executions
```bash
# Get detailed error info for failed execution
mcp__n8n__n8n_executions action=get id=<exec_id> mode=error

# This returns:
# - primaryError: message, nodeName, stackTrace
# - upstreamContext: previous node data
# - executionPath: which nodes ran and their status
```

### Workflow Backup Commands
```bash
# Create timestamped backup
cp workflows/COMPLETE_PIPELINE.json "workflows/COMPLETE_PIPELINE_BACKUP_$(date +%Y%m%d_%H%M%S).json"

# List backups
ls -la workflows/COMPLETE_PIPELINE_BACKUP_*.json

# Compare node counts
jq '.nodes | length' workflows/COMPLETE_PIPELINE.json
jq '.nodes | length' workflows/COMPLETE_PIPELINE_BACKUP_*.json
```

---

## CURRENT STATUS & PENDING FIXES

**Last Updated:** 2026-01-08

### Known Issue to Fix
The `Normalize Idea` node has a bug - uses `$input.all()` with mode `runOnceForEachItem`.
Fix needed:
```javascript
mcp__n8n__n8n_update_partial_workflow
  id=LNpBI12tR5p3NX3g
  operations=[{"type": "updateNode", "name": "Normalize Idea", "updates": {"parameters": {"mode": "runOnceForAllItems"}}}]
```

### What Has Been Completed
1. ✅ Restored HeyGen bypass nodes (Avatar Already Exists?, Skip HeyGen?, Use Existing Avatar)
2. ✅ GCS Upload configured with credential ID `LXMqHFKoSGvfqCF5`
3. ✅ Webhook flow fixed: Webhook → Respond Immediately → Specific ID?
4. ✅ Read Final Video returns binary data in `data` property
5. ✅ Test avatar video exists for script 22

### Test Content Idea
- **ID:** 141
- **Script ID:** 22
- Has voice file at `/home/node/.n8n-files/assets/audio/22_voice.mp3`
- Has test avatar at `/home/node/.n8n-files/assets/avatar/22_avatar.mp4`

### Next Steps After Fix
1. Fix `Normalize Idea` node mode
2. Trigger pipeline: `curl -X POST "http://100.83.153.43:5678/webhook/trigger-pipeline" -H "Content-Type: application/json" -d '{"content_idea_id": 141}'`
3. Monitor executions for errors
4. If errors, use `mcp__n8n__n8n_executions action=get id=<exec_id> mode=error` to debug
