# n8n Credentials Setup Guide

This document describes all credentials needed for the AI Video Content Pipeline workflows.

## Required Credentials

### 1. Apify (`apify`)
- **Type**: HTTP Header Auth
- **Header Name**: `Authorization`
- **Header Value**: `Bearer YOUR_APIFY_API_TOKEN`
- **Used by**: 01-SCRAPE workflow
- **Get API Key**: https://console.apify.com/account/integrations

### 2. OpenRouter (`openrouter`)
- **Type**: HTTP Header Auth
- **Header Name**: `Authorization`
- **Header Value**: `Bearer YOUR_OPENROUTER_API_KEY`
- **Used by**: 01-SCRAPE, 03-SCRIPT-GEN workflows
- **Get API Key**: https://openrouter.ai/keys

### 3. Pexels (`pexels`)
- **Type**: HTTP Header Auth
- **Header Name**: `Authorization`
- **Header Value**: `YOUR_PEXELS_API_KEY` (no Bearer prefix)
- **Used by**: 04-GET-VIDEO workflow
- **Get API Key**: https://www.pexels.com/api/new/

### 4. ElevenLabs (`elevenlabs`)
- **Type**: HTTP Header Auth
- **Header Name**: `xi-api-key`
- **Header Value**: `YOUR_ELEVENLABS_API_KEY`
- **Used by**: 05-CREATE-VOICE workflow
- **Get API Key**: https://elevenlabs.io/app/settings/api-keys

### 5. HeyGen (`heygen`)
- **Type**: HTTP Header Auth
- **Header Name**: `X-Api-Key`
- **Header Value**: `YOUR_HEYGEN_API_KEY`
- **Used by**: 06-CREATE-AVATAR workflow
- **Get API Key**: https://app.heygen.com/settings

### 6. OpenAI (`openai`)
- **Type**: HTTP Header Auth
- **Header Name**: `Authorization`
- **Header Value**: `Bearer YOUR_OPENAI_API_KEY`
- **Used by**: 08-CAPTION workflow (Whisper transcription)
- **Get API Key**: https://platform.openai.com/api-keys

### 7. Blotato (`blotato`)
- **Type**: HTTP Header Auth
- **Header Name**: `Authorization`
- **Header Value**: `Bearer YOUR_BLOTATO_API_KEY`
- **Used by**: 09-PUBLISH, 10-ANALYTICS workflows
- **Get API Key**: https://app.blotato.com/settings/api

## Environment Variables

Set these in your `.env` file or n8n environment:

```bash
# Voice cloning
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM  # Default: Rachel voice

# Avatar settings
HEYGEN_AVATAR_ID=Kristin_pubblic_3_20240108  # Your avatar ID

# Public URL for file serving (needed for HeyGen/Blotato to access files)
N8N_PUBLIC_URL=http://100.83.153.43:5678
```

## Setup Instructions

### Step 1: Open n8n Credentials
1. Navigate to n8n UI: http://100.83.153.43:5678
2. Click on **Settings** (gear icon)
3. Click **Credentials**

### Step 2: Create Each Credential
For each credential above:
1. Click **Add Credential**
2. Search for "Header Auth"
3. Select **Header Auth**
4. Fill in:
   - **Name**: Use the name in parentheses (e.g., `apify`)
   - **Name** (header): The header name from above
   - **Value**: Your API key/token

### Step 3: Verify Credentials
Each workflow references credentials by name. The names must match exactly:
- `apify`
- `openrouter`
- `pexels`
- `elevenlabs`
- `heygen`
- `openai`
- `blotato`

## Testing Credentials

### Test Apify
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  https://api.apify.com/v2/acts
```

### Test OpenRouter
```bash
curl -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"x-ai/grok-4-1106","messages":[{"role":"user","content":"Hello"}]}' \
  https://openrouter.ai/api/v1/chat/completions
```

### Test Pexels
```bash
curl -H "Authorization: YOUR_KEY" \
  "https://api.pexels.com/videos/search?query=house&per_page=1"
```

### Test ElevenLabs
```bash
curl -H "xi-api-key: YOUR_KEY" \
  https://api.elevenlabs.io/v1/voices
```

### Test HeyGen
```bash
curl -H "X-Api-Key: YOUR_KEY" \
  https://api.heygen.com/v1/avatar.list
```

### Test Blotato
```bash
curl -H "Authorization: Bearer YOUR_KEY" \
  https://api.blotato.com/v1/accounts
```

## Workflow Import Order

Import workflows in this order:
1. `00-file-server.json` - File serving webhooks
2. `01-scrape.json` - Content scraping
3. `02-auto-trigger.json` - Pipeline trigger
4. `03-script-gen.json` - Script generation
5. `04-get-video.json` - B-roll fetching
6. `05-create-voice.json` - Voice synthesis
7. `06-create-avatar.json` - Avatar generation
8. `07-combine-vids.json` - Video compositing
9. `08-caption.json` - Caption burning
10. `09-publish.json` - Multi-platform publishing
11. `10-analytics.json` - Analytics collection

After importing, you need to link the "Execute Workflow" nodes to the correct workflow IDs.
