# n8n Implementation Task List

Complete checklist for building all 10 workflows with exact specifications.

**RECOMMENDED APPROACH**: Use built-in nodes and community nodes where available, start from templates.

---

## PHASE 0: Prerequisites

### 0.1 Verify Services Running
- [ ] n8n accessible at http://100.83.153.43:5678
- [ ] Backend API accessible at http://100.83.153.43:8000
- [ ] Frontend accessible at http://100.83.153.43:3000
- [ ] PostgreSQL running on port 5433

### 0.2 Create Asset Directories
```bash
mkdir -p /home/canderson/n8n/assets/{fonts,logos,audio,videos,avatar,captions,output}
```
- [ ] Download Montserrat-Bold.ttf to `assets/fonts/`
- [ ] Add logo.png to `assets/logos/`

### 0.3 Install Community Nodes
In n8n: Settings → Community Nodes → Install

| Package Name | Service | Purpose |
|--------------|---------|---------|
| `@apify/n8n-nodes-apify` | Apify | Web scraping (TikTok, Reddit, YouTube, X) |
| `@elevenlabs/n8n-nodes-elevenlabs` | ElevenLabs | Voice cloning & TTS |
| `n8n-nodes-heygen` | HeyGen | AI avatar video generation |
| `n8n-nodes-pexels` | Pexels | Stock video search |
| `n8n-nodes-blotato` | Blotato | Multi-platform publishing |
| `n8n-nodes-ffmpeg` | FFmpeg | Video processing (optional - can use Execute Command) |

### 0.4 Get Required API Keys
- [ ] **Apify**: https://console.apify.com/account/integrations
- [ ] **HeyGen**: https://app.heygen.com/settings/api
- [ ] **Pexels**: https://www.pexels.com/api/new/
- [ ] **Blotato**: https://app.blotato.com/settings/api
- [ ] **ElevenLabs Voice ID**: Clone voice and get ID from https://elevenlabs.io/voice-lab
- [ ] **HeyGen Avatar ID**: Create avatar and get ID from HeyGen dashboard

### 0.5 Update .env File
Add missing keys:
```
APIFY_API_KEY=
HEYGEN_API_KEY=
HEYGEN_AVATAR_ID=
PEXELS_API_KEY=
BLOTATO_API_KEY=
ELEVENLABS_VOICE_ID=
```

### 0.6 Reference Templates to Import/Modify
These templates from n8n.io can be imported and modified:

| Template | URL | Use For |
|----------|-----|---------|
| Automated news video with HeyGen | https://n8n.io/workflows/2897 | WF6: Avatar generation |
| AI videos to social media with Blotato | https://n8n.io/workflows/3816 | WF9: Publishing |
| ElevenLabs voice synthesis | Search n8n.io/workflows | WF5: Voiceover |
| Apify scraping | Search n8n.io/workflows | WF1: Content discovery |

---

## Built-in vs Community Nodes Reference

### Built-in Nodes (No Installation Needed)
| Node | Use Case |
|------|----------|
| Schedule Trigger | Timed runs (daily 6am, every 15 min, weekly) |
| Webhook Trigger | UI-triggered events from React dashboard |
| HTTP Request | Any API without dedicated node |
| Postgres | Database queries (content_ideas, scripts, assets) |
| Execute Command | Shell commands (FFmpeg in container) |
| OpenRouter Chat Model | Grok 4.1 via OpenRouter API |
| OpenAI | Whisper transcription |
| If | Conditional logic (score >= 7, status checks) |
| Merge | Combining data from parallel paths |
| Code | Custom JavaScript transformation |
| Wait | Polling delays (HeyGen completion) |
| Execute Workflow | Chain workflows together |

### Community Nodes (Install Required)
| Node | Package | Use Case |
|------|---------|----------|
| Apify | `@apify/n8n-nodes-apify` | Run scraping actors |
| ElevenLabs | `@elevenlabs/n8n-nodes-elevenlabs` | Text-to-speech |
| HeyGen | `n8n-nodes-heygen` | Avatar video generation |
| Pexels | `n8n-nodes-pexels` | Stock video search |
| Blotato | `n8n-nodes-blotato` | Multi-platform publish |

---

## PHASE 1: n8n Credentials Setup

**NOTE**: Community nodes have their own credential types. After installing each community node, its credential type becomes available.

### 1.1 OpenRouter Credential (Built-in)
- [ ] Go to n8n → Settings → Credentials → Add Credential
- [ ] Search for: **OpenRouter**
- [ ] Type: OpenRouter API (built-in, uses OpenAI-compatible format)
- [ ] API Key: Your OpenRouter API key
- [ ] Base URL: `https://openrouter.ai/api/v1` (should be default)

### 1.2 OpenAI Credential (Built-in)
- [ ] Type: **OpenAI API**
- [ ] Name: `openai`
- [ ] API Key: Your OpenAI API key (for Whisper only)

### 1.3 ElevenLabs Credential (Community Node)
- [ ] **First**: Install `@elevenlabs/n8n-nodes-elevenlabs` community node
- [ ] Type: **ElevenLabs API** (appears after installing node)
- [ ] API Key: Your ElevenLabs API key

### 1.4 HeyGen Credential (Community Node)
- [ ] **First**: Install `n8n-nodes-heygen` community node
- [ ] Type: **HeyGen API** (appears after installing node)
- [ ] API Key: Your HeyGen API key

### 1.5 Apify Credential (Community Node)
- [ ] **First**: Install `@apify/n8n-nodes-apify` community node
- [ ] Type: **Apify API** (appears after installing node)
- [ ] API Token: Your Apify API token

### 1.6 Pexels Credential (Community Node)
- [ ] **First**: Install `n8n-nodes-pexels` community node
- [ ] Type: **Pexels API** (appears after installing node)
- [ ] API Key: Your Pexels API key

### 1.7 Blotato Credential (Community Node)
- [ ] **First**: Install `n8n-nodes-blotato` community node
- [ ] Type: **Blotato API** (appears after installing node)
- [ ] API Key: Your Blotato API key

### 1.8 HTTP Header Auth Fallback (if community nodes unavailable)
If any community node doesn't install, create Header Auth credentials:
- [ ] Type: **Header Auth**
- [ ] Configure header name/value per API documentation

---

## WORKFLOW 1: SCRAPE (Content Discovery)

**File**: `workflows/01-scrape.json`
**Trigger**: Schedule - Cron `0 6 * * *` (daily 6am UTC)

### Nodes to Create

#### Node 1: Schedule Trigger
- [ ] Type: Schedule Trigger
- [ ] Cron Expression: `0 6 * * *`
- [ ] Connects to: Nodes 2, 3, 4, 5 (parallel)

#### Node 2: Apify TikTok Scraper
- [ ] Type: HTTP Request
- [ ] Method: POST
- [ ] URL: `https://api.apify.com/v2/acts/clockworks~tiktok-scraper/runs`
- [ ] Authentication: Header Auth (apify credential)
- [ ] Body (JSON):
```json
{
  "hashtags": ["realestate", "homebuying", "realtor", "housingmarket"],
  "resultsPerPage": 20,
  "shouldDownloadVideos": false,
  "shouldDownloadCovers": false
}
```
- [ ] Connects to: Node 6 (Merge)

#### Node 3: Apify Reddit Scraper
- [ ] Type: HTTP Request
- [ ] Method: POST
- [ ] URL: `https://api.apify.com/v2/acts/trudax~reddit-scraper/runs`
- [ ] Authentication: Header Auth (apify credential)
- [ ] Body (JSON):
```json
{
  "startUrls": [
    "https://www.reddit.com/r/realestate/top/?t=week",
    "https://www.reddit.com/r/FirstTimeHomeBuyer/top/?t=week",
    "https://www.reddit.com/r/RealEstateInvesting/top/?t=week"
  ],
  "maxItems": 30,
  "sort": "top"
}
```
- [ ] Connects to: Node 6 (Merge)

#### Node 4: Apify YouTube Scraper
- [ ] Type: HTTP Request
- [ ] Method: POST
- [ ] URL: `https://api.apify.com/v2/acts/bernardo~youtube-scraper/runs`
- [ ] Authentication: Header Auth (apify credential)
- [ ] Body (JSON):
```json
{
  "searchKeywords": ["real estate tips", "home buying 2025", "realtor advice"],
  "maxResults": 20
}
```
- [ ] Connects to: Node 6 (Merge)

#### Node 5: Apify Twitter/X Scraper
- [ ] Type: HTTP Request
- [ ] Method: POST
- [ ] URL: `https://api.apify.com/v2/acts/apidojo~tweet-scraper/runs`
- [ ] Authentication: Header Auth (apify credential)
- [ ] Body (JSON):
```json
{
  "searchTerms": ["#realestate", "home buying tips", "mortgage rates"],
  "maxTweets": 30
}
```
- [ ] Connects to: Node 6 (Merge)

#### Node 6: Merge Results
- [ ] Type: Merge
- [ ] Mode: Append
- [ ] Connects from: Nodes 2, 3, 4, 5
- [ ] Connects to: Node 7

#### Node 7: Analyze with Grok 4.1
- [ ] Type: HTTP Request
- [ ] Method: POST
- [ ] URL: `https://openrouter.ai/api/v1/chat/completions`
- [ ] Authentication: Header Auth (openrouter credential)
- [ ] Headers:
  - `Content-Type`: `application/json`
  - `HTTP-Referer`: `http://100.83.153.43:5678`
  - `X-Title`: `n8n-content-scraper`
- [ ] Body (JSON):
```json
{
  "model": "x-ai/grok-4-1106",
  "messages": [
    {
      "role": "system",
      "content": "You analyze scraped social media content for a real estate agent's content pipeline. For each piece of content, determine:\n1. viral_score (1-10): How likely to go viral if adapted\n2. pillar: market_intelligence | educational_tips | lifestyle_local | brand_humanization\n3. suggested_hook: A compelling hook for this content\n4. adaptable: true/false - Can this be adapted for real estate?\n\nOnly return items with viral_score >= 7 AND adaptable = true.\nReturn as JSON array."
    },
    {
      "role": "user",
      "content": "Analyze these scraped items:\n{{ JSON.stringify($input.all()) }}"
    }
  ],
  "temperature": 0.3,
  "response_format": { "type": "json_object" }
}
```
- [ ] Connects to: Node 8

#### Node 8: Parse LLM Response
- [ ] Type: Code
- [ ] Language: JavaScript
- [ ] Code:
```javascript
const response = JSON.parse($input.first().json.choices[0].message.content);
const items = response.items || response;
return items.filter(item => item.viral_score >= 7).map(item => ({ json: item }));
```
- [ ] Connects to: Node 9

#### Node 9: Filter Score >= 7
- [ ] Type: Filter
- [ ] Condition: `{{ $json.viral_score }}` >= 7
- [ ] Connects to: Node 10

#### Node 10: Split In Batches
- [ ] Type: Split In Batches
- [ ] Batch Size: 1
- [ ] Connects to: Node 11

#### Node 11: Create Content Idea
- [ ] Type: HTTP Request
- [ ] Method: POST
- [ ] URL: `http://backend:8000/api/content-ideas`
- [ ] Headers: `Content-Type`: `application/json`
- [ ] Body (JSON):
```json
{
  "source_url": "{{ $json.url }}",
  "source_platform": "{{ $json.platform }}",
  "original_text": "{{ $json.text }}",
  "pillar": "{{ $json.pillar }}",
  "viral_score": {{ $json.viral_score }},
  "suggested_hook": "{{ $json.suggested_hook }}",
  "status": "pending"
}
```
- [ ] Connects to: Node 12

#### Node 12: Done (NoOp)
- [ ] Type: NoOp
- [ ] End of workflow

---

## WORKFLOW 2: AUTO TRIGGER (Pipeline Orchestrator)

**File**: `workflows/02-auto-trigger.json`
**Triggers**:
- Schedule: Cron `*/15 * * * *` (every 15 minutes)
- Webhook: `/webhook/trigger-pipeline`

### Nodes to Create

#### Node 1a: Schedule Trigger
- [ ] Type: Schedule Trigger
- [ ] Cron Expression: `*/15 * * * *`
- [ ] Connects to: Node 2

#### Node 1b: Webhook Trigger
- [ ] Type: Webhook
- [ ] HTTP Method: POST
- [ ] Path: `trigger-pipeline`
- [ ] Connects to: Node 2

#### Node 2: Merge Triggers
- [ ] Type: Merge
- [ ] Mode: Append
- [ ] Connects to: Node 3

#### Node 3: IF - Has Specific ID
- [ ] Type: IF
- [ ] Condition: `{{ $json.body?.content_idea_id }}` is not empty
- [ ] TRUE → Node 4a
- [ ] FALSE → Node 4b

#### Node 4a: Get Specific Idea
- [ ] Type: HTTP Request
- [ ] Method: GET
- [ ] URL: `http://backend:8000/api/content-ideas/{{ $json.body.content_idea_id }}`
- [ ] Connects to: Node 5

#### Node 4b: Get Next Approved
- [ ] Type: HTTP Request
- [ ] Method: GET
- [ ] URL: `http://backend:8000/api/content-ideas?status=approved&limit=1`
- [ ] Connects to: Node 5

#### Node 5: IF - Has Content
- [ ] Type: IF
- [ ] Condition: `{{ $json.length > 0 || $json.id }}`
- [ ] TRUE → Node 6
- [ ] FALSE → End (NoOp)

#### Node 6: Update Status to script_generating
- [ ] Type: HTTP Request
- [ ] Method: PATCH
- [ ] URL: `http://backend:8000/api/content-ideas/{{ $json.id || $json[0].id }}`
- [ ] Body: `{ "status": "script_generating" }`
- [ ] Connects to: Node 7

#### Node 7: Execute Workflow 03
- [ ] Type: Execute Workflow
- [ ] Workflow: Select `03-script-gen`
- [ ] Mode: Wait for completion
- [ ] Input Data:
```json
{
  "content_idea_id": "{{ $json.id }}",
  "original_text": "{{ $json.original_text }}",
  "pillar": "{{ $json.pillar }}",
  "suggested_hook": "{{ $json.suggested_hook }}",
  "source_platform": "{{ $json.source_platform }}"
}
```

---

## WORKFLOW 3: SCRIPT GEN (Script Generation)

**File**: `workflows/03-script-gen.json`
**Trigger**: Execute Workflow Trigger (called by WF2)

### Nodes to Create

#### Node 1: Execute Workflow Trigger
- [ ] Type: Execute Workflow Trigger
- [ ] Receives: `content_idea_id`, `original_text`, `pillar`, `suggested_hook`, `source_platform`
- [ ] Connects to: Node 2

#### Node 2: Generate Script (OpenRouter)
- [ ] Type: HTTP Request
- [ ] Method: POST
- [ ] URL: `https://openrouter.ai/api/v1/chat/completions`
- [ ] Authentication: Header Auth (openrouter)
- [ ] Headers: `Content-Type`: `application/json`
- [ ] Body (JSON):
```json
{
  "model": "x-ai/grok-4-1106",
  "messages": [
    {
      "role": "system",
      "content": "You are a viral content scriptwriter for a real estate agent.\n\nBRAND VOICE:\n- Warm, approachable, trustworthy\n- Faith and family-oriented\n- Expert but not condescending\n- Local market authority\n\nCONTENT PILLAR: {{ $json.pillar }}\n- market_intelligence: Data, trends, insights, market analysis\n- educational_tips: Practical buying/selling guidance, how-tos\n- lifestyle_local: Community highlights, local events, quality of life\n- brand_humanization: Personal stories, faith, family, behind-the-scenes\n\nOUTPUT FORMAT (JSON):\n{\n  \"hook\": \"First 3 seconds - attention grabber (15-20 words max)\",\n  \"body\": \"Main content - 20-40 seconds worth of speech\",\n  \"cta\": \"Call to action - 5-10 seconds\",\n  \"full_script\": \"Complete script combining hook + body + cta\",\n  \"duration_estimate\": 45\n}\n\nHOOK PATTERNS THAT WORK:\n- \"Here's what no one tells you about buying a home...\"\n- \"POV: You just found out [surprising fact]\"\n- \"3 things I wish I knew before [action]\"\n- \"Stop doing this when buying a home...\"\n- \"The biggest mistake [buyers/sellers] make is...\"\n\nMake the content feel authentic, not salesy."
    },
    {
      "role": "user",
      "content": "Create a viral script based on this content idea:\n\nOriginal content: {{ $json.original_text }}\nSuggested hook direction: {{ $json.suggested_hook }}\nContent pillar: {{ $json.pillar }}"
    }
  ],
  "temperature": 0.8,
  "max_tokens": 1500,
  "response_format": { "type": "json_object" }
}
```
- [ ] Connects to: Node 3

#### Node 3: Parse Script JSON
- [ ] Type: Code
- [ ] Language: JavaScript
- [ ] Code:
```javascript
const response = JSON.parse($input.first().json.choices[0].message.content);
return [{
  json: {
    content_idea_id: $('Execute Workflow Trigger').first().json.content_idea_id,
    hook: response.hook,
    body: response.body,
    cta: response.cta,
    full_script: response.full_script,
    duration_estimate: response.duration_estimate || 45
  }
}];
```
- [ ] Connects to: Node 4

#### Node 4: Generate Platform Captions (OpenRouter)
- [ ] Type: HTTP Request
- [ ] Method: POST
- [ ] URL: `https://openrouter.ai/api/v1/chat/completions`
- [ ] Authentication: Header Auth (openrouter)
- [ ] Body (JSON):
```json
{
  "model": "x-ai/grok-4-1106",
  "messages": [
    {
      "role": "system",
      "content": "Generate social media captions for each platform based on this video script.\n\nOUTPUT FORMAT (JSON):\n{\n  \"tiktok_caption\": \"Short, punchy, with 3-5 relevant hashtags (max 150 chars)\",\n  \"ig_caption\": \"Engaging caption with emojis and 5-10 hashtags\",\n  \"yt_title\": \"SEO-optimized YouTube Shorts title (max 60 chars)\",\n  \"yt_description\": \"Full description with keywords and CTA\",\n  \"linkedin_text\": \"Professional tone, value-focused, no hashtags\",\n  \"x_text\": \"Conversational, max 280 chars, 1-2 hashtags\",\n  \"facebook_text\": \"Friendly, community-focused\",\n  \"threads_text\": \"Casual, conversational, Instagram-style\"\n}"
    },
    {
      "role": "user",
      "content": "Generate platform captions for:\n\nHook: {{ $json.hook }}\nScript: {{ $json.full_script }}"
    }
  ],
  "temperature": 0.7,
  "response_format": { "type": "json_object" }
}
```
- [ ] Connects to: Node 5

#### Node 5: Merge Script + Captions
- [ ] Type: Code
- [ ] Language: JavaScript
- [ ] Code:
```javascript
const script = $('Parse Script JSON').first().json;
const captions = JSON.parse($input.first().json.choices[0].message.content);
return [{
  json: {
    ...script,
    ...captions
  }
}];
```
- [ ] Connects to: Node 6

#### Node 6: Create Script Record
- [ ] Type: HTTP Request
- [ ] Method: POST
- [ ] URL: `http://backend:8000/api/scripts`
- [ ] Headers: `Content-Type`: `application/json`
- [ ] Body (JSON):
```json
{
  "content_idea_id": {{ $json.content_idea_id }},
  "hook": "{{ $json.hook }}",
  "body": "{{ $json.body }}",
  "cta": "{{ $json.cta }}",
  "full_script": "{{ $json.full_script }}",
  "duration_estimate": {{ $json.duration_estimate }},
  "tiktok_caption": "{{ $json.tiktok_caption }}",
  "ig_caption": "{{ $json.ig_caption }}",
  "yt_title": "{{ $json.yt_title }}",
  "yt_description": "{{ $json.yt_description }}",
  "linkedin_text": "{{ $json.linkedin_text }}",
  "x_text": "{{ $json.x_text }}",
  "facebook_text": "{{ $json.facebook_text }}",
  "threads_text": "{{ $json.threads_text }}",
  "status": "script_ready"
}
```
- [ ] Connects to: Node 7

#### Node 7: Update Content Idea Status
- [ ] Type: HTTP Request
- [ ] Method: PATCH
- [ ] URL: `http://backend:8000/api/content-ideas/{{ $json.content_idea_id }}`
- [ ] Body: `{ "status": "script_ready" }`
- [ ] Connects to: Node 8

#### Node 8: Prepare Output
- [ ] Type: Set
- [ ] Fields:
  - `script_id`: `{{ $json.id }}`
  - `content_idea_id`: `{{ $json.content_idea_id }}`
- [ ] Connects to: Node 9

#### Node 9: Execute Workflow 04
- [ ] Type: Execute Workflow
- [ ] Workflow: Select `04-get-video`
- [ ] Mode: Wait for completion
- [ ] Pass data through

---

## WORKFLOW 4: GET VIDEO (Background Video Acquisition)

**File**: `workflows/04-get-video.json`
**Trigger**: Execute Workflow Trigger (called by WF3)

### Nodes to Create

#### Node 1: Execute Workflow Trigger
- [ ] Type: Execute Workflow Trigger
- [ ] Receives: `script_id`, `content_idea_id`
- [ ] Connects to: Node 2

#### Node 2: Get Content Idea Details
- [ ] Type: HTTP Request
- [ ] Method: GET
- [ ] URL: `http://backend:8000/api/content-ideas/{{ $json.content_idea_id }}`
- [ ] Connects to: Node 3

#### Node 3: Get Script Details
- [ ] Type: HTTP Request
- [ ] Method: GET
- [ ] URL: `http://backend:8000/api/scripts/{{ $json.script_id }}`
- [ ] Connects to: Node 4

#### Node 4: Switch - Source Platform
- [ ] Type: Switch
- [ ] Value: `{{ $json.source_platform }}`
- [ ] Rules:
  - `tiktok` → Node 5a
  - `youtube` → Node 5b
  - Default → Node 5c (Pexels)

#### Node 5a: TikTok Download (Apify)
- [ ] Type: HTTP Request
- [ ] Method: POST
- [ ] URL: `https://api.apify.com/v2/acts/clockworks~tiktok-scraper/runs`
- [ ] Authentication: Header Auth (apify)
- [ ] Body:
```json
{
  "postURLs": ["{{ $json.source_url }}"],
  "shouldDownloadVideos": true
}
```
- [ ] Connects to: Node 6

#### Node 5b: YouTube Download (Apify)
- [ ] Type: HTTP Request
- [ ] Method: POST
- [ ] URL: `https://api.apify.com/v2/acts/bernardo~youtube-scraper/runs`
- [ ] Similar to TikTok
- [ ] Connects to: Node 6

#### Node 5c: Pexels Stock Video Search
- [ ] Type: HTTP Request
- [ ] Method: GET
- [ ] URL: `https://api.pexels.com/videos/search`
- [ ] Authentication: Header Auth (pexels)
- [ ] Query Parameters:
  - `query`: `real estate home house` (extract from script)
  - `orientation`: `portrait`
  - `size`: `medium`
  - `per_page`: `5`
- [ ] Connects to: Node 5c-2

#### Node 5c-2: Select Best Video (Code)
- [ ] Type: Code
- [ ] Code:
```javascript
const videos = $json.videos;
const video = videos.find(v =>
  v.video_files.some(f => f.quality === 'hd' && f.height > f.width)
);
const file = video.video_files.find(f =>
  f.quality === 'hd' && f.height > f.width
);
return [{ json: { videoUrl: file.link } }];
```
- [ ] Connects to: Node 6

#### Node 6: Merge Video Sources
- [ ] Type: Merge
- [ ] Mode: Append
- [ ] Connects to: Node 7

#### Node 7: Download Video
- [ ] Type: HTTP Request
- [ ] Method: GET
- [ ] URL: `{{ $json.videoUrl }}`
- [ ] Response Format: File
- [ ] Connects to: Node 8

#### Node 8: Write Video File
- [ ] Type: Write Binary File
- [ ] File Path: `/home/node/assets/videos/{{ $json.content_idea_id }}_background.mp4`
- [ ] Connects to: Node 9

#### Node 9: Create Asset Record
- [ ] Type: HTTP Request
- [ ] Method: POST
- [ ] URL: `http://backend:8000/api/assets`
- [ ] Body:
```json
{
  "script_id": {{ $json.script_id }},
  "background_video_path": "/home/node/assets/videos/{{ $json.content_idea_id }}_background.mp4",
  "status": "pending"
}
```
- [ ] Connects to: Node 10

#### Node 10: Execute Workflow 05
- [ ] Type: Execute Workflow
- [ ] Workflow: Select `05-create-voice`
- [ ] Pass: `asset_id`, `script_id`, `content_idea_id`

---

## WORKFLOW 5: CREATE VOICE (ElevenLabs + Whisper)

**File**: `workflows/05-create-voice.json`
**Trigger**: Execute Workflow Trigger (called by WF4)

### Nodes to Create

#### Node 1: Execute Workflow Trigger
- [ ] Receives: `asset_id`, `script_id`, `content_idea_id`
- [ ] Connects to: Node 2

#### Node 2: Get Script
- [ ] Type: HTTP Request
- [ ] Method: GET
- [ ] URL: `http://backend:8000/api/scripts/{{ $json.script_id }}`
- [ ] Connects to: Node 3

#### Node 3: ElevenLabs TTS
- [ ] Type: HTTP Request
- [ ] Method: POST
- [ ] URL: `https://api.elevenlabs.io/v1/text-to-speech/{{ $env.ELEVENLABS_VOICE_ID }}`
- [ ] Authentication: Header Auth (elevenlabs)
- [ ] Headers:
  - `Content-Type`: `application/json`
  - `Accept`: `audio/mpeg`
- [ ] Body:
```json
{
  "text": "{{ $json.full_script }}",
  "model_id": "eleven_monolingual_v1",
  "voice_settings": {
    "stability": 0.5,
    "similarity_boost": 0.75,
    "style": 0.5,
    "use_speaker_boost": true
  }
}
```
- [ ] Response Format: File
- [ ] Connects to: Node 4

#### Node 4: Write Audio File
- [ ] Type: Write Binary File
- [ ] File Path: `/home/node/assets/audio/{{ $json.content_idea_id }}_voiceover.mp3`
- [ ] Connects to: Node 5

#### Node 5: Wait 3 seconds
- [ ] Type: Wait
- [ ] Duration: 3 seconds
- [ ] Connects to: Node 6

#### Node 6: Read Audio File
- [ ] Type: Read Binary File
- [ ] File Path: `/home/node/assets/audio/{{ $json.content_idea_id }}_voiceover.mp3`
- [ ] Connects to: Node 7

#### Node 7: Whisper Transcribe (OpenAI)
- [ ] Type: HTTP Request
- [ ] Method: POST
- [ ] URL: `https://api.openai.com/v1/audio/transcriptions`
- [ ] Authentication: Header Auth (OpenAI API key)
- [ ] Content-Type: `multipart/form-data`
- [ ] Body (Form Data):
  - `file`: Binary audio file
  - `model`: `whisper-1`
  - `response_format`: `verbose_json`
  - `timestamp_granularities`: `["word"]`
- [ ] Connects to: Node 8

#### Node 8: Generate SRT (Code)
- [ ] Type: Code
- [ ] Code:
```javascript
const words = $json.words;
let srt = '';
let index = 1;
let chunk = [];
let chunkStart = 0;

for (let i = 0; i < words.length; i++) {
  const word = words[i];
  if (chunk.length === 0) chunkStart = word.start;
  chunk.push(word.word);

  if (chunk.length >= 6 || (words[i+1] && words[i+1].start - word.end > 0.5)) {
    const startTime = formatSrtTime(chunkStart);
    const endTime = formatSrtTime(word.end);
    srt += `${index}\n${startTime} --> ${endTime}\n${chunk.join(' ')}\n\n`;
    index++;
    chunk = [];
  }
}

if (chunk.length > 0) {
  const startTime = formatSrtTime(chunkStart);
  const endTime = formatSrtTime(words[words.length-1].end);
  srt += `${index}\n${startTime} --> ${endTime}\n${chunk.join(' ')}\n\n`;
}

function formatSrtTime(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  const ms = Math.round((seconds % 1) * 1000);
  return `${pad(h)}:${pad(m)}:${pad(s)},${pad(ms, 3)}`;
}

function pad(n, len=2) { return n.toString().padStart(len, '0'); }

return [{ json: { srt, duration: words[words.length-1].end } }];
```
- [ ] Connects to: Node 9

#### Node 9: Write SRT File
- [ ] Type: Write File
- [ ] File Path: `/home/node/assets/captions/{{ $json.content_idea_id }}.srt`
- [ ] Content: `{{ $json.srt }}`
- [ ] Connects to: Node 10

#### Node 10: Update Asset Record
- [ ] Type: HTTP Request
- [ ] Method: PATCH
- [ ] URL: `http://backend:8000/api/assets/{{ $json.asset_id }}`
- [ ] Body:
```json
{
  "voiceover_path": "/home/node/assets/audio/{{ $json.content_idea_id }}_voiceover.mp3",
  "voiceover_duration": {{ $json.duration }},
  "srt_path": "/home/node/assets/captions/{{ $json.content_idea_id }}.srt",
  "status": "voice_ready"
}
```
- [ ] Connects to: Node 11

#### Node 11: Execute Workflow 06
- [ ] Type: Execute Workflow
- [ ] Workflow: Select `06-create-avatar`
- [ ] Pass: `asset_id`, `script_id`, `content_idea_id`

---

## WORKFLOW 6: CREATE AVATAR (HeyGen)

**File**: `workflows/06-create-avatar.json`
**Trigger**: Execute Workflow Trigger (called by WF5)

### Nodes to Create

#### Node 1: Execute Workflow Trigger
- [ ] Receives: `asset_id`, `script_id`, `content_idea_id`
- [ ] Connects to: Node 2

#### Node 2: Get Asset Record
- [ ] Type: HTTP Request
- [ ] Method: GET
- [ ] URL: `http://backend:8000/api/assets/{{ $json.asset_id }}`
- [ ] Connects to: Node 3

#### Node 3: Read Audio File
- [ ] Type: Read Binary File
- [ ] File Path: `{{ $json.voiceover_path }}`
- [ ] Connects to: Node 4

#### Node 4: Upload Audio to HeyGen
- [ ] Type: HTTP Request
- [ ] Method: POST
- [ ] URL: `https://api.heygen.com/v1/asset`
- [ ] Authentication: Header Auth (heygen)
- [ ] Content-Type: `multipart/form-data`
- [ ] Body: Audio file
- [ ] Connects to: Node 5

#### Node 5: Create Avatar Video
- [ ] Type: HTTP Request
- [ ] Method: POST
- [ ] URL: `https://api.heygen.com/v2/video/generate`
- [ ] Authentication: Header Auth (heygen)
- [ ] Body:
```json
{
  "video_inputs": [{
    "character": {
      "type": "avatar",
      "avatar_id": "{{ $env.HEYGEN_AVATAR_ID }}",
      "avatar_style": "normal"
    },
    "voice": {
      "type": "audio",
      "audio_url": "{{ $json.audio_url }}"
    },
    "background": {
      "type": "color",
      "value": "#00FF00"
    }
  }],
  "dimension": {
    "width": 1080,
    "height": 1920
  },
  "test": false
}
```
- [ ] Connects to: Node 6

#### Node 6: Store Video ID
- [ ] Type: Set
- [ ] Fields: `video_id`, `asset_id`, `content_idea_id`
- [ ] Connects to: Node 7

#### Node 7: Update Status to avatar_generating
- [ ] Type: HTTP Request
- [ ] Method: PATCH
- [ ] URL: `http://backend:8000/api/assets/{{ $json.asset_id }}`
- [ ] Body: `{ "status": "avatar_generating" }`
- [ ] Connects to: Node 8

#### Node 8: Wait 60 seconds
- [ ] Type: Wait
- [ ] Duration: 60 seconds
- [ ] Connects to: Node 9

#### Node 9: Poll Status Loop
- [ ] Type: Loop
- [ ] Max Iterations: 20
- [ ] Connects to: Node 10

#### Node 10: Check HeyGen Status
- [ ] Type: HTTP Request
- [ ] Method: GET
- [ ] URL: `https://api.heygen.com/v1/video_status.get?video_id={{ $json.video_id }}`
- [ ] Authentication: Header Auth (heygen)
- [ ] Connects to: Node 11

#### Node 11: IF - Status Check
- [ ] Type: IF
- [ ] Conditions:
  - `completed` → Node 14
  - `processing` → Node 12
  - `failed` → Node 13

#### Node 12: Wait 30 seconds (loop back)
- [ ] Type: Wait
- [ ] Duration: 30 seconds
- [ ] Connects back to: Node 9 (loop)

#### Node 13: Error Handler
- [ ] Type: HTTP Request
- [ ] Method: PATCH
- [ ] URL: `http://backend:8000/api/assets/{{ $json.asset_id }}`
- [ ] Body: `{ "status": "error", "error_message": "HeyGen video generation failed" }`
- [ ] Stop workflow

#### Node 14: Download Avatar Video
- [ ] Type: HTTP Request
- [ ] Method: GET
- [ ] URL: `{{ $json.data.video_url }}`
- [ ] Response Format: File
- [ ] Connects to: Node 15

#### Node 15: Write Avatar File
- [ ] Type: Write Binary File
- [ ] File Path: `/home/node/assets/avatar/{{ $json.content_idea_id }}_avatar.mp4`
- [ ] Connects to: Node 16

#### Node 16: Update Asset Record
- [ ] Type: HTTP Request
- [ ] Method: PATCH
- [ ] URL: `http://backend:8000/api/assets/{{ $json.asset_id }}`
- [ ] Body:
```json
{
  "avatar_video_path": "/home/node/assets/avatar/{{ $json.content_idea_id }}_avatar.mp4",
  "status": "avatar_ready"
}
```
- [ ] Connects to: Node 17

#### Node 17: Execute Workflow 07
- [ ] Type: Execute Workflow
- [ ] Workflow: Select `07-combine-vids`
- [ ] Pass: `asset_id`, `script_id`, `content_idea_id`

---

## WORKFLOW 7: COMBINE VIDS (FFmpeg Assembly)

**File**: `workflows/07-combine-vids.json`
**Trigger**: Execute Workflow Trigger (called by WF6)

### Nodes to Create

#### Node 1: Execute Workflow Trigger
- [ ] Receives: `asset_id`, `script_id`, `content_idea_id`
- [ ] Connects to: Node 2

#### Node 2: Get Asset Record
- [ ] Type: HTTP Request
- [ ] Method: GET
- [ ] URL: `http://backend:8000/api/assets/{{ $json.asset_id }}`
- [ ] Connects to: Node 3

#### Node 3: Update Status to assembling
- [ ] Type: HTTP Request
- [ ] Method: PATCH
- [ ] URL: `http://backend:8000/api/assets/{{ $json.asset_id }}`
- [ ] Body: `{ "status": "assembling" }`
- [ ] Connects to: Node 4

#### Node 4: Generate FFmpeg Command (OpenRouter)
- [ ] Type: HTTP Request
- [ ] Method: POST
- [ ] URL: `https://openrouter.ai/api/v1/chat/completions`
- [ ] Authentication: Header Auth (openrouter)
- [ ] Body:
```json
{
  "model": "x-ai/grok-4-1106",
  "messages": [
    {
      "role": "system",
      "content": "You are an FFmpeg expert. Generate a single FFmpeg command for video assembly.\n\nINPUTS:\n- Background video: {{ $json.background_video_path }}\n- Avatar video: {{ $json.avatar_video_path }} (has green screen #00FF00)\n- Duration: {{ $json.voiceover_duration }} seconds\n\nREQUIREMENTS:\n1. Scale/crop background to exactly 1080x1920 (portrait)\n2. Loop background if shorter than audio duration\n3. Chroma key the avatar video (remove green screen)\n4. Scale avatar to 1080 width, maintain aspect ratio\n5. Position avatar at bottom center (overlay at y=H-h)\n6. Use audio from avatar video (contains the voice)\n7. Output: H.264 video, AAC audio, 30fps\n8. Output path: /home/node/assets/output/{{ content_idea_id }}_combined.mp4\n\nOutput ONLY the FFmpeg command, no explanation."
    },
    {
      "role": "user",
      "content": "Generate FFmpeg command for:\nBackground: {{ $json.background_video_path }}\nAvatar: {{ $json.avatar_video_path }}\nDuration: {{ $json.voiceover_duration }}s\nOutput: /home/node/assets/output/{{ $json.content_idea_id }}_combined.mp4"
    }
  ],
  "temperature": 0.3
}
```
- [ ] Connects to: Node 5

#### Node 5: Extract Command (Code)
- [ ] Type: Code
- [ ] Code:
```javascript
const response = $json.choices[0].message.content;
let cmd = response.replace(/```bash\n?/g, '').replace(/```\n?/g, '').trim();
return [{ json: { ffmpeg_command: cmd } }];
```
- [ ] Connects to: Node 6

#### Node 6: Run FFmpeg
- [ ] Type: Execute Command
- [ ] Command: `{{ $json.ffmpeg_command }}`
- [ ] Working Directory: `/home/node/assets`
- [ ] Timeout: 300000 (5 minutes)
- [ ] Connects to: Node 7

#### Node 7: IF - FFmpeg Success
- [ ] Type: IF
- [ ] Condition: Exit code == 0
- [ ] TRUE → Node 9
- [ ] FALSE → Node 8

#### Node 8: Error Handler
- [ ] Type: HTTP Request
- [ ] Method: PATCH
- [ ] URL: `http://backend:8000/api/assets/{{ $json.asset_id }}`
- [ ] Body: `{ "status": "error", "error_message": "FFmpeg combine failed: {{ $json.stderr }}" }`
- [ ] Stop workflow

#### Node 9: Update Asset Record
- [ ] Type: HTTP Request
- [ ] Method: PATCH
- [ ] URL: `http://backend:8000/api/assets/{{ $json.asset_id }}`
- [ ] Body:
```json
{
  "combined_video_path": "/home/node/assets/output/{{ $json.content_idea_id }}_combined.mp4",
  "status": "captioning"
}
```
- [ ] Connects to: Node 10

#### Node 10: Execute Workflow 08
- [ ] Type: Execute Workflow
- [ ] Workflow: Select `08-caption`
- [ ] Pass: `asset_id`, `script_id`, `content_idea_id`

---

## WORKFLOW 8: CAPTION (Subtitle Burn-in)

**File**: `workflows/08-caption.json`
**Trigger**: Execute Workflow Trigger (called by WF7)

### Nodes to Create

#### Node 1: Execute Workflow Trigger
- [ ] Receives: `asset_id`, `script_id`, `content_idea_id`
- [ ] Connects to: Node 2

#### Node 2: Get Asset Record
- [ ] Type: HTTP Request
- [ ] Method: GET
- [ ] URL: `http://backend:8000/api/assets/{{ $json.asset_id }}`
- [ ] Connects to: Node 3

#### Node 3: Read SRT File
- [ ] Type: Read File
- [ ] File Path: `{{ $json.srt_path }}`
- [ ] Connects to: Node 4

#### Node 4: Convert SRT to ASS (Code)
- [ ] Type: Code
- [ ] Code:
```javascript
const srt = $json.data;

const assHeader = `[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Montserrat,72,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,4,0,2,20,20,350,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text`;

const blocks = srt.trim().split('\n\n');
const events = blocks.map(block => {
  const lines = block.split('\n');
  const timeMatch = lines[1].match(/(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})/);
  if (!timeMatch) return null;
  const start = `${timeMatch[1]}:${timeMatch[2]}:${timeMatch[3]}.${timeMatch[4].slice(0,2)}`;
  const end = `${timeMatch[5]}:${timeMatch[6]}:${timeMatch[7]}.${timeMatch[8].slice(0,2)}`;
  const text = lines.slice(2).join('\\N');
  return `Dialogue: 0,${start},${end},Default,,0,0,0,,${text}`;
}).filter(Boolean).join('\n');

return [{ json: { ass: assHeader + '\n' + events } }];
```
- [ ] Connects to: Node 5

#### Node 5: Write ASS File
- [ ] Type: Write File
- [ ] File Path: `/home/node/assets/captions/{{ $json.content_idea_id }}.ass`
- [ ] Content: `{{ $json.ass }}`
- [ ] Connects to: Node 6

#### Node 6: Run FFmpeg Subtitles
- [ ] Type: Execute Command
- [ ] Command:
```
ffmpeg -i {{ $json.combined_video_path }} -vf "ass=/home/node/assets/captions/{{ $json.content_idea_id }}.ass:fontsdir=/home/node/assets/fonts" -c:v libx264 -preset fast -crf 23 -c:a copy -y /home/node/assets/output/{{ $json.content_idea_id }}_final.mp4
```
- [ ] Timeout: 300000
- [ ] Connects to: Node 7

#### Node 7: IF - FFmpeg Success
- [ ] Type: IF
- [ ] Condition: Exit code == 0
- [ ] TRUE → Node 9
- [ ] FALSE → Node 8

#### Node 8: Error Handler
- [ ] Similar to WF7 error handler

#### Node 9: Update Asset Record
- [ ] Type: HTTP Request
- [ ] Method: PATCH
- [ ] URL: `http://backend:8000/api/assets/{{ $json.asset_id }}`
- [ ] Body:
```json
{
  "ass_path": "/home/node/assets/captions/{{ $json.content_idea_id }}.ass",
  "final_video_path": "/home/node/assets/output/{{ $json.content_idea_id }}_final.mp4",
  "status": "ready_to_publish"
}
```
- [ ] END of chained workflows

---

## WORKFLOW 9: PUBLISH (Multi-Platform Distribution)

**File**: `workflows/09-publish.json`
**Triggers**:
- Schedule: Cron `0 * * * *` (hourly)
- Webhook: `/webhook/publish-video`

### Nodes to Create

#### Node 1a: Schedule Trigger
- [ ] Type: Schedule Trigger
- [ ] Cron: `0 * * * *`
- [ ] Connects to: Node 2

#### Node 1b: Webhook Trigger
- [ ] Type: Webhook
- [ ] Path: `publish-video`
- [ ] Method: POST
- [ ] Connects to: Node 2

#### Node 2: Merge Triggers
- [ ] Type: Merge
- [ ] Connects to: Node 3

#### Node 3: IF - Has Specific Asset ID
- [ ] Type: IF
- [ ] Condition: `{{ $json.body?.asset_id }}` exists
- [ ] TRUE → Node 4a
- [ ] FALSE → Node 4b

#### Node 4a: Get Specific Asset
- [ ] Type: HTTP Request
- [ ] Method: GET
- [ ] URL: `http://backend:8000/api/assets/{{ $json.body.asset_id }}`

#### Node 4b: Get Next Ready Asset
- [ ] Type: HTTP Request
- [ ] Method: GET
- [ ] URL: `http://backend:8000/api/assets?status=ready_to_publish&limit=1`

#### Node 5: IF - Has Asset
- [ ] Type: IF
- [ ] Condition: Has data
- [ ] TRUE → Node 6
- [ ] FALSE → End

#### Node 6: Get Script (for captions)
- [ ] Type: HTTP Request
- [ ] Method: GET
- [ ] URL: `http://backend:8000/api/scripts/{{ $json.script_id }}`
- [ ] Connects to: Node 7

#### Node 7: Update Status to publishing
- [ ] Type: HTTP Request
- [ ] Method: PATCH
- [ ] Body: `{ "status": "publishing" }`
- [ ] Connects to: Node 8

#### Node 8: Read Final Video
- [ ] Type: Read Binary File
- [ ] File Path: `{{ $json.final_video_path }}`
- [ ] Connects to: Node 9

#### Node 9: Blotato Publish
- [ ] Type: HTTP Request
- [ ] Method: POST
- [ ] URL: `https://api.blotato.com/v1/posts`
- [ ] Authentication: Header Auth (blotato)
- [ ] Content-Type: `multipart/form-data`
- [ ] Body:
```json
{
  "media": "[Binary video]",
  "platforms": ["tiktok", "instagram", "youtube", "linkedin", "twitter", "facebook", "threads"],
  "platform_settings": {
    "tiktok": { "caption": "{{ $json.tiktok_caption }}" },
    "instagram": { "caption": "{{ $json.ig_caption }}", "share_to_feed": true, "share_to_reels": true },
    "youtube": { "title": "{{ $json.yt_title }}", "description": "{{ $json.yt_description }}", "visibility": "public", "shorts": true },
    "linkedin": { "text": "{{ $json.linkedin_text }}" },
    "twitter": { "text": "{{ $json.x_text }}" },
    "facebook": { "text": "{{ $json.facebook_text }}" },
    "threads": { "text": "{{ $json.threads_text }}" }
  },
  "schedule": "now"
}
```
- [ ] Connects to: Node 10

#### Node 10: Extract Platform URLs (Code)
- [ ] Type: Code
- [ ] Code:
```javascript
const platforms = $json.platforms;
return [{
  json: {
    tiktok_url: platforms.tiktok?.url,
    tiktok_id: platforms.tiktok?.id,
    ig_url: platforms.instagram?.url,
    ig_id: platforms.instagram?.id,
    yt_url: platforms.youtube?.url,
    yt_id: platforms.youtube?.id,
    linkedin_url: platforms.linkedin?.url,
    linkedin_id: platforms.linkedin?.id,
    x_url: platforms.twitter?.url,
    x_id: platforms.twitter?.id,
    facebook_url: platforms.facebook?.url,
    facebook_id: platforms.facebook?.id,
    threads_url: platforms.threads?.url,
    threads_id: platforms.threads?.id
  }
}];
```
- [ ] Connects to: Node 11

#### Node 11: Create Published Record
- [ ] Type: HTTP Request
- [ ] Method: POST
- [ ] URL: `http://backend:8000/api/published`
- [ ] Body: All platform URLs and IDs
- [ ] Connects to: Node 12

#### Node 12: Update Asset Status
- [ ] Type: HTTP Request
- [ ] Method: PATCH
- [ ] Body: `{ "status": "published" }`
- [ ] Connects to: Node 13

#### Node 13: Update Content Idea Status
- [ ] Type: HTTP Request
- [ ] Method: PATCH
- [ ] URL: `http://backend:8000/api/content-ideas/{{ $json.content_idea_id }}`
- [ ] Body: `{ "status": "published" }`
- [ ] END

---

## WORKFLOW 10: ANALYTICS (Performance Tracking)

**File**: `workflows/10-analytics.json`
**Trigger**: Schedule - Cron `0 0 * * 0` (weekly Sunday midnight)

### Nodes to Create

#### Node 1: Schedule Trigger
- [ ] Type: Schedule Trigger
- [ ] Cron: `0 0 * * 0`
- [ ] Connects to: Node 2

#### Node 2: Get Recent Published
- [ ] Type: HTTP Request
- [ ] Method: GET
- [ ] URL: `http://backend:8000/api/published?limit=50`
- [ ] Connects to: Node 3

#### Node 3: Split In Batches
- [ ] Type: Split In Batches
- [ ] Batch Size: 1
- [ ] Connects to: Nodes 4-7 (parallel)

#### Nodes 4-7: Platform Analytics (Parallel)
- [ ] Type: HTTP Request
- [ ] Fetch analytics from Blotato or platform APIs
- [ ] Connect to: Node 8

#### Node 8: Merge Analytics
- [ ] Type: Merge
- [ ] Connects to: Node 9

#### Node 9: Calculate Engagement (Code)
- [ ] Type: Code
- [ ] Code:
```javascript
const engagementRate = ($json.likes + $json.comments + $json.shares) / $json.views;
return [{ json: { ...$json, engagement_rate: engagementRate } }];
```
- [ ] Connects to: Node 10

#### Node 10: Create Analytics Record
- [ ] Type: HTTP Request
- [ ] Method: POST
- [ ] URL: `http://backend:8000/api/analytics`
- [ ] Body: Views, likes, comments, shares, engagement_rate
- [ ] Connects to: Node 11

#### Node 11: Generate Weekly Report (OpenRouter)
- [ ] Type: HTTP Request
- [ ] Prompt: Analyze performance, identify trends, make recommendations
- [ ] Connects to: Node 12

#### Node 12: Send Notification (Optional)
- [ ] Type: Slack/Email node
- [ ] Send weekly report summary
- [ ] END

---

## PHASE 3: Testing Checklist

### Unit Tests (Per Workflow)
- [ ] WF1: Test Apify scrapers return data
- [ ] WF1: Test LLM analysis returns valid JSON
- [ ] WF2: Test webhook trigger works
- [ ] WF2: Test schedule trigger polls correctly
- [ ] WF3: Test script generation returns all fields
- [ ] WF4: Test Pexels search returns videos
- [ ] WF5: Test ElevenLabs returns audio
- [ ] WF5: Test Whisper returns word timestamps
- [ ] WF6: Test HeyGen creates video
- [ ] WF6: Test polling loop completes
- [ ] WF7: Test FFmpeg command executes
- [ ] WF8: Test SRT to ASS conversion
- [ ] WF9: Test Blotato publishes to all platforms
- [ ] WF10: Test analytics collection

### Integration Tests
- [ ] End-to-end: Create idea → Publish (manual)
- [ ] Status flow: Verify all status transitions
- [ ] Error handling: Test failure scenarios
- [ ] UI integration: Test webhook triggers from React

### Production Checklist
- [ ] All workflows activated
- [ ] All schedules enabled
- [ ] Credentials verified
- [ ] Asset directories have write permissions
- [ ] Monitoring/alerting configured

---

## Quick Reference

### Status Flow
```
content_ideas: pending → approved → script_generating → script_ready → published | error
assets: pending → voice_ready → avatar_generating → avatar_ready → assembling → captioning → ready_to_publish → publishing → published | error
```

### Webhook URLs
```
POST http://100.83.153.43:5678/webhook/trigger-pipeline   { "content_idea_id": 123 }
POST http://100.83.153.43:5678/webhook/publish-video      { "asset_id": 456 }
POST http://100.83.153.43:5678/webhook/trigger-scrape     {}
```

### Backend API Base
```
Internal: http://backend:8000/api
External: http://100.83.153.43:8000/api
```

### Asset Paths (Container)
```
/home/node/assets/videos/      - Background videos
/home/node/assets/audio/       - Voiceovers
/home/node/assets/avatar/      - HeyGen videos
/home/node/assets/captions/    - SRT and ASS files
/home/node/assets/output/      - Final videos
/home/node/assets/fonts/       - Montserrat-Bold.ttf
```
