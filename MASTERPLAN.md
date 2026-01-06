# n8n Video Pipeline Masterplan

## Overview

The pipeline consists of **10 workflow sections** that handle the complete content lifecycle autonomously, managed from the React UI at http://100.83.153.43:3000.

```
SCRAPE → AUTO TRIGGER → SCRIPT GEN → GET VIDEO → CREATE VOICE → CREATE AVATAR → COMBINE VIDS → CAPTION → PUBLISH → ANALYTICS
```

**Key Integration Points:**
- All workflows use PostgreSQL (via FastAPI backend) instead of Airtable
- Backend API: `http://backend:8000/api/` (internal Docker network)
- Status changes trigger subsequent workflows via webhooks
- UI provides manual controls and monitoring

---

## Database-Driven Architecture

### Status Flow
```
content_ideas: pending → approved → script_generating → script_ready → error
scripts: pending → script_ready → voice_generating → voice_ready → error
assets: pending → voice_ready → avatar_generating → avatar_ready → assembling → captioning → ready_to_publish → publishing → published → error
```

### Workflow Triggers
Each workflow listens for specific status values and updates them on completion, creating an autonomous chain:

| Workflow | Triggers On | Sets Status To |
|----------|-------------|----------------|
| 1. Scrape | Schedule (6am daily) | content_ideas.status = 'pending' |
| 2. Auto Trigger | Schedule (15min) OR content_ideas.status = 'approved' | content_ideas.status = 'script_generating' |
| 3. Script Gen | content_ideas.status = 'script_generating' | scripts.status = 'script_ready' |
| 4. Get Video | scripts.status = 'script_ready' | assets.status = 'pending' |
| 5. Create Voice | assets.status = 'pending' | assets.status = 'voice_ready' |
| 6. Create Avatar | assets.status = 'voice_ready' | assets.status = 'avatar_ready' |
| 7. Combine Vids | assets.status = 'avatar_ready' | assets.status = 'assembling' → 'captioning' |
| 8. Caption | assets.status = 'captioning' | assets.status = 'ready_to_publish' |
| 9. Publish | assets.status = 'ready_to_publish' | assets.status = 'published' |
| 10. Analytics | Schedule (weekly) | Updates analytics table |

---

## Phase 1: Foundation Setup

### 1.1 n8n Credentials Setup

Configure these credentials in n8n (Settings → Credentials):

| Credential Name | Type | Required Fields |
|-----------------|------|-----------------|
| `postgres_content` | Postgres | host=n8n_postgres, port=5432, database=content_pipeline, user=n8n, password=n8n_password |
| `openrouter` | Header Auth | Name=Authorization, Value=Bearer sk-or-v1-... |
| `openai` | OpenAI API | API Key (for Whisper only) |
| `elevenlabs` | Header Auth | Name=xi-api-key, Value=sk_... |
| `heygen` | Header Auth | Name=X-Api-Key, Value=... |
| `apify` | Header Auth | Name=Authorization, Value=Bearer ... |
| `pexels` | Header Auth | Name=Authorization, Value=... |

### 1.2 Environment Variables

Already configured in `.env` and docker-compose.yml:
- `ELEVENLABS_API_KEY` - Voice generation
- `OPENROUTER_API_KEY` - LLM calls (Grok 4.1)
- `OPENAI_API_KEY` - Whisper transcription
- `DATABASE_URL` - PostgreSQL connection

Access in n8n via: `{{ $env.ELEVENLABS_API_KEY }}`

### 1.3 Backend API Endpoints

All workflows interact with the FastAPI backend:

```
Base URL (internal): http://backend:8000/api
Base URL (external): http://100.83.153.43:8000/api

GET    /content-ideas                    - List ideas (filter by status)
POST   /content-ideas                    - Create new idea
PATCH  /content-ideas/{id}               - Update idea (status, etc.)
GET    /scripts?content_idea_id={id}     - Get scripts for idea
POST   /scripts                          - Create script
PATCH  /scripts/{id}                     - Update script
GET    /assets?script_id={id}            - Get assets for script
POST   /assets                           - Create asset record
PATCH  /assets/{id}                      - Update asset (paths, status)
POST   /published                        - Create published record
GET    /pipeline/stats                   - Dashboard statistics
```

---

## Phase 2: Workflow Implementation

### Workflow 1: SCRAPE (Content Discovery)

**Purpose:** Automatically discover trending real estate content daily

**Trigger:** Schedule (daily at 6:00 AM UTC)

**Workflow File:** `workflows/01-scrape.json`

```
┌─────────────────┐
│ Schedule Trigger │ ──────────────────────────────────────────────────────────────────┐
│ Cron: 0 6 * * * │                                                                    │
└─────────────────┘                                                                    │
         │                                                                             │
         ▼                                                                             │
┌─────────────────────────────────────────────────────────────────────────────────────┐│
│ PARALLEL SCRAPE (Split In Batches)                                                  ││
├─────────────────┬─────────────────┬─────────────────┬─────────────────┐             ││
│ HTTP Request    │ HTTP Request    │ HTTP Request    │ HTTP Request    │             ││
│ Apify: TikTok   │ Apify: Reddit   │ Apify: YouTube  │ Apify: X/Twitter│             ││
│                 │                 │                 │                 │             ││
│ Actor:          │ Actor:          │ Actor:          │ Actor:          │             ││
│ clockworks/     │ trudax/         │ bernardo/       │ apidojo/        │             ││
│ tiktok-scraper  │ reddit-scraper  │ youtube-scraper │ tweet-scraper   │             ││
│                 │                 │                 │                 │             ││
│ Search Terms:   │ Subreddits:     │ Search Terms:   │ Search Terms:   │             ││
│ - real estate   │ - r/realestate  │ - real estate   │ - #realestate   │             ││
│ - home buying   │ - r/firsttime   │ - home buying   │ - home buying   │             ││
│ - realtor tips  │   homebuyer     │ - realtor tips  │ - mortgage tips │             ││
│ - mortgage      │ - r/RealEstate  │                 │                 │             ││
│                 │   Investing     │                 │                 │             ││
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘             │
         │                                                                             │
         ▼                                                                             │
┌─────────────────┐                                                                    │
│ Merge Node      │ ◄─────────────────────────────────────────────────────────────────┘
│ Combine all     │
│ scraped results │
└─────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: OpenRouter (Grok 4.1) - Analyze & Score                               │
│                                                                                     │
│ URL: https://openrouter.ai/api/v1/chat/completions                                  │
│ Method: POST                                                                        │
│ Headers:                                                                            │
│   Authorization: Bearer {{ $env.OPENROUTER_API_KEY }}                               │
│   Content-Type: application/json                                                    │
│   HTTP-Referer: http://100.83.153.43:5678                                           │
│   X-Title: n8n-content-scraper                                                      │
│                                                                                     │
│ Body:                                                                               │
│ {                                                                                   │
│   "model": "x-ai/grok-4-1106",                                                      │
│   "messages": [                                                                     │
│     {                                                                               │
│       "role": "system",                                                             │
│       "content": "You analyze scraped social media content for a real estate       │
│         agent's content pipeline. For each piece of content, determine:            │
│         1. viral_score (1-10): How likely to go viral if adapted                   │
│         2. pillar: market_intelligence | educational_tips | lifestyle_local |      │
│            brand_humanization                                                       │
│         3. suggested_hook: A compelling hook for this content                       │
│         4. adaptable: true/false - Can this be adapted for real estate?            │
│                                                                                     │
│         Only return items with viral_score >= 7 AND adaptable = true.              │
│         Return as JSON array."                                                      │
│     },                                                                              │
│     {                                                                               │
│       "role": "user",                                                               │
│       "content": "Analyze these scraped items:\n{{ JSON.stringify($input.all()) }}"│
│     }                                                                               │
│   ],                                                                                │
│   "temperature": 0.3,                                                               │
│   "response_format": { "type": "json_object" }                                      │
│ }                                                                                   │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ Code Node       │
│ Parse LLM JSON  │
│ response and    │
│ extract items   │
│ with score >= 7 │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Split In Batches│
│ Process each    │
│ filtered item   │
└─────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: Create Content Idea (Backend API)                                     │
│                                                                                     │
│ URL: http://backend:8000/api/content-ideas                                          │
│ Method: POST                                                                        │
│ Headers: Content-Type: application/json                                             │
│                                                                                     │
│ Body:                                                                               │
│ {                                                                                   │
│   "source_url": "{{ $json.url }}",                                                  │
│   "source_platform": "{{ $json.platform }}",   // tiktok, reddit, youtube, twitter │
│   "original_text": "{{ $json.text }}",                                              │
│   "pillar": "{{ $json.pillar }}",              // from LLM analysis                 │
│   "viral_score": {{ $json.viral_score }},      // from LLM analysis                 │
│   "suggested_hook": "{{ $json.suggested_hook }}",                                   │
│   "status": "pending"                                                               │
│ }                                                                                   │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ NoOp (Success)  │
│ End of workflow │
└─────────────────┘
```

**Apify Actor Configurations:**

TikTok Scraper:
```json
{
  "hashtags": ["realestate", "homebuying", "realtor", "housingmarket"],
  "resultsPerPage": 20,
  "shouldDownloadVideos": false,
  "shouldDownloadCovers": false
}
```

Reddit Scraper:
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

---

### Workflow 2: AUTO TRIGGER (Pipeline Orchestrator)

**Purpose:** Poll for approved content and trigger script generation. Also provides webhook for UI-triggered processing.

**Triggers:**
1. Schedule (every 15 minutes)
2. Webhook (POST /webhook/trigger-pipeline) - for UI "Start Processing" button

**Workflow File:** `workflows/02-auto-trigger.json`

```
┌─────────────────┐     ┌─────────────────────────────────────────┐
│ Schedule Trigger │     │ Webhook Trigger                         │
│ Cron: */15 * * * │     │ Path: /webhook/trigger-pipeline         │
│ (every 15 min)  │     │ Method: POST                            │
└────────┬────────┘     │ Auth: Header Auth or None               │
         │              │ Body: { "content_idea_id": 123 } (opt)  │
         │              └────────────────┬────────────────────────┘
         │                               │
         ▼                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ Merge Node                                                      │
│ Mode: Append                                                    │
│ Combines both trigger sources                                   │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ IF Node: Check if specific ID provided                                              │
│                                                                                     │
│ Condition: {{ $json.body?.content_idea_id }} exists                                 │
│                                                                                     │
│ TRUE  → Use provided ID                                                             │
│ FALSE → Query for next approved item                                                │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ├──── TRUE ────┐
         │              ▼
         │    ┌─────────────────────────────────────────────────────────────────────┐
         │    │ HTTP Request: Get Specific Content Idea                             │
         │    │                                                                     │
         │    │ URL: http://backend:8000/api/content-ideas/{{ $json.content_idea_id }}
         │    │ Method: GET                                                         │
         │    └─────────────────────────────────────────────────────────────────────┘
         │              │
         │              ▼
         │    ┌─────────────────────────────────────────────────────────────────────┐
         │    │ IF: Status is 'approved'                                            │
         │    │ Condition: {{ $json.status }} == 'approved'                         │
         │    └─────────────────────────────────────────────────────────────────────┘
         │              │
         ├──────────────┘
         │
         ├──── FALSE ───┐
         │              ▼
         │    ┌─────────────────────────────────────────────────────────────────────┐
         │    │ HTTP Request: Find Next Approved Content                            │
         │    │                                                                     │
         │    │ URL: http://backend:8000/api/content-ideas?status=approved&limit=1  │
         │    │ Method: GET                                                         │
         │    └─────────────────────────────────────────────────────────────────────┘
         │              │
         └──────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ IF Node: Has Approved Content                                                       │
│                                                                                     │
│ Condition: {{ $json.length > 0 || $json.id }}                                       │
│                                                                                     │
│ TRUE  → Continue processing                                                         │
│ FALSE → End workflow (nothing to process)                                           │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼ (TRUE only)
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: Update Status to 'script_generating'                                  │
│                                                                                     │
│ URL: http://backend:8000/api/content-ideas/{{ $json.id || $json[0].id }}            │
│ Method: PATCH                                                                       │
│ Body: { "status": "script_generating" }                                             │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Execute Workflow: 03-script-gen                                                     │
│                                                                                     │
│ Mode: Wait for completion                                                           │
│ Input Data:                                                                         │
│   - content_idea_id: {{ $json.id }}                                                 │
│   - original_text: {{ $json.original_text }}                                        │
│   - pillar: {{ $json.pillar }}                                                      │
│   - suggested_hook: {{ $json.suggested_hook }}                                      │
│   - source_platform: {{ $json.source_platform }}                                    │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ NoOp (Complete) │
└─────────────────┘
```

**UI Integration:**
The React frontend should call the webhook to manually trigger processing:
```javascript
// In frontend, add a "Process Now" button
const triggerPipeline = async (contentIdeaId) => {
  await fetch('http://100.83.153.43:5678/webhook/trigger-pipeline', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content_idea_id: contentIdeaId })
  });
};
```

---

### Workflow 3: SCRIPT GENERATION

**Purpose:** Generate viral scripts using Grok 4.1, with platform-specific captions

**Trigger:** Called by Workflow 2 (Execute Workflow node)

**Workflow File:** `workflows/03-script-gen.json`

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Execute Workflow Trigger                                                            │
│                                                                                     │
│ Receives:                                                                           │
│   - content_idea_id: number                                                         │
│   - original_text: string                                                           │
│   - pillar: string                                                                  │
│   - suggested_hook: string                                                          │
│   - source_platform: string                                                         │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: OpenRouter (Grok 4.1) - Generate Script                               │
│                                                                                     │
│ URL: https://openrouter.ai/api/v1/chat/completions                                  │
│ Method: POST                                                                        │
│ Headers:                                                                            │
│   Authorization: Bearer {{ $env.OPENROUTER_API_KEY }}                               │
│   Content-Type: application/json                                                    │
│                                                                                     │
│ Body:                                                                               │
│ {                                                                                   │
│   "model": "x-ai/grok-4-1106",                                                      │
│   "messages": [                                                                     │
│     {                                                                               │
│       "role": "system",                                                             │
│       "content": "You are a viral content scriptwriter for a real estate agent.    │
│                                                                                     │
│ BRAND VOICE:                                                                        │
│ - Warm, approachable, trustworthy                                                   │
│ - Faith and family-oriented                                                         │
│ - Expert but not condescending                                                      │
│ - Local market authority in [CITY]                                                  │
│                                                                                     │
│ CONTENT PILLAR: {{ $json.pillar }}                                                  │
│ - market_intelligence: Data, trends, insights, market analysis                      │
│ - educational_tips: Practical buying/selling guidance, how-tos                      │
│ - lifestyle_local: Community highlights, local events, quality of life             │
│ - brand_humanization: Personal stories, faith, family, behind-the-scenes           │
│                                                                                     │
│ OUTPUT FORMAT (JSON):                                                               │
│ {                                                                                   │
│   \"hook\": \"First 3 seconds - attention grabber (15-20 words max)\",              │
│   \"body\": \"Main content - 20-40 seconds worth of speech\",                       │
│   \"cta\": \"Call to action - 5-10 seconds\",                                       │
│   \"full_script\": \"Complete script combining hook + body + cta\",                 │
│   \"duration_estimate\": 45                                                         │
│ }                                                                                   │
│                                                                                     │
│ HOOK PATTERNS THAT WORK:                                                            │
│ - \"Here's what no one tells you about buying a home...\"                           │
│ - \"POV: You just found out [surprising fact]\"                                     │
│ - \"3 things I wish I knew before [action]\"                                        │
│ - \"Stop doing this when buying a home...\"                                         │
│ - \"The biggest mistake [buyers/sellers] make is...\"                               │
│                                                                                     │
│ Make the content feel authentic, not salesy. Speak TO the viewer, not AT them."    │
│     },                                                                              │
│     {                                                                               │
│       "role": "user",                                                               │
│       "content": "Create a viral script based on this content idea:\n\n            │
│ Original content: {{ $json.original_text }}\n                                       │
│ Suggested hook direction: {{ $json.suggested_hook }}\n                              │
│ Content pillar: {{ $json.pillar }}"                                                 │
│     }                                                                               │
│   ],                                                                                │
│   "temperature": 0.8,                                                               │
│   "max_tokens": 1500,                                                               │
│   "response_format": { "type": "json_object" }                                      │
│ }                                                                                   │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Code Node: Parse Script Response                                                    │
│                                                                                     │
│ const response = JSON.parse($input.first().json.choices[0].message.content);        │
│ return [{                                                                           │
│   json: {                                                                           │
│     content_idea_id: $('Execute Workflow Trigger').first().json.content_idea_id,    │
│     hook: response.hook,                                                            │
│     body: response.body,                                                            │
│     cta: response.cta,                                                              │
│     full_script: response.full_script,                                              │
│     duration_estimate: response.duration_estimate || 45                             │
│   }                                                                                 │
│ }];                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: OpenRouter (Grok 4.1) - Generate Platform Captions                    │
│                                                                                     │
│ URL: https://openrouter.ai/api/v1/chat/completions                                  │
│ Method: POST                                                                        │
│                                                                                     │
│ System Prompt:                                                                      │
│ "Generate social media captions for each platform based on this video script.      │
│                                                                                     │
│ OUTPUT FORMAT (JSON):                                                               │
│ {                                                                                   │
│   \"tiktok_caption\": \"Short, punchy, with 3-5 relevant hashtags (max 150 chars)\",│
│   \"ig_caption\": \"Engaging caption with emojis and 5-10 hashtags\",               │
│   \"yt_title\": \"SEO-optimized YouTube Shorts title (max 60 chars)\",              │
│   \"yt_description\": \"Full description with keywords and CTA\",                   │
│   \"linkedin_text\": \"Professional tone, value-focused, no hashtags\",             │
│   \"x_text\": \"Conversational, max 280 chars, 1-2 hashtags\",                      │
│   \"facebook_text\": \"Friendly, community-focused\",                               │
│   \"threads_text\": \"Casual, conversational, Instagram-style\"                     │
│ }"                                                                                  │
│                                                                                     │
│ User Prompt:                                                                        │
│ "Generate platform captions for:\n\nHook: {{ $json.hook }}\n                        │
│ Script: {{ $json.full_script }}"                                                    │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Code Node: Merge Script + Captions                                                  │
│                                                                                     │
│ const script = $('Code Node').first().json;                                         │
│ const captions = JSON.parse($input.first().json.choices[0].message.content);        │
│ return [{                                                                           │
│   json: {                                                                           │
│     ...script,                                                                      │
│     ...captions                                                                     │
│   }                                                                                 │
│ }];                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: Create Script Record                                                  │
│                                                                                     │
│ URL: http://backend:8000/api/scripts                                                │
│ Method: POST                                                                        │
│ Body:                                                                               │
│ {                                                                                   │
│   "content_idea_id": {{ $json.content_idea_id }},                                   │
│   "hook": "{{ $json.hook }}",                                                       │
│   "body": "{{ $json.body }}",                                                       │
│   "cta": "{{ $json.cta }}",                                                         │
│   "full_script": "{{ $json.full_script }}",                                         │
│   "duration_estimate": {{ $json.duration_estimate }},                               │
│   "tiktok_caption": "{{ $json.tiktok_caption }}",                                   │
│   "ig_caption": "{{ $json.ig_caption }}",                                           │
│   "yt_title": "{{ $json.yt_title }}",                                               │
│   "yt_description": "{{ $json.yt_description }}",                                   │
│   "linkedin_text": "{{ $json.linkedin_text }}",                                     │
│   "x_text": "{{ $json.x_text }}",                                                   │
│   "facebook_text": "{{ $json.facebook_text }}",                                     │
│   "threads_text": "{{ $json.threads_text }}",                                       │
│   "status": "script_ready"                                                          │
│ }                                                                                   │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: Update Content Idea Status                                            │
│                                                                                     │
│ URL: http://backend:8000/api/content-ideas/{{ $json.content_idea_id }}              │
│ Method: PATCH                                                                       │
│ Body: { "status": "script_ready" }                                                  │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Set Node: Prepare Output                                                            │
│                                                                                     │
│ Output: { script_id: {{ $json.id }}, content_idea_id: {{ $json.content_idea_id }} } │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Execute Workflow: 04-get-video                                                      │
│                                                                                     │
│ Pass: script_id, content_idea_id                                                    │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Workflow 4: GET VIDEO (Background Video Acquisition)

**Purpose:** Download or find appropriate background video for the content

**Trigger:** Called by Workflow 3

**Workflow File:** `workflows/04-get-video.json`

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Execute Workflow Trigger                                                            │
│                                                                                     │
│ Receives: script_id, content_idea_id                                                │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: Get Content Idea Details                                              │
│                                                                                     │
│ URL: http://backend:8000/api/content-ideas/{{ $json.content_idea_id }}              │
│ Method: GET                                                                         │
│                                                                                     │
│ Need: source_url, source_platform, pillar                                           │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: Get Script Details                                                    │
│                                                                                     │
│ URL: http://backend:8000/api/scripts/{{ $json.script_id }}                          │
│ Method: GET                                                                         │
│                                                                                     │
│ Need: hook, body (for keyword extraction)                                           │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Switch Node: Source Platform                                                        │
│                                                                                     │
│ Routing based on: {{ $json.source_platform }}                                       │
│                                                                                     │
│ - "tiktok"  → Branch A: Download TikTok video via Apify                             │
│ - "youtube" → Branch B: Download YouTube video via Apify                            │
│ - "reddit"  → Branch C: Use Pexels stock video                                      │
│ - "twitter" → Branch C: Use Pexels stock video                                      │
│ - default   → Branch C: Use Pexels stock video                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ├──── Branch A (TikTok) ────┐
         │                           ▼
         │    ┌─────────────────────────────────────────────────────────────────────┐
         │    │ HTTP Request: Apify TikTok Downloader                               │
         │    │                                                                     │
         │    │ URL: https://api.apify.com/v2/acts/clockworks~tiktok-scraper/runs   │
         │    │ Method: POST                                                        │
         │    │ Body: {                                                             │
         │    │   "postURLs": ["{{ $json.source_url }}"],                           │
         │    │   "shouldDownloadVideos": true,                                     │
         │    │   "shouldDownloadCovers": false                                     │
         │    │ }                                                                   │
         │    │                                                                     │
         │    │ Wait for completion, get video URL                                  │
         │    └─────────────────────────────────────────────────────────────────────┘
         │                           │
         │                           ▼
         │    ┌─────────────────────────────────────────────────────────────────────┐
         │    │ HTTP Request: Download Video                                        │
         │    │                                                                     │
         │    │ URL: {{ $json.videoUrl }}                                           │
         │    │ Response: Binary                                                    │
         │    └─────────────────────────────────────────────────────────────────────┘
         │                           │
         │                           ▼
         ├───────────────────────────┤
         │                           │
         ├──── Branch B (YouTube) ───┤
         │                           │
         │    (Similar to TikTok, use YouTube downloader)                           │
         │                           │
         ├───────────────────────────┤
         │                           │
         ├──── Branch C (Pexels) ────┐
         │                           ▼
         │    ┌─────────────────────────────────────────────────────────────────────┐
         │    │ HTTP Request: OpenRouter - Extract Keywords                         │
         │    │                                                                     │
         │    │ Prompt: "Extract 3-5 keywords for stock video search from:          │
         │    │ {{ $json.hook }} {{ $json.body }}                                   │
         │    │ Focus on visual concepts. Return JSON: { keywords: ['...'] }"       │
         │    └─────────────────────────────────────────────────────────────────────┘
         │                           │
         │                           ▼
         │    ┌─────────────────────────────────────────────────────────────────────┐
         │    │ HTTP Request: Pexels Video Search                                   │
         │    │                                                                     │
         │    │ URL: https://api.pexels.com/videos/search                           │
         │    │ Method: GET                                                         │
         │    │ Headers: Authorization: {{ $env.PEXELS_API_KEY }}                   │
         │    │ Query:                                                              │
         │    │   query: {{ $json.keywords.join(' ') }}                             │
         │    │   orientation: portrait                                             │
         │    │   size: medium                                                      │
         │    │   per_page: 5                                                       │
         │    └─────────────────────────────────────────────────────────────────────┘
         │                           │
         │                           ▼
         │    ┌─────────────────────────────────────────────────────────────────────┐
         │    │ Code Node: Select Best Video                                        │
         │    │                                                                     │
         │    │ // Select HD portrait video                                         │
         │    │ const videos = $json.videos;                                        │
         │    │ const video = videos.find(v =>                                      │
         │    │   v.video_files.some(f => f.quality === 'hd' && f.height > f.width) │
         │    │ );                                                                  │
         │    │ const file = video.video_files.find(f =>                            │
         │    │   f.quality === 'hd' && f.height > f.width                          │
         │    │ );                                                                  │
         │    │ return [{ json: { videoUrl: file.link } }];                         │
         │    └─────────────────────────────────────────────────────────────────────┘
         │                           │
         │                           ▼
         │    ┌─────────────────────────────────────────────────────────────────────┐
         │    │ HTTP Request: Download Pexels Video                                 │
         │    │                                                                     │
         │    │ URL: {{ $json.videoUrl }}                                           │
         │    │ Response: Binary                                                    │
         │    └─────────────────────────────────────────────────────────────────────┘
         │                           │
         └───────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Merge Node: Combine All Branches                                                    │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Write Binary File                                                                   │
│                                                                                     │
│ Path: /home/node/assets/videos/{{ $json.content_idea_id }}_background.mp4           │
│ Data: Binary from HTTP Request                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: Create Asset Record                                                   │
│                                                                                     │
│ URL: http://backend:8000/api/assets                                                 │
│ Method: POST                                                                        │
│ Body: {                                                                             │
│   "script_id": {{ $json.script_id }},                                               │
│   "background_video_path": "/home/node/assets/videos/{{ $json.content_idea_id }}_background.mp4",
│   "status": "pending"                                                               │
│ }                                                                                   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Execute Workflow: 05-create-voice                                                   │
│                                                                                     │
│ Pass: asset_id, script_id, content_idea_id                                          │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Workflow 5: CREATE VOICE (ElevenLabs TTS + Whisper)

**Purpose:** Generate voiceover audio and word-level timestamps for captions

**Trigger:** Called by Workflow 4

**Workflow File:** `workflows/05-create-voice.json`

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Execute Workflow Trigger                                                            │
│                                                                                     │
│ Receives: asset_id, script_id, content_idea_id                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: Get Script                                                            │
│                                                                                     │
│ URL: http://backend:8000/api/scripts/{{ $json.script_id }}                          │
│ Method: GET                                                                         │
│                                                                                     │
│ Need: full_script                                                                   │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: ElevenLabs Text-to-Speech                                             │
│                                                                                     │
│ URL: https://api.elevenlabs.io/v1/text-to-speech/{{ $env.ELEVENLABS_VOICE_ID }}     │
│ Method: POST                                                                        │
│ Headers:                                                                            │
│   xi-api-key: {{ $env.ELEVENLABS_API_KEY }}                                         │
│   Content-Type: application/json                                                    │
│   Accept: audio/mpeg                                                                │
│                                                                                     │
│ Body: {                                                                             │
│   "text": "{{ $json.full_script }}",                                                │
│   "model_id": "eleven_monolingual_v1",                                              │
│   "voice_settings": {                                                               │
│     "stability": 0.5,                                                               │
│     "similarity_boost": 0.75,                                                       │
│     "style": 0.5,                                                                   │
│     "use_speaker_boost": true                                                       │
│   }                                                                                 │
│ }                                                                                   │
│                                                                                     │
│ Response Type: Binary (audio/mpeg)                                                  │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Write Binary File: Save Audio                                                       │
│                                                                                     │
│ Path: /home/node/assets/audio/{{ $json.content_idea_id }}_voiceover.mp3             │
│ Data: Binary from ElevenLabs                                                        │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Wait Node: 3 seconds                                                                │
│                                                                                     │
│ Ensure file is fully written before reading                                         │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Read Binary File                                                                    │
│                                                                                     │
│ Path: /home/node/assets/audio/{{ $json.content_idea_id }}_voiceover.mp3             │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: OpenAI Whisper - Transcribe with Timestamps                           │
│                                                                                     │
│ URL: https://api.openai.com/v1/audio/transcriptions                                 │
│ Method: POST                                                                        │
│ Headers:                                                                            │
│   Authorization: Bearer {{ $env.OPENAI_API_KEY }}                                   │
│                                                                                     │
│ Body (Form Data):                                                                   │
│   file: [Binary audio file]                                                         │
│   model: whisper-1                                                                  │
│   response_format: verbose_json                                                     │
│   timestamp_granularities: ["word"]                                                 │
│                                                                                     │
│ Response includes word-level timestamps:                                            │
│ { "words": [{ "word": "Hello", "start": 0.0, "end": 0.5 }, ...] }                   │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Code Node: Generate SRT from Whisper Response                                       │
│                                                                                     │
│ // Convert word timestamps to SRT format                                            │
│ const words = $json.words;                                                          │
│ let srt = '';                                                                       │
│ let index = 1;                                                                      │
│ let chunk = [];                                                                     │
│ let chunkStart = 0;                                                                 │
│                                                                                     │
│ for (let i = 0; i < words.length; i++) {                                            │
│   const word = words[i];                                                            │
│   if (chunk.length === 0) chunkStart = word.start;                                  │
│   chunk.push(word.word);                                                            │
│                                                                                     │
│   // Create subtitle every 5-7 words or at natural pauses                           │
│   if (chunk.length >= 6 || (words[i+1] && words[i+1].start - word.end > 0.5)) {     │
│     const startTime = formatSrtTime(chunkStart);                                    │
│     const endTime = formatSrtTime(word.end);                                        │
│     srt += `${index}\n${startTime} --> ${endTime}\n${chunk.join(' ')}\n\n`;         │
│     index++;                                                                        │
│     chunk = [];                                                                     │
│   }                                                                                 │
│ }                                                                                   │
│                                                                                     │
│ // Handle remaining words                                                           │
│ if (chunk.length > 0) {                                                             │
│   const startTime = formatSrtTime(chunkStart);                                      │
│   const endTime = formatSrtTime(words[words.length-1].end);                         │
│   srt += `${index}\n${startTime} --> ${endTime}\n${chunk.join(' ')}\n\n`;           │
│ }                                                                                   │
│                                                                                     │
│ function formatSrtTime(seconds) {                                                   │
│   const h = Math.floor(seconds / 3600);                                             │
│   const m = Math.floor((seconds % 3600) / 60);                                      │
│   const s = Math.floor(seconds % 60);                                               │
│   const ms = Math.round((seconds % 1) * 1000);                                      │
│   return `${pad(h)}:${pad(m)}:${pad(s)},${pad(ms, 3)}`;                             │
│ }                                                                                   │
│                                                                                     │
│ function pad(n, len=2) { return n.toString().padStart(len, '0'); }                  │
│                                                                                     │
│ return [{ json: { srt, duration: words[words.length-1].end } }];                    │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Write File: Save SRT                                                                │
│                                                                                     │
│ Path: /home/node/assets/captions/{{ $json.content_idea_id }}.srt                    │
│ Content: {{ $json.srt }}                                                            │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: Update Asset Record                                                   │
│                                                                                     │
│ URL: http://backend:8000/api/assets/{{ $json.asset_id }}                            │
│ Method: PATCH                                                                       │
│ Body: {                                                                             │
│   "voiceover_path": "/home/node/assets/audio/{{ $json.content_idea_id }}_voiceover.mp3",
│   "voiceover_duration": {{ $json.duration }},                                       │
│   "srt_path": "/home/node/assets/captions/{{ $json.content_idea_id }}.srt",         │
│   "status": "voice_ready"                                                           │
│ }                                                                                   │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Execute Workflow: 06-create-avatar                                                  │
│                                                                                     │
│ Pass: asset_id, script_id, content_idea_id                                          │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Workflow 6: CREATE AVATAR (HeyGen)

**Purpose:** Generate AI avatar video with lip-synced speech

**Trigger:** Called by Workflow 5

**Workflow File:** `workflows/06-create-avatar.json`

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Execute Workflow Trigger                                                            │
│                                                                                     │
│ Receives: asset_id, script_id, content_idea_id                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: Get Asset Record                                                      │
│                                                                                     │
│ URL: http://backend:8000/api/assets/{{ $json.asset_id }}                            │
│ Method: GET                                                                         │
│                                                                                     │
│ Need: voiceover_path                                                                │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Read Binary File: Voiceover                                                         │
│                                                                                     │
│ Path: {{ $json.voiceover_path }}                                                    │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: Upload Audio to HeyGen (or use public URL)                            │
│                                                                                     │
│ Option A: Upload to HeyGen                                                          │
│ URL: https://api.heygen.com/v1/asset                                                │
│ Method: POST                                                                        │
│ Headers: X-Api-Key: {{ $env.HEYGEN_API_KEY }}                                       │
│ Body: Form-data with audio file                                                     │
│                                                                                     │
│ Option B: Use publicly accessible URL                                               │
│ (Would need to expose assets via nginx or similar)                                  │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: HeyGen - Create Avatar Video                                          │
│                                                                                     │
│ URL: https://api.heygen.com/v2/video/generate                                       │
│ Method: POST                                                                        │
│ Headers:                                                                            │
│   X-Api-Key: {{ $env.HEYGEN_API_KEY }}                                              │
│   Content-Type: application/json                                                    │
│                                                                                     │
│ Body: {                                                                             │
│   "video_inputs": [{                                                                │
│     "character": {                                                                  │
│       "type": "avatar",                                                             │
│       "avatar_id": "{{ $env.HEYGEN_AVATAR_ID }}",                                   │
│       "avatar_style": "normal"                                                      │
│     },                                                                              │
│     "voice": {                                                                      │
│       "type": "audio",                                                              │
│       "audio_url": "{{ $json.audio_url }}"                                          │
│     },                                                                              │
│     "background": {                                                                 │
│       "type": "color",                                                              │
│       "value": "#00FF00"    // Green screen for chroma keying                       │
│     }                                                                               │
│   }],                                                                               │
│   "dimension": {                                                                    │
│     "width": 1080,                                                                  │
│     "height": 1920          // Vertical/portrait for social media                   │
│   },                                                                                │
│   "test": false                                                                     │
│ }                                                                                   │
│                                                                                     │
│ Response: { "video_id": "abc123" }                                                  │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Set Node: Store Video ID                                                            │
│                                                                                     │
│ video_id: {{ $json.video_id }}                                                      │
│ asset_id: (pass through)                                                            │
│ content_idea_id: (pass through)                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: Update Asset Status                                                   │
│                                                                                     │
│ URL: http://backend:8000/api/assets/{{ $json.asset_id }}                            │
│ Method: PATCH                                                                       │
│ Body: { "status": "avatar_generating" }                                             │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Wait Node: Initial Wait                                                             │
│                                                                                     │
│ Duration: 60 seconds                                                                │
│ HeyGen typically takes 1-3 minutes for avatar generation                            │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Loop Node: Poll for Completion                                                      │
│                                                                                     │
│ Max Iterations: 20                                                                  │
│ (Total max wait: ~10 minutes with 30s intervals)                                    │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: Check Video Status                                                    │
│                                                                                     │
│ URL: https://api.heygen.com/v1/video_status.get?video_id={{ $json.video_id }}       │
│ Method: GET                                                                         │
│ Headers: X-Api-Key: {{ $env.HEYGEN_API_KEY }}                                       │
│                                                                                     │
│ Response: {                                                                         │
│   "status": "completed" | "processing" | "failed",                                  │
│   "video_url": "https://..."                                                        │
│ }                                                                                   │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ IF Node: Check Status                                                               │
│                                                                                     │
│ Condition: {{ $json.data.status }}                                                  │
│                                                                                     │
│ "completed" → Continue to download                                                  │
│ "processing" → Wait 30 seconds, loop again                                          │
│ "failed" → Error handling branch                                                    │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ├──── "processing" ───┐
         │                     ▼
         │    ┌─────────────────────────────────────────────────────────────────────┐
         │    │ Wait Node: 30 seconds                                               │
         │    │                                                                     │
         │    │ Then loop back to "Check Video Status"                              │
         │    └─────────────────────────────────────────────────────────────────────┘
         │
         ├──── "failed" ───────┐
         │                     ▼
         │    ┌─────────────────────────────────────────────────────────────────────┐
         │    │ HTTP Request: Update Asset with Error                               │
         │    │                                                                     │
         │    │ URL: http://backend:8000/api/assets/{{ $json.asset_id }}            │
         │    │ Body: { "status": "error", "error_message": "HeyGen failed" }       │
         │    │                                                                     │
         │    │ Stop Node: End workflow with error                                  │
         │    └─────────────────────────────────────────────────────────────────────┘
         │
         ▼ ("completed")
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: Download Avatar Video                                                 │
│                                                                                     │
│ URL: {{ $json.data.video_url }}                                                     │
│ Method: GET                                                                         │
│ Response Type: Binary                                                               │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Write Binary File: Save Avatar Video                                                │
│                                                                                     │
│ Path: /home/node/assets/avatar/{{ $json.content_idea_id }}_avatar.mp4               │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: Update Asset Record                                                   │
│                                                                                     │
│ URL: http://backend:8000/api/assets/{{ $json.asset_id }}                            │
│ Method: PATCH                                                                       │
│ Body: {                                                                             │
│   "avatar_video_path": "/home/node/assets/avatar/{{ $json.content_idea_id }}_avatar.mp4",
│   "status": "avatar_ready"                                                          │
│ }                                                                                   │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Execute Workflow: 07-combine-vids                                                   │
│                                                                                     │
│ Pass: asset_id, script_id, content_idea_id                                          │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Workflow 7: COMBINE VIDS (FFmpeg Assembly)

**Purpose:** Composite avatar over background video using FFmpeg

**Trigger:** Called by Workflow 6

**Workflow File:** `workflows/07-combine-vids.json`

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Execute Workflow Trigger                                                            │
│                                                                                     │
│ Receives: asset_id, script_id, content_idea_id                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: Get Asset Record                                                      │
│                                                                                     │
│ URL: http://backend:8000/api/assets/{{ $json.asset_id }}                            │
│ Method: GET                                                                         │
│                                                                                     │
│ Need: background_video_path, avatar_video_path, voiceover_duration                  │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: Update Status to 'assembling'                                         │
│                                                                                     │
│ URL: http://backend:8000/api/assets/{{ $json.asset_id }}                            │
│ Method: PATCH                                                                       │
│ Body: { "status": "assembling" }                                                    │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: OpenRouter (Grok 4.1) - Generate FFmpeg Command                       │
│                                                                                     │
│ URL: https://openrouter.ai/api/v1/chat/completions                                  │
│ Headers: Authorization: Bearer {{ $env.OPENROUTER_API_KEY }}                        │
│                                                                                     │
│ System Prompt:                                                                      │
│ "You are an FFmpeg expert. Generate a single FFmpeg command for video assembly.    │
│                                                                                     │
│ INPUTS:                                                                             │
│ - Background video: {{ $json.background_video_path }}                               │
│ - Avatar video: {{ $json.avatar_video_path }} (has green screen #00FF00)            │
│ - Duration: {{ $json.voiceover_duration }} seconds                                  │
│                                                                                     │
│ REQUIREMENTS:                                                                       │
│ 1. Scale/crop background to exactly 1080x1920 (portrait)                            │
│ 2. Loop background if shorter than audio duration                                   │
│ 3. Chroma key the avatar video (remove green screen)                                │
│ 4. Scale avatar to 1080 width, maintain aspect ratio                                │
│ 5. Position avatar at bottom center (overlay at y=H-h)                              │
│ 6. Use audio from avatar video (contains the voice)                                 │
│ 7. Output: H.264 video, AAC audio, 30fps                                            │
│ 8. Output path: /home/node/assets/output/{{ content_idea_id }}_combined.mp4         │
│                                                                                     │
│ Output ONLY the FFmpeg command, no explanation."                                    │
│                                                                                     │
│ User Prompt:                                                                        │
│ "Generate FFmpeg command for:                                                       │
│ Background: {{ $json.background_video_path }}                                       │
│ Avatar: {{ $json.avatar_video_path }}                                               │
│ Duration: {{ $json.voiceover_duration }}s                                           │
│ Output: /home/node/assets/output/{{ $json.content_idea_id }}_combined.mp4"          │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Code Node: Extract Command                                                          │
│                                                                                     │
│ const response = $json.choices[0].message.content;                                  │
│ // Clean up the command (remove markdown code blocks if present)                    │
│ let cmd = response.replace(/```bash\n?/g, '').replace(/```\n?/g, '').trim();        │
│ return [{ json: { ffmpeg_command: cmd } }];                                         │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Execute Command Node: Run FFmpeg                                                    │
│                                                                                     │
│ Command: {{ $json.ffmpeg_command }}                                                 │
│ Working Directory: /home/node/assets                                                │
│ Timeout: 300000 (5 minutes)                                                         │
│                                                                                     │
│ IMPORTANT: n8n container must have FFmpeg installed (done in Dockerfile)            │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ IF Node: Check FFmpeg Success                                                       │
│                                                                                     │
│ Condition: Exit code == 0                                                           │
│                                                                                     │
│ TRUE  → Continue                                                                    │
│ FALSE → Error handling                                                              │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ├──── FALSE (Error) ───┐
         │                      ▼
         │    ┌─────────────────────────────────────────────────────────────────────┐
         │    │ HTTP Request: Update Asset with Error                               │
         │    │                                                                     │
         │    │ Body: {                                                             │
         │    │   "status": "error",                                                │
         │    │   "error_message": "FFmpeg combine failed: {{ $json.stderr }}"      │
         │    │ }                                                                   │
         │    └─────────────────────────────────────────────────────────────────────┘
         │
         ▼ (TRUE - Success)
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: Update Asset Record                                                   │
│                                                                                     │
│ URL: http://backend:8000/api/assets/{{ $json.asset_id }}                            │
│ Method: PATCH                                                                       │
│ Body: {                                                                             │
│   "combined_video_path": "/home/node/assets/output/{{ $json.content_idea_id }}_combined.mp4",
│   "status": "captioning"                                                            │
│ }                                                                                   │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Execute Workflow: 08-caption                                                        │
│                                                                                     │
│ Pass: asset_id, script_id, content_idea_id                                          │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

**Example FFmpeg Command Generated:**
```bash
ffmpeg -stream_loop -1 -i /home/node/assets/videos/123_background.mp4 \
  -i /home/node/assets/avatar/123_avatar.mp4 \
  -filter_complex "
    [0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1[bg];
    [1:v]colorkey=0x00FF00:0.3:0.2,scale=1080:-1[avatar];
    [bg][avatar]overlay=(W-w)/2:H-h[outv]
  " \
  -map "[outv]" -map 1:a \
  -c:v libx264 -preset fast -crf 23 \
  -c:a aac -b:a 128k \
  -t 45 -r 30 \
  -y /home/node/assets/output/123_combined.mp4
```

---

### Workflow 8: CAPTION (Subtitle Burn-in)

**Purpose:** Burn styled captions into the video

**Trigger:** Called by Workflow 7

**Workflow File:** `workflows/08-caption.json`

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Execute Workflow Trigger                                                            │
│                                                                                     │
│ Receives: asset_id, script_id, content_idea_id                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: Get Asset Record                                                      │
│                                                                                     │
│ URL: http://backend:8000/api/assets/{{ $json.asset_id }}                            │
│ Method: GET                                                                         │
│                                                                                     │
│ Need: combined_video_path, srt_path                                                 │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Read File: Get SRT Content                                                          │
│                                                                                     │
│ Path: {{ $json.srt_path }}                                                          │
│ Output: Text content of SRT file                                                    │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Code Node: Convert SRT to ASS (Styled Subtitles)                                    │
│                                                                                     │
│ // ASS format allows for advanced styling                                           │
│ const srt = $json.data;                                                             │
│                                                                                     │
│ const assHeader = `[Script Info]                                                    │
│ ScriptType: v4.00+                                                                  │
│ PlayResX: 1080                                                                      │
│ PlayResY: 1920                                                                      │
│ WrapStyle: 0                                                                        │
│                                                                                     │
│ [V4+ Styles]                                                                        │
│ Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
│ Style: Default,Montserrat,72,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,4,0,2,20,20,350,1
│                                                                                     │
│ [Events]                                                                            │
│ Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text`;   │
│                                                                                     │
│ // Parse SRT and convert to ASS events                                              │
│ const blocks = srt.trim().split('\n\n');                                            │
│ const events = blocks.map(block => {                                                │
│   const lines = block.split('\n');                                                  │
│   const timeMatch = lines[1].match(/(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})/);
│   if (!timeMatch) return null;                                                      │
│   const start = `${timeMatch[1]}:${timeMatch[2]}:${timeMatch[3]}.${timeMatch[4].slice(0,2)}`;
│   const end = `${timeMatch[5]}:${timeMatch[6]}:${timeMatch[7]}.${timeMatch[8].slice(0,2)}`;
│   const text = lines.slice(2).join('\\N');  // ASS line break                       │
│   return `Dialogue: 0,${start},${end},Default,,0,0,0,,${text}`;                     │
│ }).filter(Boolean).join('\n');                                                      │
│                                                                                     │
│ return [{ json: { ass: assHeader + '\n' + events } }];                              │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Write File: Save ASS                                                                │
│                                                                                     │
│ Path: /home/node/assets/captions/{{ $json.content_idea_id }}.ass                    │
│ Content: {{ $json.ass }}                                                            │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Execute Command Node: FFmpeg Subtitle Burn-in                                       │
│                                                                                     │
│ Command:                                                                            │
│ ffmpeg -i {{ $json.combined_video_path }} \                                         │
│   -vf "ass=/home/node/assets/captions/{{ $json.content_idea_id }}.ass:fontsdir=/home/node/assets/fonts" \
│   -c:v libx264 -preset fast -crf 23 \                                               │
│   -c:a copy \                                                                       │
│   -y /home/node/assets/output/{{ $json.content_idea_id }}_final.mp4                 │
│                                                                                     │
│ Working Directory: /home/node/assets                                                │
│ Timeout: 300000 (5 minutes)                                                         │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ IF Node: Check FFmpeg Success                                                       │
│                                                                                     │
│ Condition: Exit code == 0                                                           │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼ (Success)
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: Update Asset Record                                                   │
│                                                                                     │
│ URL: http://backend:8000/api/assets/{{ $json.asset_id }}                            │
│ Method: PATCH                                                                       │
│ Body: {                                                                             │
│   "ass_path": "/home/node/assets/captions/{{ $json.content_idea_id }}.ass",         │
│   "final_video_path": "/home/node/assets/output/{{ $json.content_idea_id }}_final.mp4",
│   "status": "ready_to_publish"                                                      │
│ }                                                                                   │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ END: Video ready for publishing                                                     │
│                                                                                     │
│ The video is now complete and marked "ready_to_publish"                             │
│ Workflow 9 (Publish) will pick it up on its next scheduled run                      │
│ Or user can trigger publishing manually from UI                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Workflow 9: PUBLISH (Multi-Platform Distribution)

**Purpose:** Publish completed videos to all social media platforms via Blotato

**Trigger:**
1. Schedule (hourly check for ready_to_publish)
2. Webhook (manual trigger from UI)

**Workflow File:** `workflows/09-publish.json`

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Schedule Trigger                           │ Webhook Trigger                        │
│ Cron: 0 * * * * (every hour)               │ Path: /webhook/publish-video           │
│                                            │ Body: { "asset_id": 123 }              │
└────────────────────────────────────────────┴────────────────────────────────────────┘
         │                                                    │
         └───────────────────────┬────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ IF Node: Manual trigger with specific ID?                                           │
│                                                                                     │
│ Condition: {{ $json.body?.asset_id }} exists                                        │
│                                                                                     │
│ TRUE  → Use provided asset_id                                                       │
│ FALSE → Query for next ready_to_publish                                             │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: Get Ready Asset                                                       │
│                                                                                     │
│ URL: http://backend:8000/api/assets?status=ready_to_publish&limit=1                 │
│ (or specific ID if provided)                                                        │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ IF Node: Has Asset to Publish                                                       │
│                                                                                     │
│ Condition: {{ $json.length > 0 || $json.id }}                                       │
│                                                                                     │
│ FALSE → End workflow (nothing to publish)                                           │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼ (TRUE)
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: Get Script (for captions)                                             │
│                                                                                     │
│ URL: http://backend:8000/api/scripts/{{ $json.script_id }}                          │
│                                                                                     │
│ Need: tiktok_caption, ig_caption, yt_title, yt_description, linkedin_text,          │
│       x_text, facebook_text, threads_text                                           │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: Update Asset Status                                                   │
│                                                                                     │
│ Body: { "status": "publishing" }                                                    │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Read Binary File: Final Video                                                       │
│                                                                                     │
│ Path: {{ $json.final_video_path }}                                                  │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: Blotato - Multi-Platform Post                                         │
│                                                                                     │
│ URL: https://api.blotato.com/v1/posts                                               │
│ Method: POST                                                                        │
│ Headers:                                                                            │
│   Authorization: Bearer {{ $env.BLOTATO_API_KEY }}                                  │
│   Content-Type: multipart/form-data                                                 │
│                                                                                     │
│ Body:                                                                               │
│ {                                                                                   │
│   "media": [Binary video file],                                                     │
│   "platforms": ["tiktok", "instagram", "youtube", "linkedin", "twitter",            │
│                 "facebook", "threads"],                                             │
│   "platform_settings": {                                                            │
│     "tiktok": {                                                                     │
│       "caption": "{{ $json.tiktok_caption }}"                                       │
│     },                                                                              │
│     "instagram": {                                                                  │
│       "caption": "{{ $json.ig_caption }}",                                          │
│       "share_to_feed": true,                                                        │
│       "share_to_reels": true                                                        │
│     },                                                                              │
│     "youtube": {                                                                    │
│       "title": "{{ $json.yt_title }}",                                              │
│       "description": "{{ $json.yt_description }}",                                  │
│       "visibility": "public",                                                       │
│       "shorts": true                                                                │
│     },                                                                              │
│     "linkedin": {                                                                   │
│       "text": "{{ $json.linkedin_text }}"                                           │
│     },                                                                              │
│     "twitter": {                                                                    │
│       "text": "{{ $json.x_text }}"                                                  │
│     },                                                                              │
│     "facebook": {                                                                   │
│       "text": "{{ $json.facebook_text }}"                                           │
│     },                                                                              │
│     "threads": {                                                                    │
│       "text": "{{ $json.threads_text }}"                                            │
│     }                                                                               │
│   },                                                                                │
│   "schedule": "now"  // or ISO timestamp for scheduled posting                      │
│ }                                                                                   │
│                                                                                     │
│ Response: {                                                                         │
│   "post_id": "...",                                                                 │
│   "platforms": {                                                                    │
│     "tiktok": { "url": "...", "id": "..." },                                        │
│     "instagram": { "url": "...", "id": "..." },                                     │
│     ...                                                                             │
│   }                                                                                 │
│ }                                                                                   │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Code Node: Extract Platform URLs                                                    │
│                                                                                     │
│ const platforms = $json.platforms;                                                  │
│ return [{                                                                           │
│   json: {                                                                           │
│     tiktok_url: platforms.tiktok?.url,                                              │
│     tiktok_id: platforms.tiktok?.id,                                                │
│     ig_url: platforms.instagram?.url,                                               │
│     ig_id: platforms.instagram?.id,                                                 │
│     yt_url: platforms.youtube?.url,                                                 │
│     yt_id: platforms.youtube?.id,                                                   │
│     linkedin_url: platforms.linkedin?.url,                                          │
│     linkedin_id: platforms.linkedin?.id,                                            │
│     x_url: platforms.twitter?.url,                                                  │
│     x_id: platforms.twitter?.id,                                                    │
│     facebook_url: platforms.facebook?.url,                                          │
│     facebook_id: platforms.facebook?.id,                                            │
│     threads_url: platforms.threads?.url,                                            │
│     threads_id: platforms.threads?.id                                               │
│   }                                                                                 │
│ }];                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: Create Published Record                                               │
│                                                                                     │
│ URL: http://backend:8000/api/published                                              │
│ Method: POST                                                                        │
│ Body: {                                                                             │
│   "asset_id": {{ $json.asset_id }},                                                 │
│   "tiktok_url": "{{ $json.tiktok_url }}",                                           │
│   "tiktok_id": "{{ $json.tiktok_id }}",                                             │
│   "ig_url": "{{ $json.ig_url }}",                                                   │
│   "ig_id": "{{ $json.ig_id }}",                                                     │
│   "yt_url": "{{ $json.yt_url }}",                                                   │
│   "yt_id": "{{ $json.yt_id }}",                                                     │
│   "linkedin_url": "{{ $json.linkedin_url }}",                                       │
│   "linkedin_id": "{{ $json.linkedin_id }}",                                         │
│   "x_url": "{{ $json.x_url }}",                                                     │
│   "x_id": "{{ $json.x_id }}",                                                       │
│   "facebook_url": "{{ $json.facebook_url }}",                                       │
│   "facebook_id": "{{ $json.facebook_id }}",                                         │
│   "threads_url": "{{ $json.threads_url }}",                                         │
│   "threads_id": "{{ $json.threads_id }}"                                            │
│ }                                                                                   │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: Update Asset Status                                                   │
│                                                                                     │
│ URL: http://backend:8000/api/assets/{{ $json.asset_id }}                            │
│ Method: PATCH                                                                       │
│ Body: { "status": "published" }                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: Update Content Idea Status                                            │
│                                                                                     │
│ URL: http://backend:8000/api/content-ideas/{{ $json.content_idea_id }}              │
│ Method: PATCH                                                                       │
│ Body: { "status": "published" }                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ (Optional) Send Notification                                                        │
│                                                                                     │
│ Slack/Email notification with published URLs                                        │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Workflow 10: ANALYTICS (Performance Tracking)

**Purpose:** Collect analytics from all platforms weekly

**Trigger:** Schedule (weekly, Sunday at midnight)

**Workflow File:** `workflows/10-analytics.json`

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Schedule Trigger                                                                    │
│                                                                                     │
│ Cron: 0 0 * * 0 (Sunday at midnight)                                                │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: Get Recent Published Content                                          │
│                                                                                     │
│ URL: http://backend:8000/api/published?limit=50                                     │
│ Method: GET                                                                         │
│                                                                                     │
│ Get published items from the last 7 days                                            │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Split In Batches                                                                    │
│                                                                                     │
│ Process each published item                                                         │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ PARALLEL: Fetch Analytics from Each Platform                                        │
│                                                                                     │
│ Note: This requires platform-specific API access or Blotato analytics               │
├─────────────────┬─────────────────┬─────────────────┬─────────────────┐             │
│ TikTok API      │ Instagram API   │ YouTube API     │ LinkedIn API    │             │
│ (if available)  │ (Insights)      │ (Analytics)     │ (Analytics)     │             │
│                 │                 │                 │                 │             │
│ Or use Blotato  │ Get: views,     │ Get: views,     │ Get: views,     │             │
│ analytics API   │ likes, comments │ likes, comments │ likes, comments │             │
│ for all         │ shares          │ watch time      │ shares          │             │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘             │
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Code Node: Calculate Engagement Rate                                                │
│                                                                                     │
│ // For each platform                                                                │
│ const engagementRate = (likes + comments + shares) / views;                         │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: Create Analytics Records                                              │
│                                                                                     │
│ URL: http://backend:8000/api/analytics                                              │
│ Method: POST                                                                        │
│ Body: {                                                                             │
│   "published_id": {{ $json.published_id }},                                         │
│   "platform": "{{ $json.platform }}",                                               │
│   "views": {{ $json.views }},                                                       │
│   "likes": {{ $json.likes }},                                                       │
│   "comments": {{ $json.comments }},                                                 │
│   "shares": {{ $json.shares }},                                                     │
│   "engagement_rate": {{ $json.engagement_rate }}                                    │
│ }                                                                                   │
│                                                                                     │
│ (Repeat for each platform)                                                          │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: OpenRouter - Generate Weekly Report                                   │
│                                                                                     │
│ Prompt: "Analyze this week's content performance and provide insights:              │
│ - Which pillar performed best?                                                      │
│ - Which platform has highest engagement?                                            │
│ - Best performing hooks?                                                            │
│ - Recommendations for next week                                                     │
│                                                                                     │
│ Data: {{ JSON.stringify($json.analytics) }}"                                        │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ (Optional) Send Weekly Report                                                       │
│                                                                                     │
│ Email or Slack notification with:                                                   │
│ - Top performing content                                                            │
│ - Aggregate stats                                                                   │
│ - AI recommendations                                                                │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 3: UI Integration Points

### React Frontend Additions

Add these features to the existing React frontend to control the pipeline:

**1. Dashboard Actions:**
```javascript
// Add to Dashboard.js
const triggerScrape = () => fetch('/webhook/trigger-scrape', { method: 'POST' });
const triggerPublish = () => fetch('/webhook/publish-video', { method: 'POST' });
```

**2. Content Ideas Page Actions:**
```javascript
// Process single idea immediately
const processIdea = (id) => fetch('/webhook/trigger-pipeline', {
  method: 'POST',
  body: JSON.stringify({ content_idea_id: id })
});

// Bulk approve and process
const bulkProcess = async (ids) => {
  await bulkApproveIdeas(ids);
  for (const id of ids) {
    await processIdea(id);
  }
};
```

**3. Assets Page Actions:**
```javascript
// Force publish specific asset
const publishNow = (assetId) => fetch('/webhook/publish-video', {
  method: 'POST',
  body: JSON.stringify({ asset_id: assetId })
});

// Retry failed asset
const retryAsset = (assetId, fromStage) => fetch('/webhook/retry-asset', {
  method: 'POST',
  body: JSON.stringify({ asset_id: assetId, from_stage: fromStage })
});
```

### Webhook Endpoints Summary

| Webhook Path | Method | Purpose | Body |
|--------------|--------|---------|------|
| `/webhook/trigger-scrape` | POST | Manual content scraping | `{}` |
| `/webhook/trigger-pipeline` | POST | Start processing an idea | `{ content_idea_id: number }` |
| `/webhook/publish-video` | POST | Publish specific or next ready | `{ asset_id?: number }` |
| `/webhook/retry-asset` | POST | Retry failed processing | `{ asset_id, from_stage }` |

---

## Phase 4: Error Handling & Recovery

### Global Error Handling Pattern

Add to EVERY workflow after API calls:

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ IF Node: Check Success                                                              │
│                                                                                     │
│ Condition: {{ $json.error || $response.statusCode >= 400 }}                         │
│                                                                                     │
│ TRUE (Error) →                                                                      │
│   1. HTTP Request: Update status to "error"                                         │
│   2. HTTP Request: Set error_message                                                │
│   3. (Optional) Send Slack notification                                             │
│   4. Stop Node: End workflow                                                        │
│                                                                                     │
│ FALSE (Success) → Continue                                                          │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Retry Logic

For critical API calls (HeyGen, ElevenLabs), wrap in retry pattern:

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Loop Node: Retry Up to 3 Times                                                      │
│                                                                                     │
│ Max Iterations: 3                                                                   │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ HTTP Request: API Call                                                              │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ IF Node: Success?                                                                   │
│                                                                                     │
│ TRUE  → Break loop, continue                                                        │
│ FALSE → Wait 10 seconds, retry                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## API Costs Estimate (Per Video)

| Service | Cost | Notes |
|---------|------|-------|
| Apify | $0.01-0.05 | Scraping |
| OpenRouter (Grok 4.1) | $0.01-0.03 | ~3-5 LLM calls per video |
| ElevenLabs | $0.05-0.15 | 45 sec voice |
| Whisper API | $0.003 | 45 sec transcription |
| HeyGen | $0.50-1.00 | Avatar video |
| Pexels | Free | Stock video |
| FFmpeg | Free | Self-hosted |
| Blotato | ~$0.15 | Per post across platforms |

**Total per video: ~$0.75-1.40**
**Daily (2 videos): ~$1.50-2.80**
**Monthly (60 videos): ~$45-85**

---

## Files Structure

```
/home/canderson/n8n/
├── docker-compose.yml
├── Dockerfile
├── .env                      # API keys
├── CLAUDE.md
├── MASTERPLAN.md
├── readme.md
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py
│   ├── models.py
│   ├── schemas.py
│   └── init.sql
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   └── src/
├── workflows/
│   ├── 01-scrape.json
│   ├── 02-auto-trigger.json
│   ├── 03-script-gen.json
│   ├── 04-get-video.json
│   ├── 05-create-voice.json
│   ├── 06-create-avatar.json
│   ├── 07-combine-vids.json
│   ├── 08-caption.json
│   ├── 09-publish.json
│   └── 10-analytics.json
└── assets/
    ├── fonts/
    │   └── Montserrat-Bold.ttf
    ├── logos/
    ├── audio/
    ├── videos/
    ├── avatar/
    ├── captions/
    └── output/
```

---

## Implementation Checklist

### Foundation
- [ ] All n8n credentials configured
- [ ] Backend API running and accessible from n8n
- [ ] Asset directories created with proper permissions
- [ ] Montserrat font installed in assets/fonts/

### Workflows
- [ ] 01-scrape.json imported and tested
- [ ] 02-auto-trigger.json imported and tested
- [ ] 03-script-gen.json imported and tested
- [ ] 04-get-video.json imported and tested
- [ ] 05-create-voice.json imported and tested
- [ ] 06-create-avatar.json imported and tested
- [ ] 07-combine-vids.json imported and tested
- [ ] 08-caption.json imported and tested
- [ ] 09-publish.json imported and tested
- [ ] 10-analytics.json imported and tested

### Integration Testing
- [ ] End-to-end test: Scrape → Publish
- [ ] Error handling verified
- [ ] UI webhook triggers working
- [ ] Status updates flowing correctly

### Production
- [ ] All workflows activated
- [ ] Schedules enabled
- [ ] Monitoring/alerting configured

---

## Appendix A: Node-by-Node Connection Map

This section provides the exact connections for each node in every workflow to ensure flawless builds.

### Workflow 1: SCRAPE - Node Connections

| Node # | Node Name | Node Type | Connects FROM | Connects TO | Output Data |
|--------|-----------|-----------|---------------|-------------|-------------|
| 1 | Schedule Trigger | Schedule Trigger | (start) | 2,3,4,5 (parallel) | `{ timestamp }` |
| 2 | Apify TikTok | HTTP Request | 1 | 6 | `{ items: [...] }` |
| 3 | Apify Reddit | HTTP Request | 1 | 6 | `{ items: [...] }` |
| 4 | Apify YouTube | HTTP Request | 1 | 6 | `{ items: [...] }` |
| 5 | Apify Twitter | HTTP Request | 1 | 6 | `{ items: [...] }` |
| 6 | Merge Results | Merge | 2,3,4,5 | 7 | `[...all items...]` |
| 7 | Analyze with Grok | HTTP Request | 6 | 8 | `{ choices: [...] }` |
| 8 | Parse LLM Response | Code | 7 | 9 | `[{ url, text, score, pillar }]` |
| 9 | Filter Score >= 7 | Filter | 8 | 10 | `[...filtered items...]` |
| 10 | Split In Batches | Split In Batches | 9 | 11 | `{ url, text, score, pillar }` |
| 11 | Create Content Idea | HTTP Request | 10 | 12 | `{ id, status }` |
| 12 | Done | NoOp | 11 | (end) | - |

### Workflow 2: AUTO TRIGGER - Node Connections

| Node # | Node Name | Node Type | Connects FROM | Connects TO | Output Data |
|--------|-----------|-----------|---------------|-------------|-------------|
| 1a | Schedule Trigger | Schedule Trigger | (start) | 2 | `{}` |
| 1b | Webhook Trigger | Webhook | (start) | 2 | `{ body: { content_idea_id? } }` |
| 2 | Merge Triggers | Merge | 1a,1b | 3 | merged data |
| 3 | IF: Has Specific ID | IF | 2 | 4a OR 4b | - |
| 4a | Get Specific Idea | HTTP Request | 3 (true) | 5 | `{ id, status, ... }` |
| 4b | Get Next Approved | HTTP Request | 3 (false) | 5 | `[{ id, status, ... }]` |
| 5 | IF: Has Content | IF | 4a,4b | 6 OR end | - |
| 6 | Update to script_generating | HTTP Request | 5 (true) | 7 | `{ id }` |
| 7 | Execute Workflow 03 | Execute Workflow | 6 | (end) | passes to WF3 |

### Workflow 3: SCRIPT GEN - Node Connections

| Node # | Node Name | Node Type | Connects FROM | Connects TO | Output Data |
|--------|-----------|-----------|---------------|-------------|-------------|
| 1 | Workflow Trigger | Execute Workflow Trigger | WF2 | 2 | `{ content_idea_id, original_text, pillar, suggested_hook }` |
| 2 | Generate Script | HTTP Request (OpenRouter) | 1 | 3 | `{ choices: [{ message: { content } }] }` |
| 3 | Parse Script JSON | Code | 2 | 4 | `{ hook, body, cta, full_script, duration_estimate }` |
| 4 | Generate Captions | HTTP Request (OpenRouter) | 3 | 5 | `{ choices: [...] }` |
| 5 | Merge Script+Captions | Code | 4 | 6 | `{ ...script, tiktok_caption, ig_caption, ... }` |
| 6 | Create Script Record | HTTP Request (POST /scripts) | 5 | 7 | `{ id, content_idea_id }` |
| 7 | Update Idea Status | HTTP Request (PATCH) | 6 | 8 | `{ status: 'script_ready' }` |
| 8 | Prepare Output | Set | 7 | 9 | `{ script_id, content_idea_id }` |
| 9 | Execute Workflow 04 | Execute Workflow | 8 | (end) | passes to WF4 |

### Workflow 4: GET VIDEO - Node Connections

| Node # | Node Name | Node Type | Connects FROM | Connects TO | Output Data |
|--------|-----------|-----------|---------------|-------------|-------------|
| 1 | Workflow Trigger | Execute Workflow Trigger | WF3 | 2 | `{ script_id, content_idea_id }` |
| 2 | Get Content Idea | HTTP Request | 1 | 3 | `{ source_url, source_platform, pillar }` |
| 3 | Get Script | HTTP Request | 2 | 4 | `{ hook, body }` |
| 4 | Switch: Platform | Switch | 3 | 5a,5b,5c | routes by platform |
| 5a | TikTok Download | HTTP Request (Apify) | 4 | 6 | `{ videoUrl }` |
| 5b | YouTube Download | HTTP Request (Apify) | 4 | 6 | `{ videoUrl }` |
| 5c | Pexels Search | HTTP Request | 4 | 6 | `{ videos: [...] }` |
| 6 | Merge Video Sources | Merge | 5a,5b,5c | 7 | `{ videoUrl }` |
| 7 | Download Video | HTTP Request | 6 | 8 | binary data |
| 8 | Write Video File | Write Binary File | 7 | 9 | file path |
| 9 | Create Asset Record | HTTP Request (POST /assets) | 8 | 10 | `{ id, status: 'pending' }` |
| 10 | Execute Workflow 05 | Execute Workflow | 9 | (end) | passes to WF5 |

### Workflow 5: CREATE VOICE - Node Connections

| Node # | Node Name | Node Type | Connects FROM | Connects TO | Output Data |
|--------|-----------|-----------|---------------|-------------|-------------|
| 1 | Workflow Trigger | Execute Workflow Trigger | WF4 | 2 | `{ asset_id, script_id, content_idea_id }` |
| 2 | Get Script | HTTP Request | 1 | 3 | `{ full_script }` |
| 3 | ElevenLabs TTS | HTTP Request | 2 | 4 | binary audio |
| 4 | Write Audio File | Write Binary File | 3 | 5 | file path |
| 5 | Wait 3s | Wait | 4 | 6 | - |
| 6 | Read Audio File | Read Binary File | 5 | 7 | binary data |
| 7 | Whisper Transcribe | HTTP Request (OpenAI) | 6 | 8 | `{ words: [{ word, start, end }] }` |
| 8 | Generate SRT | Code | 7 | 9 | `{ srt, duration }` |
| 9 | Write SRT File | Write File | 8 | 10 | file path |
| 10 | Update Asset | HTTP Request (PATCH) | 9 | 11 | `{ status: 'voice_ready' }` |
| 11 | Execute Workflow 06 | Execute Workflow | 10 | (end) | passes to WF6 |

### Workflow 6: CREATE AVATAR - Node Connections

| Node # | Node Name | Node Type | Connects FROM | Connects TO | Output Data |
|--------|-----------|-----------|---------------|-------------|-------------|
| 1 | Workflow Trigger | Execute Workflow Trigger | WF5 | 2 | `{ asset_id, script_id, content_idea_id }` |
| 2 | Get Asset | HTTP Request | 1 | 3 | `{ voiceover_path }` |
| 3 | Read Audio | Read Binary File | 2 | 4 | binary data |
| 4 | Upload to HeyGen | HTTP Request | 3 | 5 | `{ audio_url }` |
| 5 | Create Avatar Video | HTTP Request (HeyGen) | 4 | 6 | `{ video_id }` |
| 6 | Store Video ID | Set | 5 | 7 | `{ video_id, asset_id }` |
| 7 | Update Status | HTTP Request (PATCH) | 6 | 8 | `{ status: 'avatar_generating' }` |
| 8 | Wait 60s | Wait | 7 | 9 | - |
| 9 | Poll Status Loop | Loop | 8 | 10 | iteration data |
| 10 | Check HeyGen Status | HTTP Request | 9 | 11 | `{ status, video_url }` |
| 11 | IF: Status Check | IF | 10 | 12,13,14 | routes by status |
| 12 | Wait 30s (processing) | Wait | 11 | 9 (loop back) | - |
| 13 | Error Handler (failed) | HTTP Request | 11 | (end) | error |
| 14 | Download Video (completed) | HTTP Request | 11 | 15 | binary data |
| 15 | Write Avatar File | Write Binary File | 14 | 16 | file path |
| 16 | Update Asset | HTTP Request (PATCH) | 15 | 17 | `{ status: 'avatar_ready' }` |
| 17 | Execute Workflow 07 | Execute Workflow | 16 | (end) | passes to WF7 |

### Workflow 7: COMBINE VIDS - Node Connections

| Node # | Node Name | Node Type | Connects FROM | Connects TO | Output Data |
|--------|-----------|-----------|---------------|-------------|-------------|
| 1 | Workflow Trigger | Execute Workflow Trigger | WF6 | 2 | `{ asset_id, script_id, content_idea_id }` |
| 2 | Get Asset | HTTP Request | 1 | 3 | `{ background_video_path, avatar_video_path, voiceover_duration }` |
| 3 | Update to assembling | HTTP Request (PATCH) | 2 | 4 | `{ status: 'assembling' }` |
| 4 | Generate FFmpeg Cmd | HTTP Request (OpenRouter) | 3 | 5 | `{ choices: [...] }` |
| 5 | Extract Command | Code | 4 | 6 | `{ ffmpeg_command }` |
| 6 | Run FFmpeg | Execute Command | 5 | 7 | `{ stdout, stderr, exitCode }` |
| 7 | IF: Success | IF | 6 | 8,9 | routes by exit code |
| 8 | Error Handler | HTTP Request (PATCH) | 7 (false) | (end) | error |
| 9 | Update Asset | HTTP Request (PATCH) | 7 (true) | 10 | `{ status: 'captioning' }` |
| 10 | Execute Workflow 08 | Execute Workflow | 9 | (end) | passes to WF8 |

### Workflow 8: CAPTION - Node Connections

| Node # | Node Name | Node Type | Connects FROM | Connects TO | Output Data |
|--------|-----------|-----------|---------------|-------------|-------------|
| 1 | Workflow Trigger | Execute Workflow Trigger | WF7 | 2 | `{ asset_id, script_id, content_idea_id }` |
| 2 | Get Asset | HTTP Request | 1 | 3 | `{ combined_video_path, srt_path }` |
| 3 | Read SRT | Read File | 2 | 4 | SRT text content |
| 4 | Convert to ASS | Code | 3 | 5 | `{ ass }` |
| 5 | Write ASS File | Write File | 4 | 6 | file path |
| 6 | Run FFmpeg Subtitles | Execute Command | 5 | 7 | `{ exitCode }` |
| 7 | IF: Success | IF | 6 | 8,9 | routes by exit code |
| 8 | Error Handler | HTTP Request (PATCH) | 7 (false) | (end) | error |
| 9 | Update Asset | HTTP Request (PATCH) | 7 (true) | (end) | `{ status: 'ready_to_publish' }` |

### Workflow 9: PUBLISH - Node Connections

| Node # | Node Name | Node Type | Connects FROM | Connects TO | Output Data |
|--------|-----------|-----------|---------------|-------------|-------------|
| 1a | Schedule Trigger | Schedule Trigger | (start) | 2 | `{}` |
| 1b | Webhook Trigger | Webhook | (start) | 2 | `{ body: { asset_id? } }` |
| 2 | Merge Triggers | Merge | 1a,1b | 3 | merged data |
| 3 | IF: Has Specific ID | IF | 2 | 4a,4b | - |
| 4a | Get Specific Asset | HTTP Request | 3 (true) | 5 | asset data |
| 4b | Get Next Ready | HTTP Request | 3 (false) | 5 | asset data |
| 5 | IF: Has Asset | IF | 4a,4b | 6 OR end | - |
| 6 | Get Script | HTTP Request | 5 | 7 | captions data |
| 7 | Update to publishing | HTTP Request (PATCH) | 6 | 8 | `{ status: 'publishing' }` |
| 8 | Read Video | Read Binary File | 7 | 9 | binary data |
| 9 | Blotato Publish | HTTP Request | 8 | 10 | `{ platforms: {...} }` |
| 10 | Extract URLs | Code | 9 | 11 | platform URLs |
| 11 | Create Published | HTTP Request (POST /published) | 10 | 12 | `{ id }` |
| 12 | Update Asset | HTTP Request (PATCH) | 11 | 13 | `{ status: 'published' }` |
| 13 | Update Idea | HTTP Request (PATCH) | 12 | (end) | `{ status: 'published' }` |

### Workflow 10: ANALYTICS - Node Connections

| Node # | Node Name | Node Type | Connects FROM | Connects TO | Output Data |
|--------|-----------|-----------|---------------|-------------|-------------|
| 1 | Schedule Trigger | Schedule Trigger | (start) | 2 | `{}` |
| 2 | Get Recent Published | HTTP Request | 1 | 3 | `[...published items...]` |
| 3 | Split In Batches | Split In Batches | 2 | 4,5,6,7 | single item |
| 4 | TikTok Analytics | HTTP Request | 3 | 8 | `{ views, likes, ... }` |
| 5 | IG Analytics | HTTP Request | 3 | 8 | `{ views, likes, ... }` |
| 6 | YT Analytics | HTTP Request | 3 | 8 | `{ views, likes, ... }` |
| 7 | Other Analytics | HTTP Request | 3 | 8 | `{ views, likes, ... }` |
| 8 | Merge Analytics | Merge | 4,5,6,7 | 9 | combined stats |
| 9 | Calculate Engagement | Code | 8 | 10 | `{ engagement_rate }` |
| 10 | Create Analytics | HTTP Request (POST /analytics) | 9 | 11 | `{ id }` |
| 11 | Generate Report | HTTP Request (OpenRouter) | 10 | 12 | report text |
| 12 | Send Notification | Slack/Email | 11 | (end) | - |

---

## Appendix B: Data Flow Between Workflows

This shows exactly what data passes between workflows via Execute Workflow nodes.

```
WF1 (SCRAPE)
    │
    └── Stores in DB: content_ideas (status='pending')

                    ─── MANUAL: User approves in UI ───
                                    │
                                    ▼
WF2 (AUTO TRIGGER) ─── Reads: content_ideas WHERE status='approved'
    │
    │   PASSES TO WF3:
    │   {
    │     content_idea_id: number,
    │     original_text: string,
    │     pillar: string,
    │     suggested_hook: string,
    │     source_platform: string
    │   }
    │
    ▼
WF3 (SCRIPT GEN) ─── Creates: scripts record
    │
    │   PASSES TO WF4:
    │   {
    │     script_id: number,
    │     content_idea_id: number
    │   }
    │
    ▼
WF4 (GET VIDEO) ─── Creates: assets record, downloads background video
    │
    │   PASSES TO WF5:
    │   {
    │     asset_id: number,
    │     script_id: number,
    │     content_idea_id: number
    │   }
    │
    ▼
WF5 (CREATE VOICE) ─── Updates: assets (voiceover_path, srt_path)
    │
    │   PASSES TO WF6:
    │   {
    │     asset_id: number,
    │     script_id: number,
    │     content_idea_id: number
    │   }
    │
    ▼
WF6 (CREATE AVATAR) ─── Updates: assets (avatar_video_path)
    │
    │   PASSES TO WF7:
    │   {
    │     asset_id: number,
    │     script_id: number,
    │     content_idea_id: number
    │   }
    │
    ▼
WF7 (COMBINE VIDS) ─── Updates: assets (combined_video_path)
    │
    │   PASSES TO WF8:
    │   {
    │     asset_id: number,
    │     script_id: number,
    │     content_idea_id: number
    │   }
    │
    ▼
WF8 (CAPTION) ─── Updates: assets (ass_path, final_video_path, status='ready_to_publish')
    │
    └── END OF CHAIN (Video ready)

                    ─── SCHEDULED or MANUAL TRIGGER ───
                                    │
                                    ▼
WF9 (PUBLISH) ─── Reads: assets WHERE status='ready_to_publish'
    │             Creates: published record
    │             Updates: assets (status='published')
    │
    └── END

                    ─── WEEKLY SCHEDULE ───
                                    │
                                    ▼
WF10 (ANALYTICS) ─── Reads: published (last 7 days)
     │               Creates: analytics records
     │
     └── END
```

---

## Appendix C: Environment Variables per Workflow

| Workflow | Required Environment Variables |
|----------|-------------------------------|
| WF1: Scrape | `OPENROUTER_API_KEY`, `APIFY_API_KEY` |
| WF2: Auto Trigger | (none - uses backend API only) |
| WF3: Script Gen | `OPENROUTER_API_KEY` |
| WF4: Get Video | `APIFY_API_KEY`, `PEXELS_API_KEY`, `OPENROUTER_API_KEY` |
| WF5: Create Voice | `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`, `OPENAI_API_KEY` |
| WF6: Create Avatar | `HEYGEN_API_KEY`, `HEYGEN_AVATAR_ID` |
| WF7: Combine Vids | `OPENROUTER_API_KEY` |
| WF8: Caption | (none - uses FFmpeg only) |
| WF9: Publish | `BLOTATO_API_KEY` |
| WF10: Analytics | `BLOTATO_API_KEY` (or platform-specific keys) |

---

## Appendix D: n8n Credential Configuration

### 1. OpenRouter Credential
```
Type: Header Auth
Name: openrouter
Header Name: Authorization
Header Value: Bearer {{ $env.OPENROUTER_API_KEY }}
```

### 2. OpenAI Credential
```
Type: OpenAI API
Name: openai
API Key: {{ $env.OPENAI_API_KEY }}
```

### 3. ElevenLabs Credential
```
Type: Header Auth
Name: elevenlabs
Header Name: xi-api-key
Header Value: {{ $env.ELEVENLABS_API_KEY }}
```

### 4. HeyGen Credential
```
Type: Header Auth
Name: heygen
Header Name: X-Api-Key
Header Value: {{ $env.HEYGEN_API_KEY }}
```

### 5. Apify Credential
```
Type: Header Auth
Name: apify
Header Name: Authorization
Header Value: Bearer {{ $env.APIFY_API_KEY }}
```

### 6. Pexels Credential
```
Type: Header Auth
Name: pexels
Header Name: Authorization
Header Value: {{ $env.PEXELS_API_KEY }}
```

### 7. Blotato Credential
```
Type: Header Auth
Name: blotato
Header Name: Authorization
Header Value: Bearer {{ $env.BLOTATO_API_KEY }}
```

---

## Appendix E: Troubleshooting Common Issues

### FFmpeg Errors

**Error**: `ffmpeg: command not found`
**Solution**: Ensure n8n container is built with custom Dockerfile that includes FFmpeg

**Error**: `No such file or directory`
**Solution**: Check asset paths use `/home/node/assets/` (n8n container path)

**Error**: `colorkey filter failed`
**Solution**: Verify HeyGen video uses exact green #00FF00 background

### API Errors

**Error**: `401 Unauthorized` on OpenRouter
**Solution**: Check OPENROUTER_API_KEY in .env, ensure format is `sk-or-v1-...`

**Error**: `402 Payment Required` on ElevenLabs
**Solution**: Check character quota, upgrade plan if needed

**Error**: `HeyGen video stuck in processing`
**Solution**: Maximum poll time is 10 minutes; if exceeded, video may have failed silently

### Database Errors

**Error**: `connection refused` to backend
**Solution**: Use `http://backend:8000` (Docker network name), not localhost

**Error**: `relation does not exist`
**Solution**: Run `docker exec -i n8n_postgres psql -U n8n -d content_pipeline < backend/init.sql`

### Workflow Errors

**Error**: Workflow stops without error
**Solution**: Check IF nodes have both TRUE and FALSE branches connected

**Error**: Data not passing between workflows
**Solution**: Ensure Execute Workflow nodes have "Wait for completion" enabled
