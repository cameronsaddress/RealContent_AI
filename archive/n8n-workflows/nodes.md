# AI ContentGenerator Workflow - Node Documentation

This document provides a comprehensive analysis of every node in the AI ContentGenerator workflow, evaluating data flow, connections, and identifying any issues.

**Workflow ID:** `rTAhapEWXNxRAElT` (file) / `LNpBI12tR5p3NX3g` (deployed)
**Total Nodes:** 104
**Trigger Count:** 6

---

## Workflow Overview

The workflow is organized into the following major sections:

1. **SCRAPE** - Discover trending content from TikTok, Instagram, YouTube
2. **AUTO TRIGGER** - Schedule (15min) + Webhook to start main pipeline
3. **SCRIPT GENERATION** - Grok 4.1 creates viral script
4. **GET VIDEO** - Pexels B-roll search & download
5. **CREATE VOICE** - ElevenLabs TTS + duration check
6. **CREATE AVATAR** - HeyGen lip-sync video generation
7. **COMBINE VIDEOS** - FFmpeg chromakey compositing
8. **CAPTION** - Generate SRT + burn captions with FFmpeg
9. **PUBLISH** - Blotato multi-platform publishing
10. **FILE SERVER** - Serve audio/video files to external APIs

---

## Section 1: SCRAPE (Content Discovery)

### Node 1: Scrape Section (Sticky Note)
| Property | Value |
|----------|-------|
| **ID** | `note-scrape` |
| **Type** | `n8n-nodes-base.stickyNote` |
| **Purpose** | Visual label for scrape section |
| **Inputs** | None (decorative) |
| **Outputs** | None (decorative) |
| **Issues** | None - decorative node |

---

### Node 2: Daily 6am Scrape
| Property | Value |
|----------|-------|
| **ID** | `scrape-schedule` |
| **Type** | `n8n-nodes-base.scheduleTrigger` |
| **TypeVersion** | 1.2 |
| **Purpose** | Triggers scraping workflow daily at 6am |
| **Configuration** | `triggerAtHour: 6` |
| **Inputs** | None (trigger) |
| **Outputs** | `Parse Scrape Params` |
| **Data Passed** | Empty trigger payload (no body) |

**Analysis:**
- This is an entry point trigger
- Fires daily at 6:00 AM
- Passes empty JSON to `Parse Scrape Params`
- The downstream node handles defaults when no body is provided

**Issues:** None - correctly configured

---

### Node 3: Scrape Webhook
| Property | Value |
|----------|-------|
| **ID** | `scrape-webhook` |
| **Type** | `n8n-nodes-base.webhook` |
| **TypeVersion** | 2 |
| **Purpose** | HTTP endpoint to manually trigger scraping |
| **Path** | `scrape-trends` |
| **Method** | POST |
| **Authentication** | None |
| **Response Mode** | `responseNode` |
| **Inputs** | External HTTP POST |
| **Outputs** | `Parse Scrape Params` |

**Expected Input Body:**
```json
{
  "niche": "real estate",
  "hashtags": ["realestate", "realtor"],
  "platforms": ["tiktok", "instagram"],
  "time_range": "24h",
  "results_per_platform": 50
}
```

**Analysis:**
- Second entry point for scraping
- Uses `responseNode` mode - requires `Respond to Webhook` node later
- No authentication - publicly accessible

**Issues:**
- **SECURITY**: No authentication configured - anyone can trigger scrapes
- Otherwise correctly wired

---

### Node 4: Parse Scrape Params
| Property | Value |
|----------|-------|
| **ID** | `scrape-parse-params` |
| **Type** | `n8n-nodes-base.code` |
| **TypeVersion** | 2 |
| **Purpose** | Parse incoming parameters with defaults |
| **Inputs** | `Daily 6am Scrape`, `Scrape Webhook` |
| **Outputs** | `TikTok?`, `Instagram?`, `YouTube?` (parallel) |

**Code Logic:**
```javascript
const body = $input.first().json.body || $input.first().json;
// Sets defaults for niche, hashtags, platforms, timeRange, resultsPerPlatform
```

**Output Data:**
```json
{
  "niche": "real estate",
  "nicheTerms": ["real estate"],
  "hashtags": ["realestate", "realtor", ...],
  "platforms": ["tiktok", "instagram"],
  "timeRange": "24h",
  "resultsPerPlatform": 50,
  "scrapeStartedAt": "ISO timestamp"
}
```

**Analysis:**
- Correctly handles both scheduled (empty) and webhook (with body) triggers
- Provides sensible defaults
- Fans out to 3 parallel platform checks

**Issues:** None - correctly implemented

---

### Node 5: TikTok?
| Property | Value |
|----------|-------|
| **ID** | `check-tiktok` |
| **Type** | `n8n-nodes-base.if` |
| **TypeVersion** | 2 |
| **Purpose** | Check if TikTok is in platforms array |
| **Condition** | `$json.platforms.includes('tiktok') === true` |
| **Inputs** | `Parse Scrape Params` |
| **Outputs (true)** | `Apify TikTok` |
| **Outputs (false)** | `Skip TikTok` |

**Analysis:**
- Correctly checks if 'tiktok' is in the platforms array
- Routes to appropriate branch

**Issues:** None

---

### Node 6: Instagram?
| Property | Value |
|----------|-------|
| **ID** | `check-instagram` |
| **Type** | `n8n-nodes-base.if` |
| **TypeVersion** | 2 |
| **Purpose** | Check if Instagram is in platforms array |
| **Condition** | `$json.platforms.includes('instagram') === true` |
| **Inputs** | `Parse Scrape Params` |
| **Outputs (true)** | `Apify Instagram Reels` |
| **Outputs (false)** | `Skip Instagram` |

**Issues:** None

---

### Node 7: YouTube?
| Property | Value |
|----------|-------|
| **ID** | `check-youtube` |
| **Type** | `n8n-nodes-base.if` |
| **TypeVersion** | 2 |
| **Purpose** | Check if YouTube is in platforms array |
| **Condition** | `$json.platforms.includes('youtube') === true` |
| **Inputs** | `Parse Scrape Params` |
| **Outputs (true)** | `Apify YouTube` |
| **Outputs (false)** | `Skip YouTube` |

**Issues:** None

---

### Node 8: Apify TikTok
| Property | Value |
|----------|-------|
| **ID** | `apify-tiktok` |
| **Type** | `n8n-nodes-base.httpRequest` |
| **TypeVersion** | 4.2 |
| **Purpose** | Scrape TikTok via Apify |
| **URL** | `https://api.apify.com/v2/acts/clockworks~free-tiktok-scraper/run-sync-get-dataset-items` |
| **Method** | POST |
| **Auth** | HTTP Header Auth (credential: `apify`) |
| **Timeout** | 180000ms (3 min) |
| **Inputs** | `TikTok?` (true branch) |
| **Outputs** | `Merge Results` |

**Request Body:**
```json
{
  "hashtags": ["from Parse Scrape Params"],
  "resultsPerPage": 50
}
```

**Analysis:**
- References `$('Parse Scrape Params')` to get hashtags - correct
- Uses free TikTok scraper actor

**Issues:** None - correctly wired

---

### Node 9: Apify Instagram Reels
| Property | Value |
|----------|-------|
| **ID** | `apify-instagram` |
| **Type** | `n8n-nodes-base.httpRequest` |
| **TypeVersion** | 4.2 |
| **Purpose** | Scrape Instagram Reels via Apify |
| **URL** | `https://api.apify.com/v2/acts/apify~instagram-hashtag-scraper/run-sync-get-dataset-items` |
| **Method** | POST |
| **Auth** | HTTP Header Auth (credential: `apify`) |
| **Inputs** | `Instagram?` (true branch) |
| **Outputs** | `Merge Results` |

**Request Body:**
```json
{
  "hashtags": ["from Parse Scrape Params"],
  "resultsType": "reels",
  "resultsLimit": 50
}
```

**Issues:** None

---

### Node 10: Apify YouTube
| Property | Value |
|----------|-------|
| **ID** | `apify-youtube` |
| **Type** | `n8n-nodes-base.httpRequest` |
| **TypeVersion** | 4.2 |
| **Purpose** | Scrape YouTube via Apify |
| **URL** | `https://api.apify.com/v2/acts/apify~youtube-scraper/run-sync-get-dataset-items` |
| **Method** | POST |
| **Auth** | HTTP Header Auth (credential: `apify`) |
| **Inputs** | `YouTube?` (true branch) |
| **Outputs** | `Merge Results` |

**Request Body:**
```json
{
  "searchKeywords": ["nicheTerms from Parse Scrape Params"],
  "sort": "relevance",
  "maxResults": 50
}
```

**Issues:** None

---

### Node 11: Skip TikTok
| Property | Value |
|----------|-------|
| **ID** | `skip-tiktok` |
| **Type** | `n8n-nodes-base.noOp` |
| **Purpose** | Pass-through when TikTok not selected |
| **Inputs** | `TikTok?` (false branch) |
| **Outputs** | `Merge Results` |

**Issues:** None

---

### Node 12: Skip Instagram
| Property | Value |
|----------|-------|
| **ID** | `skip-instagram` |
| **Type** | `n8n-nodes-base.noOp` |
| **Purpose** | Pass-through when Instagram not selected |
| **Inputs** | `Instagram?` (false branch) |
| **Outputs** | `Merge Results` |

**Issues:** None

---

### Node 13: Skip YouTube
| Property | Value |
|----------|-------|
| **ID** | `skip-youtube` |
| **Type** | `n8n-nodes-base.noOp` |
| **Purpose** | Pass-through when YouTube not selected |
| **Inputs** | `YouTube?` (false branch) |
| **Outputs** | `Merge Results` |

**Issues:** None

---

### Node 14: Merge Results
| Property | Value |
|----------|-------|
| **ID** | `scrape-merge` |
| **Type** | `n8n-nodes-base.merge` |
| **TypeVersion** | 3 |
| **Mode** | `append` |
| **Purpose** | Combine results from all platform scrapers |
| **Inputs** | `Apify TikTok`, `Apify Instagram Reels`, `Apify YouTube`, `Skip TikTok`, `Skip Instagram`, `Skip YouTube` |
| **Outputs** | `Normalize Data` |

**Analysis:**
- Receives data from 6 sources (3 scrapers + 3 skip nodes)
- Appends all items together
- All inputs go to index 0 (single input)

**Issues:**
- **POTENTIAL ISSUE**: Merge node has only 1 input configured, but 6 nodes connect to it. With `append` mode this works but may cause timing issues if some branches complete before others.

---

### Node 15: Normalize Data
| Property | Value |
|----------|-------|
| **ID** | `scrape-normalize` |
| **Type** | `n8n-nodes-base.code` |
| **TypeVersion** | 2 |
| **Purpose** | Normalize scraped data from different platforms into unified format |
| **Inputs** | `Merge Results` |
| **Outputs** | `Analyze with Grok` |

**Code Logic:**
- Iterates through all merged items
- Detects platform by URL patterns or data structure
- Normalizes to common schema: `platform`, `url`, `title`, `views`, `likes`, `comments`, `shares`, `engagement`, `author`, `created`, `thumbnail`
- Sorts by engagement (highest first)

**Output:**
```json
{
  "niche": "real estate",
  "platforms": ["tiktok", "instagram"],
  "scrapedAt": "ISO timestamp",
  "totalResults": 100,
  "results": [/* normalized items */]
}
```

**Issues:** None - well implemented

---

### Node 16: Analyze with Grok
| Property | Value |
|----------|-------|
| **ID** | `scrape-analyze-grok` |
| **Type** | `n8n-nodes-base.httpRequest` |
| **TypeVersion** | 4.2 |
| **Purpose** | Use Grok LLM to analyze trends and score virality |
| **URL** | `https://openrouter.ai/api/v1/chat/completions` |
| **Method** | POST |
| **Auth** | HTTP Header Auth (credential: `openrouter`) |
| **Model** | `x-ai/grok-4.1-fast` |
| **Timeout** | 90000ms |
| **Inputs** | `Normalize Data` |
| **Outputs** | `Format Response` |

**Headers:**
- `Content-Type: application/json`
- `HTTP-Referer: http://100.83.153.43:5678`
- `X-Title: n8n-trend-analyzer`

**Analysis:**
- Sends top 30 scraped results to Grok for analysis
- Requests JSON response with viral_score, pillar, suggested_hook, why_viral, adaptable
- Uses temperature 0.3 for consistent output

**Issues:**
- **MODEL MISMATCH**: Uses `x-ai/grok-4.1-fast` but CLAUDE.md specifies `x-ai/grok-4-1106`. Should verify correct model name.

---

### Node 17: Format Response
| Property | Value |
|----------|-------|
| **ID** | `scrape-format-response` |
| **Type** | `n8n-nodes-base.code` |
| **TypeVersion** | 2 |
| **Purpose** | Parse LLM response and format for API/UI |
| **Inputs** | `Analyze with Grok` |
| **Outputs** | `Process & Save Trends` |

**Code Logic:**
- Parses JSON from LLM response
- Filters for viral_score >= 6 and adaptable !== false
- Returns success status with filtered trends

**Issues:** None

---

### Node 18: Process & Save Trends
| Property | Value |
|----------|-------|
| **ID** | `scrape-process-save` |
| **Type** | `n8n-nodes-base.code` |
| **TypeVersion** | 2 |
| **Purpose** | Save analyzed trends to backend database |
| **Inputs** | `Format Response` |
| **Outputs** | `Respond to Webhook` |

**Code Logic:**
- Iterates through trends
- POSTs each to `http://backend:8000/api/content-ideas`
- Uses `this.helpers.request()` for HTTP calls
- Returns summary with saved count

**Issues:** None - correctly saves to backend

---

### Node 19: Respond to Webhook
| Property | Value |
|----------|-------|
| **ID** | `scrape-respond` |
| **Type** | `n8n-nodes-base.respondToWebhook` |
| **TypeVersion** | 1.1 |
| **Purpose** | Return response to Scrape Webhook caller |
| **Response Type** | JSON |
| **Response Body** | `$json` (full output from Process & Save Trends) |
| **Inputs** | `Process & Save Trends` |
| **Outputs** | None (terminal) |

**Issues:** None

---

## SCRAPE SECTION SUMMARY

**Flow:**
```
Daily 6am Scrape ─┬─> Parse Scrape Params ─┬─> TikTok? ─┬─> Apify TikTok ──────┬─> Merge Results
Scrape Webhook ───┘                        │            └─> Skip TikTok ───────┤
                                           ├─> Instagram? ─┬─> Apify Instagram ┤
                                           │               └─> Skip Instagram ─┤
                                           └─> YouTube? ─┬─> Apify YouTube ────┤
                                                         └─> Skip YouTube ─────┘
                                                                               │
                                                                               v
Respond to Webhook <── Process & Save <── Format Response <── Analyze with Grok <── Normalize Data
```

**Issues Found in SCRAPE Section:**
1. **SECURITY**: Scrape Webhook has no authentication
2. **MODEL NAME**: Grok model name may be incorrect (`grok-4.1-fast` vs `grok-4-1106`)

---

## Section 2: AUTO TRIGGER / MAIN PIPELINE ENTRY

### Node 20: Auto Trigger (Sticky Note)
| Property | Value |
|----------|-------|
| **ID** | `note-auto-trigger` |
| **Type** | `n8n-nodes-base.stickyNote` |
| **Purpose** | Visual label for auto trigger section |
| **Issues** | None - decorative node |

---

### Node 21: Schedule Trigger
| Property | Value |
|----------|-------|
| **ID** | `schedule-trigger` |
| **Type** | `n8n-nodes-base.scheduleTrigger` |
| **TypeVersion** | 1.2 |
| **Purpose** | Auto-trigger pipeline every 15 minutes |
| **Configuration** | `minutesInterval: 15` |
| **Disabled** | **YES** |
| **Inputs** | None (trigger) |
| **Outputs** | `Get Approved Idea` |

**Analysis:**
- Currently DISABLED - pipeline won't auto-run
- Would trigger every 15 minutes when enabled
- Goes directly to fetch approved content idea

**Issues:**
- **DISABLED**: This is intentionally disabled, but note for production activation

---

### Node 22: Webhook (Pipeline Trigger)
| Property | Value |
|----------|-------|
| **ID** | `webhook-trigger` |
| **Type** | `n8n-nodes-base.webhook` |
| **TypeVersion** | 2 |
| **Purpose** | HTTP endpoint to manually trigger video pipeline |
| **Path** | `trigger-pipeline` |
| **Method** | POST |
| **Authentication** | None |
| **Response Mode** | `responseNode` |
| **Inputs** | External HTTP POST |
| **Outputs** | `Respond Immediately` |

**Expected Input Body:**
```json
{
  "content_idea_id": 123  // Optional - specific ID to process
}
```

**Analysis:**
- Main entry point for video pipeline
- Can optionally specify a content_idea_id in body
- Uses responseNode mode for async response

**Issues:**
- **SECURITY**: No authentication - publicly accessible
- **WIRING ISSUE**: According to connections, Webhook goes to `Respond Immediately`, but there's no connection to `Specific ID?` check. Let me verify...

Looking at connections:
- `Webhook` -> `Respond Immediately` -> `Get Approved Idea`

**CRITICAL ISSUE**: The `Specific ID?` node exists but is NOT connected to the main flow! This means the `content_idea_id` parameter in the webhook body is **never used**.

---

### Node 23: Respond Immediately
| Property | Value |
|----------|-------|
| **ID** | `trigger-respond-immediate` |
| **Type** | `n8n-nodes-base.respondToWebhook` |
| **TypeVersion** | 1.1 |
| **Purpose** | Return immediate response to webhook caller |
| **Response Type** | Text |
| **Response Body** | `"Pipeline Triggered"` |
| **Inputs** | `Webhook` |
| **Outputs** | `Get Approved Idea` |

**Analysis:**
- Immediately responds to caller so they don't wait for full pipeline
- Pipeline continues asynchronously

**Issues:** None for this node itself

---

### Node 24: Specific ID? (IF Node)
| Property | Value |
|----------|-------|
| **ID** | `if-specific-id` |
| **Type** | `n8n-nodes-base.if` |
| **TypeVersion** | 2 |
| **Purpose** | Check if specific content_idea_id was provided |
| **Condition** | `$json.body?.content_idea_id` exists (number) |
| **Inputs** | **NONE - DISCONNECTED** |
| **Outputs (true)** | `Get Specific Idea` |
| **Outputs (false)** | `Merge` (input 1) |

**CRITICAL ISSUE:**
- **ORPHANED NODE**: This node has NO incoming connections!
- It was designed to check if webhook has a specific ID
- True branch -> Get Specific Idea
- False branch -> Merge (to use approved idea from queue)

**This is a BROKEN flow path!**

---

### Node 25: Get Approved Idea
| Property | Value |
|----------|-------|
| **ID** | `get-approved-idea` |
| **Type** | `n8n-nodes-base.httpRequest` |
| **TypeVersion** | 4.2 |
| **Purpose** | Fetch next approved content idea from queue |
| **URL** | `http://backend:8000/api/content-ideas?status=approved&limit=1` |
| **Method** | GET |
| **Inputs** | `Respond Immediately`, `Schedule Trigger` |
| **Outputs** | `Merge` |

**Analysis:**
- Fetches the oldest approved content idea
- Returns array with single item (or empty array)

**Issues:** None

---

### Node 26: Get Specific Idea
| Property | Value |
|----------|-------|
| **ID** | `get-specific-idea` |
| **Type** | `n8n-nodes-base.httpRequest` |
| **TypeVersion** | 4.2 |
| **Purpose** | Fetch specific content idea by ID |
| **URL** | `http://backend:8000/api/content-ideas/{{ $json.body.content_idea_id }}` |
| **Method** | GET |
| **Inputs** | `Specific ID?` (true branch) |
| **Outputs** | `Merge` |

**CRITICAL ISSUE:**
- This node receives input from `Specific ID?` which is disconnected
- The URL expression `$json.body.content_idea_id` would fail without proper input
- **This entire branch is dead code**

---

### Node 27: Merge
| Property | Value |
|----------|-------|
| **ID** | `merge-idea-sources` |
| **Type** | `n8n-nodes-base.merge` |
| **TypeVersion** | 3 |
| **Mode** | `append` |
| **Purpose** | Combine results from different idea sources |
| **Inputs** | `Get Approved Idea` (index 0), `Specific ID?` false branch (index 1), `Get Specific Idea` (index 0) |
| **Outputs** | `Normalize Idea` |

**Analysis:**
- Receives from Get Approved Idea and Get Specific Idea (both to index 0)
- Also receives from Specific ID? false branch (to index 1)
- With `append` mode, just combines whatever arrives

**Issues:**
- **WIRING CONFUSION**: Multiple inputs to same index, but the Specific ID path is dead anyway

---

### Node 28: Normalize Idea
| Property | Value |
|----------|-------|
| **ID** | `normalize-idea` |
| **Type** | `n8n-nodes-base.code` |
| **TypeVersion** | 2 |
| **Purpose** | Normalize and validate content idea format |
| **Inputs** | `Merge` |
| **Outputs** | `Has Content?` |

**Code Logic:**
```javascript
// Handles both array response and single object
// Returns { hasContent: true/false, content_idea_id, original_text, pillar, suggested_hook, source_platform, source_url }
```

**Output (success):**
```json
{
  "hasContent": true,
  "content_idea_id": 123,
  "original_text": "...",
  "pillar": "educational_tips",
  "suggested_hook": "...",
  "source_platform": "tiktok",
  "source_url": "https://..."
}
```

**Output (no content):**
```json
{
  "hasContent": false,
  "message": "No approved content found"
}
```

**Issues:** None - handles edge cases well

---

### Node 29: Has Content?
| Property | Value |
|----------|-------|
| **ID** | `if-has-content` |
| **Type** | `n8n-nodes-base.if` |
| **TypeVersion** | 2 |
| **Purpose** | Check if content idea was found |
| **Condition** | `$json.hasContent === true` |
| **Inputs** | `Normalize Idea` |
| **Outputs (true)** | `Update Status` (continues pipeline) |
| **Outputs (false)** | `No Content` (ends) |

**Issues:** None

---

### Node 30: No Content
| Property | Value |
|----------|-------|
| **ID** | `no-content-end` |
| **Type** | `n8n-nodes-base.noOp` |
| **Purpose** | Terminal node when no content found |
| **Inputs** | `Has Content?` (false branch) |
| **Outputs** | None (terminal) |

**Issues:** None - graceful termination

---

## AUTO TRIGGER SECTION SUMMARY

**Actual Flow (Current):**
```
Schedule Trigger (disabled) ──┬──> Get Approved Idea ──> Merge ──> Normalize Idea ──> Has Content?
                              │                                                        ├─> Update Status (continue)
Webhook ──> Respond Immediately ──┘                                                    └─> No Content (end)
```

**Intended Flow (Broken):**
```
Webhook ──> Specific ID? ──┬─> Get Specific Idea ──┬──> Merge ──> ...
                           └─> Get Approved Idea ──┘
```

**CRITICAL Issues Found in AUTO TRIGGER Section:**
1. **ORPHANED NODE**: `Specific ID?` node is completely disconnected - no inputs!
2. **DEAD CODE**: `Get Specific Idea` node never executes because `Specific ID?` is orphaned
3. **BROKEN FEATURE**: Cannot process specific content_idea_id from webhook - always uses queue
4. **SECURITY**: Webhook has no authentication

---

## Section 3: SCRIPT GENERATION

### Node 31: Script Gen (Sticky Note)
| Property | Value |
|----------|-------|
| **ID** | `note-script-gen` |
| **Type** | `n8n-nodes-base.stickyNote` |
| **Purpose** | Visual label for script generation section |
| **Issues** | None - decorative node |

---

### Node 32: Update Status
| Property | Value |
|----------|-------|
| **ID** | `update-status-processing` |
| **Type** | `n8n-nodes-base.httpRequest` |
| **TypeVersion** | 4.2 |
| **Purpose** | Update content idea status to "script_generating" |
| **URL** | `http://backend:8000/api/content-ideas/{{ $json.content_idea_id }}` |
| **Method** | PATCH |
| **Inputs** | `Has Content?` (true branch) |
| **Outputs** | `Generate Script (Grok)` |

**Request Body:**
```json
{
  "status": "script_generating"
}
```

**Input Data Expected:**
- `$json.content_idea_id` from `Normalize Idea`

**Analysis:**
- Updates DB status before starting script generation
- Correctly references content_idea_id from previous node

**Issues:** None

---

### Node 33: Generate Script (Grok)
| Property | Value |
|----------|-------|
| **ID** | `generate-script-llm` |
| **Type** | `n8n-nodes-base.httpRequest` |
| **TypeVersion** | 4.2 |
| **Purpose** | Generate video script using Grok LLM |
| **URL** | `https://openrouter.ai/api/v1/chat/completions` |
| **Method** | POST |
| **Auth** | HTTP Header Auth (credential: `openrouter`) |
| **Model** | `x-ai/grok-4.1-fast` |
| **Timeout** | 60000ms |
| **Inputs** | `Update Status` |
| **Outputs** | `Parse Script` |

**Headers:**
- `Content-Type: application/json`
- `HTTP-Referer: http://100.83.153.43:5678`
- `X-Title: n8n-video-pipeline`

**System Prompt Key Points:**
- Persona: Sarah Mitchell, realtor in Pacific Northwest, Idaho and Pacific Northwest, Washington
- Structure: Hook, Body, Call to Action
- Markers: `[PAUSE]` for pauses, `*asterisks*` for emphasis
- Length: 150-250 words
- Returns JSON with: hook, script, cta, estimated_duration_seconds, suggested_broll

**Input References:**
- `$json.pillar` - from Normalize Idea (via Update Status output)
- `$json.suggested_hook` - from Normalize Idea
- `$json.original_text` - from Normalize Idea
- `$json.source_platform` - from Normalize Idea

**Analysis:**
- References fields from `Normalize Idea` but receives input from `Update Status`
- The PATCH response from Update Status may not include these fields!

**Issues:**
- **POTENTIAL DATA LOSS**: Update Status returns the PATCH response from backend, not the original Normalize Idea data. The script generation may receive empty/missing fields for `pillar`, `suggested_hook`, `original_text`, `source_platform`.

---

### Node 34: Parse Script
| Property | Value |
|----------|-------|
| **ID** | `parse-script` |
| **Type** | `n8n-nodes-base.code` |
| **TypeVersion** | 2 |
| **Purpose** | Parse LLM response and extract script data |
| **Inputs** | `Generate Script (Grok)` |
| **Outputs** | `Save Script` |

**Code Logic:**
```javascript
const response = $input.first().json;
const prevData = $('Update Status').first().json;  // Gets PATCH response
const content = response.choices[0].message.content;
const scriptData = JSON.parse(content);

return [{
  json: {
    content_idea_id: prevData.id || $('Normalize Idea').first().json.content_idea_id,
    hook: scriptData.hook,
    script: scriptData.script,
    cta: scriptData.cta,
    estimated_duration: scriptData.estimated_duration_seconds || 45,
    suggested_broll: scriptData.suggested_broll || [],
    pillar: $('Normalize Idea').first().json.pillar  // References Normalize Idea directly
  }
}];
```

**Analysis:**
- Correctly uses `$('Normalize Idea')` to get pillar (not from prevData)
- Falls back to Normalize Idea for content_idea_id if not in PATCH response
- Good defensive coding

**Issues:** None - handles data flow correctly

---

### Node 35: Save Script
| Property | Value |
|----------|-------|
| **ID** | `save-script` |
| **Type** | `n8n-nodes-base.httpRequest` |
| **TypeVersion** | 4.2 |
| **Purpose** | Save generated script to database |
| **URL** | `http://backend:8000/api/scripts` |
| **Method** | POST |
| **Inputs** | `Parse Script` |
| **Outputs** | `Create Asset Record` |

**Request Body:**
```json
{
  "content_idea_id": "$json.content_idea_id",
  "hook": "$json.hook",
  "body": "$json.script",
  "cta": "$json.cta",
  "full_script": "hook + script + cta concatenated",
  "duration_estimate": "$json.estimated_duration",
  "status": "script_ready"
}
```

**Output Data:**
- Returns saved script object with `id` field

**Issues:** None

---

### Node 36: Create Asset Record
| Property | Value |
|----------|-------|
| **ID** | `create-asset-record` |
| **Type** | `n8n-nodes-base.httpRequest` |
| **TypeVersion** | 4.2 |
| **Purpose** | Create asset tracking record in database |
| **URL** | `http://backend:8000/api/assets` |
| **Method** | POST |
| **Inputs** | `Save Script` |
| **Outputs** | `Prepare B-Roll Searches` |

**Request Body:**
```json
{
  "script_id": "$json.id",
  "status": "pending"
}
```

**Input Expected:**
- `$json.id` - script ID from Save Script response

**Output Data:**
- Returns created asset object with `id` field

**Issues:** None

---

## SCRIPT GENERATION SECTION SUMMARY

**Flow:**
```
Has Content? (true) ──> Update Status ──> Generate Script (Grok) ──> Parse Script ──> Save Script ──> Create Asset Record
```

**Data Flow:**
1. `Normalize Idea` outputs: `content_idea_id`, `pillar`, `suggested_hook`, `original_text`, `source_platform`
2. `Update Status` sends PATCH, returns updated idea (may have different structure)
3. `Generate Script` uses expression to access Normalize Idea fields directly
4. `Parse Script` extracts LLM response, combines with `Normalize Idea` data
5. `Save Script` creates script record, returns with `id`
6. `Create Asset Record` creates asset record using script `id`

**Issues Found in SCRIPT GENERATION Section:**
1. **MODEL NAME**: Uses `x-ai/grok-4.1-fast` but CLAUDE.md specifies `x-ai/grok-4-1106`
2. **POTENTIAL DATA LOSS**: `Generate Script` reads from `Update Status` output which is PATCH response - may not have original fields. However, the expression uses `$json` which would be empty/wrong - but LLM prompt construction might work since it uses string interpolation before POST.

**Needs verification:** Check if expression `$json.pillar` in Generate Script body refers to Update Status output or works differently in expression context.

---

## Section 4: GET VIDEO / B-ROLL

### Node 37: Get Video (Sticky Note)
| Property | Value |
|----------|-------|
| **ID** | `note-get-video` |
| **Type** | `n8n-nodes-base.stickyNote` |
| **Purpose** | Visual label for B-roll search section |
| **Issues** | None - decorative node |

---

### Node 38: Prepare B-Roll Searches
| Property | Value |
|----------|-------|
| **ID** | `prepare-broll-searches` |
| **Type** | `n8n-nodes-base.code` |
| **TypeVersion** | 2 |
| **Purpose** | Generate search queries for B-roll videos |
| **Inputs** | `Create Asset Record` |
| **Outputs** | `Search Pexels` (multiple items) |

**Code Logic:**
```javascript
const asset = $input.first().json;
const parseData = $('Parse Script').first().json;
const script = $('Save Script').first().json;
const suggestedBroll = parseData.suggested_broll || [];

// Default searches by pillar if none provided
const defaultSearches = {
  market_intelligence: ['real estate market', 'house exterior', 'city skyline aerial'],
  educational_tips: ['home inspection', 'signing contract documents', 'house tour interior'],
  // ...
};

const searches = suggestedBroll.length > 0 ? suggestedBroll : defaultSearches[pillar];

// Returns array of items (up to 3)
return searches.slice(0, 3).map((query, index) => ({
  json: {
    asset_id: asset.id,
    script_id: script.id,
    content_idea_id: parseData.content_idea_id,
    scene_index: index,
    search_query: query
  }
}));
```

**Output:**
- Returns 1-3 items, each with a search query
- Passes `asset_id`, `script_id`, `content_idea_id`, `scene_index`, `search_query`

**Issues:** None - correctly references upstream nodes

---

### Node 39: Search Pexels
| Property | Value |
|----------|-------|
| **ID** | `search-pexels` |
| **Type** | `n8n-nodes-base.httpRequest` |
| **TypeVersion** | 4.2 |
| **Purpose** | Search Pexels API for stock videos |
| **URL** | `https://api.pexels.com/videos/search?query={{ encodeURIComponent($json.search_query) }}&per_page=3&orientation=portrait` |
| **Method** | GET |
| **Auth** | HTTP Header Auth (credential: `pexels`) |
| **Timeout** | 30000ms |
| **Inputs** | `Prepare B-Roll Searches` |
| **Outputs** | `Select Videos` |

**Analysis:**
- Runs for each search query item
- Searches for portrait-oriented videos (9:16)
- Returns 3 results per search

**Issues:** None

---

### Node 40: Select Videos
| Property | Value |
|----------|-------|
| **ID** | `select-videos` |
| **Type** | `n8n-nodes-base.code` |
| **TypeVersion** | 2 |
| **Purpose** | Select best video from each Pexels search result |
| **Inputs** | `Search Pexels` |
| **Outputs** | `Video Found?` |

**Code Logic:**
```javascript
const items = $input.all();
// For each search result, select first HD video
// Returns array with video_found, pexels_id, video_url, duration, etc.
```

**Output Fields:**
- `asset_id`, `script_id`, `content_idea_id`, `scene_index`
- `video_found`: boolean
- `pexels_id`, `video_url`, `duration` (if found)

**Issues:**
- **REFERENCE ERROR**: Code tries to find matching item from `Prepare B-Roll Searches` by scene_index, but `$('Prepare B-Roll Searches').all().find()` may not work correctly if items aren't paired 1:1.

---

### Node 41: Video Found?
| Property | Value |
|----------|-------|
| **ID** | `if-video-found` |
| **Type** | `n8n-nodes-base.if` |
| **TypeVersion** | 2 |
| **Purpose** | Check if video was found for this search |
| **Condition** | `$json.video_found === true` |
| **Inputs** | `Select Videos` |
| **Outputs (true)** | `Download B-Roll` |
| **Outputs (false)** | `Skip B-Roll` |

**Issues:** None

---

### Node 42: Download B-Roll
| Property | Value |
|----------|-------|
| **ID** | `download-broll` |
| **Type** | `n8n-nodes-base.httpRequest` |
| **TypeVersion** | 4.2 |
| **Purpose** | Download video file from Pexels |
| **URL** | `{{ $json.video_url }}` |
| **Method** | GET |
| **Response Format** | File (binary) |
| **Timeout** | 120000ms |
| **Inputs** | `Video Found?` (true branch) |
| **Outputs** | `Save B-Roll` |

**Issues:** None

---

### Node 43: Skip B-Roll
| Property | Value |
|----------|-------|
| **ID** | `skip-broll` |
| **Type** | `n8n-nodes-base.noOp` |
| **Purpose** | Pass-through when no video found |
| **Inputs** | `Video Found?` (false branch) |
| **Outputs** | `Merge B-Roll` (index 1) |

**Issues:** None

---

### Node 44: Save B-Roll
| Property | Value |
|----------|-------|
| **ID** | `save-broll-file` |
| **Type** | `n8n-nodes-base.code` |
| **TypeVersion** | 2 |
| **Mode** | `runOnceForEachItem` |
| **Purpose** | Save downloaded video to filesystem |
| **Inputs** | `Download B-Roll` |
| **Outputs** | `Merge B-Roll` (index 0) |

**Code Logic:**
```javascript
const fs = require('fs');
const item = $input.item;
const scriptId = $('Select Videos').first().json.script_id;
const sceneIndex = $('Select Videos').first().json.scene_index;
const fileName = `/home/node/.n8n-files/assets/videos/${scriptId}_scene${sceneIndex}.mp4`;

if (item.binary && item.binary.data) {
  fs.writeFileSync(fileName, Buffer.from(item.binary.data.data, 'base64'));
  item.json.saved_path = fileName;
}
return item;
```

**File Path:** `/home/node/.n8n-files/assets/videos/{script_id}_scene{scene_index}.mp4`

**Issues:**
- **BUG**: Uses `$('Select Videos').first()` which always gets first item, not the current item being processed. When multiple videos are downloaded, all will try to use the same scene_index from first item!
- Should use the current item's data, not reference back to Select Videos

---

### Node 45: Merge B-Roll
| Property | Value |
|----------|-------|
| **ID** | `merge-broll` |
| **Type** | `n8n-nodes-base.merge` |
| **TypeVersion** | 3 |
| **Mode** | `append` |
| **Purpose** | Combine saved B-roll and skipped items |
| **Inputs** | `Save B-Roll` (index 0), `Skip B-Roll` (index 1) |
| **Outputs** | `Get Character Config` |

**Issues:** None

---

## GET VIDEO / B-ROLL SECTION SUMMARY

**Flow:**
```
Create Asset Record ──> Prepare B-Roll Searches ──> Search Pexels ──> Select Videos ──> Video Found?
                                                                                         ├─> Download B-Roll ──> Save B-Roll ──┬──> Merge B-Roll
                                                                                         └─> Skip B-Roll ───────────────────────┘
```

**Issues Found in B-ROLL Section:**
1. **BUG in Save B-Roll**: Uses `$('Select Videos').first()` instead of current item data - will save all videos with same scene_index
2. **REFERENCE ISSUE in Select Videos**: May not correctly pair search results with original queries

---

## Section 5: CREATE VOICE / TTS

### Node 46: Create Voice (Sticky Note)
| Property | Value |
|----------|-------|
| **ID** | `note-create-voice` |
| **Type** | `n8n-nodes-base.stickyNote` |
| **Purpose** | Visual label for voice generation section |
| **Issues** | None - decorative node |

---

### Node 47: Get Character Config
| Property | Value |
|----------|-------|
| **ID** | `get-character-config` |
| **Type** | `n8n-nodes-base.httpRequest` |
| **TypeVersion** | 4.2 |
| **Purpose** | Fetch character/avatar configuration from backend |
| **URL** | `http://backend:8000/api/config/character` |
| **Method** | GET |
| **Inputs** | `Merge B-Roll` |
| **Outputs** | `Prepare TTS Text` |

**Expected Response:**
```json
{
  "voice_id": "ElevenLabs voice ID",
  "avatar_id": "HeyGen avatar ID or 'static_image'",
  "avatar_type": "avatar" or "talking_photo",
  "image_url": "URL if using static image"
}
```

**Issues:** None

---

### Node 48: Prepare TTS Text
| Property | Value |
|----------|-------|
| **ID** | `prepare-tts-text` |
| **Type** | `n8n-nodes-base.code` |
| **TypeVersion** | 2 |
| **Purpose** | Prepare script text for TTS and aggregate character config |
| **Inputs** | `Get Character Config` |
| **Outputs** | `Check Voice Exists` |

**Code Logic:**
```javascript
const brollItems = $('Merge B-Roll').all();
const charConfig = $input.first().json || {};
const scriptData = $('Save Script').first().json;
const assetData = $('Create Asset Record').first().json;
const parseData = $('Parse Script').first().json;

// Clean script for TTS
let text = parseData.script || '';
text = text.replace(/\[PAUSE\]/gi, '...');
text = text.replace(/\*(.*?)\*/g, '$1');  // Remove emphasis markers
if (parseData.cta) text = text + ' ... ' + parseData.cta;

// Voice ID fallback
let voiceId = charConfig.voice_id;
if (!voiceId || voiceId.trim().length < 5) {
  voiceId = '21m00Tcm4TlvDq8ikWAM'; // Rachel (default)
}

return [{
  json: {
    asset_id: assetData.id,
    script_id: scriptData.id,
    content_idea_id: parseData.content_idea_id,
    text_for_tts: text,
    voice_id: voiceId,
    avatar_id: charConfig.avatar_id || 'Kristin_pubblic_3_20240108',
    avatar_type: charConfig.avatar_type || 'avatar',
    config_image_url: charConfig.image_url,
    broll_paths: brollItems.filter(i => i.json.fileName).map(...)
  }
}];
```

**Output Fields:**
- `asset_id`, `script_id`, `content_idea_id`
- `text_for_tts` - cleaned script text
- `voice_id` - ElevenLabs voice ID
- `avatar_id`, `avatar_type`, `config_image_url` - HeyGen config
- `broll_paths` - array of saved B-roll file paths

**Issues:**
- **BUG in broll_paths**: Filters by `i.json.fileName` but Save B-Roll sets `saved_path`, not `fileName`. This will always return empty array!

---

### Node 49: Check Voice Exists
| Property | Value |
|----------|-------|
| **ID** | `check-voice-exists` |
| **Type** | `n8n-nodes-base.code` |
| **TypeVersion** | 2 |
| **Purpose** | Check if voice file already exists (idempotency) |
| **Inputs** | `Prepare TTS Text` |
| **Outputs** | `Voice Exists?` |

**Code Logic:**
```javascript
const fs = require('fs');
const sourceData = $('Prepare TTS Text').first().json;
const scriptId = sourceData.script_id;
const filePath = `/home/node/.n8n-files/assets/audio/${scriptId}_voice.mp3`;
const exists = fs.existsSync(filePath);

return [{
  json: {
    ...sourceData,  // Pass all data through
    exists,
    filePath,
    script_id: scriptId,
    voice_id: sourceData.voice_id
  }
}];
```

**Issues:** None - correctly passes data through

---

### Node 50: Voice Exists?
| Property | Value |
|----------|-------|
| **ID** | `if-voice-exists` |
| **Type** | `n8n-nodes-base.if` |
| **TypeVersion** | 2 |
| **Purpose** | Check if voice file already exists |
| **Condition** | `$json.exists === true` |
| **Inputs** | `Check Voice Exists` |
| **Outputs (true)** | `Get Duration` (skip generation) |
| **Outputs (false)** | `Generate Voice (ElevenLabs)` |

**Issues:** None - good idempotency pattern

---

### Node 51: Generate Voice (ElevenLabs)
| Property | Value |
|----------|-------|
| **ID** | `generate-voice` |
| **Type** | `n8n-nodes-base.httpRequest` |
| **TypeVersion** | 4.2 |
| **Purpose** | Generate voice audio using ElevenLabs API |
| **URL** | `https://api.elevenlabs.io/v1/text-to-speech/{{ $json.voice_id }}` |
| **Method** | POST |
| **Auth** | HTTP Header Auth (credential: `elevenlabs`) |
| **Timeout** | 120000ms |
| **Inputs** | `Voice Exists?` (false branch) |
| **Outputs** | `Save Voice` |

**Headers:**
- `Content-Type: application/json`
- `Accept: audio/mpeg`

**Request Body:**
```json
{
  "text": "$json.text_for_tts",
  "model_id": "eleven_multilingual_v2",
  "voice_settings": {
    "stability": 0.5,
    "similarity_boost": 0.75,
    "style": 0.5,
    "use_speaker_boost": true
  }
}
```

**Response Format:** File (binary audio)

**Issues:** None

---

### Node 52: Save Voice
| Property | Value |
|----------|-------|
| **ID** | `save-voice-file` |
| **Type** | `n8n-nodes-base.writeBinaryFile` |
| **TypeVersion** | 1 |
| **Purpose** | Save voice audio to filesystem |
| **File Name** | `/home/node/.n8n-files/assets/audio/{{ $('Prepare TTS Text').first().json.script_id }}_voice.mp3` |
| **Binary Property** | `data` |
| **Inputs** | `Generate Voice (ElevenLabs)` |
| **Outputs** | `Get Duration` |

**Issues:** None - correctly references Prepare TTS Text for script_id

---

### Node 53: Get Duration
| Property | Value |
|----------|-------|
| **ID** | `get-voice-duration` |
| **Type** | `n8n-nodes-base.code` |
| **TypeVersion** | 2 |
| **Purpose** | Get audio file duration using ffprobe |
| **Inputs** | `Voice Exists?` (true branch), `Save Voice` |
| **Outputs** | `Update Asset Voice` |

**Code Logic:**
```javascript
const { execSync } = require('child_process');
const ttsData = $('Prepare TTS Text').first().json;
const audioPath = `/home/node/.n8n-files/assets/audio/${ttsData.script_id}_voice.mp3`;

const result = execSync(`ffprobe -v quiet -show_entries format=duration -of csv=p=0 "${audioPath}"`, { encoding: 'utf-8' });
return [{ json: { stdout: result.trim(), success: true } }];
```

**Issues:** None - uses ffprobe correctly

---

### Node 54: Update Asset Voice
| Property | Value |
|----------|-------|
| **ID** | `update-asset-voice` |
| **Type** | `n8n-nodes-base.httpRequest` |
| **TypeVersion** | 4.2 |
| **Purpose** | Update asset record with voice file path |
| **URL** | `http://backend:8000/api/assets/{{ $('Prepare TTS Text').first().json.asset_id }}` |
| **Method** | PATCH |
| **Inputs** | `Get Duration` |
| **Outputs** | `Read Audio File` |

**Request Body:**
```json
{
  "voiceover_path": "/home/node/.n8n-files/assets/audio/{script_id}_voice.mp3",
  "status": "voice_ready"
}
```

**Issues:** None

---

### Node 55: Read Audio File
| Property | Value |
|----------|-------|
| **ID** | `read-audio-file` |
| **Type** | `n8n-nodes-base.readBinaryFile` |
| **TypeVersion** | 1 |
| **Purpose** | Read voice file for HeyGen upload |
| **File Path** | `/home/node/.n8n-files/assets/audio/{{ $json.script_id }}_voice.mp3` |
| **Inputs** | `Update Asset Voice` |
| **Outputs** | `Upload HeyGen Audio` |

**Issues:**
- **DATA REFERENCE**: Uses `$json.script_id` but input is from Update Asset Voice which returns PATCH response. The PATCH response may not include `script_id`!
- Should reference `$('Prepare TTS Text').first().json.script_id` instead

---

## CREATE VOICE / TTS SECTION SUMMARY

**Flow:**
```
Merge B-Roll ──> Get Character Config ──> Prepare TTS Text ──> Check Voice Exists ──> Voice Exists?
                                                                                        ├─(true)─> Get Duration ──┬──> Update Asset Voice ──> Read Audio File
                                                                                        └─(false)─> Generate Voice ──> Save Voice ──┘
```

**Issues Found in TTS Section:**
1. **BUG in Prepare TTS Text**: `broll_paths` filter uses `fileName` but saved field is `saved_path` - always empty array
2. **DATA REFERENCE in Read Audio File**: Uses `$json.script_id` from PATCH response which may not have this field

---

## Section 6: CREATE AVATAR / HEYGEN

### Node 56: Create Avatar (Sticky Note)
| Property | Value |
|----------|-------|
| **ID** | `note-create-avatar` |
| **Type** | `n8n-nodes-base.stickyNote` |
| **Purpose** | Visual label for avatar generation section |
| **Issues** | None - decorative node |

---

### Node 57: Upload HeyGen Audio
| Property | Value |
|----------|-------|
| **ID** | `upload-heygen-audio-http` |
| **Type** | `n8n-nodes-base.httpRequest` |
| **TypeVersion** | 4.2 |
| **Purpose** | Upload audio file to HeyGen |
| **URL** | `https://upload.heygen.com/v1/asset?name={{ $('Read Audio File').first().json.script_id }}_voice.mp3` |
| **Method** | POST |
| **Auth** | HTTP Header Auth (credential: `heygen`) |
| **Content Type** | Binary Data |
| **Binary Property** | `data` |
| **Inputs** | `Read Audio File` |
| **Outputs** | `Is Talking Photo?` |

**Response:**
```json
{
  "data": {
    "id": "uploaded-asset-id"
  }
}
```

**Issues:**
- **DATA REFERENCE**: Uses `$('Read Audio File').first().json.script_id` but Read Audio File outputs binary file read result which may not have `script_id` in json
- This could fail if script_id is not available

---

### Node 58: Is Talking Photo?
| Property | Value |
|----------|-------|
| **ID** | `check-is-photo` |
| **Type** | `n8n-nodes-base.if` |
| **TypeVersion** | 2 |
| **Purpose** | Check if avatar is static image (talking photo) |
| **Condition** | `$('Prepare TTS Text').first().json.avatar_id === 'static_image'` |
| **Inputs** | `Upload HeyGen Audio` |
| **Outputs (true)** | `Download Character Image` |
| **Outputs (false)** | `Prepare HeyGen Data` |

**Issues:** None - correctly references Prepare TTS Text

---

### Node 59: Download Character Image
| Property | Value |
|----------|-------|
| **ID** | `download-char-image` |
| **Type** | `n8n-nodes-base.httpRequest` |
| **TypeVersion** | 4.2 |
| **Purpose** | Download character image for talking photo |
| **URL** | `{{ $('Prepare TTS Text').first().json.config_image_url }}` |
| **Method** | GET |
| **Response Format** | File (binary) |
| **Inputs** | `Is Talking Photo?` (true branch) |
| **Outputs** | `Upload Talking Photo` |

**Issues:** None

---

### Node 60: Upload Talking Photo
| Property | Value |
|----------|-------|
| **ID** | `upload-talking-photo` |
| **Type** | `n8n-nodes-base.httpRequest` |
| **TypeVersion** | 4.2 |
| **Purpose** | Upload image as HeyGen talking photo |
| **URL** | `https://upload.heygen.com/v1/talking_photo?name=avatar.jpg` |
| **Method** | POST |
| **Auth** | HTTP Header Auth (credential: `heygen`) |
| **Content Type** | Binary Data |
| **Inputs** | `Download Character Image` |
| **Outputs** | `Prepare HeyGen Data` |

**Response:**
```json
{
  "data": {
    "talking_photo_id": "uploaded-photo-id"
  }
}
```

**Issues:** None

---

### Node 61: Prepare HeyGen Data
| Property | Value |
|----------|-------|
| **ID** | `prepare-heygen-data` |
| **Type** | `n8n-nodes-base.code` |
| **TypeVersion** | 2 |
| **Purpose** | Prepare data for HeyGen video generation |
| **Inputs** | `Is Talking Photo?` (false), `Upload Talking Photo` |
| **Outputs** | `Check Video Exists` |

**Code Logic:**
```javascript
const uploadResult = $('Upload HeyGen Audio').first().json;
const durationResult = $('Get Duration').first().json;
const ttsData = $('Prepare TTS Text').first().json;

// Check for Dynamic Photo Upload
let dynamicPhotoId = null;
try {
  const photoUpload = $('Upload Talking Photo').first();
  if (photoUpload && photoUpload.json?.data) {
    dynamicPhotoId = photoUpload.json.data.talking_photo_id || photoUpload.json.data.id;
  }
} catch(e) {}

// Avatar Logic
let avatarId = ttsData.avatar_id;
let avatarType = ttsData.avatar_type;

if (avatarId === 'static_image') {
  if (dynamicPhotoId) {
    avatarType = 'talking_photo';
    avatarId = dynamicPhotoId;
  } else {
    // Fallback to default avatar
    avatarId = 'Anna_public_3_20240108';
    avatarType = 'avatar';
  }
}

return [{
  json: {
    asset_id: ttsData.asset_id,
    script_id: ttsData.script_id,
    content_idea_id: ttsData.content_idea_id,
    voice_path: `...`,
    duration_seconds: parseFloat(durationResult.stdout.trim()) || 45,
    broll_paths: ttsData.broll_paths,
    avatar_id: avatarId,
    avatar_type: avatarType,
    audio_asset_id: uploadResult.data.id
  }
}];
```

**Issues:** None - handles both paths (avatar vs talking photo) well

---

### Node 62: Check Video Exists
| Property | Value |
|----------|-------|
| **ID** | `check-video-exists` |
| **Type** | `n8n-nodes-base.code` |
| **TypeVersion** | 2 |
| **Purpose** | Check if avatar video already exists (idempotency) |
| **Inputs** | `Prepare HeyGen Data` |
| **Outputs** | `Video Exists?` |

**Code Logic:**
```javascript
const fs = require('fs');
const sourceData = $('Prepare HeyGen Data').first().json;
const scriptId = sourceData.script_id;
const filePath = `/home/node/.n8n-files/assets/avatar/${scriptId}_avatar.mp4`;
const exists = fs.existsSync(filePath);

return [{
  json: {
    ...sourceData,
    exists,
    filePath,
    script_id: scriptId
  }
}];
```

**Issues:** None

---

### Node 63: Video Exists? (Avatar)
| Property | Value |
|----------|-------|
| **ID** | `if-video-exists` |
| **Type** | `n8n-nodes-base.if` |
| **TypeVersion** | 2 |
| **Purpose** | Check if avatar video already exists |
| **Condition** | `$json.exists === true` |
| **Inputs** | `Check Video Exists` |
| **Outputs (true)** | `Build FFmpeg Command` (skip generation) |
| **Outputs (false)** | `Create HeyGen Video` |

**Issues:** None

---

### Node 64: Create HeyGen Video
| Property | Value |
|----------|-------|
| **ID** | `create-heygen-video` |
| **Type** | `n8n-nodes-base.httpRequest` |
| **TypeVersion** | 4.2 |
| **Purpose** | Submit video generation request to HeyGen |
| **URL** | `https://api.heygen.com/v2/video/generate` |
| **Method** | POST |
| **Auth** | HTTP Header Auth (credential: `heygen`) |
| **Timeout** | 60000ms |
| **Inputs** | `Video Exists?` (false branch) |
| **Outputs** | `Extract Video ID` |

**Request Body:**
```json
{
  "video_inputs": [{
    "character": {
      "type": "$json.avatar_type",  // "avatar" or "talking_photo"
      "avatar_id": "$json.avatar_id" // or "talking_photo_id"
    },
    "voice": {
      "type": "audio",
      "audio_asset_id": "$json.audio_asset_id"
    },
    "background": {
      "type": "color",
      "value": "#00FF00"  // Green screen
    }
  }],
  "dimension": { "width": 720, "height": 1280 },
  "aspect_ratio": "9:16"
}
```

**Issues:** None

---

### Node 65: Extract Video ID
| Property | Value |
|----------|-------|
| **ID** | `extract-heygen-id` |
| **Type** | `n8n-nodes-base.code` |
| **TypeVersion** | 2 |
| **Purpose** | Extract video ID and initialize retry counter |
| **Inputs** | `Create HeyGen Video` |
| **Outputs** | `Wait 30s` |

**Code Logic:**
```javascript
const response = $input.first().json;
const prevData = $('Prepare HeyGen Data').first().json;

if (!response.data?.video_id) {
  throw new Error(`HeyGen error: ${JSON.stringify(response)}`);
}

return [{
  json: {
    ...prevData,
    heygen_video_id: response.data.video_id,
    retry_count: 0,
    max_retries: 20
  }
}];
```

**Issues:** None

---

### Node 66: Wait 30s
| Property | Value |
|----------|-------|
| **ID** | `wait-heygen` |
| **Type** | `n8n-nodes-base.wait` |
| **TypeVersion** | 1.1 |
| **Purpose** | Initial wait before first status check |
| **Duration** | 30 seconds |
| **Inputs** | `Extract Video ID` |
| **Outputs** | `Check Status` |

**Issues:** None

---

### Node 67: Check Status
| Property | Value |
|----------|-------|
| **ID** | `check-heygen-status` |
| **Type** | `n8n-nodes-base.httpRequest` |
| **TypeVersion** | 4.2 |
| **Purpose** | Check HeyGen video generation status |
| **URL** | `https://api.heygen.com/v1/video_status.get?video_id={{ $json.heygen_video_id }}` |
| **Method** | GET |
| **Auth** | HTTP Header Auth (credential: `heygen`) |
| **Inputs** | `Wait 30s`, `Wait & Retry` |
| **Outputs** | `Route Status` |

**Response:**
```json
{
  "data": {
    "status": "completed" | "processing" | "pending" | "failed",
    "video_url": "https://..." // when completed
  }
}
```

**Issues:** None

---

### Node 68: Route Status
| Property | Value |
|----------|-------|
| **ID** | `route-status-switch` |
| **Type** | `n8n-nodes-base.switch` |
| **TypeVersion** | 3 |
| **Purpose** | Route based on HeyGen status |
| **Input** | `$json.data.status` |
| **Inputs** | `Check Status` |
| **Outputs** | |
| - Output 0 (`completed`) | `Download Avatar` |
| - Output 1 (`processing`/`pending`) | `Increment Retry` |
| - Output 2 (fallback/`failed`) | `Stop on Failure` |

**Issues:** None - good switch pattern

---

### Node 69: Increment Retry
| Property | Value |
|----------|-------|
| **ID** | `increment-retry` |
| **Type** | `n8n-nodes-base.code` |
| **TypeVersion** | 2 |
| **Purpose** | Increment retry counter, check max |
| **Inputs** | `Route Status` (output 1) |
| **Outputs** | `Wait & Retry` |

**Code Logic:**
```javascript
const prevData = $('Extract Video ID').first().json;
const checkData = $input.first().json;
const retryCount = (prevData.retry_count || 0) + 1;
const maxRetries = prevData.max_retries || 20;

if (retryCount > maxRetries) {
  throw new Error(`HeyGen timeout: Video did not complete after ${maxRetries} retries`);
}

return [{
  json: {
    ...prevData,
    retry_count: retryCount,
    last_status: checkData.data?.status
  }
}];
```

**Issues:** None

---

### Node 70: Wait & Retry
| Property | Value |
|----------|-------|
| **ID** | `wait-retry` |
| **Type** | `n8n-nodes-base.wait` |
| **TypeVersion** | 1.1 |
| **Purpose** | Wait before next status check |
| **Duration** | 60 seconds |
| **Inputs** | `Increment Retry` |
| **Outputs** | `Check Status` |

**Issues:** None - creates polling loop

---

### Node 71: Stop on Failure
| Property | Value |
|----------|-------|
| **ID** | `stop-heygen-error` |
| **Type** | `n8n-nodes-base.stopAndError` |
| **TypeVersion** | 1 |
| **Purpose** | Stop workflow on HeyGen failure |
| **Error Message** | `HeyGen Generation Failed` |
| **Error Description** | `$json.data.error || $json.data.detail || 'Unknown Error'` |
| **Inputs** | `Route Status` (fallback) |
| **Outputs** | None (terminal) |

**Issues:** None

---

### Node 72: Download Avatar
| Property | Value |
|----------|-------|
| **ID** | `download-avatar` |
| **Type** | `n8n-nodes-base.httpRequest` |
| **TypeVersion** | 4.2 |
| **Purpose** | Download completed avatar video from HeyGen |
| **URL** | `{{ $('Check Status').first().json.data.video_url }}` |
| **Method** | GET |
| **Response Format** | File (binary) |
| **Timeout** | 180000ms |
| **Inputs** | `Route Status` (output 0) |
| **Outputs** | `Save Avatar` |

**Issues:** None

---

### Node 73: Save Avatar
| Property | Value |
|----------|-------|
| **ID** | `save-avatar-file` |
| **Type** | `n8n-nodes-base.code` |
| **TypeVersion** | 2 |
| **Mode** | `runOnceForEachItem` |
| **Purpose** | Save avatar video to filesystem |
| **Inputs** | `Download Avatar` |
| **Outputs** | `Build FFmpeg Command` |

**Code Logic:**
```javascript
const fs = require('fs');
const item = $input.item;
const scriptId = $('Extract Video ID').first().json.script_id;
const fileName = `/home/node/.n8n-files/assets/avatar/${scriptId}_avatar.mp4`;

if (item.binary && item.binary.data) {
  fs.writeFileSync(fileName, Buffer.from(item.binary.data.data, 'base64'));
  item.json.saved_path = fileName;
}
return item;
```

**File Path:** `/home/node/.n8n-files/assets/avatar/{script_id}_avatar.mp4`

**Issues:** None - correctly references Extract Video ID for script_id

---

## CREATE AVATAR / HEYGEN SECTION SUMMARY

**Flow:**
```
Read Audio File ──> Upload HeyGen Audio ──> Is Talking Photo?
                                            ├─(true)─> Download Character Image ──> Upload Talking Photo ──┐
                                            └─(false)─────────────────────────────────────────────────────────┴──> Prepare HeyGen Data

Prepare HeyGen Data ──> Check Video Exists ──> Video Exists?
                                                ├─(true)─> Build FFmpeg Command
                                                └─(false)─> Create HeyGen Video ──> Extract Video ID ──> Wait 30s ──> Check Status ──> Route Status
                                                                                                                                        ├─(completed)─> Download Avatar ──> Save Avatar ──> Build FFmpeg
                                                                                                                                        ├─(processing)─> Increment Retry ──> Wait & Retry ──┘ (loop back to Check Status)
                                                                                                                                        └─(failed)─> Stop on Failure
```

**Issues Found in HEYGEN Section:**
1. **DATA REFERENCE in Upload HeyGen Audio**: Uses `$('Read Audio File').first().json.script_id` but Read Audio File may not have script_id in its output

---

## Section 7: COMBINE VIDEOS / FFMPEG

### Node 74: Combine Vids (Sticky Note)
| Property | Value |
|----------|-------|
| **ID** | `note-combine` |
| **Type** | `n8n-nodes-base.stickyNote` |
| **Purpose** | Visual label for video combining section |
| **Issues** | None - decorative node |

---

### Node 75: Build FFmpeg Command
| Property | Value |
|----------|-------|
| **ID** | `build-ffmpeg-cmd` |
| **Type** | `n8n-nodes-base.code` |
| **TypeVersion** | 2 |
| **Purpose** | Build FFmpeg command for video compositing |
| **Inputs** | `Video Exists?` (true), `Save Avatar` |
| **Outputs** | `Run FFmpeg` |

**Code Logic:**
```javascript
const fs = require('fs');
const extractNode = $('Extract Video ID').first();
const sourceData = extractNode ? extractNode.json : $('Prepare HeyGen Data').first().json;
const scriptId = sourceData.script_id;

const avatarPath = `/home/node/.n8n-files/assets/avatar/${scriptId}_avatar.mp4`;
const outputPath = `/home/node/.n8n-files/assets/output/${scriptId}_combined.mp4`;
const musicPath = `/home/node/.n8n-files/assets/music/background_music.mp3`;
const brollPaths = sourceData.broll_paths || [];

// Build FFmpeg filter complex:
// 1. Concatenate B-roll clips
// 2. Chromakey avatar (green screen removal)
// 3. Overlay avatar on B-roll
// 4. Mix audio (voice + background music)

const ffmpegCmd = `ffmpeg -y ${inputs} -filter_complex "${fc}" -map "[outv]" -map "[outa]" -c:v libx264 ...`;

return [{
  json: {
    asset_id: sourceData.asset_id,
    script_id: scriptId,
    content_idea_id: sourceData.content_idea_id,
    ffmpeg_command: ffmpegCmd,
    output_path: outputPath
  }
}];
```

**Issues:**
- **DEPENDS ON broll_paths**: Uses `sourceData.broll_paths` which was set in `Prepare TTS Text` but has the bug where it filters by `fileName` instead of `saved_path` - will always be empty array, resulting in fallback to solid background

---

### Node 76: Run FFmpeg
| Property | Value |
|----------|-------|
| **ID** | `run-ffmpeg-combine` |
| **Type** | `n8n-nodes-base.code` |
| **TypeVersion** | 2 |
| **Purpose** | Execute FFmpeg command |
| **Inputs** | `Build FFmpeg Command` |
| **Outputs** | `Success?` |

**Code Logic:**
```javascript
const { execSync } = require('child_process');
const ffmpegCmd = $input.first().json.ffmpeg_command;

try {
  const result = execSync(ffmpegCmd, { encoding: 'utf-8', timeout: 300000 });
  return [{ json: { stdout: result, exitCode: 0, success: true } }];
} catch (error) {
  return [{ json: { stderr: error.message, exitCode: error.status || 1, success: false } }];
}
```

**Issues:** None

---

### Node 77: Success? (FFmpeg)
| Property | Value |
|----------|-------|
| **ID** | `if-ffmpeg-success` |
| **Type** | `n8n-nodes-base.if` |
| **TypeVersion** | 2 |
| **Purpose** | Check if FFmpeg succeeded |
| **Condition** | `$json.exitCode === 0` |
| **Inputs** | `Run FFmpeg` |
| **Outputs (true)** | `Prepare Whisper` |
| **Outputs (false)** | `Handle Error` |

**Issues:** None

---

### Node 78: Handle Error (FFmpeg)
| Property | Value |
|----------|-------|
| **ID** | `handle-ffmpeg-error` |
| **Type** | `n8n-nodes-base.code` |
| **TypeVersion** | 2 |
| **Purpose** | Throw error on FFmpeg failure |
| **Inputs** | `Success?` (false branch) |
| **Outputs** | None (throws) |

**Code:** `throw new Error(`FFmpeg failed: ${$input.first().json.stderr}`);`

**Issues:** None

---

## COMBINE VIDEOS SECTION SUMMARY

**Flow:**
```
Video Exists? (true) ──┬──> Build FFmpeg Command ──> Run FFmpeg ──> Success?
Save Avatar ───────────┘                                            ├─(true)─> Prepare Whisper
                                                                    └─(false)─> Handle Error
```

**Issues Found:**
1. **broll_paths always empty**: Due to upstream bug in Prepare TTS Text, B-roll videos are never used in compositing

---

## Section 8: CAPTION / WHISPER

### Node 79: Caption (Sticky Note)
| Property | Value |
|----------|-------|
| **ID** | `note-caption` |
| **Type** | `n8n-nodes-base.stickyNote` |
| **Purpose** | Visual label for caption section |
| **Issues** | None - decorative node |

---

### Node 80: Prepare Whisper
| Property | Value |
|----------|-------|
| **ID** | `prepare-whisper` |
| **Type** | `n8n-nodes-base.code` |
| **TypeVersion** | 2 |
| **Purpose** | Prepare data for Whisper transcription |
| **Inputs** | `Success?` (true branch) |
| **Outputs** | `Read Audio for Whisper` |

**Code Logic:**
```javascript
const ffmpegData = $('Build FFmpeg Command').first().json;
const ttsData = $('Prepare TTS Text').first().json;

return [{
  json: {
    asset_id: ffmpegData.asset_id,
    script_id: ffmpegData.script_id,
    content_idea_id: ffmpegData.content_idea_id,
    combined_video_path: ffmpegData.output_path,
    audio_path: `/home/node/.n8n-files/assets/audio/${ffmpegData.script_id}_voice.mp3`
  }
}];
```

**Issues:** None

---

### Node 81: Read Audio for Whisper
| Property | Value |
|----------|-------|
| **ID** | `read-audio-for-whisper` |
| **Type** | `n8n-nodes-base.code` |
| **TypeVersion** | 2 |
| **Purpose** | Read audio file as binary for Whisper API |
| **Inputs** | `Prepare Whisper` |
| **Outputs** | `Whisper Transcribe` |

**Code Logic:**
```javascript
const fs = require('fs');
const filePath = $input.first().json.audio_path;
const buffer = fs.readFileSync(filePath);

return [{
  json: $input.first().json,
  binary: {
    data: {
      data: buffer.toString('base64'),
      mimeType: 'audio/mpeg',
      fileName: filePath.split('/').pop()
    }
  }
}];
```

**Issues:** None

---

### Node 82: Whisper Transcribe
| Property | Value |
|----------|-------|
| **ID** | `whisper-transcribe` |
| **Type** | `n8n-nodes-base.httpRequest` |
| **TypeVersion** | 4.2 |
| **Purpose** | Transcribe audio using OpenAI Whisper |
| **URL** | `https://api.openai.com/v1/audio/transcriptions` |
| **Method** | POST |
| **Auth** | HTTP Header Auth (credential: `openai`) |
| **Content Type** | `multipart-form-data` |
| **Timeout** | 120000ms |
| **Inputs** | `Read Audio for Whisper` |
| **Outputs** | `Parse Whisper Response` |

**Form Data:**
- `model`: `whisper-1`
- `response_format`: `srt`
- `file`: binary audio data

**Issues:** None

---

### Node 83: Parse Whisper Response
| Property | Value |
|----------|-------|
| **ID** | `parse-whisper-response` |
| **Type** | `n8n-nodes-base.code` |
| **TypeVersion** | 2 |
| **Purpose** | Extract SRT content from Whisper response |
| **Inputs** | `Whisper Transcribe` |
| **Outputs** | `Save SRT` |

**Code Logic:**
```javascript
const response = $input.first().json;
const prevData = $('Prepare Whisper').first().json;

// Whisper returns SRT as plain text
const srtContent = typeof response === 'string' ? response : (response.text || response.data || JSON.stringify(response));

return [{
  json: {
    asset_id: prevData.asset_id,
    script_id: prevData.script_id,
    content_idea_id: prevData.content_idea_id,
    combined_video_path: prevData.combined_video_path,
    srt_content: srtContent
  }
}];
```

**Issues:** None

---

### Node 84: Save SRT
| Property | Value |
|----------|-------|
| **ID** | `save-srt` |
| **Type** | `n8n-nodes-base.code` |
| **TypeVersion** | 2 |
| **Mode** | `runOnceForEachItem` |
| **Purpose** | Save SRT file to filesystem |
| **Inputs** | `Parse Whisper Response` |
| **Outputs** | `Build Caption Cmd` |

**File Path:** `/home/node/.n8n-files/assets/captions/{script_id}_captions.srt`

**Issues:** None

---

### Node 85: Build Caption Cmd
| Property | Value |
|----------|-------|
| **ID** | `build-caption-cmd` |
| **Type** | `n8n-nodes-base.code` |
| **TypeVersion** | 2 |
| **Purpose** | Build FFmpeg command to burn captions |
| **Inputs** | `Save SRT` |
| **Outputs** | `Burn Captions` |

**Code Logic:**
```javascript
const prevData = $('Parse Whisper Response').first().json;
const scriptId = prevData.script_id;

const inputPath = prevData.combined_video_path;
const srtPath = `/home/node/.n8n-files/assets/captions/${scriptId}_captions.srt`;
const outputPath = `/home/node/.n8n-files/assets/output/${scriptId}_final.mp4`;

// Styled subtitles for social media
const subtitleFilter = `subtitles=${srtPath}:force_style='FontName=Arial,FontSize=28,PrimaryColour=&HFFFFFF...'`;

const ffmpegCmd = `ffmpeg -y -i ${inputPath} -vf "${subtitleFilter}" -c:v libx264 -preset fast -crf 23 -c:a copy ${outputPath}`;

return [{
  json: {
    asset_id: prevData.asset_id,
    script_id: scriptId,
    content_idea_id: prevData.content_idea_id,
    ffmpeg_command: ffmpegCmd,
    final_video_path: outputPath
  }
}];
```

**Issues:** None

---

### Node 86: Burn Captions
| Property | Value |
|----------|-------|
| **ID** | `run-ffmpeg-caption` |
| **Type** | `n8n-nodes-base.code` |
| **TypeVersion** | 2 |
| **Purpose** | Execute FFmpeg caption burning |
| **Inputs** | `Build Caption Cmd` |
| **Outputs** | `Caption Success?` |

**Issues:** None - same pattern as Run FFmpeg

---

### Node 87: Caption Success?
| Property | Value |
|----------|-------|
| **ID** | `if-caption-success` |
| **Type** | `n8n-nodes-base.if` |
| **TypeVersion** | 2 |
| **Purpose** | Check if caption burning succeeded |
| **Condition** | `$json.exitCode === 0` |
| **Inputs** | `Burn Captions` |
| **Outputs (true)** | `Prepare Publish Data` |
| **Outputs (false)** | `Handle Caption Error` |

**Issues:** None

---

### Node 88: Handle Caption Error
| Property | Value |
|----------|-------|
| **ID** | `handle-caption-error` |
| **Type** | `n8n-nodes-base.code` |
| **TypeVersion** | 2 |
| **Purpose** | Throw error on caption failure |
| **Inputs** | `Caption Success?` (false branch) |
| **Outputs** | None (throws) |

**Issues:** None

---

## CAPTION SECTION SUMMARY

**Flow:**
```
Success? (true) ──> Prepare Whisper ──> Read Audio for Whisper ──> Whisper Transcribe ──> Parse Whisper Response ──> Save SRT ──> Build Caption Cmd ──> Burn Captions ──> Caption Success?
                                                                                                                                                                              ├─(true)─> Prepare Publish Data
                                                                                                                                                                              └─(false)─> Handle Caption Error
```

**Issues Found:** None

---

## Section 9: PUBLISH / BLOTATO

### Node 89: Publish (Sticky Note)
| Property | Value |
|----------|-------|
| **ID** | `note-publish` |
| **Type** | `n8n-nodes-base.stickyNote` |
| **Purpose** | Visual label for publish section |
| **Issues** | None - decorative node |

---

### Node 90: Prepare Publish Data
| Property | Value |
|----------|-------|
| **ID** | `prepare-publish` |
| **Type** | `n8n-nodes-base.code` |
| **TypeVersion** | 2 |
| **Purpose** | Prepare caption and hashtags for publishing |
| **Inputs** | `Caption Success?` (true branch) |
| **Outputs** | `Read Final Video` |

**Code Logic:**
```javascript
const captionData = $('Build Caption Cmd').first().json;
const parseData = $('Parse Script').first().json;

const pillarHashtags = {
  market_intelligence: ['#realestate', '#housingmarket', ...],
  educational_tips: ['#realestatetips', '#homebuying101', ...],
  // ...
};

const hashtags = pillarHashtags[parseData.pillar] || pillarHashtags.educational_tips;
const caption = `${parseData.hook}\n\n${parseData.cta}\n\n${hashtags.join(' ')}`;

return [{
  json: {
    asset_id: captionData.asset_id,
    script_id: captionData.script_id,
    content_idea_id: captionData.content_idea_id,
    final_video_path: captionData.final_video_path,
    caption: caption,
    platforms: ['tiktok', 'instagram', 'youtube_shorts']
  }
}];
```

**Issues:** None

---

### Node 91: Read Final Video
| Property | Value |
|----------|-------|
| **ID** | `read-final-video` |
| **Type** | `n8n-nodes-base.code` |
| **TypeVersion** | 2 |
| **Purpose** | Read final video file as binary |
| **Inputs** | `Prepare Publish Data` |
| **Outputs** | `GCS Upload HTTP` |

**Issues:** None

---

### Node 92: GCS Upload HTTP
| Property | Value |
|----------|-------|
| **ID** | `gcs-upload-http` |
| **Type** | `n8n-nodes-base.httpRequest` |
| **TypeVersion** | 4.1 |
| **Purpose** | Upload video to Google Cloud Storage |
| **URL** | `https://storage.googleapis.com/upload/storage/v1/b/content-pipeline-assets/o` |
| **Method** | POST |
| **Auth** | Google Service Account |
| **Content Type** | Binary Data |
| **Inputs** | `Read Final Video` |
| **Outputs** | `Format GCS URL` |

**Query Parameters:**
- `uploadType`: `media`
- `name`: `{script_id}_final.mp4`
- `predefinedAcl`: `publicRead`

**Issues:** None

---

### Node 93: Format GCS URL
| Property | Value |
|----------|-------|
| **ID** | `format-gcs-url` |
| **Type** | `n8n-nodes-base.code` |
| **TypeVersion** | 2 |
| **Purpose** | Format public GCS URL for video |
| **Inputs** | `GCS Upload HTTP` |
| **Outputs** | `Publish (Blotato)` |

**Code Logic:**
```javascript
const fileName = $input.first().json.name;
const publishData = $('Prepare Publish Data').first().json;
const videoUrl = `https://storage.googleapis.com/content-pipeline-assets/${fileName}`;

return [{
  json: {
    ...publishData,
    video_url: videoUrl
  }
}];
```

**Issues:** None

---

### Node 94: Publish (Blotato)
| Property | Value |
|----------|-------|
| **ID** | `publish-blotato` |
| **Type** | `n8n-nodes-base.httpRequest` |
| **TypeVersion** | 4.2 |
| **Purpose** | Publish video to multiple platforms via Blotato |
| **URL** | `https://api.blotato.com/v1/posts` |
| **Method** | POST |
| **Auth** | HTTP Header Auth (credential: `blotato`) |
| **Timeout** | 120000ms |
| **Inputs** | `Format GCS URL` |
| **Outputs** | `Parse Response` |

**Request Body:**
```json
{
  "platforms": ["tiktok", "instagram", "youtube_shorts"],
  "content": {
    "caption": "...",
    "video_url": "https://storage.googleapis.com/..."
  },
  "schedule": { "type": "now" },
  "options": { "auto_hashtags": false, "cross_post": true }
}
```

**Issues:** None

---

### Node 95: Parse Response
| Property | Value |
|----------|-------|
| **ID** | `parse-publish-response` |
| **Type** | `n8n-nodes-base.code` |
| **TypeVersion** | 2 |
| **Purpose** | Parse Blotato response and map platform URLs |
| **Inputs** | `Publish (Blotato)` |
| **Outputs** | `Save Publish Record` |

**Code Logic:**
```javascript
const response = $input.first().json;
const publishData = $('Prepare Publish Data').first().json;
const posts = response.posts || response.data?.posts || [];

// Map each platform post to specific columns
const platformData = {
  tiktok_url: null, tiktok_id: null,
  ig_url: null, ig_id: null,
  yt_url: null, yt_id: null,
  // ...other platforms...
};

posts.forEach(post => {
  const platform = post.platform?.toLowerCase();
  if (platform === 'tiktok') {
    platformData.tiktok_url = post.url;
    platformData.tiktok_id = post.id;
  }
  // ... etc
});

return [{
  json: {
    asset_id: publishData.asset_id,
    content_idea_id: publishData.content_idea_id,
    ...platformData,
    published_at: new Date().toISOString()
  }
}];
```

**Issues:** None

---

### Node 96: Save Publish Record
| Property | Value |
|----------|-------|
| **ID** | `save-publish-record` |
| **Type** | `n8n-nodes-base.httpRequest` |
| **TypeVersion** | 4.2 |
| **Purpose** | Save publish record to database |
| **URL** | `http://backend:8000/api/published` |
| **Method** | POST |
| **Inputs** | `Parse Response` |
| **Outputs** | `Update Status Published` |

**Issues:** None

---

### Node 97: Update Status Published
| Property | Value |
|----------|-------|
| **ID** | `update-final-status` |
| **Type** | `n8n-nodes-base.httpRequest` |
| **TypeVersion** | 4.2 |
| **Purpose** | Update content idea status to "published" |
| **URL** | `http://backend:8000/api/content-ideas/{{ $('Parse Response').first().json.content_idea_id }}` |
| **Method** | PATCH |
| **Inputs** | `Save Publish Record` |
| **Outputs** | None (terminal) |

**Request Body:**
```json
{
  "status": "published",
  "published_at": "ISO timestamp"
}
```

**Issues:** None - correctly marks content idea as published

---

## PUBLISH SECTION SUMMARY

**Flow:**
```
Caption Success? (true) ──> Prepare Publish Data ──> Read Final Video ──> GCS Upload HTTP ──> Format GCS URL ──> Publish (Blotato) ──> Parse Response ──> Save Publish Record ──> Update Status Published
```

**Issues Found:** None

---

## Section 10: FILE SERVER (Utility Webhooks)

### Node 98: File Server (Sticky Note)
| Property | Value |
|----------|-------|
| **ID** | `note-file-server` |
| **Type** | `n8n-nodes-base.stickyNote` |
| **Purpose** | Visual label for file server section |
| **Issues** | None - decorative node |

---

### Node 99: Serve Audio (Webhook)
| Property | Value |
|----------|-------|
| **ID** | `webhook-serve-audio` |
| **Type** | `n8n-nodes-base.webhook` |
| **TypeVersion** | 2 |
| **Purpose** | HTTP endpoint to serve audio files |
| **Path** | `serve-audio/:script_id` |
| **Method** | GET |
| **Authentication** | None |
| **Response Mode** | `responseNode` |
| **Inputs** | External HTTP GET |
| **Outputs** | `Read Audio` |

**Issues:**
- **SECURITY**: No authentication - publicly serves audio files

---

### Node 100: Read Audio
| Property | Value |
|----------|-------|
| **ID** | `read-audio-file` (duplicate name with Node 55) |
| **Type** | `n8n-nodes-base.code` |
| **TypeVersion** | 2 |
| **Purpose** | Read audio file from filesystem |
| **Inputs** | `Serve Audio` |
| **Outputs** | `Return Audio` |

**Code Logic:**
```javascript
const fs = require('fs');
const scriptId = $input.first().json.params.script_id;
const filePath = `/home/node/.n8n-files/assets/audio/${scriptId}_voice.mp3`;

const buffer = fs.readFileSync(filePath);
return [{
  json: $input.first().json,
  binary: {
    data: {
      data: buffer.toString('base64'),
      mimeType: 'audio/mpeg',
      fileName: `${scriptId}_voice.mp3`
    }
  }
}];
```

**Issues:** None

---

### Node 101: Return Audio
| Property | Value |
|----------|-------|
| **ID** | `respond-audio` |
| **Type** | `n8n-nodes-base.respondToWebhook` |
| **TypeVersion** | 1.1 |
| **Purpose** | Return audio file as HTTP response |
| **Response Type** | Binary |
| **Headers** | `Content-Type: audio/mpeg` |
| **Inputs** | `Read Audio` |
| **Outputs** | None (terminal) |

**Issues:** None

---

### Node 102: Serve Video (Webhook)
| Property | Value |
|----------|-------|
| **ID** | `webhook-serve-video` |
| **Type** | `n8n-nodes-base.webhook` |
| **TypeVersion** | 2 |
| **Purpose** | HTTP endpoint to serve video files |
| **Path** | `serve-video/:script_id` |
| **Method** | GET |
| **Authentication** | None |
| **Response Mode** | `responseNode` |
| **Inputs** | External HTTP GET |
| **Outputs** | `Read Video` |

**Issues:**
- **SECURITY**: No authentication - publicly serves video files

---

### Node 103: Read Video
| Property | Value |
|----------|-------|
| **ID** | `read-video-file` |
| **Type** | `n8n-nodes-base.code` |
| **TypeVersion** | 2 |
| **Purpose** | Read video file from filesystem |
| **Inputs** | `Serve Video` |
| **Outputs** | `Return Video` |

**Issues:** None

---

### Node 104: Return Video
| Property | Value |
|----------|-------|
| **ID** | `respond-video` |
| **Type** | `n8n-nodes-base.respondToWebhook` |
| **TypeVersion** | 1.1 |
| **Purpose** | Return video file as HTTP response |
| **Response Type** | Binary |
| **Headers** | `Content-Type: video/mp4` |
| **Inputs** | `Read Video` |
| **Outputs** | None (terminal) |

**Issues:** None

---

## FILE SERVER SECTION SUMMARY

**Flow:**
```
Serve Audio (GET /serve-audio/:script_id) ──> Read Audio ──> Return Audio
Serve Video (GET /serve-video/:script_id) ──> Read Video ──> Return Video
```

**Issues Found:**
1. **SECURITY**: Both webhooks have no authentication - anyone can access files

---

# CONSOLIDATED ISSUES SUMMARY

## CRITICAL Issues (Workflow Breaking)

### 1. ORPHANED NODE - Specific ID? (Section 2)
**Location:** Node 24 `Specific ID?`
**Problem:** This IF node has NO incoming connections - completely disconnected from the workflow
**Impact:**
- Cannot process specific content_idea_id from webhook
- `Get Specific Idea` node is dead code
- Always falls back to queue-based processing
**Fix Required:** Connect `Webhook` → `Specific ID?` instead of directly to `Respond Immediately`

### 2. BUG - Save B-Roll uses wrong reference (Section 4)
**Location:** Node 44 `Save B-Roll`
**Problem:** Uses `$('Select Videos').first()` instead of current item data
**Impact:** When multiple B-roll videos are downloaded, ALL will be saved with the same scene_index (0), overwriting each other
**Fix Required:** Use the current item's json data instead of referencing back to Select Videos

### 3. BUG - broll_paths filter uses wrong field (Section 5)
**Location:** Node 48 `Prepare TTS Text`
**Problem:** Filters B-roll items by `i.json.fileName` but Save B-Roll sets `saved_path`, not `fileName`
**Impact:** `broll_paths` is ALWAYS empty, meaning:
- B-roll videos are never passed to FFmpeg compositing
- Videos always use solid color background instead of B-roll
**Fix Required:** Change filter from `fileName` to `saved_path`

---

## HIGH Priority Issues (Data Flow Problems)

### 4. DATA REFERENCE - Read Audio File (Section 5)
**Location:** Node 55 `Read Audio File`
**Problem:** Uses `$json.script_id` but input is PATCH response from `Update Asset Voice` which may not contain `script_id`
**Impact:** Could fail to read audio file with undefined path
**Fix Required:** Use `$('Prepare TTS Text').first().json.script_id` instead

### 5. DATA REFERENCE - Upload HeyGen Audio (Section 6)
**Location:** Node 57 `Upload HeyGen Audio`
**Problem:** Uses `$('Read Audio File').first().json.script_id` but Read Audio File outputs binary read result
**Impact:** May fail to construct proper filename in upload URL
**Fix Required:** Reference `$('Prepare TTS Text').first().json.script_id` instead

### 6. POTENTIAL DATA LOSS - Generate Script (Section 3)
**Location:** Node 33 `Generate Script (Grok)`
**Problem:** Body references `$json.pillar`, `$json.suggested_hook`, etc. but input is PATCH response
**Impact:** LLM prompt may have empty/missing context fields
**Fix Required:** Reference `$('Normalize Idea').first().json` for these fields

---

## MEDIUM Priority Issues (Configuration)

### 7. MODEL NAME MISMATCH (Sections 1 & 3)
**Location:** Nodes 16 & 33 (Analyze with Grok, Generate Script)
**Problem:** Uses `x-ai/grok-4.1-fast` but CLAUDE.md specifies `x-ai/grok-4-1106`
**Impact:** May use wrong model or fail if model name invalid
**Fix Required:** Verify correct model name and update

### 8. SCHEDULE TRIGGER DISABLED (Section 2)
**Location:** Node 21 `Schedule Trigger`
**Problem:** Pipeline auto-trigger is disabled
**Impact:** Pipeline won't auto-run every 15 minutes - only manual webhook trigger works
**Fix Required:** Enable when ready for production

---

## LOW Priority Issues (Security)

### 9. NO AUTHENTICATION on Webhooks
**Locations:**
- Node 3 `Scrape Webhook`
- Node 22 `Webhook` (Pipeline Trigger)
- Node 99 `Serve Audio`
- Node 102 `Serve Video`

**Problem:** All webhooks have `authentication: none`
**Impact:** Anyone can:
- Trigger scrapes (API costs)
- Trigger pipeline runs (API costs, resource usage)
- Access audio/video files
**Fix Required:** Add authentication (API key, basic auth, or header auth)

---

## Summary Statistics

| Category | Count |
|----------|-------|
| **CRITICAL** | 3 |
| **HIGH** | 3 |
| **MEDIUM** | 2 |
| **LOW (Security)** | 4 webhooks |
| **Total Nodes** | 104 |
| **Nodes with Issues** | ~12 |

---

## Recommended Fix Priority

1. **First**: Fix orphaned `Specific ID?` node connection
2. **Second**: Fix `broll_paths` filter to use `saved_path`
3. **Third**: Fix `Save B-Roll` to use current item data
4. **Fourth**: Fix data references for `script_id` in TTS/HeyGen sections
5. **Fifth**: Verify and fix model names
6. **Last**: Add authentication to webhooks

