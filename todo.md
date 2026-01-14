# Video Pipeline TODO

## Video Download Integration

### Current Status (yt-dlp Local Docker)

| Platform | Status | Issue |
|----------|--------|-------|
| YouTube | Works | Some format warnings but downloads work |
| TikTok | Blocked | "IP address is blocked" - datacenter IP block |
| Instagram | Blocked | "API is not granting access" - requires cookies/login |
| Twitter/X | Partial | Works only if video exists in tweet, some failures |
| Facebook | Untested | Likely similar issues to Instagram |

### Recommended Apify Providers for Gaps

| Platform | Best Provider | Cost | Notes |
|----------|--------------|------|-------|
| TikTok | [clockworks/tiktok-video-scraper](https://apify.com/clockworks/tiktok-video-scraper) (6,292 users) | ~$0.01/video | Most popular, PPR model |
| Instagram | [apify/instagram-reel-scraper](https://apify.com/apify/instagram-reel-scraper) (67,733 users) | ~$2.60/1000 ($0.0026/reel) | Official Apify actor |
| Twitter/X | [bytepulselabs/x-video-downloader](https://apify.com/bytepulselabs/x-video-downloader) | $0.006/start + $0.06/10MB | Pay-per-event model |
| Facebook | [apify/facebook-reels-scraper](https://apify.com/apify/facebook-reels-scraper) | ~$2.60/1000 (estimated) | Official Apify actor |

### Estimated Monthly Costs (100 videos/month)

| Platform | Cost for 100 videos |
|----------|---------------------|
| TikTok | ~$1.00 |
| Instagram | ~$0.26 |
| Twitter/X | ~$1-2 (depends on video size) |
| Facebook | ~$0.26 |
| **Total** | **~$2.50-3.50/month** |

> Apify gives **$5 free credits/month** on the free plan, so light usage (~150-200 videos) could be free.

### Action Items

- [ ] Integrate Apify TikTok scraper as fallback in video-processor
- [ ] Integrate Apify Instagram scraper as fallback in video-processor
- [ ] Integrate Apify Twitter/X scraper as fallback in video-processor
- [ ] Integrate Apify Facebook scraper as fallback in video-processor
- [ ] Add APIFY_API_KEY to video-processor container environment
- [ ] Test end-to-end with real content from each platform
