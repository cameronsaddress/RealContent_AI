import axios from 'axios';

export const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Content Ideas
export const getContentIdeas = async (params = {}) => {
  const { data } = await api.get('/api/content-ideas', { params });
  return data;
};

export const getContentIdea = async (id) => {
  const { data } = await api.get(`/api/content-ideas/${id}`);
  return data;
};

export const createContentIdea = async (idea) => {
  const { data } = await api.post('/api/content-ideas', idea);
  return data;
};

export const updateContentIdea = async (id, idea) => {
  const { data } = await api.patch(`/api/content-ideas/${id}`, idea);
  return data;
};

export const deleteContentIdea = async (id) => {
  const { data } = await api.delete(`/api/content-ideas/${id}`);
  return data;
};

export const bulkApproveIdeas = async (ids) => {
  const { data } = await api.post('/api/content-ideas/bulk-approve', ids);
  return data;
};

export const bulkRejectIdeas = async (ids) => {
  const { data } = await api.post('/api/content-ideas/bulk-reject', ids);
  return data;
};

// Scripts
export const getScripts = async (params = {}) => {
  const { data } = await api.get('/api/scripts', { params });
  return data;
};

export const getScript = async (id) => {
  const { data } = await api.get(`/api/scripts/${id}`);
  return data;
};

export const createScript = async (script) => {
  const { data } = await api.post('/api/scripts', script);
  return data;
};

export const updateScript = async (id, script) => {
  const { data } = await api.patch(`/api/scripts/${id}`, script);
  return data;
};

export const deleteScript = async (id) => {
  const { data } = await api.delete(`/api/scripts/${id}`);
  return data;
};

// Assets
export const getAssets = async (params = {}) => {
  const { data } = await api.get('/api/assets', { params });
  return data;
};

export const getAsset = async (id) => {
  const { data } = await api.get(`/api/assets/${id}`);
  return data;
};

export const updateAsset = async (id, asset) => {
  const { data } = await api.patch(`/api/assets/${id}`, asset);
  return data;
};

// Published
export const getPublished = async (params = {}) => {
  const { data } = await api.get('/api/published', { params });
  return data;
};

// Analytics
export const getAnalytics = async (params = {}) => {
  const { data } = await api.get('/api/analytics', { params });
  return data;
};

// Pipeline
export const getPipelineOverview = async (params = {}) => {
  const { data } = await api.get('/api/pipeline/overview', { params });
  return data;
};

export const getPipelineStats = async () => {
  const { data } = await api.get('/api/pipeline/stats');
  return data;
};

// Scraper
export const runScrape = async (scrapeConfig) => {
  const { data } = await api.post('/api/scrape/run', scrapeConfig);
  return data;
};

export const triggerTestPipeline = async (data = {}) => {
  // Direct call to n8n webhook, bypassing the backend API wrapper for this specific test case
  // Use axios directly to avoid the baseURL if needed, but here we can just post to the absolute URL
  const response = await axios.post('http://100.83.153.43:5678/webhook-test/trigger-pipeline', data, {
    headers: { 'Content-Type': 'application/json' }
  });
  return response.data;
};

export const getScrapeRuns = async (params = {}) => {
  const { data } = await api.get('/api/scrape/runs', { params });
  return data;
};

export const getScrapeRun = async (id) => {
  const { data } = await api.get(`/api/scrape/runs/${id}`);
  return data;
};

export const useTrendAsIdea = async (trend) => {
  const { data } = await api.post('/api/scrape/use-trend', trend);
  return data;
};

// Niche Presets
export const getNichePresets = async () => {
  const { data } = await api.get('/api/niche-presets');
  return data;
};

export const createNichePreset = async (preset) => {
  const { data } = await api.post('/api/niche-presets', preset);
  return data;
};

export const deleteNichePreset = async (id) => {
  const { data } = await api.delete(`/api/niche-presets/${id}`);
  return data;
};

// Character & Settings
export const getVoices = async () => {
  const { data } = await api.get('/api/settings/voices');
  return data;
};

export const getAvatars = async () => {
  const { data } = await api.get('/api/settings/avatars');
  return data;
};

export const getCharacterConfig = async () => {
  const { data } = await api.get('/api/config/character');
  return data;
};

export const saveCharacterConfig = async (config) => {
  const { data } = await api.post('/api/config/character', config);
  return data;
};

export const generateAvatarImage = async (promptEnhancements) => {
  const { data } = await api.post('/api/generate-avatar-image', { prompt_enhancements: promptEnhancements });
  return data;
};

// Music
export const uploadMusic = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  // axios automatically sets boundary content-type for FormData, but we can be explicit or just let interceptor handle it?
  // Our api instance has application/json default. We must override.
  const { data } = await api.post('/api/music/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return data;
};

export const getMusicInfo = async () => {
  const { data } = await api.get('/api/music/info');
  return data;
};

export const getMusicFiles = async () => {
  const { data } = await api.get('/api/music/files');
  return data;
};

export const activateMusic = async (filename) => {
  const { data } = await api.post('/api/music/activate', { filename });
  return data;
};

export const deleteMusic = async (filename) => {
  const { data } = await api.delete(`/api/music/${filename}`);
  return data;
};

// Character & Settings (Upload Extensions)
export const uploadAvatarImage = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const { data } = await api.post('/api/upload/avatar', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return data;
};

export const getAvatarImages = async () => {
  const { data } = await api.get('/api/avatar-images');
  return data;
};

// Pipeline Settings
export const getAudioSettings = async () => {
  const { data } = await api.get('/api/settings/audio');
  return data;
};

export const updateAudioSettings = async (settings) => {
  const { data } = await api.put('/api/settings/audio', settings);
  return data;
};

export const getVideoSettings = async () => {
  const { data } = await api.get('/api/settings/video');
  return data;
};

export const updateVideoSettings = async (settings) => {
  const { data } = await api.put('/api/settings/video', settings);
  return data;
};

export const getLLMSettings = async () => {
  const { data } = await api.get('/api/settings/llm');
  return data;
};

export const updateLLMSetting = async (key, update) => {
  const { data } = await api.put(`/api/settings/llm/${key}`, update);
  return data;
};

export const getAllSettings = async () => {
  const { data } = await api.get('/api/settings/all');
  return data;
};

// Brand Persona Settings
export const getBrandPersona = async () => {
  const { data } = await api.get('/api/settings/persona');
  return data;
};

export const updateBrandPersona = async (persona) => {
  const { data } = await api.put('/api/settings/persona', persona);
  return data;
};

// New categorized endpoints for Character page
export const getHeyGenAvatars = async (params = {}) => {
  const { data } = await api.get('/api/settings/heygen-avatars', { params });
  return data;
};

export const getElevenLabsVoices = async () => {
  const { data } = await api.get('/api/settings/elevenlabs-voices');
  return data;
};

// API Credits
export const getApiCredits = async () => {
  const { data } = await api.get('/api/credits');
  return data;
};

// Viral Clip Factory
export const getInfluencers = async () => {
  const { data } = await api.get('/api/viral/influencers');
  return data;
};

export const createInfluencer = async (influencer) => {
  const { data } = await api.post('/api/viral/influencers', influencer);
  return data;
};

export const fetchInfluencerVideos = async (id) => {
  const { data } = await api.post(`/api/viral/influencers/${id}/fetch`);
  return data;
};

export const getVideoDetails = async (id) => {
  const { data } = await api.get(`/api/viral/videos/${id}/details`);
  return data;
};

export const analyzeVideo = async (id) => {
  const { data } = await api.post(`/api/viral/videos/${id}/analyze`);
  return data;
};

export const getInfluencerVideos = async (id) => {
  const { data } = await api.get(`/api/viral/influencers/${id}/videos`);
  return data;
};

export const getViralClips = async () => {
  const { data } = await api.get(`/api/viral/viral-clips`);
  return data;
};

export default api;

// Viral Factory
export const getViralMusic = async () => {
  const { data } = await api.get('/api/viral/music');
  return data;
};

// Font Management
export const getViralFonts = async () => {
  const { data } = await api.get('/api/viral/fonts');
  return data;
};

export const deleteViralFont = async (filename) => {
  const { data } = await api.delete(`/api/viral/fonts/${filename}`);
  return data;
};

export const downloadGoogleFont = async (fontName) => {
  const { data } = await api.post('/api/viral/fonts/google', { font_name: fontName });
  return data;
};

// B-Roll Management
export const getBrollClips = async () => {
  const { data } = await api.get('/api/viral/broll');
  return data;
};

export const uploadBrollFromYoutube = async (youtubeUrl, category = null) => {
  const { data } = await api.post('/api/viral/broll/upload-youtube', { youtube_url: youtubeUrl, category });
  return data;
};

export const getBrollUploadStatus = async (jobId) => {
  const { data } = await api.get(`/api/viral/broll/status/${jobId}`);
  return data;
};

export const retagBrollClips = async (force = false, limit = 0) => {
  const { data } = await api.post('/api/viral/broll/retag', { force, limit });
  return data;
};

export const deleteBrollClip = async (filename) => {
  const { data } = await api.delete(`/api/viral/broll/${filename}`);
  return data;
};
