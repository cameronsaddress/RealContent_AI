import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

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

export default api;
