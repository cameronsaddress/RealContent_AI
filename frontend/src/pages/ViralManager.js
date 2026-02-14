import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import api, {
    getInfluencers, createInfluencer, deleteInfluencer, fetchInfluencerVideos,
    analyzeVideo, getVideoDetails, getInfluencerVideos, getViralClips, API_URL,
    getBrollClips, uploadBrollFromYoutube, getBrollUploadStatus, retagBrollClips, deleteBrollClip,
    getEffectsCatalog, updateClipEffects, getDownloadProgress,
    getInfluencerAutoMode, updateInfluencerAutoMode,
    getPublishingStats, getPublishingQueue, getPublishingConfigs,
    createPublishingConfig, updatePublishingConfig, deletePublishingConfig,
    approveQueueItem, rejectQueueItem, publishNow, getBlotatoAccounts,
    publishClipNow, getPublishingConfigsForInfluencer
} from '../api';
import './ViralManager.css'; // We'll assume standard styling or create it

const ViralManager = () => {
    const [activeTab, setActiveTab] = useState('influencers');
    const [influencers, setInfluencers] = useState([]);
    const [showAddModal, setShowAddModal] = useState(false);
    const [newInfluencer, setNewInfluencer] = useState({ name: '', platform: 'youtube', channel_url: '', persona_id: 1 });

    // For Video Browsing
    const [selectedInfluencer, setSelectedInfluencer] = useState(null);
    const [videos, setVideos] = useState([]);
    const [loadingVideos, setLoadingVideos] = useState(false);
    const [renderingClips, setRenderingClips] = useState({}); // Track rendering state by clip ID

    // For Clips
    const [clips, setClips] = useState([]);
    const [viewingClip, setViewingClip] = useState(null);
    const [clipsFilterInfluencer, setClipsFilterInfluencer] = useState(null); // Filter clips by influencer

    // For Templates
    const [templates, setTemplates] = useState([]);

    // For Effects
    const [effectsModalClip, setEffectsModalClip] = useState(null);
    const [effectsCatalog, setEffectsCatalog] = useState(null);
    const [effectsLocalState, setEffectsLocalState] = useState({}); // Lifted from EffectsModal to fix Rules of Hooks

    // For B-Roll
    const [brollClips, setBrollClips] = useState([]);
    const [brollMetadata, setBrollMetadata] = useState(null);
    const [brollYoutubeUrl, setBrollYoutubeUrl] = useState('');
    const [brollCategory, setBrollCategory] = useState('');
    const [brollUploadJobs, setBrollUploadJobs] = useState([]);
    const [brollLoading, setBrollLoading] = useState(false);
    const [brollTagging, setBrollTagging] = useState(false);
    const [brollFilter, setBrollFilter] = useState('all');
    const [brollHasMore, setBrollHasMore] = useState(true);
    const [brollLoadingMore, setBrollLoadingMore] = useState(false);
    const brollLoadMoreRef = useRef(null); // Intersection observer ref

    // For Auto-Mode
    const [autoModeModalInfluencer, setAutoModeModalInfluencer] = useState(null);
    const [autoModeSettings, setAutoModeSettings] = useState(null);
    const [autoModeLoading, setAutoModeLoading] = useState(false);

    // For Publishing
    const [publishingStats, setPublishingStats] = useState(null);
    const [publishingQueue, setPublishingQueue] = useState([]);
    const [publishingConfigs, setPublishingConfigs] = useState([]);
    const [queueFilter, setQueueFilter] = useState('all');
    const [showConfigModal, setShowConfigModal] = useState(false);
    const [editingConfig, setEditingConfig] = useState(null);
    const [configForm, setConfigForm] = useState({
        blotato_account_id: '',
        platforms: ['tiktok'],
        posts_per_day: 3,
        posting_hours: [9, 12, 18],
        days_active: ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'],
        require_approval: true,
        is_active: true
    });

    // Download progress tracking
    const [downloadProgress, setDownloadProgress] = useState({}); // { videoId: { bytes, speed, fragment } }

    // Delete confirmation
    const [deleteConfirmInfluencer, setDeleteConfirmInfluencer] = useState(null);
    const [deleteLoading, setDeleteLoading] = useState(false);

    // Refs for cleanup on unmount
    const mountedRef = useRef(true);
    const pollTimeoutsRef = useRef([]);

    // Refs for callback functions to avoid stale closures in intervals
    const loadVideosRef = useRef(null);
    const loadClipsRef = useRef(null);

    useEffect(() => {
        mountedRef.current = true;
        return () => {
            mountedRef.current = false;
            // Clear all active polling timeouts on unmount
            pollTimeoutsRef.current.forEach(clearTimeout);
            pollTimeoutsRef.current = [];
        };
    }, []);

    useEffect(() => {
        loadInfluencers();
        loadClips();
        loadTemplates();
        loadBroll();
        loadPublishingData();
        // Load auto-publish setting
        api.get('/api/viral/settings/auto-publish').then(res => {
            setAutoPublishEnabled(res.data?.enabled || false);
        }).catch(() => {});
    }, []);

    const loadPublishingData = async () => {
        try {
            const [statsData, queueData, configsData] = await Promise.all([
                getPublishingStats(),
                getPublishingQueue(),
                getPublishingConfigs()
            ]);
            setPublishingStats(statsData);
            setPublishingQueue(queueData);
            setPublishingConfigs(configsData);
        } catch (e) {
            console.error('Failed to load publishing data:', e);
        }
    };

    const handleApproveQueueItem = async (itemId) => {
        try {
            await approveQueueItem(itemId);
            loadPublishingData();
        } catch (e) {
            alert('Failed to approve: ' + (e.response?.data?.detail || e.message));
        }
    };

    const handleRejectQueueItem = async (itemId) => {
        const reason = window.prompt('Rejection reason (optional):');
        try {
            await rejectQueueItem(itemId, reason);
            loadPublishingData();
        } catch (e) {
            alert('Failed to reject: ' + (e.response?.data?.detail || e.message));
        }
    };

    const handlePublishNow = async (itemId) => {
        if (!window.confirm('Publish this clip immediately?')) return;
        try {
            await publishNow(itemId);
            loadPublishingData();
        } catch (e) {
            alert('Failed to publish: ' + (e.response?.data?.detail || e.message));
        }
    };

    const openConfigModal = (config = null) => {
        if (config) {
            setEditingConfig(config);
            setConfigForm({
                blotato_account_id: config.blotato_account_id || '',
                platforms: config.platforms || ['tiktok'],
                posts_per_day: config.posts_per_day || 3,
                posting_hours: config.posting_hours || [9, 12, 18],
                days_active: config.days_active || ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'],
                require_approval: config.require_approval !== false,
                is_active: config.is_active !== false
            });
        } else {
            setEditingConfig(null);
            setConfigForm({
                blotato_account_id: '',
                platforms: ['tiktok'],
                posts_per_day: 3,
                posting_hours: [9, 12, 18],
                days_active: ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'],
                require_approval: true,
                is_active: true
            });
        }
        setShowConfigModal(true);
    };

    const savePublishingConfig = async () => {
        try {
            if (editingConfig) {
                await updatePublishingConfig(editingConfig.id, configForm);
            } else {
                await createPublishingConfig(configForm);
            }
            setShowConfigModal(false);
            loadPublishingData();
        } catch (e) {
            alert('Failed to save config: ' + (e.response?.data?.detail || e.message));
        }
    };

    const handleDeleteConfig = async (configId) => {
        if (!window.confirm('Delete this publishing configuration?')) return;
        try {
            await deletePublishingConfig(configId);
            loadPublishingData();
        } catch (e) {
            alert('Failed to delete: ' + (e.response?.data?.detail || e.message));
        }
    };

    const togglePlatform = (platform) => {
        const platforms = configForm.platforms.includes(platform)
            ? configForm.platforms.filter(p => p !== platform)
            : [...configForm.platforms, platform];
        setConfigForm({ ...configForm, platforms });
    };

    const toggleDay = (day) => {
        const days = configForm.days_active.includes(day)
            ? configForm.days_active.filter(d => d !== day)
            : [...configForm.days_active, day];
        setConfigForm({ ...configForm, days_active: days });
    };

    const updatePostingHours = (value) => {
        const hours = value.split(',').map(h => parseInt(h.trim())).filter(h => !isNaN(h) && h >= 0 && h <= 23);
        setConfigForm({ ...configForm, posting_hours: hours });
    };

    const filteredQueue = useMemo(() => {
        return publishingQueue.filter(item => {
            if (queueFilter === 'all') return true;
            return item.status === queueFilter;
        });
    }, [publishingQueue, queueFilter]);

    // Memoized sorted videos list (deduplicated by URL)
    const sortedVideos = useMemo(() => {
        return Array.from(new Map(videos.map(item => [item.url, item])).values())
            .sort((a, b) => new Date(b.publication_date || b.created_at || 0) - new Date(a.publication_date || a.created_at || 0));
    }, [videos]);

    // Memoized filtered and sorted clips list
    const filteredSortedClips = useMemo(() => {
        return [...clips]
            .filter(c => !clipsFilterInfluencer || c.influencer_id === clipsFilterInfluencer.id)
            .sort((a, b) => {
                // Sort by updated_at (render time) first, then created_at
                const aTime = new Date(a.updated_at || a.created_at || 0);
                const bTime = new Date(b.updated_at || b.created_at || 0);
                return bTime - aTime;
            });
    }, [clips, clipsFilterInfluencer]);

    // Load more B-roll clips (server-side pagination)
    const loadMoreBroll = useCallback(async () => {
        if (brollLoadingMore || !brollHasMore) return;
        setBrollLoadingMore(true);
        try {
            const category = brollFilter === 'all' ? null : brollFilter;
            const data = await getBrollClips(50, brollClips.length, category);
            setBrollClips(prev => [...prev, ...(data.clips || [])]);
            setBrollHasMore(data.pagination?.has_more || false);
        } catch (e) {
            console.error('Failed to load more B-roll:', e);
        } finally {
            setBrollLoadingMore(false);
        }
    }, [brollLoadingMore, brollHasMore, brollFilter, brollClips.length]);

    // Reset and reload when filter changes
    useEffect(() => {
        const loadFiltered = async () => {
            setBrollClips([]);
            setBrollHasMore(true);
            try {
                const category = brollFilter === 'all' ? null : brollFilter;
                const data = await getBrollClips(50, 0, category);
                setBrollClips(data.clips || []);
                setBrollMetadata(data.metadata || null);
                setBrollHasMore(data.pagination?.has_more || false);
            } catch (e) {
                console.error('Failed to load B-roll:', e);
            }
        };
        if (activeTab === 'broll') {
            loadFiltered();
        }
    }, [brollFilter, activeTab]);

    // Intersection Observer for B-roll lazy loading
    useEffect(() => {
        if (activeTab !== 'broll' || !brollHasMore) return;

        const observer = new IntersectionObserver(
            (entries) => {
                if (entries[0].isIntersecting) {
                    loadMoreBroll();
                }
            },
            { threshold: 0.1 }
        );

        const loadMoreElement = brollLoadMoreRef.current;
        if (loadMoreElement) {
            observer.observe(loadMoreElement);
        }

        return () => {
            if (loadMoreElement) {
                observer.unobserve(loadMoreElement);
            }
        };
    }, [activeTab, brollHasMore, loadMoreBroll]);

    const formatPublishDate = (dateStr) => {
        if (!dateStr) return '-';
        return new Date(dateStr).toLocaleString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: 'numeric',
            minute: '2-digit',
            hour12: true
        });
    };

    // Poll download progress for videos in "downloading" state
    useEffect(() => {
        const downloadingVideos = videos.filter(v => v.status?.toLowerCase() === 'downloading');
        if (downloadingVideos.length === 0) return;

        const pollProgress = async () => {
            for (const v of downloadingVideos) {
                try {
                    const data = await getDownloadProgress(v.id);
                    setDownloadProgress(prev => {
                        const prevBytes = prev[v.id]?.bytes_downloaded || 0;
                        const speed = prevBytes > 0 ? (data.bytes_downloaded - prevBytes) / 5 : 0; // bytes per second (5s interval)
                        return { ...prev, [v.id]: { ...data, speed: speed > 0 ? speed : prev[v.id]?.speed || 0 } };
                    });
                } catch (e) { /* ignore */ }
            }
        };
        pollProgress();
        const interval = setInterval(pollProgress, 5000);
        return () => clearInterval(interval);
    }, [videos]);

    const loadClips = async () => {
        try {
            const data = await getViralClips();
            setClips(data);
        } catch (e) { console.error(e); }
    };

    // Keep refs in sync with latest function references (avoids stale closures in intervals)
    useEffect(() => {
        loadClipsRef.current = loadClips;
    });

    const loadBroll = async () => {
        try {
            const category = brollFilter === 'all' ? null : brollFilter;
            const data = await getBrollClips(50, 0, category);
            setBrollClips(data.clips || []);
            setBrollMetadata(data.metadata || null);
            setBrollHasMore(data.pagination?.has_more || false);
        } catch (e) { console.error('Failed to load B-roll:', e); }
    };

    const handleBrollUpload = async (e) => {
        e.preventDefault();
        if (!brollYoutubeUrl.trim()) return;

        setBrollLoading(true);
        try {
            const result = await uploadBrollFromYoutube(brollYoutubeUrl, brollCategory || null);
            setBrollUploadJobs(prev => [...prev, { job_id: result.job_id, status: 'started', url: brollYoutubeUrl }]);
            setBrollYoutubeUrl('');
            setBrollCategory('');
            // Start polling for status
            pollBrollJob(result.job_id);
        } catch (e) {
            alert('Upload failed: ' + (e.response?.data?.detail || e.message));
        } finally {
            setBrollLoading(false);
        }
    };

    const pollBrollJob = async (jobId) => {
        const poll = async () => {
            // Check if component is still mounted before proceeding
            if (!mountedRef.current) return;

            try {
                const status = await getBrollUploadStatus(jobId);
                // Check again after async call
                if (!mountedRef.current) return;

                setBrollUploadJobs(prev => prev.map(j =>
                    j.job_id === jobId ? { ...j, ...status } : j
                ));
                if (status.status === 'complete' || status.status === 'error') {
                    loadBroll(); // Refresh clips list
                    return;
                }
                // Store timeout ID for cleanup on unmount
                const timeoutId = setTimeout(poll, 3000);
                pollTimeoutsRef.current.push(timeoutId);
            } catch (e) {
                console.error('Poll error:', e);
            }
        };
        poll();
    };

    const handleRetagBroll = async (force = false) => {
        setBrollTagging(true);
        try {
            const result = await retagBrollClips(force, 0);
            alert(`Tagged ${result.clips_tagged} clips`);
            loadBroll();
        } catch (e) {
            alert('Tagging failed: ' + (e.response?.data?.detail || e.message));
        } finally {
            setBrollTagging(false);
        }
    };

    const handleDeleteBroll = async (filename) => {
        if (!window.confirm(`Delete ${filename}?`)) return;
        try {
            await deleteBrollClip(filename);
            loadBroll();
        } catch (e) {
            alert('Delete failed: ' + (e.response?.data?.detail || e.message));
        }
    };

    // Auto-Mode Functions
    const handleAutoModeToggle = async (influencer, e) => {
        e.stopPropagation();
        try {
            const newState = !influencer.auto_mode_enabled;
            await updateInfluencerAutoMode(influencer.id, { auto_mode_enabled: newState });
            loadInfluencers();
        } catch (err) {
            alert('Failed to toggle auto-mode: ' + (err.response?.data?.detail || err.message));
        }
    };

    const openAutoModeSettings = async (influencer, e) => {
        e.stopPropagation();
        setAutoModeLoading(true);
        try {
            const settings = await getInfluencerAutoMode(influencer.id);
            setAutoModeSettings(settings);
            setAutoModeModalInfluencer(influencer);
        } catch (err) {
            alert('Failed to load auto-mode settings: ' + (err.response?.data?.detail || err.message));
        } finally {
            setAutoModeLoading(false);
        }
    };

    const saveAutoModeSettings = async () => {
        if (!autoModeModalInfluencer || !autoModeSettings) return;
        setAutoModeLoading(true);
        try {
            await updateInfluencerAutoMode(autoModeModalInfluencer.id, autoModeSettings);
            setAutoModeModalInfluencer(null);
            setAutoModeSettings(null);
            loadInfluencers();
        } catch (err) {
            alert('Failed to save settings: ' + (err.response?.data?.detail || err.message));
        } finally {
            setAutoModeLoading(false);
        }
    };

    const loadTemplates = async () => {
        try {
            const res = await api.get('/api/viral/templates');
            setTemplates(res.data);
        } catch (e) { console.error('Failed to load templates:', e); }
    };

    const handleTemplateChange = async (clipId, templateId) => {
        try {
            await api.put(`/api/viral/viral-clips/${clipId}/template?template_id=${templateId}`);
            // Refresh both video clips and standalone clips list
            if (selectedInfluencer) {
                loadVideos(selectedInfluencer.id, true);
            }
            loadClips();
        } catch (e) {
            console.error('Failed to update template:', e);
        }
    };

    // Effect badges display
    const EffectBadges = ({ clip, compact = false }) => {
        const effects = clip.render_metadata?.director_effects;
        if (!effects || Object.keys(effects).length === 0) return null;

        const badges = [];
        if (effects.color_grade) badges.push({ label: effects.color_grade, color: '#8b5cf6' });
        if (effects.camera_shake) badges.push({ label: 'shake', color: '#f59e0b' });
        if (effects.retro_glow) badges.push({ label: 'glow', color: '#ec4899' });
        if (effects.temporal_trail) badges.push({ label: 'trail', color: '#6366f1' });
        if (effects.wave_displacement) badges.push({ label: 'wave', color: '#14b8a6' });
        if (effects.speed_ramps?.length) badges.push({ label: `${effects.speed_ramps.length}x ramp`, color: '#f97316' });
        if (effects.caption_style && effects.caption_style !== 'standard') badges.push({ label: effects.caption_style, color: '#06b6d4' });
        if (effects.beat_sync) badges.push({ label: 'beat', color: '#22c55e' });
        if (effects.datamosh_segments?.length) badges.push({ label: 'datamosh', color: '#dc2626' });
        if (effects.pixel_sort_segments?.length) badges.push({ label: 'pxsort', color: '#7c3aed' });
        if (effects.transition) badges.push({ label: effects.transition, color: '#a855f7' });

        if (badges.length === 0) return null;

        return (
            <div className={`effect-badges ${compact ? 'compact' : ''}`}>
                {badges.slice(0, compact ? 3 : 6).map((b, i) => (
                    <span key={i} className="effect-badge" style={{ backgroundColor: b.color }}
                          onClick={(e) => { e.stopPropagation(); if (!compact) openEffectsModal(clip); }}>
                        {b.label}
                    </span>
                ))}
                {!compact && <button className="fx-edit-btn" onClick={(e) => { e.stopPropagation(); openEffectsModal(clip); }}>FX</button>}
            </div>
        );
    };

    // Open effects modal
    const openEffectsModal = async (clip) => {
        if (!effectsCatalog) {
            try {
                const catalog = await getEffectsCatalog();
                setEffectsCatalog(catalog);
            } catch (e) { console.error('Failed to load effects catalog:', e); }
        }
        // Initialize local effects state when opening modal (fixes Rules of Hooks)
        const currentEffects = clip.render_metadata?.director_effects || {};
        setEffectsLocalState({...currentEffects});
        setEffectsModalClip(clip);
    };

    // Save effects override
    const saveEffectsOverride = async (clipId, effects) => {
        try {
            await updateClipEffects(clipId, effects);
            setEffectsModalClip(null);
            // Refresh clips list
            if (selectedInfluencer) {
                const data = await getInfluencerVideos(selectedInfluencer.id);
                setVideos(data);
            }
            const allClips = await getViralClips();
            setClips(allClips);
        } catch (e) { console.error('Failed to save effects:', e); }
    };

    // Effects override modal component (uses lifted state to fix Rules of Hooks)
    const EffectsModal = () => {
        if (!effectsModalClip) return null;
        // Use parent's state instead of local useState (fixes Rules of Hooks violation)
        const localEffects = effectsLocalState;
        const setLocalEffects = setEffectsLocalState;

        const updateLocal = (key, value) => setLocalEffects(prev => ({...prev, [key]: value}));

        return (
            <div className="modal-overlay" onClick={() => setEffectsModalClip(null)}>
                <div className="effects-modal" onClick={e => e.stopPropagation()}>
                    <div className="modal-header">
                        <h3>Effects Override</h3>
                        <button className="close-btn" onClick={() => setEffectsModalClip(null)}>X</button>
                    </div>
                    <div className="effects-form">
                        <div className="effect-group">
                            <label>Color Grade</label>
                            <select value={localEffects.color_grade || ''} onChange={e => updateLocal('color_grade', e.target.value || null)}>
                                <option value="">AI Default</option>
                                {(effectsCatalog?.color_grades || []).map(g => (
                                    <option key={g.id} value={g.id}>{g.label} - {g.description}</option>
                                ))}
                            </select>
                        </div>
                        <div className="effect-group">
                            <label>Caption Style</label>
                            <select value={localEffects.caption_style || ''} onChange={e => updateLocal('caption_style', e.target.value || null)}>
                                <option value="">AI Default</option>
                                {(effectsCatalog?.caption_styles || []).map(s => (
                                    <option key={s.id} value={s.id}>{s.label}</option>
                                ))}
                            </select>
                        </div>
                        <div className="effect-group">
                            <label>B-Roll Transition</label>
                            <select value={localEffects.transition || ''} onChange={e => updateLocal('transition', e.target.value || null)}>
                                <option value="">AI Default</option>
                                {(effectsCatalog?.transitions || []).map(t => (
                                    <option key={t.id} value={t.id}>{t.label}</option>
                                ))}
                            </select>
                        </div>
                        <div className="effect-group toggles">
                            <label><input type="checkbox" checked={!!localEffects.camera_shake} onChange={e => updateLocal('camera_shake', e.target.checked ? {intensity: 8, frequency: 2.0} : null)} /> Camera Shake</label>
                            <label><input type="checkbox" checked={!!localEffects.retro_glow} onChange={e => updateLocal('retro_glow', e.target.checked ? 0.3 : null)} /> Retro Glow</label>
                            <label><input type="checkbox" checked={!!localEffects.beat_sync} onChange={e => updateLocal('beat_sync', e.target.checked || null)} /> Beat Sync</label>
                            <label><input type="checkbox" checked={!!localEffects.audio_saturation} onChange={e => updateLocal('audio_saturation', e.target.checked || null)} /> Audio Saturation</label>
                        </div>
                        <div className="effect-group">
                            <label>Pulse Intensity ({localEffects.pulse_intensity || 0.25})</label>
                            <input type="range" min="0.05" max="0.5" step="0.05"
                                value={localEffects.pulse_intensity || 0.25}
                                onChange={e => updateLocal('pulse_intensity', parseFloat(e.target.value))} />
                        </div>
                        <div className="effect-group">
                            <label>VHS Intensity ({localEffects.vhs_intensity || 1.0})</label>
                            <input type="range" min="0" max="2" step="0.1"
                                value={localEffects.vhs_intensity || 1.0}
                                onChange={e => updateLocal('vhs_intensity', parseFloat(e.target.value))} />
                        </div>
                        <div className="effect-group rare-effects">
                            <label className="group-label">Rare Effects (expensive, use sparingly)</label>
                            <label><input type="checkbox"
                                checked={!!(localEffects.datamosh_segments && localEffects.datamosh_segments.length)}
                                onChange={e => updateLocal('datamosh_segments', e.target.checked ? [{start: 5.0, end: 7.0}] : null)}
                            /> Datamosh (frame melt)</label>
                            <label><input type="checkbox"
                                checked={!!(localEffects.pixel_sort_segments && localEffects.pixel_sort_segments.length)}
                                onChange={e => updateLocal('pixel_sort_segments', e.target.checked ? [{start: 10.0, end: 12.0}] : null)}
                            /> Pixel Sort (glitch art)</label>
                        </div>
                    </div>
                    <div className="modal-actions">
                        <button className="reset-btn" onClick={() => setLocalEffects({})}>Reset to AI</button>
                        <button className="save-btn" onClick={() => saveEffectsOverride(effectsModalClip.id, localEffects)}>Save & Re-render</button>
                    </div>
                </div>
            </div>
        );
    };

    // Template selector dropdown component
    const TemplateSelector = ({ clip, compact = false }) => {
        const currentTemplate = templates.find(t => t.id === clip.template_id);
        const recommendedTemplate = templates.find(t => t.id === clip.recommended_template_id);

        return (
            <div className={`template-selector ${compact ? 'compact' : ''}`}>
                <select
                    value={clip.template_id || ''}
                    onChange={(e) => handleTemplateChange(clip.id, parseInt(e.target.value) || 0)}
                    onClick={(e) => e.stopPropagation()}
                    className="template-dropdown"
                    title={currentTemplate ? currentTemplate.description : 'No template selected'}
                >
                    <option value="">No Template</option>
                    {templates.map(t => (
                        <option key={t.id} value={t.id}>
                            {t.name}{t.id === clip.recommended_template_id ? ' ★' : ''}
                        </option>
                    ))}
                </select>
                {recommendedTemplate && !compact && clip.template_id !== clip.recommended_template_id && (
                    <span className="template-recommendation" title={`AI recommended: ${recommendedTemplate.name}`}>
                        💡 {recommendedTemplate.name}
                    </span>
                )}
            </div>
        );
    };

    const loadInfluencers = async () => {
        try {
            const data = await getInfluencers();
            setInfluencers(data);
        } catch (error) {
            console.error('Failed to load influencers:', error);
        }
    };

    const handleAddInfluencer = async (e) => {
        e.preventDefault();
        try {
            await createInfluencer(newInfluencer);
            setShowAddModal(false);
            setNewInfluencer({ name: '', platform: 'youtube', channel_url: '', persona_id: 1 });
            loadInfluencers();
        } catch (error) {
            console.error('Failed to add influencer:', error);
        }
    };

    const getThumbnail = (video) => {
        if (video.thumbnail_url) return video.thumbnail_url;
        // Fallback for YouTube logic if URL structure known, or just placeholder
        return 'https://via.placeholder.com/320x180?text=No+Thumbnail';
    };

    const formatDuration = (seconds) => {
        if (!seconds) return '';
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}:${s.toString().padStart(2, '0')}`;
    };

    // Calculate transcription progress based on elapsed time and video duration
    const getTranscriptionProgress = (video) => {
        if (!video.processing_started_at || !video.duration) return null;
        const started = new Date(video.processing_started_at);
        const elapsed = (Date.now() - started.getTime()) / 1000; // seconds
        // GPU Whisper is roughly 20-30x real-time, use 25x as estimate
        const estimatedTotal = video.duration / 25;
        const progress = Math.min(99, Math.round((elapsed / estimatedTotal) * 100));
        return progress;
    };

    const handleSelectInfluencer = async (inf) => {
        setSelectedInfluencer(inf);
        setActiveTab('videos');
        loadVideos(inf.id);
    };

    const handleViewInfluencerClips = (inf, e) => {
        e.stopPropagation();
        setClipsFilterInfluencer(inf);
        setActiveTab('clips');
    };

    const handleDeleteInfluencer = async () => {
        if (!deleteConfirmInfluencer) return;
        setDeleteLoading(true);
        try {
            const result = await deleteInfluencer(deleteConfirmInfluencer.id);
            console.log('Deleted influencer:', result);
            // Clear selection if deleted influencer was selected
            if (selectedInfluencer?.id === deleteConfirmInfluencer.id) {
                setSelectedInfluencer(null);
                setVideos([]);
            }
            // Clear clips filter if filtering by deleted influencer
            if (clipsFilterInfluencer?.id === deleteConfirmInfluencer.id) {
                setClipsFilterInfluencer(null);
            }
            setDeleteConfirmInfluencer(null);
            loadInfluencers();
            loadClips();
        } catch (e) {
            console.error('Failed to delete influencer:', e);
            alert('Failed to delete influencer: ' + (e.response?.data?.detail || e.message));
        } finally {
            setDeleteLoading(false);
        }
    };

    const loadVideos = async (id, silent = false) => {
        try {
            const data = await getInfluencerVideos(id);
            setVideos(data);
        } catch (e) {
            console.error(e);
        }
    }

    // Keep ref in sync with latest function reference (avoids stale closure in interval)
    useEffect(() => {
        loadVideosRef.current = loadVideos;
    });

    // Auto-refresh poll - uses ref to avoid stale closure
    useEffect(() => {
        if (!selectedInfluencer) return;

        const hasActiveTasks = videos.some(v => {
            const s = v.status?.toLowerCase() || '';
            const videoActive = s.includes('downloading') || s.includes('transcribing') || s.includes('analyzing') || s.includes('processing');
            const clipsActive = v.clips?.some(c => {
                const cs = c.status?.toLowerCase() || '';
                return cs.includes('rendering') || cs.includes('processing') || cs.includes('queued');
            });
            return videoActive || clipsActive;
        });

        let interval;
        if (hasActiveTasks) {
            interval = setInterval(() => {
                loadVideosRef.current?.(selectedInfluencer.id, true);
            }, 3000);
        }
        return () => clearInterval(interval);
    }, [videos, selectedInfluencer]);

    // Auto-refresh for clips tab - uses ref to avoid stale closure
    useEffect(() => {
        if (activeTab !== 'clips') return;

        const hasActiveClips = clips.some(c => {
            const cs = c.status?.toLowerCase() || '';
            return cs.includes('rendering') || cs.includes('processing') || cs.includes('queued');
        });

        let interval;
        if (hasActiveClips) {
            interval = setInterval(() => {
                loadClipsRef.current?.();
            }, 3000);
        }
        return () => clearInterval(interval);
    }, [clips, activeTab]);

    const handleFetchVideos = async () => {
        if (!selectedInfluencer) return;
        setLoadingVideos(true);
        try {
            const res = await fetchInfluencerVideos(selectedInfluencer.id);
            alert(res.message);
            loadVideos(selectedInfluencer.id);
        } catch (e) {
            alert('Error fetching videos: ' + e.message);
        } finally {
            setLoadingVideos(false);
        }
    };

    const handleAnalyze = async (videoId) => {
        // Removed confirmation as requested
        // if (!window.confirm("...")) return; 
        try {
            const res = await analyzeVideo(videoId);
            console.log(res.message);
            // Refresh videos to trigger the polling loop (which checks for 'downloading' etc)
            if (selectedInfluencer) {
                loadVideos(selectedInfluencer.id);
            }
            loadClips();
        } catch (e) {
            alert(e.message);
        }
    };

    const [publishingClips, setPublishingClips] = useState({});
    const [autoPublishEnabled, setAutoPublishEnabled] = useState(false);

    const handlePublish = async (clipId, recommendedPlatform = null) => {
        // Build prompt with AI recommendation highlighted
        const aiRec = recommendedPlatform ?
            `\n🤖 AI recommends: ${recommendedPlatform === 'tiktok' ? 'TikTok' : 'Reels'}\n` : '';

        // Default to AI recommendation (1=TikTok, 2=Reels) or 3=All
        const defaultChoice = recommendedPlatform === 'tiktok' ? '1' :
                              recommendedPlatform === 'reels' ? '2' : '3';

        const choice = window.prompt(
            `Publish to which platforms?${aiRec}\n` +
            "1 = TikTok only\n" +
            "2 = Reels only\n" +
            "3 = Both TikTok + Reels\n" +
            "4 = X (Twitter) only\n\n" +
            "Enter 1, 2, 3, or 4:",
            defaultChoice
        );
        if (!choice) return;

        let platforms = null; // null = all configured
        if (choice.trim() === "1") platforms = ["tiktok"];
        else if (choice.trim() === "2") platforms = ["reels", "instagram_reels"];
        else if (choice.trim() === "3") platforms = ["tiktok", "reels", "instagram_reels"];
        else if (choice.trim() === "4") platforms = ["twitter"];
        // anything else = all configured (null)

        setPublishingClips(prev => ({ ...prev, [clipId]: true }));
        try {
            const result = await publishClipNow(clipId, platforms);
            alert(`Published to ${result.queued} account(s): ${result.configs.join(', ')}`);
            if (selectedInfluencer) loadVideos(selectedInfluencer.id, true);
            loadClips();
        } catch (e) {
            alert("Publish error: " + (e.response?.data?.detail || e.message));
        } finally {
            setPublishingClips(prev => ({ ...prev, [clipId]: false }));
        }
    };

    const handleRender = async (clipId) => {
        setRenderingClips(prev => ({ ...prev, [clipId]: true }));
        try {
            await api.post(`/api/viral/viral-clips/${clipId}/render`);
            // Refresh both views
            if (selectedInfluencer) {
                loadVideos(selectedInfluencer.id, true);
            }
            loadClips(); // Always refresh clips list
        } catch (e) {
            alert("Render error: " + e.message);
        } finally {
            setRenderingClips(prev => ({ ...prev, [clipId]: false }));
        }
    };

    const [renderingAll, setRenderingAll] = useState(false);

    const handleRenderAll = async () => {
        // Get all clips that can be rendered (pending, error, failed, or ready for re-render)
        const renderableClips = filteredSortedClips.filter(c =>
            c.status === 'pending' || c.status === 'error' || c.status === 'failed' || c.status === 'ready'
        );

        if (renderableClips.length === 0) {
            alert('No clips to render');
            return;
        }

        const confirmed = window.confirm(
            `Render ${renderableClips.length} clip(s)?\n\n` +
            `This will queue all pending/error clips and re-render ready clips.`
        );
        if (!confirmed) return;

        setRenderingAll(true);
        let queued = 0;
        let errors = 0;

        for (const clip of renderableClips) {
            try {
                await api.post(`/api/viral/viral-clips/${clip.id}/render`);
                queued++;
            } catch (e) {
                console.error(`Failed to queue clip ${clip.id}:`, e);
                errors++;
            }
        }

        setRenderingAll(false);
        loadClips();
        if (selectedInfluencer) loadVideos(selectedInfluencer.id, true);

        alert(`Queued ${queued} clip(s) for rendering${errors > 0 ? `, ${errors} failed` : ''}`);
    };

    return (
        <div className="viral-manager-container">
            <div className="header">
                <h2>Content Discovery</h2>
                <div className="tabs">
                    <button className={activeTab === 'influencers' ? 'active' : ''} onClick={() => setActiveTab('influencers')}>Content Sources</button>
                    <button className={activeTab === 'videos' ? 'active' : ''} onClick={() => setActiveTab('videos')}>Videos</button>
                    <button className={activeTab === 'clips' ? 'active' : ''} onClick={() => setActiveTab('clips')}>Clips</button>
                    <button className={activeTab === 'broll' ? 'active' : ''} onClick={() => setActiveTab('broll')}>B-Roll</button>
                    <button className={activeTab === 'publishing' ? 'active' : ''} onClick={() => setActiveTab('publishing')}>
                        Publishing {publishingStats?.pending_approval > 0 && <span className="tab-badge">{publishingStats.pending_approval}</span>}
                    </button>
                </div>
            </div>

            <div className="content">
                {activeTab === 'influencers' && (
                    <div className="influencers-panel">
                        <div className="influencer-grid">
                            {/* Add Source Card */}
                            <div className="add-influencer-card" onClick={() => setShowAddModal(true)}>
                                <div className="add-icon">+</div>
                                <span>Add Source</span>
                            </div>
                            {influencers.map(inf => (
                                <div key={inf.id} className="influencer-card-v2" onClick={() => handleSelectInfluencer(inf)}>
                                    {/* Banner/Thumbnail */}
                                    <div className="inf-card-banner">
                                        {inf.thumbnail_url ? (
                                            <img src={inf.thumbnail_url} alt="" onError={(e) => e.target.style.display = 'none'} />
                                        ) : (
                                            <div className="inf-banner-placeholder">
                                                <span>{inf.name.charAt(0).toUpperCase()}</span>
                                            </div>
                                        )}
                                        <div className="inf-banner-overlay"></div>
                                        {/* Auto-mode toggle in corner */}
                                        <div className="inf-auto-toggle" onClick={(e) => e.stopPropagation()}>
                                            <label className="toggle-switch-sm" title={inf.auto_mode_enabled ? 'Auto-mode ON' : 'Auto-mode OFF'}>
                                                <input
                                                    type="checkbox"
                                                    checked={inf.auto_mode_enabled || false}
                                                    onChange={(e) => handleAutoModeToggle(inf, e)}
                                                />
                                                <span className="slider-sm"></span>
                                            </label>
                                        </div>
                                        {/* Platform badge */}
                                        <span className={`inf-platform-badge ${inf.platform}`}>{inf.platform}</span>
                                    </div>

                                    {/* Card Content */}
                                    <div className="inf-card-content">
                                        <div className="inf-card-header">
                                            <h3>{inf.name}</h3>
                                            {inf.auto_mode_enabled && <span className="inf-auto-badge">AUTO</span>}
                                        </div>

                                        {/* Stats Row */}
                                        <div className="inf-stats-row">
                                            <div className="inf-stat">
                                                <span className="inf-stat-value">{inf.video_count || 0}</span>
                                                <span className="inf-stat-label">Videos</span>
                                            </div>
                                            <div className="inf-stat">
                                                <span className="inf-stat-value">{inf.clip_count || 0}</span>
                                                <span className="inf-stat-label">Clips</span>
                                            </div>
                                            <div className="inf-stat">
                                                <span className="inf-stat-value ready">{inf.ready_clips || 0}</span>
                                                <span className="inf-stat-label">Ready</span>
                                            </div>
                                        </div>

                                        {/* Last Activity */}
                                        {inf.last_fetch_at && (
                                            <div className="inf-last-activity">
                                                Last fetched {new Date(inf.last_fetch_at).toLocaleDateString()}
                                            </div>
                                        )}

                                        {/* Actions */}
                                        <div className="inf-card-actions">
                                            <button
                                                className="inf-settings-btn"
                                                onClick={(e) => openAutoModeSettings(inf, e)}
                                            >
                                                Settings
                                            </button>
                                            <button
                                                className="inf-clips-btn"
                                                onClick={(e) => handleViewInfluencerClips(inf, e)}
                                            >
                                                Clips
                                            </button>
                                            <button
                                                className="inf-view-btn"
                                                onClick={(e) => { e.stopPropagation(); handleSelectInfluencer(inf); }}
                                            >
                                                View Videos
                                            </button>
                                            <button
                                                className="inf-delete-btn"
                                                onClick={(e) => { e.stopPropagation(); setDeleteConfirmInfluencer(inf); }}
                                            >
                                                Delete
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Delete Confirmation Modal */}
                {deleteConfirmInfluencer && (
                    <div className="modal-overlay" onClick={() => !deleteLoading && setDeleteConfirmInfluencer(null)}>
                        <div className="modal delete-confirm-modal" onClick={e => e.stopPropagation()}>
                            <h2>Delete Source</h2>
                            <p className="delete-warning">
                                Are you sure you want to delete <strong>{deleteConfirmInfluencer.name}</strong>?
                            </p>
                            <p className="delete-details">
                                This will permanently delete:
                            </p>
                            <ul className="delete-list">
                                <li>{deleteConfirmInfluencer.video_count || 0} videos</li>
                                <li>{deleteConfirmInfluencer.clip_count || 0} clips</li>
                            </ul>
                            <p className="delete-warning-text">This action cannot be undone.</p>
                            <div className="modal-actions">
                                <button
                                    className="cancel-btn"
                                    onClick={() => setDeleteConfirmInfluencer(null)}
                                    disabled={deleteLoading}
                                >
                                    Cancel
                                </button>
                                <button
                                    className="confirm-delete-btn"
                                    onClick={handleDeleteInfluencer}
                                    disabled={deleteLoading}
                                >
                                    {deleteLoading ? 'Deleting...' : 'Delete'}
                                </button>
                            </div>
                        </div>
                    </div>
                )}

                {activeTab === 'videos' && (
                    <div className="videos-panel">
                        {!selectedInfluencer ? (
                            <p>Please select a content source first.</p>
                        ) : (
                            <div>
                                <div className="panel-header">
                                    <h3>Videos for {selectedInfluencer.name}</h3>
                                    <button disabled={loadingVideos} onClick={handleFetchVideos}>
                                        {loadingVideos ? 'Fetching...' : 'Fetch Latest Videos'}
                                    </button>
                                </div>
                                <div className="video-list">
                                    {sortedVideos.map(v => (
                                            <div key={v.id} className="video-item card">
                                                <div className="video-card-content">
                                                    <a href={v.url} target="_blank" rel="noreferrer" className="thumbnail-link">
                                                        <div className="thumbnail-wrapper">
                                                            <img src={getThumbnail(v)} alt={v.title} onError={(e) => e.target.src = 'https://via.placeholder.com/320x180?text=No+Thumbnail'} />
                                                            <div className="play-overlay">
                                                                <div className="play-icon">▶</div>
                                                            </div>
                                                        </div>
                                                    </a>
                                                    <div className="info">
                                                        <h4>{v.title}</h4>
                                                        <div className="meta">
                                                            {v.duration && <span className="duration">{formatDuration(v.duration)}</span>}
                                                            {v.influencer && <span className="influencer-name"> | {v.influencer.name}</span>}
                                                        </div>

                                                        {v.status !== 'pending' && (
                                                            <div className={`status-badge ${v.status?.toLowerCase().split(' ')[0]}`}>
                                                                {(v.status?.toLowerCase().includes('downloading') ||
                                                                    v.status?.toLowerCase().includes('transcribing') ||
                                                                    v.status?.toLowerCase().includes('analyzing')) && (
                                                                        <span className="spinner-small"></span>
                                                                    )}
                                                                <span className="status-text-large">
                                                                    {v.status}
                                                                    {v.status?.toLowerCase().includes('transcribing') && getTranscriptionProgress(v) !== null && (
                                                                        <span className="progress-percent"> ({getTranscriptionProgress(v)}%)</span>
                                                                    )}
                                                                </span>
                                                            </div>
                                                        )}
                                                        {v.status?.toLowerCase() === 'downloading' && downloadProgress[v.id] && (
                                                            <div className="download-progress">
                                                                <span className="dl-size">
                                                                    {(downloadProgress[v.id].bytes_downloaded / (1024*1024*1024)).toFixed(2)} GB
                                                                </span>
                                                                {downloadProgress[v.id].speed > 0 && (
                                                                    <span className="dl-speed">
                                                                        {' '}@ {(downloadProgress[v.id].speed / (1024*1024)).toFixed(1)} MB/s
                                                                    </span>
                                                                )}
                                                                {downloadProgress[v.id].fragment_info && (
                                                                    <span className="dl-frags">
                                                                        {' '}(frag {downloadProgress[v.id].fragment_info.current}{downloadProgress[v.id].fragment_info.total ? `/${downloadProgress[v.id].fragment_info.total}` : ''})
                                                                    </span>
                                                                )}
                                                            </div>
                                                        )}
                                                        {v.status === 'error' && <p className="error-text" title={v.error_message}>{v.error_message}</p>}

                                                        <div className="actions-row">
                                                            {!['downloaded', 'transcribed', 'analyzed', 'pending', 'error'].some(s => v.status?.toLowerCase().includes(s) === false) ? null :
                                                                /* Actually, simpler logic: Show if NOT active */
                                                                !(v.status?.toLowerCase().includes('downloading') ||
                                                                    v.status?.toLowerCase().includes('transcribing') ||
                                                                    v.status?.toLowerCase().includes('analyzing') ||
                                                                    v.status?.toLowerCase().includes('processing') ||
                                                                    v.status?.toLowerCase().includes('rendering')) && (
                                                                    <button
                                                                        className="analyze-btn"
                                                                        style={{ marginTop: '10px', width: '100%' }}
                                                                        onClick={() => handleAnalyze(v.id)}
                                                                    >
                                                                        {v.status === 'analyzed' || v.status === 'completed' ? 'Re-Generate Clips' : 'Auto-Generate Clips'}
                                                                    </button>
                                                                )}
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        ))}
                                </div>
                            </div>
                        )}
                    </div>
                )
                }

                {
                    activeTab === 'clips' && (
                        <div className="clips-panel">
                            <div className="panel-header">
                                <h2>{clipsFilterInfluencer ? `Clips - ${clipsFilterInfluencer.name}` : 'All Clips'}</h2>
                                <div className="header-actions">
                                    <div className="auto-publish-toggle" onClick={(e) => e.stopPropagation()}>
                                        <label className="toggle-switch-sm" title={autoPublishEnabled ? 'Auto-publish ON' : 'Auto-publish OFF'}>
                                            <input
                                                type="checkbox"
                                                checked={autoPublishEnabled}
                                                onChange={async (e) => {
                                                    const newVal = e.target.checked;
                                                    setAutoPublishEnabled(newVal);
                                                    try {
                                                        await api.put('/api/viral/settings/auto-publish', { enabled: newVal });
                                                    } catch (err) {
                                                        setAutoPublishEnabled(!newVal);
                                                        alert('Failed to update auto-publish: ' + (err.response?.data?.detail || err.message));
                                                    }
                                                }}
                                            />
                                            <span className="toggle-slider-sm"></span>
                                        </label>
                                        <span className="toggle-label-sm">Auto-Publish</span>
                                        {autoPublishEnabled && <span className="inf-auto-badge">ON</span>}
                                    </div>
                                    {clipsFilterInfluencer && (
                                        <button className="clear-filter-btn" onClick={() => setClipsFilterInfluencer(null)}>
                                            Clear Filter ✕
                                        </button>
                                    )}
                                    <button
                                        className="render-all-btn"
                                        onClick={handleRenderAll}
                                        disabled={renderingAll}
                                    >
                                        {renderingAll ? 'Queuing...' : 'Render All'}
                                    </button>
                                    <button onClick={loadClips}>Refresh</button>
                                </div>
                            </div>
                            <div className="card-grid">
                                {filteredSortedClips.map(c => {
                                    const isReady = ['completed', 'ready'].includes(c.status?.toLowerCase());
                                    const isRendering = ['rendering', 'processing', 'queued'].some(s => c.status?.toLowerCase().includes(s));
                                    const canRender = c.status === 'pending' || c.status === 'error' || c.status === 'failed';

                                    // Format timestamp for display
                                    const timestamp = c.updated_at || c.created_at;
                                    const formattedTime = timestamp ? new Date(timestamp).toLocaleString('en-US', {
                                        month: 'short',
                                        day: 'numeric',
                                        hour: 'numeric',
                                        minute: '2-digit',
                                        hour12: true
                                    }) : '';

                                    return (
                                        <div key={c.id} className="card clip-card">
                                            {isReady && c.publishing_status !== 'published' && (
                                                <button
                                                    className="publish-icon-btn"
                                                    onClick={() => handlePublish(c.id, c.render_metadata?.recommended_platform)}
                                                    disabled={publishingClips[c.id]}
                                                    title="Publish"
                                                >
                                                    {publishingClips[c.id] ? '...' : '↑'}
                                                </button>
                                            )}
                                            {c.influencer_name && (
                                                <div className="clip-influencer">
                                                    <span className="influencer-badge">{c.influencer_name}</span>
                                                </div>
                                            )}
                                            <div className="clip-header">
                                                <h4>{c.title}</h4>
                                            </div>
                                            <div className="status-row">
                                                {isRendering && <span className="spinner-small"></span>}
                                                <span className={`status ${c.status?.split(':')[0]?.split(' ')[0]?.toLowerCase() || 'pending'}`}>
                                                    {c.status?.includes('Processing:')
                                                        ? c.status.replace('Processing:', '').trim()
                                                        : c.status}
                                                </span>
                                            </div>
                                            <div className="clip-meta-row">
                                                <span className="clip-type">Type: {c.clip_type}</span>
                                                {(() => {
                                                    const platform = c.render_metadata?.recommended_platform || 'tiktok';
                                                    return (
                                                        <span className={`platform-badge ${platform}`}>
                                                            {platform === 'tiktok' ? '🎵 TikTok' : '📷 Reels'}
                                                        </span>
                                                    );
                                                })()}
                                            </div>
                                            {c.render_metadata?.effect_failures?.length > 0 && (
                                                <div className="effect-failures" title={c.render_metadata.effect_failures.join(', ')}>
                                                    <span className="failure-badge">⚠ {c.render_metadata.effect_failures.length} effect{c.render_metadata.effect_failures.length > 1 ? 's' : ''} failed</span>
                                                    <span className="failure-list">{c.render_metadata.effect_failures.join(', ')}</span>
                                                </div>
                                            )}
                                            <TemplateSelector clip={c} />
                                            <div className="actions">
                                                {canRender && (
                                                    <button
                                                        onClick={() => handleRender(c.id)}
                                                        disabled={renderingClips[c.id]}
                                                        className={['error', 'failed'].includes(c.status) ? 'retry-btn' : ''}
                                                    >
                                                        {renderingClips[c.id] ? 'Starting...' : (['error', 'failed'].includes(c.status) ? 'Retry' : 'Render')}
                                                    </button>
                                                )}
                                                {isReady && (
                                                    <>
                                                        <button onClick={() => setViewingClip(c)}>▶ Play</button>
                                                        <button onClick={async () => {
                                                            const filename = c.edited_video_path?.split('/').pop();
                                                            if (!filename) return;
                                                            const downloadUrl = new URL(`/api/viral/file/${filename}`, API_URL).toString();
                                                            const response = await fetch(downloadUrl);
                                                            const blob = await response.blob();
                                                            const blobUrl = URL.createObjectURL(blob);
                                                            const a = document.createElement('a');
                                                            a.href = blobUrl;
                                                            a.download = filename;
                                                            a.click();
                                                            URL.revokeObjectURL(blobUrl);
                                                        }}>Download</button>
                                                        <button
                                                            onClick={() => handleRender(c.id)}
                                                            disabled={renderingClips[c.id]}
                                                            className="rerender-btn"
                                                        >
                                                            {renderingClips[c.id] ? 'Starting...' : 'Re-render'}
                                                        </button>
                                                    </>
                                                )}
                                            </div>
                                            <div className="clip-footer">
                                                {c.publishing_status === 'published' && (
                                                    <span className="published-pill">Published</span>
                                                )}
                                                {formattedTime && <span className="clip-timestamp">{formattedTime}</span>}
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    )
                }

                {activeTab === 'broll' && (
                    <div className="broll-panel">
                        <div className="panel-header">
                            <h3>B-Roll Library</h3>
                            <div className="header-actions">
                                <button onClick={loadBroll}>Refresh</button>
                                <button
                                    onClick={() => handleRetagBroll(false)}
                                    disabled={brollTagging}
                                >
                                    {brollTagging ? 'Tagging...' : 'Tag Untagged'}
                                </button>
                                <button
                                    onClick={() => handleRetagBroll(true)}
                                    disabled={brollTagging}
                                    className="secondary-btn"
                                >
                                    Re-Tag All
                                </button>
                            </div>
                        </div>

                        {/* Upload Form */}
                        <div className="broll-upload-section card">
                            <h4>Add B-Roll from YouTube</h4>
                            <form onSubmit={handleBrollUpload} className="upload-form">
                                <input
                                    type="text"
                                    placeholder="YouTube URL (video will be split into clips)"
                                    value={brollYoutubeUrl}
                                    onChange={e => setBrollYoutubeUrl(e.target.value)}
                                    className="url-input"
                                />
                                <select
                                    value={brollCategory}
                                    onChange={e => setBrollCategory(e.target.value)}
                                    className="category-select"
                                >
                                    <option value="">Auto-detect category</option>
                                    <option value="war">War / Military</option>
                                    <option value="wealth">Wealth / Success</option>
                                    <option value="faith">Faith / Religion</option>
                                    <option value="strength">Strength / Fitness</option>
                                    <option value="nature">Nature / Landscapes</option>
                                    <option value="people">People / Crowds</option>
                                    <option value="chaos">Chaos / Destruction</option>
                                    <option value="victory">Victory / Celebration</option>
                                    <option value="power">Power / Authority</option>
                                    <option value="history">History / Archive</option>
                                </select>
                                <button type="submit" disabled={brollLoading || !brollYoutubeUrl.trim()}>
                                    {brollLoading ? 'Starting...' : 'Upload & Process'}
                                </button>
                            </form>

                            {/* Active Jobs */}
                            {brollUploadJobs.length > 0 && (
                                <div className="upload-jobs">
                                    <h5>Processing Jobs</h5>
                                    {brollUploadJobs.map(job => (
                                        <div key={job.job_id} className={`job-item ${job.status}`}>
                                            <span className="job-url">{job.url?.substring(0, 50)}...</span>
                                            <span className="job-status">
                                                {job.status === 'complete' ? '✓ Complete' :
                                                    job.status === 'error' ? '✗ Error' :
                                                        <><span className="spinner-small"></span> {job.status}</>}
                                            </span>
                                            {job.clips_created && <span className="job-clips">{job.clips_created} clips</span>}
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>

                        {/* Stats */}
                        {brollMetadata && (
                            <div className="broll-stats card">
                                <h4>Library Stats</h4>
                                <div className="stats-grid">
                                    <div className="stat">
                                        <span className="stat-value">{brollMetadata.total_clips || 0}</span>
                                        <span className="stat-label">Total Clips</span>
                                    </div>
                                    {brollMetadata.category_counts && Object.entries(brollMetadata.category_counts).map(([cat, count]) => (
                                        <div
                                            key={cat}
                                            className={`stat category-stat ${brollFilter === cat ? 'active' : ''}`}
                                            onClick={() => setBrollFilter(brollFilter === cat ? 'all' : cat)}
                                        >
                                            <span className="stat-value">{count}</span>
                                            <span className="stat-label">{cat}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Filter */}
                        <div className="broll-filter">
                            <span>Filter:</span>
                            <select value={brollFilter} onChange={e => setBrollFilter(e.target.value)}>
                                <option value="all">All Categories</option>
                                <option value="war">War</option>
                                <option value="wealth">Wealth</option>
                                <option value="faith">Faith</option>
                                <option value="strength">Strength</option>
                                <option value="nature">Nature</option>
                                <option value="people">People</option>
                                <option value="chaos">Chaos</option>
                                <option value="victory">Victory</option>
                                <option value="power">Power</option>
                                <option value="history">History</option>
                                <option value="untagged">Untagged</option>
                            </select>
                            <span className="filter-count">
                                {brollMetadata?.filtered_clips || brollClips.length} clips
                            </span>
                        </div>

                        {/* Clips Grid - Server-side Paginated */}
                        <div className="broll-grid">
                            {brollClips.map(clip => (
                                    <div key={clip.filename} className="broll-card card">
                                        <div className="broll-preview">
                                            <video
                                                src={`${API_URL}/api/viral/broll/file/${clip.filename}`}
                                                muted
                                                loop
                                                preload="none"
                                                onMouseEnter={e => e.target.play().catch(() => {})}
                                                onMouseLeave={e => { e.target.pause(); e.target.currentTime = 0; }}
                                            />
                                        </div>
                                        <div className="broll-info">
                                            <span className="broll-filename" title={clip.filename}>
                                                {clip.filename.length > 25 ? clip.filename.substring(0, 22) + '...' : clip.filename}
                                            </span>
                                            {clip.duration && <span className="broll-duration">{clip.duration.toFixed(1)}s</span>}
                                            {clip.caption && (
                                                <p className="broll-caption" title={clip.caption}>
                                                    {clip.caption.length > 50 ? clip.caption.substring(0, 47) + '...' : clip.caption}
                                                </p>
                                            )}
                                            <div className="broll-categories">
                                                {clip.categories?.length > 0 ? (
                                                    clip.categories.map(cat => (
                                                        <span key={cat} className={`category-tag ${cat}`}>{cat}</span>
                                                    ))
                                                ) : (
                                                    <span className="category-tag untagged">untagged</span>
                                                )}
                                            </div>
                                        </div>
                                        <div className="broll-actions">
                                            <button
                                                className="delete-btn"
                                                onClick={() => handleDeleteBroll(clip.filename)}
                                            >
                                                Delete
                                            </button>
                                        </div>
                                    </div>
                                ))}

                            {/* Skeleton placeholders while loading more */}
                            {brollHasMore && (
                                [...Array(4)].map((_, i) => (
                                    <div key={`skeleton-${i}`} className="broll-card card skeleton">
                                        <div className="broll-preview skeleton-preview"></div>
                                        <div className="broll-info">
                                            <span className="skeleton-text skeleton-filename"></span>
                                            <span className="skeleton-text skeleton-duration"></span>
                                            <div className="skeleton-text skeleton-caption"></div>
                                            <div className="broll-categories">
                                                <span className="skeleton-text skeleton-tag"></span>
                                                <span className="skeleton-text skeleton-tag"></span>
                                            </div>
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>

                        {/* Load More Trigger */}
                        {brollHasMore && (
                            <div ref={brollLoadMoreRef} className="load-more-trigger">
                                <span className="loading-text">{brollLoadingMore ? 'Loading...' : 'Scroll for more'}</span>
                                <span className="load-count">{brollClips.length} of {brollMetadata?.filtered_clips || '?'}</span>
                            </div>
                        )}

                        {brollClips.length === 0 && (
                            <div className="empty-state">
                                <p>No B-roll clips yet. Upload a YouTube video to get started!</p>
                            </div>
                        )}
                    </div>
                )}

                {/* Publishing Tab */}
                {activeTab === 'publishing' && (
                    <div className="publishing-panel">
                        {/* Stats Bar */}
                        {publishingStats && (
                            <div className="publishing-stats-bar">
                                <div className="pub-stat-item">
                                    <span className="pub-stat-value pending">{publishingStats.pending_approval || 0}</span>
                                    <span className="pub-stat-label">Pending</span>
                                </div>
                                <div className="pub-stat-item">
                                    <span className="pub-stat-value scheduled">{publishingStats.scheduled || 0}</span>
                                    <span className="pub-stat-label">Scheduled</span>
                                </div>
                                <div className="pub-stat-item">
                                    <span className="pub-stat-value published">{publishingStats.published_today || 0}</span>
                                    <span className="pub-stat-label">Today</span>
                                </div>
                                <div className="pub-stat-item">
                                    <span className="pub-stat-value failed">{publishingStats.failed_recent || 0}</span>
                                    <span className="pub-stat-label">Failed</span>
                                </div>
                            </div>
                        )}

                        {/* Queue Section */}
                        <div className="publishing-section">
                            <div className="section-header">
                                <h3>Publishing Queue</h3>
                                <div className="section-controls">
                                    <select value={queueFilter} onChange={(e) => setQueueFilter(e.target.value)}>
                                        <option value="all">All Status</option>
                                        <option value="pending_approval">Pending Approval</option>
                                        <option value="approved">Approved</option>
                                        <option value="scheduled">Scheduled</option>
                                        <option value="published">Published</option>
                                        <option value="failed">Failed</option>
                                    </select>
                                    <button onClick={loadPublishingData}>Refresh</button>
                                </div>
                            </div>

                            {filteredQueue.length === 0 ? (
                                <div className="empty-state">
                                    <p>No items in the queue.</p>
                                </div>
                            ) : (
                                <div className="queue-list">
                                    {filteredQueue.map(item => (
                                        <div key={item.id} className={`queue-item ${item.status}`}>
                                            <div className="queue-item-thumb">
                                                {item.clip?.edited_video_path && (
                                                    <video
                                                        src={`${API_URL}/api/viral/file/${item.clip.edited_video_path.split('/').pop()}`}
                                                        muted
                                                        onMouseEnter={e => e.target.play().catch(() => {})}
                                                        onMouseLeave={e => { e.target.pause(); e.target.currentTime = 0; }}
                                                    />
                                                )}
                                            </div>
                                            <div className="queue-item-info">
                                                <h4>{item.clip?.title || `Clip #${item.clip_id}`}</h4>
                                                <div className="queue-item-meta">
                                                    <span className={`queue-status-badge ${item.status}`}>{item.status?.replace('_', ' ')}</span>
                                                    <span className="queue-platforms">{item.platforms?.join(', ') || 'Not set'}</span>
                                                    {item.scheduled_time && (
                                                        <span className="queue-scheduled">{formatPublishDate(item.scheduled_time)}</span>
                                                    )}
                                                </div>
                                                {item.rejection_reason && <p className="queue-rejection">Rejected: {item.rejection_reason}</p>}
                                                {item.error_message && <p className="queue-error">{item.error_message}</p>}
                                            </div>
                                            <div className="queue-item-actions">
                                                {item.status === 'pending_approval' && (
                                                    <>
                                                        <button className="approve-btn" onClick={() => handleApproveQueueItem(item.id)}>Approve</button>
                                                        <button className="reject-btn" onClick={() => handleRejectQueueItem(item.id)}>Reject</button>
                                                    </>
                                                )}
                                                {['approved', 'scheduled'].includes(item.status) && (
                                                    <button className="publish-now-btn" onClick={() => handlePublishNow(item.id)}>Publish Now</button>
                                                )}
                                                {item.status === 'failed' && (
                                                    <button className="retry-btn" onClick={() => handlePublishNow(item.id)}>Retry</button>
                                                )}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>

                        {/* Configurations Section */}
                        <div className="publishing-section">
                            <div className="section-header">
                                <h3>Publishing Configurations</h3>
                                <button className="add-btn" onClick={() => openConfigModal()}>+ Add Config</button>
                            </div>

                            {publishingConfigs.length === 0 ? (
                                <div className="empty-state">
                                    <p>No publishing configurations yet.</p>
                                </div>
                            ) : (
                                <div className="configs-grid">
                                    {publishingConfigs.map(config => (
                                        <div key={config.id} className={`config-card ${config.is_active ? 'active' : 'inactive'}`}>
                                            <div className="config-card-header">
                                                <h4>{config.blotato_account_id || 'Unnamed'}</h4>
                                                <span className={`config-status ${config.is_active ? 'active' : ''}`}>
                                                    {config.is_active ? 'Active' : 'Paused'}
                                                </span>
                                            </div>
                                            <div className="config-details">
                                                <p><strong>Platforms:</strong> {config.platforms?.join(', ') || 'None'}</p>
                                                <p><strong>Posts/Day:</strong> {config.posts_per_day}</p>
                                                <p><strong>Hours:</strong> {config.posting_hours?.join(', ')}</p>
                                                <p><strong>Approval:</strong> {config.require_approval ? 'Required' : 'Auto'}</p>
                                            </div>
                                            <div className="config-card-actions">
                                                <button onClick={() => openConfigModal(config)}>Edit</button>
                                                <button className="delete-btn" onClick={() => handleDeleteConfig(config.id)}>Delete</button>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div >
            {
                showAddModal && (
                    <div className="modal-overlay">
                        <div className="modal">
                            <h3>Add Content Source</h3>
                            <form onSubmit={handleAddInfluencer}>
                                <input
                                    placeholder="Name"
                                    value={newInfluencer.name}
                                    onChange={e => setNewInfluencer({ ...newInfluencer, name: e.target.value })}
                                    required
                                />
                                <select
                                    value={newInfluencer.platform}
                                    onChange={e => setNewInfluencer({ ...newInfluencer, platform: e.target.value })}
                                >
                                    <option value="youtube">YouTube</option>
                                    <option value="rumble">Rumble</option>
                                </select>
                                <input
                                    placeholder="Channel URL"
                                    value={newInfluencer.channel_url}
                                    onChange={e => setNewInfluencer({ ...newInfluencer, channel_url: e.target.value })}
                                    required
                                />
                                <div className="actions">
                                    <button type="button" onClick={() => setShowAddModal(false)}>Cancel</button>
                                    <button type="submit">Save</button>
                                </div>
                            </form>
                        </div>
                    </div>
                )
            }
            {
                viewingClip && (
                    <div className="modal-overlay" onClick={() => setViewingClip(null)}>
                        <div className="modal video-modal" onClick={e => e.stopPropagation()}>
                            <div className="modal-header">
                                <h3>{viewingClip.title}</h3>
                                <button className="close-btn" onClick={() => setViewingClip(null)}>×</button>
                            </div>
                            <video
                                controls
                                src={`${API_URL}/api/viral/file/${viewingClip.edited_video_path?.split('/').pop()}`}
                                className="modal-video-player"
                                onLoadedData={e => e.target.play().catch(() => {})}
                                onError={e => console.warn('Video load error:', e)}
                            />
                        </div>
                    </div>
                )
            }
            <EffectsModal />

            {/* Auto-Mode Settings Modal */}
            {autoModeModalInfluencer && autoModeSettings && (
                <div className="modal-overlay" onClick={() => { setAutoModeModalInfluencer(null); setAutoModeSettings(null); }}>
                    <div className="modal auto-mode-modal" onClick={e => e.stopPropagation()}>
                        <div className="modal-header">
                            <h3>Auto-Mode Settings: {autoModeModalInfluencer.name}</h3>
                            <button className="close-btn" onClick={() => { setAutoModeModalInfluencer(null); setAutoModeSettings(null); }}>×</button>
                        </div>
                        <div className="auto-mode-form">
                            <div className="form-group">
                                <label>
                                    <input
                                        type="checkbox"
                                        checked={autoModeSettings.auto_mode_enabled || false}
                                        onChange={(e) => setAutoModeSettings({ ...autoModeSettings, auto_mode_enabled: e.target.checked })}
                                    />
                                    Enable Auto-Mode
                                </label>
                                <p className="help-text">When enabled, automatically fetches new videos from this channel</p>
                            </div>

                            <div className="form-group">
                                <label>Fetch Frequency (hours)</label>
                                <input
                                    type="number"
                                    min="1"
                                    max="168"
                                    value={autoModeSettings.fetch_frequency_hours || 24}
                                    onChange={(e) => setAutoModeSettings({ ...autoModeSettings, fetch_frequency_hours: parseInt(e.target.value) || 24 })}
                                />
                                <p className="help-text">How often to check for new videos (1-168 hours)</p>
                            </div>

                            <div className="form-group">
                                <label>Max Videos Per Fetch</label>
                                <input
                                    type="number"
                                    min="1"
                                    max="20"
                                    value={autoModeSettings.max_videos_per_fetch || 5}
                                    onChange={(e) => setAutoModeSettings({ ...autoModeSettings, max_videos_per_fetch: parseInt(e.target.value) || 5 })}
                                />
                                <p className="help-text">Maximum new videos to process per fetch cycle</p>
                            </div>

                            <div className="form-divider"></div>

                            <div className="form-group">
                                <label>
                                    <input
                                        type="checkbox"
                                        checked={autoModeSettings.auto_analyze_enabled !== false}
                                        onChange={(e) => setAutoModeSettings({ ...autoModeSettings, auto_analyze_enabled: e.target.checked })}
                                    />
                                    Auto-Analyze Videos
                                </label>
                                <p className="help-text">Automatically analyze new videos with Grok AI</p>
                            </div>

                            <div className="form-group">
                                <label>
                                    <input
                                        type="checkbox"
                                        checked={autoModeSettings.auto_render_enabled !== false}
                                        onChange={(e) => setAutoModeSettings({ ...autoModeSettings, auto_render_enabled: e.target.checked })}
                                    />
                                    Auto-Render Clips
                                </label>
                                <p className="help-text">Automatically render identified viral clips</p>
                            </div>

                            {autoModeSettings.auto_mode_enabled_at && (
                                <div className="auto-mode-info">
                                    <p><strong>Enabled since:</strong> {new Date(autoModeSettings.auto_mode_enabled_at).toLocaleString()}</p>
                                    <p className="help-text">Only videos published after this date will be processed</p>
                                </div>
                            )}
                        </div>
                        <div className="modal-actions">
                            <button className="cancel-btn" onClick={() => { setAutoModeModalInfluencer(null); setAutoModeSettings(null); }}>Cancel</button>
                            <button className="save-btn" onClick={saveAutoModeSettings} disabled={autoModeLoading}>
                                {autoModeLoading ? 'Saving...' : 'Save Settings'}
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Publishing Config Modal */}
            {showConfigModal && (
                <div className="modal-overlay" onClick={() => setShowConfigModal(false)}>
                    <div className="modal config-modal" onClick={e => e.stopPropagation()}>
                        <div className="modal-header">
                            <h3>{editingConfig ? 'Edit Configuration' : 'New Publishing Configuration'}</h3>
                            <button className="close-btn" onClick={() => setShowConfigModal(false)}>×</button>
                        </div>
                        <div className="config-form">
                            <div className="form-group">
                                <label>Blotato Account ID</label>
                                <input
                                    type="text"
                                    value={configForm.blotato_account_id}
                                    onChange={e => setConfigForm({ ...configForm, blotato_account_id: e.target.value })}
                                    placeholder="e.g., main_account"
                                />
                            </div>

                            <div className="form-group">
                                <label>Platforms</label>
                                <div className="platform-toggles">
                                    {['tiktok', 'instagram', 'youtube', 'twitter', 'facebook'].map(platform => (
                                        <button
                                            key={platform}
                                            type="button"
                                            className={`platform-toggle ${configForm.platforms.includes(platform) ? 'active' : ''}`}
                                            onClick={() => togglePlatform(platform)}
                                        >
                                            {platform}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            <div className="form-group">
                                <label>Posts Per Day</label>
                                <input
                                    type="number"
                                    min="1"
                                    max="20"
                                    value={configForm.posts_per_day}
                                    onChange={e => setConfigForm({ ...configForm, posts_per_day: parseInt(e.target.value) || 1 })}
                                />
                            </div>

                            <div className="form-group">
                                <label>Posting Hours (comma-separated, 0-23)</label>
                                <input
                                    type="text"
                                    value={configForm.posting_hours.join(', ')}
                                    onChange={e => updatePostingHours(e.target.value)}
                                    placeholder="e.g., 9, 12, 18"
                                />
                            </div>

                            <div className="form-group">
                                <label>Active Days</label>
                                <div className="day-toggles">
                                    {['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'].map(day => (
                                        <button
                                            key={day}
                                            type="button"
                                            className={`day-toggle ${configForm.days_active.includes(day) ? 'active' : ''}`}
                                            onClick={() => toggleDay(day)}
                                        >
                                            {day.substring(0, 3)}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            <div className="form-group checkbox-group">
                                <label>
                                    <input
                                        type="checkbox"
                                        checked={configForm.require_approval}
                                        onChange={e => setConfigForm({ ...configForm, require_approval: e.target.checked })}
                                    />
                                    Require Manual Approval
                                </label>
                            </div>

                            <div className="form-group checkbox-group">
                                <label>
                                    <input
                                        type="checkbox"
                                        checked={configForm.is_active}
                                        onChange={e => setConfigForm({ ...configForm, is_active: e.target.checked })}
                                    />
                                    Configuration Active
                                </label>
                            </div>
                        </div>
                        <div className="modal-actions">
                            <button className="cancel-btn" onClick={() => setShowConfigModal(false)}>Cancel</button>
                            <button className="save-btn" onClick={savePublishingConfig}>
                                {editingConfig ? 'Save Changes' : 'Create'}
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div >
    );
};

export default ViralManager;
