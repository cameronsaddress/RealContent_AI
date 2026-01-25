import React, { useState, useEffect } from 'react';
import api, {
    getInfluencers, createInfluencer, fetchInfluencerVideos,
    analyzeVideo, getVideoDetails, getInfluencerVideos, getViralClips, API_URL,
    getBrollClips, uploadBrollFromYoutube, getBrollUploadStatus, retagBrollClips, deleteBrollClip,
    getEffectsCatalog, updateClipEffects, getDownloadProgress
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

    // For Templates
    const [templates, setTemplates] = useState([]);

    // For Effects
    const [effectsModalClip, setEffectsModalClip] = useState(null);
    const [effectsCatalog, setEffectsCatalog] = useState(null);

    // For B-Roll
    const [brollClips, setBrollClips] = useState([]);
    const [brollMetadata, setBrollMetadata] = useState(null);
    const [brollYoutubeUrl, setBrollYoutubeUrl] = useState('');
    const [brollCategory, setBrollCategory] = useState('');
    const [brollUploadJobs, setBrollUploadJobs] = useState([]);
    const [brollLoading, setBrollLoading] = useState(false);
    const [brollTagging, setBrollTagging] = useState(false);
    const [brollFilter, setBrollFilter] = useState('all');

    // Download progress tracking
    const [downloadProgress, setDownloadProgress] = useState({}); // { videoId: { bytes, speed, fragment } }

    useEffect(() => {
        loadInfluencers();
        loadClips();
        loadTemplates();
        loadBroll();
    }, []);

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

    const loadBroll = async () => {
        try {
            const data = await getBrollClips();
            setBrollClips(data.clips || []);
            setBrollMetadata(data.metadata || null);
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
            try {
                const status = await getBrollUploadStatus(jobId);
                setBrollUploadJobs(prev => prev.map(j =>
                    j.job_id === jobId ? { ...j, ...status } : j
                ));
                if (status.status === 'complete' || status.status === 'error') {
                    loadBroll(); // Refresh clips list
                    return;
                }
                setTimeout(poll, 3000);
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

    // Effects override modal component
    const EffectsModal = () => {
        if (!effectsModalClip) return null;
        const currentEffects = effectsModalClip.render_metadata?.director_effects || {};
        const [localEffects, setLocalEffects] = useState({...currentEffects});

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

    const loadVideos = async (id, silent = false) => {
        try {
            const data = await getInfluencerVideos(id);
            setVideos(data);
        } catch (e) {
            console.error(e);
        }
    }

    // Auto-refresh poll
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
                loadVideos(selectedInfluencer.id, true);
            }, 3000);
        }
        return () => clearInterval(interval);
    }, [videos, selectedInfluencer]);

    // Auto-refresh for clips tab
    useEffect(() => {
        if (activeTab !== 'clips') return;

        const hasActiveClips = clips.some(c => {
            const cs = c.status?.toLowerCase() || '';
            return cs.includes('rendering') || cs.includes('processing') || cs.includes('queued');
        });

        let interval;
        if (hasActiveClips) {
            interval = setInterval(() => {
                loadClips();
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

    return (
        <div className="viral-manager-container">
            <div className="header">
                <h2>Viral Clip Factory</h2>
                <div className="tabs">
                    <button className={activeTab === 'influencers' ? 'active' : ''} onClick={() => setActiveTab('influencers')}>Influencers</button>
                    <button className={activeTab === 'videos' ? 'active' : ''} onClick={() => setActiveTab('videos')}>Videos</button>
                    <button className={activeTab === 'clips' ? 'active' : ''} onClick={() => setActiveTab('clips')}>Clips</button>
                    <button className={activeTab === 'broll' ? 'active' : ''} onClick={() => setActiveTab('broll')}>B-Roll</button>
                </div>
            </div>

            <div className="content">
                {activeTab === 'influencers' && (
                    <div className="influencers-panel">
                        <button className="add-btn" onClick={() => setShowAddModal(true)}>+ Add Influencer</button>
                        <div className="card-grid">
                            {influencers.map(inf => (
                                <div key={inf.id} className="card influencer-card" onClick={() => handleSelectInfluencer(inf)}>
                                    <h3>{inf.name}</h3>
                                    <p className="platform-tag">{inf.platform}</p>
                                    <p className="url">{inf.channel_url}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {activeTab === 'videos' && (
                    <div className="videos-panel">
                        {!selectedInfluencer ? (
                            <p>Please select an influencer first.</p>
                        ) : (
                            <div>
                                <div className="panel-header">
                                    <h3>Videos for {selectedInfluencer.name}</h3>
                                    <button disabled={loadingVideos} onClick={handleFetchVideos}>
                                        {loadingVideos ? 'Fetching...' : 'Fetch Latest Videos'}
                                    </button>
                                </div>
                                <div className="video-list">
                                    {Array.from(new Map(videos.map(item => [item.url, item])).values())
                                        .sort((a, b) => new Date(b.publication_date || b.created_at || 0) - new Date(a.publication_date || a.created_at || 0))
                                        .map(v => (
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
                                <h2>All Clips</h2>
                                <button onClick={loadClips}>Refresh</button>
                            </div>
                            <div className="card-grid">
                                {[...clips]
                                    .sort((a, b) => {
                                        // Sort by updated_at (render time) first, then created_at
                                        const aTime = new Date(a.updated_at || a.created_at || 0);
                                        const bTime = new Date(b.updated_at || b.created_at || 0);
                                        return bTime - aTime; // Newest first
                                    })
                                    .map(c => {
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
                                            <div className="clip-header">
                                                <h4>{c.title}</h4>
                                                {formattedTime && <span className="clip-timestamp">{formattedTime}</span>}
                                            </div>
                                            <div className="status-row">
                                                {isRendering && <span className="spinner-small"></span>}
                                                <span className={`status ${c.status}`}>{c.status}</span>
                                            </div>
                                            <p>Type: {c.clip_type}</p>
                                            <EffectBadges clip={c} />
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
                                                        <button onClick={() => {
                                                            const filename = c.edited_video_path?.split('/').pop();
                                                            if (!filename) return;
                                                            const downloadUrl = new URL(`/api/viral/file/${filename}`, API_URL).toString();
                                                            window.open(downloadUrl, '_blank');
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
                                {brollClips.filter(c =>
                                    brollFilter === 'all' ? true :
                                        brollFilter === 'untagged' ? (!c.categories || c.categories.length === 0) :
                                            c.categories?.includes(brollFilter)
                                ).length} clips
                            </span>
                        </div>

                        {/* Clips Grid */}
                        <div className="broll-grid">
                            {brollClips
                                .filter(c =>
                                    brollFilter === 'all' ? true :
                                        brollFilter === 'untagged' ? (!c.categories || c.categories.length === 0) :
                                            c.categories?.includes(brollFilter)
                                )
                                .map(clip => (
                                    <div key={clip.filename} className="broll-card card">
                                        <div className="broll-preview">
                                            <video
                                                src={`${API_URL}/api/viral/broll/file/${clip.filename}`}
                                                muted
                                                loop
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
                        </div>

                        {brollClips.length === 0 && (
                            <div className="empty-state">
                                <p>No B-roll clips yet. Upload a YouTube video to get started!</p>
                            </div>
                        )}
                    </div>
                )}
            </div >
            {
                showAddModal && (
                    <div className="modal-overlay">
                        <div className="modal">
                            <h3>Add Influencer</h3>
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
                                autoPlay
                                src={`${API_URL}/api/viral/file/${viewingClip.edited_video_path?.split('/').pop()}`}
                                className="modal-video-player"
                            />
                        </div>
                    </div>
                )
            }
            <EffectsModal />
        </div >
    );
};

export default ViralManager;
