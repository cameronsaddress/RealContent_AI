import React, { useState, useEffect } from 'react';
import api, {
    getInfluencers, createInfluencer, fetchInfluencerVideos,
    analyzeVideo, getVideoDetails, getInfluencerVideos, getViralClips, API_URL
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

    useEffect(() => {
        loadInfluencers();
        loadClips();
    }, []);

    const loadClips = async () => {
        try {
            const data = await getViralClips();
            setClips(data);
        } catch (e) { console.error(e); }
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
        if (!window.confirm("Start analysis? This will download, transcribe, and use Grok credits.")) return;
        try {
            const res = await analyzeVideo(videoId);
            alert(res.message);
            setActiveTab('clips'); // Switch to clips to see result coming in
            loadClips();
        } catch (e) {
            alert(e.message);
        }
    };

    const handleRender = async (clipId) => {
        setRenderingClips(prev => ({ ...prev, [clipId]: true }));
        try {
            await api.post(`/api/viral/viral-clips/${clipId}/render`);
            // Optimistic update or just wait for poll
            // Force refresh of video list to see status change
            if (selectedInfluencer) {
                loadVideos(selectedInfluencer.id, true);
            }
        } catch (e) {
            alert("Render error: " + e.message);
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
                                                <img src={v.thumbnail_url} alt="thumb" />
                                                <div className="info">
                                                    <h4>{v.title}</h4>
                                                    {v.status !== 'pending' && (
                                                        <div className={`status-badge ${v.status?.toLowerCase().split(' ')[0]}`}>
                                                            {(v.status?.toLowerCase().includes('downloading') ||
                                                                v.status?.toLowerCase().includes('transcribing') ||
                                                                v.status?.toLowerCase().includes('analyzing')) && (
                                                                    <span className="spinner-small"></span>
                                                                )}
                                                            {v.status}
                                                        </div>
                                                    )}
                                                    {v.status === 'error' && <p className="error-text" title={v.error_message}>{v.error_message}</p>}
                                                    <a href={v.url} target="_blank" rel="noreferrer">Watch</a>
                                                </div>
                                                <div className="actions">
                                                    {v.clips && v.clips.length > 0 && (
                                                        <div className="clips-mini-list">
                                                            <h5>Generated Clips ({v.clips.length})</h5>
                                                            {v.clips.map(c => {
                                                                const isReady = ['completed', 'ready'].includes(c.status?.toLowerCase());
                                                                return (
                                                                    <div key={c.id} className="clip-mini-item">
                                                                        <div className="clip-info">
                                                                            <span className="clip-type">{c.clip_type}</span>:
                                                                            <span className="clip-title" title={c.title}>{c.title}</span>
                                                                        </div>
                                                                        <div className="mini-actions">
                                                                            {(['rendering', 'processing'].some(s => c.status?.toLowerCase().includes(s))) && <span className="spinner-small"></span>}
                                                                            {isReady && (
                                                                                <button
                                                                                    className="play-clip-btn"
                                                                                    onClick={(e) => {
                                                                                        e.stopPropagation();
                                                                                        setViewingClip(c);
                                                                                    }}
                                                                                >
                                                                                    ▶️ Play
                                                                                </button>
                                                                            )}
                                                                            {!isReady && !['rendering', 'processing'].some(s => c.status?.toLowerCase().includes(s)) && (
                                                                                <div className="pending-actions">
                                                                                    <span className="status-text">{c.status}</span>
                                                                                    {(c.status === 'pending' || c.status === 'error' || c.status === 'failed') && (
                                                                                        <button
                                                                                            className={`render-btn-mini ${['error', 'failed'].includes(c.status) ? 'retry-btn' : ''}`}
                                                                                            disabled={renderingClips[c.id]}
                                                                                            onClick={(e) => {
                                                                                                e.stopPropagation();
                                                                                                handleRender(c.id);
                                                                                            }}
                                                                                        >
                                                                                            {renderingClips[c.id] ? 'Starting...' : (['error', 'failed'].includes(c.status) ? 'Retry' : 'Render')}
                                                                                        </button>
                                                                                    )}
                                                                                </div>
                                                                            )}
                                                                        </div>
                                                                    </div>
                                                                );
                                                            })}
                                                        </div>
                                                    )}
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
                            <button onClick={loadClips}>Refresh Clips</button>
                            <div className="card-grid">
                                {clips.map(c => (
                                    <div key={c.id} className="card clip-card">
                                        <h4>{c.title}</h4>
                                        <span className={`status ${c.status}`}>{c.status}</span>
                                        <p>Type: {c.clip_type}</p>
                                        <div className="actions">
                                            {c.status === 'pending' && (
                                                <button onClick={() => handleRender(c.id)}>Render (CapCut Style)</button>
                                            )}
                                            {c.status === 'ready' && (
                                                <button onClick={() => window.open(`/api/viral/file/${c.edited_video_path?.split('/').pop()}`, '_blank')}>Download</button>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )
                }
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
        </div >
    );
};

export default ViralManager;
