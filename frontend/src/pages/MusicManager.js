
import React, { useState, useEffect, useRef } from 'react';
import { uploadMusic, getMusicInfo, getMusicFiles, activateMusic, deleteMusic } from '../api';

const MusicManager = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [currentTrack, setCurrentTrack] = useState(null);
    const [musicFiles, setMusicFiles] = useState([]);
    const [status, setStatus] = useState('');
    const [isUploading, setIsUploading] = useState(false);
    const [isLoadingInfo, setIsLoadingInfo] = useState(true);
    const [playingPreview, setPlayingPreview] = useState(null); // Filename of currently playing track
    const audioRef = useRef(null);

    const fetchInfo = async () => {
        setIsLoadingInfo(true);
        try {
            const [info, files] = await Promise.all([
                getMusicInfo(),
                getMusicFiles()
            ]);
            setCurrentTrack(info);
            setMusicFiles(files.files || []);
        } catch (error) {
            console.error("Failed to load music info", error);
        } finally {
            setIsLoadingInfo(false);
        }
    };

    useEffect(() => {
        fetchInfo();
    }, []);

    // Audio Playback Logic
    useEffect(() => {
        if (playingPreview && audioRef.current) {
            audioRef.current.play().catch(e => console.error("Play error", e));
        } else if (!playingPreview && audioRef.current) {
            audioRef.current.pause();
            audioRef.current.currentTime = 0;
        }
    }, [playingPreview]);

    const handlePlayPause = (e, file) => {
        e.stopPropagation(); // Prevent card click
        if (playingPreview === file.filename) {
            setPlayingPreview(null); // Stop
        } else {
            setPlayingPreview(file.filename); // Play new
        }
    };

    // Auto-Upload Handler
    const handleAutoUpload = async (e) => {
        if (!e.target.files || !e.target.files[0]) return;

        const file = e.target.files[0];
        // Validate
        if (file.type !== 'audio/mpeg' && !file.name.endsWith('.mp3')) {
            setStatus('Error: Only MP3 files are allowed.');
            e.target.value = ''; // Reset
            return;
        }

        setIsUploading(true);
        setStatus(`Uploading ${file.name}...`);

        try {
            await uploadMusic(file);
            setStatus(`Success! ${file.name} uploaded & activated.`);
            await fetchInfo();
        } catch (error) {
            console.error(error);
            setStatus('Error: ' + (error.response?.data?.detail || error.message));
        } finally {
            setIsUploading(false);
            e.target.value = ''; // Reset input to allow re-uploading same file
        }
    };

    const handleActivate = async (filename) => {
        // Optimistic UI update could go here, but fetch is fast enough
        try {
            await activateMusic(filename);
            await fetchInfo();
        } catch (error) {
            console.error("Failed to activate track", error);
            alert("Failed to activate track: " + error.message);
        }
    };

    const formatBytes = (bytes, decimals = 2) => {
        if (!+bytes) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
    };

    const formatDate = (isoString) => {
        return new Date(isoString).toLocaleDateString();
    };

    return (
        <div style={{ maxWidth: '1000px', margin: '0 auto', paddingBottom: '4rem' }}>
            <div className="page-header">
                <div>
                    <h1 className="page-title">Audio Studio</h1>
                    <p style={{ color: 'var(--text-secondary)', marginTop: '8px' }}>Manage background music for your AI videos.</p>
                </div>
            </div>

            {/* Hidden Audio Element for Previews */}
            <audio
                ref={audioRef}
                src={playingPreview ? `${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/assets/music/${playingPreview}` : ''}
                onEnded={() => setPlayingPreview(null)}
            />

            {/* Top Section: Single Auto-Upload Button */}
            <div className="card" style={{ marginBottom: '32px' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>

                    {/* Hidden File Input */}
                    <input
                        type="file"
                        accept=".mp3"
                        onChange={handleAutoUpload}
                        id="file-upload"
                        style={{ display: 'none' }}
                    />

                    {/* Main Upload Trigger */}
                    <button
                        onClick={() => document.getElementById('file-upload').click()}
                        disabled={isUploading}
                        className="btn btn-primary"
                        style={{
                            fontSize: '16px',
                            padding: '24px',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '16px',
                            width: '100%',
                            justifyContent: 'center',
                            borderRadius: '12px'
                        }}
                    >
                        <span style={{ fontSize: '32px' }}>{isUploading ? '⏳' : '☁️'}</span>
                        <span style={{ fontSize: '18px', fontWeight: 'bold' }}>
                            {isUploading ? 'Uploading & Processing...' : 'Upload New Track'}
                        </span>
                    </button>
                </div>

                {status && (
                    <div style={{
                        marginTop: '16px',
                        padding: '12px',
                        borderRadius: '8px',
                        fontSize: '14px',
                        fontWeight: '500',
                        textAlign: 'center',
                        backgroundColor: status.includes('Error') ? 'rgba(239, 68, 68, 0.1)' : 'rgba(16, 185, 129, 0.1)',
                        color: status.includes('Error') ? 'var(--error)' : 'var(--success)'
                    }}>
                        {status}
                    </div>
                )}
            </div>

            {/* Bottom Section: Library */}
            <div>
                <h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', marginBottom: '1.5rem', color: 'var(--text-primary)' }}>Music Library</h2>

                {isLoadingInfo ? (
                    <div className="loading">Loading library...</div>
                ) : musicFiles.length === 0 ? (
                    <div className="empty-state">No tracks found. Upload one to get started.</div>
                ) : (
                    <div style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', // Wider cards
                        gap: '24px'
                    }}>
                        {musicFiles.map((file, idx) => {
                            const isActive = file.is_active;
                            const isPlaying = playingPreview === file.filename;

                            return (
                                <div
                                    key={idx}
                                    onClick={() => handleActivate(file.filename)}
                                    style={{
                                        backgroundColor: 'var(--bg-secondary)',
                                        borderRadius: '16px',
                                        padding: '20px',
                                        cursor: 'pointer',
                                        border: isActive ? '2px solid var(--accent-primary)' : '2px solid transparent',
                                        boxShadow: isActive ? '0 0 0 2px rgba(139, 92, 246, 0.2)' : 'none',
                                        transition: 'all 0.2s',
                                        position: 'relative',
                                        display: 'flex',
                                        flexDirection: 'column',
                                        gap: '16px'
                                    }}
                                    className="music-card"
                                >
                                    {/* Top Row: Icon + Meta */}
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                                        <div style={{
                                            width: '48px', height: '48px',
                                            borderRadius: '12px',
                                            backgroundColor: isActive ? 'var(--accent-primary)' : 'var(--bg-tertiary)',
                                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                                            fontSize: '24px',
                                            color: isActive ? 'white' : 'var(--text-secondary)'
                                        }}>
                                            🎵
                                        </div>
                                        <div style={{ flex: 1, minWidth: 0 }}>
                                            <h3 style={{
                                                fontWeight: 'bold',
                                                fontSize: '16px',
                                                marginBottom: '4px',
                                                whiteSpace: 'nowrap',
                                                overflow: 'hidden',
                                                textOverflow: 'ellipsis',
                                                color: isActive ? 'var(--accent-primary)' : 'var(--text-primary)'
                                            }} title={file.filename}>
                                                {file.filename}
                                            </h3>
                                            <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                                                {formatBytes(file.size)} • {formatDate(file.created_at)}
                                            </div>
                                        </div>
                                    </div>

                                    {/* Bottom Row: Controls */}
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 'auto' }}>
                                        <div style={{ display: 'flex', gap: '8px' }}>
                                            {/* Play/Pause Button */}
                                            <button
                                                onClick={(e) => handlePlayPause(e, file)}
                                                style={{
                                                    background: isPlaying ? 'var(--accent-primary)' : 'var(--bg-tertiary)',
                                                    color: isPlaying ? 'white' : 'var(--text-primary)',
                                                    border: 'none',
                                                    borderRadius: '50%',
                                                    width: '36px', height: '36px',
                                                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                                                    cursor: 'pointer',
                                                    transition: 'all 0.2s'
                                                }}
                                                title="Preview"
                                            >
                                                {isPlaying ? '⏸' : '▶'}
                                            </button>

                                            {/* Delete Button */}
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    if (window.confirm(`Delete "${file.filename}"?`)) {
                                                        const { deleteMusic } = require('../api');
                                                        deleteMusic(file.filename).then(() => fetchInfo()).catch(err => alert("Failed to delete: " + err));
                                                    }
                                                }}
                                                style={{
                                                    background: 'rgba(239, 68, 68, 0.1)',
                                                    color: 'var(--error)',
                                                    border: 'none',
                                                    borderRadius: '50%',
                                                    width: '36px', height: '36px',
                                                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                                                    cursor: 'pointer',
                                                    transition: 'all 0.2s'
                                                }}
                                                title="Delete"
                                            >
                                                🗑️
                                            </button>
                                        </div>

                                        {/* Status Indicator */}
                                        {isActive ? (
                                            <div style={{
                                                fontSize: '12px', fontWeight: 'bold',
                                                color: 'var(--accent-primary)',
                                                display: 'flex', alignItems: 'center', gap: '6px'
                                            }}>
                                                <span>✓ ACTIVE</span>
                                            </div>
                                        ) : (
                                            <div style={{ fontSize: '12px', color: 'var(--text-secondary)', opacity: 0 }}>
                                                Click to Activate
                                            </div>
                                        )}
                                    </div>

                                    {/* Hover effect for non-active cards could go here via CSS, 
                                         but inline style limitation makes it tricky. 
                                         However, the cursor pointer implies action. */}
                                </div>
                            );
                        })}
                    </div>
                )}
            </div>
        </div>
    );
};

export default MusicManager;
