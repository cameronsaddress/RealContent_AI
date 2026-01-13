import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
    getCharacterConfig, saveCharacterConfig, generateAvatarImage,
    uploadAvatarImage, getAvatarImages, getHeyGenAvatars, getElevenLabsVoices
} from '../api';
import './Character.css';

const Character = () => {
    // Character configuration state
    const [config, setConfig] = useState({
        voice_id: '',
        avatar_id: '',
        avatar_type: 'video_avatar',
        voice_name: '',
        avatar_name: '',
        image_url: '',
        is_cloned_voice: false
    });

    // Avatar data with pagination
    const [videoAvatars, setVideoAvatars] = useState([]);
    const [talkingPhotos, setTalkingPhotos] = useState([]);
    const [uploadedImages, setUploadedImages] = useState([]);

    // Pagination state
    const [videoPage, setVideoPage] = useState(1);
    const [photoPage, setPhotoPage] = useState(1);
    const [videoHasMore, setVideoHasMore] = useState(true);
    const [photoHasMore, setPhotoHasMore] = useState(true);
    const [videoTotal, setVideoTotal] = useState(0);
    const [photoTotal, setPhotoTotal] = useState(0);
    const [isLoadingMore, setIsLoadingMore] = useState(false);

    // Search
    const [avatarSearch, setAvatarSearch] = useState('');
    const searchTimeoutRef = useRef(null);

    // Voice data
    const [clonedVoices, setClonedVoices] = useState([]);
    const [libraryVoices, setLibraryVoices] = useState([]);

    // UI state
    const [isLoading, setIsLoading] = useState(true);
    const [isSaving, setIsSaving] = useState(false);
    const [isGenerating, setIsGenerating] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [generatedImage, setGeneratedImage] = useState(null);

    // Tabs
    const [avatarTab, setAvatarTab] = useState('video'); // video, photo, upload
    const [voiceTab, setVoiceTab] = useState('cloned'); // cloned, library

    // Filters
    const [voiceSearch, setVoiceSearch] = useState('');
    const [voiceGender, setVoiceGender] = useState('all');

    // AI Generator prompt
    const [avatarPrompt, setAvatarPrompt] = useState(
        "Raw unedited photo of a real human woman, news anchor, looking at camera. Shot on Sony A7R IV. Hyper-realistic, 8k, pore-level skin detail, slight imperfections. Green screen background. Natural studio lighting. NOT an illustration, NOT 3D render. 100% photograph."
    );

    // Refs for infinite scroll
    const videoGridRef = useRef(null);
    const photoGridRef = useRef(null);

    // Load avatars with pagination
    const loadAvatars = useCallback(async (type, page, search = '', append = false) => {
        try {
            if (page === 1) setIsLoading(true);
            else setIsLoadingMore(true);

            const data = await getHeyGenAvatars({
                avatar_type: type === 'video' ? 'video' : 'talking_photo',
                page,
                limit: 24,
                search: search || undefined
            });

            if (type === 'video') {
                setVideoAvatars(prev => append ? [...prev, ...data.items] : data.items);
                setVideoHasMore(data.has_more);
                setVideoTotal(data.video_avatars_total);
                setPhotoTotal(data.talking_photos_total);
                setVideoPage(page);
            } else {
                setTalkingPhotos(prev => append ? [...prev, ...data.items] : data.items);
                setPhotoHasMore(data.has_more);
                setPhotoTotal(data.talking_photos_total);
                setVideoTotal(data.video_avatars_total);
                setPhotoPage(page);
            }
        } catch (error) {
            console.error(`Failed to load ${type} avatars`, error);
        } finally {
            setIsLoading(false);
            setIsLoadingMore(false);
        }
    }, []);

    // Handle scroll for infinite loading
    const handleScroll = useCallback((e, type) => {
        const { scrollTop, scrollHeight, clientHeight } = e.target;
        const isNearBottom = scrollHeight - scrollTop - clientHeight < 200;

        if (isNearBottom && !isLoadingMore) {
            if (type === 'video' && videoHasMore) {
                loadAvatars('video', videoPage + 1, avatarSearch, true);
            } else if (type === 'photo' && photoHasMore) {
                loadAvatars('photo', photoPage + 1, avatarSearch, true);
            }
        }
    }, [isLoadingMore, videoHasMore, photoHasMore, videoPage, photoPage, avatarSearch, loadAvatars]);

    // Handle search with debounce
    const handleAvatarSearch = (value) => {
        setAvatarSearch(value);
        if (searchTimeoutRef.current) clearTimeout(searchTimeoutRef.current);

        searchTimeoutRef.current = setTimeout(() => {
            if (avatarTab === 'video') {
                setVideoPage(1);
                loadAvatars('video', 1, value, false);
            } else if (avatarTab === 'photo') {
                setPhotoPage(1);
                loadAvatars('photo', 1, value, false);
            }
        }, 300);
    };

    useEffect(() => {
        loadData();
    }, []);

    const loadData = async () => {
        try {
            setIsLoading(true);
            const [voiceData, configData, imagesData] = await Promise.all([
                getElevenLabsVoices(),
                getCharacterConfig(),
                getAvatarImages()
            ]);

            // Load first page of video avatars
            await loadAvatars('video', 1, '', false);

            setUploadedImages(imagesData.images || []);

            // Set voice data
            setClonedVoices(voiceData.cloned_voices || []);
            setLibraryVoices(voiceData.library_voices || []);

            // Set config and determine initial tab
            if (configData) {
                setConfig(configData);

                // Set voice tab based on whether it's a cloned voice
                if (configData.is_cloned_voice || voiceData.cloned_voices?.some(v => v.voice_id === configData.voice_id)) {
                    setVoiceTab('cloned');
                } else {
                    setVoiceTab(voiceData.cloned_voices?.length > 0 ? 'cloned' : 'library');
                }

                // Set avatar tab based on type
                if (configData.avatar_type === 'talking_photo') {
                    setAvatarTab('photo');
                } else if (configData.avatar_type === 'custom_photo' || (configData.image_url && configData.avatar_id === 'static_image')) {
                    setAvatarTab('upload');
                }
            }
        } catch (error) {
            console.error("Failed to load character data", error);
        } finally {
            setIsLoading(false);
        }
    };

    // Load avatars when tab changes
    useEffect(() => {
        if (avatarTab === 'video' && videoAvatars.length === 0) {
            loadAvatars('video', 1, avatarSearch, false);
        } else if (avatarTab === 'photo' && talkingPhotos.length === 0) {
            loadAvatars('photo', 1, avatarSearch, false);
        }
    }, [avatarTab, avatarSearch, loadAvatars, videoAvatars.length, talkingPhotos.length]);

    const handleSave = async () => {
        try {
            setIsSaving(true);
            await saveCharacterConfig(config);
        } catch (error) {
            console.error("Failed to save", error);
            alert("Failed to save configuration");
        } finally {
            setIsSaving(false);
        }
    };

    // Avatar selection handlers
    const selectVideoAvatar = (avatar) => {
        setConfig({
            ...config,
            avatar_id: avatar.avatar_id,
            avatar_name: avatar.avatar_name,
            avatar_type: 'video_avatar',
            image_url: avatar.preview_image_url
        });
    };

    const selectTalkingPhoto = (avatar) => {
        setConfig({
            ...config,
            avatar_id: avatar.avatar_id,
            avatar_name: avatar.avatar_name,
            avatar_type: 'talking_photo',
            image_url: avatar.preview_image_url
        });
    };

    const selectGalleryImage = (img) => {
        let fullUrl = img.url;
        if (fullUrl.startsWith('/')) {
            fullUrl = `http://${window.location.hostname}:8000${fullUrl}`;
        }
        setConfig({
            ...config,
            avatar_id: 'static_image',
            avatar_type: 'custom_photo',
            image_url: fullUrl,
            avatar_name: 'Custom Upload'
        });
    };

    // Voice selection handlers
    const selectVoice = (voice, isCloned) => {
        setConfig({
            ...config,
            voice_id: voice.voice_id,
            voice_name: voice.name,
            is_cloned_voice: isCloned
        });
    };

    // AI Generation
    const handleGenerateAvatar = async () => {
        try {
            setIsGenerating(true);
            const res = await generateAvatarImage(avatarPrompt);
            if (res.image_url) {
                let fullUrl = res.image_url;
                if (fullUrl.startsWith('/')) {
                    fullUrl = `http://${window.location.hostname}:8000${fullUrl}`;
                }
                setGeneratedImage(fullUrl);
                refreshImages();
            }
        } catch (error) {
            console.error("Failed to generate", error);
            alert("Generation failed. Check backend logs.");
        } finally {
            setIsGenerating(false);
        }
    };

    const confirmGeneratedAvatar = () => {
        setConfig({
            ...config,
            avatar_id: 'static_image',
            avatar_type: 'custom_photo',
            image_url: generatedImage,
            avatar_name: 'AI Generated Avatar'
        });
        setGeneratedImage(null);
    };

    // Upload handling
    const handleUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        try {
            setIsUploading(true);
            const res = await uploadAvatarImage(file);
            if (res.image_url) {
                let fullUrl = res.image_url;
                if (fullUrl.startsWith('/')) {
                    fullUrl = `http://${window.location.hostname}:8000${fullUrl}`;
                }
                setConfig({
                    ...config,
                    avatar_id: 'static_image',
                    avatar_type: 'custom_photo',
                    image_url: fullUrl,
                    avatar_name: 'Custom Upload'
                });
                refreshImages();
            }
        } catch (error) {
            console.error("Upload failed", error);
            alert("Upload failed.");
        } finally {
            setIsUploading(false);
        }
    };

    const refreshImages = async () => {
        try {
            const data = await getAvatarImages();
            setUploadedImages(data.images || []);
        } catch (e) {
            console.error("Failed to refresh images", e);
        }
    };

    // Filter voices
    const filterVoices = (voices) => {
        return voices.filter(v => {
            const matchesSearch = v.name.toLowerCase().includes(voiceSearch.toLowerCase());
            const gender = v.labels?.gender || 'unknown';
            const matchesGender = voiceGender === 'all' || gender === voiceGender;
            return matchesSearch && matchesGender;
        });
    };

    const filteredClonedVoices = filterVoices(clonedVoices);
    const filteredLibraryVoices = filterVoices(libraryVoices);

    if (isLoading) {
        return (
            <div className="character-page">
                <div className="loading-state">Loading character configuration...</div>
            </div>
        );
    }

    return (
        <div className="character-page">
            {/* Header */}
            <header className="character-header">
                <div>
                    <h1>AI Character Setup</h1>
                    <p>Configure your pipeline's persona and voice</p>
                </div>
                <button
                    onClick={handleSave}
                    disabled={isSaving}
                    className="save-btn"
                >
                    {isSaving ? 'Saving...' : 'Save Configuration'}
                </button>
            </header>

            {/* Preview Card */}
            <div className="preview-card">
                {config.image_url ? (
                    <img src={config.image_url} alt="Selected Avatar" className="preview-avatar" />
                ) : (
                    <div className="preview-avatar-placeholder">?</div>
                )}
                <div className="preview-info">
                    <h3>{config.avatar_name || 'No Avatar Selected'}</h3>
                    <div>
                        <span className="type-badge">{config.avatar_type?.replace(/_/g, ' ')}</span>
                        {config.is_cloned_voice && <span className="cloned-badge">Cloned Voice</span>}
                    </div>
                    <p className="preview-voice">
                        Voice: {config.voice_name || 'None selected'}
                    </p>
                </div>
            </div>

            {/* Main Grid */}
            <div className="character-grid">
                {/* Avatar Section */}
                <section className="section-card">
                    <h2>
                        <span className="step-number">1</span>
                        Choose Avatar
                    </h2>

                    {/* Avatar Tabs */}
                    <div className="tabs">
                        <button
                            className={`tab-btn ${avatarTab === 'video' ? 'active' : ''}`}
                            onClick={() => setAvatarTab('video')}
                        >
                            Video Avatars
                            <span className="count">{videoTotal || videoAvatars.length}</span>
                        </button>
                        <button
                            className={`tab-btn ${avatarTab === 'photo' ? 'active' : ''}`}
                            onClick={() => setAvatarTab('photo')}
                        >
                            Talking Photos
                            <span className="count">{photoTotal || talkingPhotos.length}</span>
                        </button>
                        <button
                            className={`tab-btn ${avatarTab === 'upload' ? 'active' : ''}`}
                            onClick={() => setAvatarTab('upload')}
                        >
                            Upload / AI Gen
                        </button>
                    </div>

                    {/* Search bar for avatars */}
                    {(avatarTab === 'video' || avatarTab === 'photo') && (
                        <div className="avatar-search">
                            <input
                                type="text"
                                placeholder="Search avatars..."
                                value={avatarSearch}
                                onChange={(e) => handleAvatarSearch(e.target.value)}
                                className="avatar-search-input"
                            />
                        </div>
                    )}

                    {/* Video Avatars Tab */}
                    {avatarTab === 'video' && (
                        <div
                            className="avatar-grid-container"
                            ref={videoGridRef}
                            onScroll={(e) => handleScroll(e, 'video')}
                        >
                            <div className="avatar-grid">
                                {videoAvatars.length === 0 && !isLoading ? (
                                    <div className="empty-state">
                                        <p>No video avatars found</p>
                                        <p>{avatarSearch ? 'Try a different search' : 'Check your HeyGen account'}</p>
                                    </div>
                                ) : (
                                    videoAvatars.map(avatar => (
                                        <div
                                            key={avatar.avatar_id}
                                            className={`avatar-card ${config.avatar_id === avatar.avatar_id ? 'selected' : ''}`}
                                            onClick={() => selectVideoAvatar(avatar)}
                                        >
                                            <img
                                                src={avatar.preview_image_url}
                                                alt={avatar.avatar_name}
                                                loading="lazy"
                                                onError={(e) => {
                                                    e.target.onerror = null;
                                                    e.target.src = `https://ui-avatars.com/api/?name=${encodeURIComponent(avatar.avatar_name)}&background=random`;
                                                }}
                                            />
                                            <span className="avatar-name">{avatar.avatar_name}</span>
                                            {avatar.preview_video_url && <span className="video-badge">VIDEO</span>}
                                        </div>
                                    ))
                                )}
                            </div>
                            {isLoadingMore && (
                                <div className="loading-more">Loading more avatars...</div>
                            )}
                            {!videoHasMore && videoAvatars.length > 0 && (
                                <div className="end-of-list">All {videoTotal} avatars loaded</div>
                            )}
                        </div>
                    )}

                    {/* Talking Photos Tab */}
                    {avatarTab === 'photo' && (
                        <div
                            className="avatar-grid-container"
                            ref={photoGridRef}
                            onScroll={(e) => handleScroll(e, 'photo')}
                        >
                            <div className="avatar-grid">
                                {talkingPhotos.length === 0 && !isLoading ? (
                                    <div className="empty-state">
                                        <p>No talking photos found</p>
                                        <p>{avatarSearch ? 'Try a different search' : 'Upload a photo or use the AI generator'}</p>
                                    </div>
                                ) : (
                                    talkingPhotos.map(avatar => (
                                        <div
                                            key={avatar.avatar_id}
                                            className={`avatar-card ${config.avatar_id === avatar.avatar_id ? 'selected' : ''}`}
                                            onClick={() => selectTalkingPhoto(avatar)}
                                        >
                                            <img
                                                src={avatar.preview_image_url}
                                                alt={avatar.avatar_name}
                                                loading="lazy"
                                                onError={(e) => {
                                                    e.target.onerror = null;
                                                    e.target.src = `https://ui-avatars.com/api/?name=${encodeURIComponent(avatar.avatar_name)}&background=random`;
                                                }}
                                            />
                                            <span className="avatar-name">{avatar.avatar_name}</span>
                                        </div>
                                    ))
                                )}
                            </div>
                            {isLoadingMore && (
                                <div className="loading-more">Loading more photos...</div>
                            )}
                            {!photoHasMore && talkingPhotos.length > 0 && (
                                <div className="end-of-list">All {photoTotal} photos loaded</div>
                            )}
                        </div>
                    )}

                    {/* Upload / AI Gen Tab */}
                    {avatarTab === 'upload' && (
                        <div className="upload-zone">
                            {/* Upload Section */}
                            <div className="upload-dropzone">
                                <p>Upload a photo to create a talking avatar</p>
                                <input
                                    type="file"
                                    accept="image/*"
                                    onChange={handleUpload}
                                    disabled={isUploading}
                                    className="upload-input"
                                    id="avatar-upload-input"
                                />
                                <label htmlFor="avatar-upload-input" className="upload-btn">
                                    {isUploading ? 'Uploading...' : 'Select Image'}
                                </label>
                            </div>

                            {/* AI Generator */}
                            <div className="ai-gen-section">
                                <p>Or generate an AI avatar with a prompt</p>
                                <label className="prompt-label">Generation Prompt</label>
                                <textarea
                                    value={avatarPrompt}
                                    onChange={(e) => setAvatarPrompt(e.target.value)}
                                    className="prompt-textarea"
                                />

                                {generatedImage ? (
                                    <div className="generated-preview">
                                        <img src={generatedImage} alt="Generated" />
                                        <div className="generated-actions">
                                            <button onClick={confirmGeneratedAvatar} className="use-btn">
                                                Use This Avatar
                                            </button>
                                            <button onClick={() => setGeneratedImage(null)} className="discard-btn">
                                                Discard
                                            </button>
                                        </div>
                                    </div>
                                ) : (
                                    <button
                                        onClick={handleGenerateAvatar}
                                        disabled={isGenerating}
                                        className="generate-btn"
                                    >
                                        {isGenerating ? 'Generating...' : 'Generate AI Avatar'}
                                    </button>
                                )}
                            </div>

                            {/* Uploaded Images Gallery */}
                            {uploadedImages.length > 0 && (
                                <div className="gallery-section">
                                    <h3>My Uploaded Images</h3>
                                    <div className="gallery-grid">
                                        {uploadedImages.map((img, idx) => {
                                            let fullUrl = img.url;
                                            if (fullUrl.startsWith('/')) {
                                                fullUrl = `http://${window.location.hostname}:8000${fullUrl}`;
                                            }
                                            const isSelected = config.image_url === fullUrl;
                                            return (
                                                <div
                                                    key={idx}
                                                    className={`gallery-item ${isSelected ? 'selected' : ''}`}
                                                    onClick={() => selectGalleryImage(img)}
                                                >
                                                    <img src={fullUrl} alt={img.name} />
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </section>

                {/* Voice Section */}
                <section className="section-card">
                    <h2>
                        <span className="step-number">2</span>
                        Choose Voice
                    </h2>

                    {/* Voice Tabs */}
                    <div className="tabs">
                        <button
                            className={`tab-btn ${voiceTab === 'cloned' ? 'active' : ''}`}
                            onClick={() => setVoiceTab('cloned')}
                        >
                            My Voices
                            <span className="count">{clonedVoices.length}</span>
                        </button>
                        <button
                            className={`tab-btn ${voiceTab === 'library' ? 'active' : ''}`}
                            onClick={() => setVoiceTab('library')}
                        >
                            Library
                            <span className="count">{libraryVoices.length}</span>
                        </button>
                    </div>

                    {/* Voice Filters */}
                    <div className="voice-filters">
                        <input
                            type="text"
                            placeholder="Search voices..."
                            value={voiceSearch}
                            onChange={(e) => setVoiceSearch(e.target.value)}
                            className="voice-search"
                        />
                        <select
                            value={voiceGender}
                            onChange={(e) => setVoiceGender(e.target.value)}
                            className="voice-gender-filter"
                        >
                            <option value="all">All Genders</option>
                            <option value="male">Male</option>
                            <option value="female">Female</option>
                        </select>
                    </div>

                    {/* Cloned Voices */}
                    {voiceTab === 'cloned' && (
                        <div className="voice-list">
                            {filteredClonedVoices.length === 0 ? (
                                <div className="empty-state">
                                    <p>No cloned voices found</p>
                                    <p>Clone a voice in ElevenLabs to see it here</p>
                                </div>
                            ) : (
                                filteredClonedVoices.map(voice => (
                                    <div
                                        key={voice.voice_id}
                                        className={`voice-item ${config.voice_id === voice.voice_id ? 'selected' : ''}`}
                                        onClick={() => selectVoice(voice, true)}
                                    >
                                        <div className="voice-info">
                                            <span className="star-icon">&#11088;</span>
                                            <span className="voice-name">{voice.name}</span>
                                            <span className="voice-meta">
                                                {voice.labels?.accent || 'Unknown'} {voice.labels?.gender || ''}
                                            </span>
                                        </div>
                                        {voice.preview_url && (
                                            <audio
                                                controls
                                                src={voice.preview_url}
                                                className="voice-preview"
                                                onClick={(e) => e.stopPropagation()}
                                            />
                                        )}
                                    </div>
                                ))
                            )}
                        </div>
                    )}

                    {/* Library Voices */}
                    {voiceTab === 'library' && (
                        <div className="voice-list">
                            {filteredLibraryVoices.length === 0 ? (
                                <div className="empty-state">
                                    <p>No voices match your search</p>
                                </div>
                            ) : (
                                filteredLibraryVoices.map(voice => (
                                    <div
                                        key={voice.voice_id}
                                        className={`voice-item ${config.voice_id === voice.voice_id ? 'selected' : ''}`}
                                        onClick={() => selectVoice(voice, false)}
                                    >
                                        <div className="voice-info">
                                            <span className="voice-name">{voice.name}</span>
                                            <span className="voice-meta">
                                                {voice.category} {voice.labels?.accent || ''} {voice.labels?.gender || ''}
                                            </span>
                                        </div>
                                        {voice.preview_url && (
                                            <audio
                                                controls
                                                src={voice.preview_url}
                                                className="voice-preview"
                                                onClick={(e) => e.stopPropagation()}
                                            />
                                        )}
                                    </div>
                                ))
                            )}
                        </div>
                    )}
                </section>
            </div>
        </div>
    );
};

export default Character;
