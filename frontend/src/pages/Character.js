import React, { useState, useEffect } from 'react';
import {
    getVoices, getAvatars, getCharacterConfig, saveCharacterConfig, generateAvatarImage,
    uploadAvatarImage, getAvatarImages
} from '../api';

const Character = () => {
    const [config, setConfig] = useState({
        voice_id: '',
        avatar_id: '',
        avatar_type: 'pretrained',
        voice_name: '',
        avatar_name: '',
        image_url: ''
    });

    const [voices, setVoices] = useState([]);
    const [avatars, setAvatars] = useState([]);
    const [uploadedImages, setUploadedImages] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [isSaving, setIsSaving] = useState(false);
    const [isGenerating, setIsGenerating] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [generatedImage, setGeneratedImage] = useState(null);

    // Filters
    const [voiceSearch, setVoiceSearch] = useState('');
    const [voiceGender, setVoiceGender] = useState('all');
    const [activeTab, setActiveTab] = useState('pretrained'); // pretrained, ai_gen, upload

    // Prompt State
    const [avatarPrompt, setAvatarPrompt] = useState(
        "Raw unedited photo of a real human woman, news anchor, looking at camera. Shot on Sony A7R IV. Hyper-realistic, 8k, pore-level skin detail, slight imperfections. Green screen background. Natural studio lighting. NOT an illustration, NOT 3D render. 100% photograph."
    );

    useEffect(() => {
        loadData();
    }, []);

    const loadData = async () => {
        try {
            setIsLoading(true);
            const [voicesData, avatarsData, configData, imagesData] = await Promise.all([
                getVoices(),
                getAvatars(),
                getCharacterConfig(),
                getAvatarImages()
            ]);

            setVoices(voicesData.voices || []);
            setAvatars(avatarsData.avatars || []);
            setUploadedImages(imagesData.images || []);

            if (configData) {
                setConfig(configData);
                if (configData.avatar_type === 'generated' || configData.avatar_type === 'static' || configData.avatar_type === 'talking_photo') {
                    // Start on AI Gen tab if it's a custom image, or keep default
                    // We might want to switch tabs based on type, but for now logic is fine.
                    if (configData.image_url && !avatarsData.avatars.find(a => a.avatar_id === configData.avatar_id)) {
                        setActiveTab('upload');
                    }
                }
            }
        } catch (error) {
            console.error("Failed to load character data", error);
        } finally {
            setIsLoading(false);
        }
    };

    const handleSave = async () => {
        try {
            setIsSaving(true);
            await saveCharacterConfig(config);
            // Optional: Show success toast
        } catch (error) {
            console.error("Failed to save", error);
        } finally {
            setIsSaving(false);
        }
    };

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
                // Refresh list to incude new generated image
                refreshImages();
            }
        } catch (error) {
            console.error("Failed to generate", error);
            alert("Generation failed. Check backend logs.");
        } finally {
            setIsGenerating(false);
        }
    };

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

                // Auto-select uploaded image
                setConfig({
                    ...config,
                    avatar_id: 'static_image',
                    avatar_type: 'talking_photo', // Use talking_photo for custom uploads
                    image_url: fullUrl,
                    avatar_name: 'Custom Upload'
                });

                // Refresh gallery
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

    const confirmGeneratedAvatar = () => {
        setConfig({
            ...config,
            avatar_id: 'static_image',
            avatar_type: 'talking_photo',
            image_url: generatedImage,
            avatar_name: 'AI Generated Avatar'
        });
        setGeneratedImage(null); // Clear preview, it's now selected
    };

    const selectGalleryImage = (img) => {
        // Construct full URL if relative
        let fullUrl = img.url;
        if (fullUrl.startsWith('/')) {
            fullUrl = `http://${window.location.hostname}:8000${fullUrl}`;
        }

        setConfig({
            ...config,
            avatar_id: 'static_image',
            avatar_type: 'talking_photo',
            image_url: fullUrl,
            avatar_name: 'Custom Avatar'
        });
    };

    const filteredVoices = voices.filter(v => {
        const matchesSearch = v.name.toLowerCase().includes(voiceSearch.toLowerCase());
        const gender = v.labels?.gender || 'unknown';
        const matchesGender = voiceGender === 'all' || gender === voiceGender;
        return matchesSearch && matchesGender;
    });

    return (
        <div style={{ padding: '2rem', maxWidth: '1200px', margin: '0 auto', color: 'var(--text-primary)' }}>
            <header style={{ marginBottom: '2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                    <h1 style={{ fontSize: '2rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>AI Character Setup</h1>
                    <p style={{ color: 'var(--text-secondary)' }}>Configure your pipeline's persona and voice.</p>
                </div>
                <button
                    onClick={handleSave}
                    disabled={isSaving}
                    style={{
                        padding: '0.75rem 1.5rem',
                        backgroundColor: 'var(--accent-primary)',
                        color: 'white',
                        border: 'none',
                        borderRadius: '8px',
                        cursor: isSaving ? 'wait' : 'pointer',
                        fontWeight: 'bold'
                    }}
                >
                    {isSaving ? 'Saving...' : 'Save Configuration'}
                </button>
            </header>

            {isLoading ? <div style={{ textAlign: 'center', padding: '2rem' }}>Loading assets...</div> : (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>

                    {/* LEFT: AVATAR SECTION */}
                    <section className="card">
                        <h2 style={{ fontSize: '1.25rem', marginBottom: '1rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem' }}>
                            1. Choose Avatar
                        </h2>

                        <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem' }}>
                            <button
                                onClick={() => setActiveTab('pretrained')}
                                style={{
                                    padding: '0.5rem 1rem',
                                    borderRadius: '6px',
                                    border: 'none',
                                    backgroundColor: activeTab === 'pretrained' ? 'var(--bg-tertiary)' : 'transparent',
                                    color: activeTab === 'pretrained' ? 'var(--accent-primary)' : 'var(--text-secondary)',
                                    cursor: 'pointer',
                                    fontWeight: 'bold'
                                }}
                            >
                                Pre-trained
                            </button>
                            <button
                                onClick={() => setActiveTab('ai_gen')}
                                style={{
                                    padding: '0.5rem 1rem',
                                    borderRadius: '6px',
                                    border: 'none',
                                    backgroundColor: activeTab === 'ai_gen' ? 'var(--bg-tertiary)' : 'transparent',
                                    color: activeTab === 'ai_gen' ? 'var(--accent-primary)' : 'var(--text-secondary)',
                                    cursor: 'pointer',
                                    fontWeight: 'bold'
                                }}
                            >
                                AI Generator
                            </button>
                            <button
                                onClick={() => setActiveTab('upload')}
                                style={{
                                    padding: '0.5rem 1rem',
                                    borderRadius: '6px',
                                    border: 'none',
                                    backgroundColor: activeTab === 'upload' ? 'var(--bg-tertiary)' : 'transparent',
                                    color: activeTab === 'upload' ? 'var(--accent-primary)' : 'var(--text-secondary)',
                                    cursor: 'pointer',
                                    fontWeight: 'bold'
                                }}
                            >
                                Upload
                            </button>
                        </div>

                        {activeTab === 'pretrained' && (
                            <div style={{
                                display: 'grid',
                                gridTemplateColumns: 'repeat(auto-fill, minmax(100px, 1fr))',
                                gap: '1rem',
                                maxHeight: '500px',
                                overflowY: 'auto',
                                paddingRight: '0.5rem'
                            }}>
                                {avatars.map(avatar => (
                                    <div
                                        key={avatar.avatar_id}
                                        onClick={() => setConfig({ ...config, avatar_id: avatar.avatar_id, avatar_name: avatar.name, avatar_type: 'pretrained', image_url: avatar.preview_image_url })}
                                        style={{
                                            cursor: 'pointer',
                                            border: config.avatar_id === avatar.avatar_id ? '2px solid var(--accent-primary)' : '2px solid transparent',
                                            borderRadius: '8px',
                                            overflow: 'hidden',
                                            position: 'relative'
                                        }}
                                    >
                                        <img
                                            src={avatar.preview_image_url || avatar.preview_url}
                                            alt={avatar.name}
                                            style={{ width: '100%', aspectRatio: '1', objectFit: 'cover' }}
                                            onError={(e) => {
                                                e.target.onerror = null;
                                                e.target.src = 'https://ui-avatars.com/api/?name=' + encodeURIComponent(avatar.name) + '&background=random';
                                            }}
                                        />
                                        <div style={{
                                            position: 'absolute', bottom: 0, left: 0, right: 0,
                                            background: 'rgba(0,0,0,0.7)', padding: '4px', fontSize: '0.75rem', textAlign: 'center'
                                        }}>
                                            {avatar.name}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}

                        {activeTab === 'ai_gen' && (
                            <div style={{ textAlign: 'center', padding: '2rem', background: 'var(--bg-tertiary)', borderRadius: '8px' }}>
                                <p style={{ marginBottom: '1.5rem', color: 'var(--text-secondary)' }}>
                                    Generate a unique AI avatar seeded with extensive movement prompts.
                                </p>

                                <div style={{ marginBottom: '1.5rem', textAlign: 'left' }}>
                                    <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                                        Generation Prompt
                                    </label>
                                    <textarea
                                        value={avatarPrompt}
                                        onChange={(e) => setAvatarPrompt(e.target.value)}
                                        style={{
                                            width: '100%',
                                            minHeight: '120px',
                                            padding: '0.75rem',
                                            borderRadius: '8px',
                                            border: '1px solid var(--border-color)',
                                            background: 'var(--bg-secondary)',
                                            color: 'white',
                                            lineHeight: '1.5',
                                            resize: 'vertical'
                                        }}
                                    />
                                </div>

                                {generatedImage ? (
                                    <div style={{ marginBottom: '1.5rem' }}>
                                        <img src={generatedImage} alt="Generated" style={{ maxWidth: '100%', borderRadius: '8px', boxShadow: '0 4px 12px rgba(0,0,0,0.3)' }} />
                                        <div style={{ marginTop: '1rem', display: 'flex', gap: '1rem', justifyContent: 'center' }}>
                                            <button
                                                onClick={confirmGeneratedAvatar}
                                                style={{ padding: '0.5rem 1rem', background: 'var(--accent-primary)', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
                                            >
                                                Use This Avatar
                                            </button>
                                            <button
                                                onClick={() => setGeneratedImage(null)}
                                                style={{ padding: '0.5rem 1rem', background: 'transparent', color: 'var(--text-secondary)', border: '1px solid var(--border-color)', borderRadius: '4px', cursor: 'pointer' }}
                                            >
                                                Discard
                                            </button>
                                        </div>
                                    </div>
                                ) : (
                                    <button
                                        onClick={handleGenerateAvatar}
                                        disabled={isGenerating}
                                        style={{
                                            padding: '1rem 2rem',
                                            fontSize: '1.1rem',
                                            background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))',
                                            color: 'white',
                                            border: 'none',
                                            borderRadius: '8px',
                                            cursor: isGenerating ? 'wait' : 'pointer'
                                        }}
                                    >
                                        {isGenerating ? 'Generating Magic...' : 'Generate AI Avatar'}
                                    </button>
                                )}
                            </div>
                        )}

                        {activeTab === 'upload' && (
                            <div style={{ padding: '2rem', background: 'var(--bg-tertiary)', borderRadius: '8px', textAlign: 'center' }}>
                                <div style={{ border: '2px dashed var(--border-color)', borderRadius: '8px', padding: '2rem', marginBottom: '1rem' }}>
                                    <p style={{ marginBottom: '1rem', color: 'var(--text-secondary)' }}>Upload your own photo to use as a dynamic Talking Photo.</p>
                                    <input
                                        type="file"
                                        accept="image/*"
                                        onChange={handleUpload}
                                        disabled={isUploading}
                                        style={{ display: 'none' }}
                                        id="avatar-upload-input"
                                    />
                                    <label
                                        htmlFor="avatar-upload-input"
                                        style={{
                                            display: 'inline-block',
                                            padding: '0.75rem 1.5rem',
                                            backgroundColor: 'var(--accent-primary)',
                                            color: 'white',
                                            borderRadius: '6px',
                                            cursor: isUploading ? 'wait' : 'pointer',
                                            fontWeight: 'bold'
                                        }}
                                    >
                                        {isUploading ? 'Uploading...' : 'Select Image File'}
                                    </label>
                                </div>
                            </div>
                        )}
                    </section>

                    {/* RIGHT: VOICE SECTION */}
                    <section className="card">
                        <h2 style={{ fontSize: '1.25rem', marginBottom: '1rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem' }}>
                            2. Choose Voice
                        </h2>
                        {/* No changes to voice section */}
                        <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem' }}>
                            <input
                                type="text"
                                placeholder="Search voices..."
                                value={voiceSearch}
                                onChange={(e) => setVoiceSearch(e.target.value)}
                                style={{
                                    flex: 1,
                                    padding: '0.5rem',
                                    borderRadius: '6px',
                                    border: '1px solid var(--border-color)',
                                    background: 'var(--bg-tertiary)',
                                    color: 'white'
                                }}
                            />
                            <select
                                value={voiceGender}
                                onChange={(e) => setVoiceGender(e.target.value)}
                                style={{
                                    padding: '0.5rem',
                                    borderRadius: '6px',
                                    border: '1px solid var(--border-color)',
                                    background: 'var(--bg-tertiary)',
                                    color: 'white'
                                }}
                            >
                                <option value="all">All Genders</option>
                                <option value="male">Male</option>
                                <option value="female">Female</option>
                            </select>
                        </div>

                        <div style={{
                            display: 'flex',
                            flexDirection: 'column',
                            gap: '0.5rem',
                            maxHeight: '500px',
                            overflowY: 'auto'
                        }}>
                            {filteredVoices.map(voice => (
                                <div
                                    key={voice.voice_id}
                                    onClick={() => setConfig({ ...config, voice_id: voice.voice_id, voice_name: voice.name })}
                                    style={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'space-between',
                                        padding: '0.75rem',
                                        borderRadius: '6px',
                                        background: config.voice_id === voice.voice_id ? 'rgba(74, 222, 128, 0.1)' : 'var(--bg-tertiary)',
                                        border: config.voice_id === voice.voice_id ? '1px solid var(--accent-primary)' : '1px solid transparent',
                                        cursor: 'pointer'
                                    }}
                                >
                                    <div>
                                        <div style={{ fontWeight: 'bold' }}>{voice.name}</div>
                                        <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                                            {voice.category} • {voice.labels?.gender || 'Unknown'} • {voice.labels?.accent || 'US'}
                                        </div>
                                    </div>
                                    {voice.preview_url && (
                                        <audio
                                            controls
                                            src={voice.preview_url}
                                            style={{ height: '30px', maxWidth: '100px' }}
                                            onClick={(e) => e.stopPropagation()}
                                        />
                                    )}
                                </div>
                            ))}
                        </div>
                    </section>

                </div>
            )}

            {/* Current Selection Bar */}
            <div style={{
                marginTop: '2rem',
                padding: '1.5rem',
                background: 'var(--bg-secondary)',
                borderRadius: '8px',
                border: '1px solid var(--border-color)',
                display: 'flex',
                alignItems: 'center',
                gap: '2rem'
            }}>
                <div>
                    <h3 style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>Selected Persona</h3>
                    <div style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>
                        {config.avatar_name || 'None'} <span style={{ color: 'var(--text-secondary)' }}>+</span> {config.voice_name || 'None'}
                    </div>
                </div>
                {config.image_url && (
                    <img src={config.image_url} alt="Selected Avatar" style={{ width: '60px', height: '60px', borderRadius: '50%', objectFit: 'cover' }} />
                )}
            </div>

            {/* New Image Gallery Collage */}
            {uploadedImages.length > 0 && (
                <div style={{ marginTop: '2rem' }}>
                    <h3 style={{ fontSize: '1.1rem', marginBottom: '1rem', color: 'var(--text-primary)' }}>My Avatar Gallery</h3>
                    <div style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fill, minmax(100px, 1fr))',
                        gap: '1rem'
                    }}>
                        {uploadedImages.map((img, idx) => {
                            let fullUrl = img.url;
                            if (fullUrl.startsWith('/')) {
                                fullUrl = `http://${window.location.hostname}:8000${fullUrl}`;
                            }
                            const isSelected = config.image_url === fullUrl;

                            return (
                                <div
                                    key={idx}
                                    onClick={() => selectGalleryImage(img)}
                                    style={{
                                        cursor: 'pointer',
                                        borderRadius: '8px',
                                        overflow: 'hidden',
                                        border: isSelected ? '2px solid var(--accent-primary)' : '2px solid transparent',
                                        position: 'relative',
                                        aspectRatio: '1',
                                        transition: 'transform 0.2s',
                                        boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
                                    }}
                                    className="gallery-item"
                                >
                                    <img
                                        src={fullUrl}
                                        alt={img.name}
                                        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                                    />
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}
        </div>
    );
};

export default Character;
