import React, { useState, useEffect } from 'react';
import {
  getAudioSettings, updateAudioSettings,
  getVideoSettings, updateVideoSettings,
  getLLMSettings, updateLLMSetting,
  getBrandPersona, updateBrandPersona,
  getViralMusic, getViralFonts, deleteViralFont, downloadGoogleFont,
  API_URL
} from '../api';

export default function Settings() {
  // Audio settings state
  const [audio, setAudio] = useState({
    original_volume: 0.7,
    avatar_volume: 1.0,
    music_volume: 0.3,
    ducking_enabled: true,
    avatar_delay_seconds: 3.0,
    duck_to_percent: 0.5,
    music_autoduck: true
  });

  // Video settings state
  const [video, setVideo] = useState({
    output_width: 1080,
    output_height: 1920,
    crf: 18,
    preset: 'slow',
    greenscreen_enabled: true,
    greenscreen_color: '#00FF00',
    // Avatar composition
    avatar_position: 'bottom-left',
    avatar_scale: 0.8,
    avatar_offset_x: -200,
    avatar_offset_y: 500,
    // Caption settings
    caption_style: 'karaoke',
    caption_font_size: 96,
    caption_font: 'Arial',
    caption_color: '#FFFFFF',
    caption_highlight_color: '#FFFF00',
    caption_outline_color: '#000000',
    caption_outline_width: 5,
    caption_position_y: 850
  });

  // LLM settings state
  const [llmPrompts, setLlmPrompts] = useState([]);
  const [editingPrompt, setEditingPrompt] = useState(null);
  const [editValue, setEditValue] = useState('');

  // Music state
  const [musicFiles, setMusicFiles] = useState([]);

  // Font state
  const [fonts, setFonts] = useState([]);
  const [newGoogleFont, setNewGoogleFont] = useState('');
  const [downloadingFont, setDownloadingFont] = useState(false);

  // Brand Persona state
  const [persona, setPersona] = useState({
    name: '',
    title: '',
    location: '',
    bio: '',
    tone: 'professional',
    energy_level: 'warm',
    humor_style: 'light',
    core_values: [],
    content_boundaries: [],
    response_style: '',
    signature_intro: '',
    signature_cta: '',
    hashtags: []
  });
  const [newValue, setNewValue] = useState('');
  const [newBoundary, setNewBoundary] = useState('');
  const [newHashtag, setNewHashtag] = useState('');

  // Loading states
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState({ text: '', type: '' });

  useEffect(() => {
    loadAllSettings();
  }, []);

  const loadAllSettings = async () => {
    setLoading(true);
    try {
      const [audioData, videoData, llmData, personaData, musicData, fontsData] = await Promise.all([
        getAudioSettings(),
        getVideoSettings(),
        getLLMSettings(),
        getBrandPersona(),
        getViralMusic(),
        getViralFonts()
      ]);
      setAudio(audioData);
      setVideo(videoData);
      setLlmPrompts(llmData);
      setMusicFiles(musicData);
      setFonts(fontsData.fonts || []);
      if (personaData) {
        setPersona(personaData);
      }
    } catch (err) {
      console.error('Failed to load settings:', err);
      showMessage('Failed to load settings', 'error');
    }
    setLoading(false);
  };

  const showMessage = (text, type = 'success') => {
    setMessage({ text, type });
    setTimeout(() => setMessage({ text: '', type: '' }), 3000);
  };

  const saveAudioSettings = async () => {
    setSaving(true);
    try {
      await updateAudioSettings(audio);
      showMessage('Audio settings saved!', 'success');
    } catch (err) {
      showMessage('Error saving audio settings', 'error');
    }
    setSaving(false);
  };

  const saveVideoSettings = async () => {
    setSaving(true);
    try {
      await updateVideoSettings(video);
      showMessage('Video settings saved!', 'success');
    } catch (err) {
      showMessage('Error saving video settings', 'error');
    }
    setSaving(false);
  };

  const handleDeleteFont = async (filename) => {
    if (!window.confirm(`Delete font "${filename}"?`)) return;
    try {
      await deleteViralFont(filename);
      setFonts(fonts.filter(f => f.filename !== filename));
      showMessage('Font deleted!', 'success');
    } catch (err) {
      showMessage('Failed to delete font', 'error');
    }
  };

  const handleDownloadGoogleFont = async () => {
    if (!newGoogleFont.trim()) return;
    setDownloadingFont(true);
    try {
      const result = await downloadGoogleFont(newGoogleFont.trim());
      showMessage(`Downloaded: ${result.files.join(', ')}`, 'success');
      setNewGoogleFont('');
      // Reload fonts
      const fontsData = await getViralFonts();
      setFonts(fontsData.fonts || []);
    } catch (err) {
      showMessage(err.response?.data?.detail || 'Failed to download font', 'error');
    }
    setDownloadingFont(false);
  };

  const saveLLMPrompt = async (key, val) => {
    setSaving(true);
    try {
      await updateLLMSetting(key, { value: val || editValue });
      if (!val) setEditingPrompt(null);
      loadAllSettings();
      showMessage('Setting saved!', 'success');
    } catch (err) {
      showMessage('Error saving setting', 'error');
    }
    setSaving(false);
  };

  const savePersonaSettings = async () => {
    setSaving(true);
    try {
      await updateBrandPersona(persona);
      showMessage('Brand persona saved!', 'success');
    } catch (err) {
      showMessage('Error saving brand persona', 'error');
    }
    setSaving(false);
  };

  // Helpers for array fields
  const addToArray = (field, value, setter) => {
    if (value.trim()) {
      setPersona({ ...persona, [field]: [...(persona[field] || []), value.trim()] });
      setter('');
    }
  };

  const removeFromArray = (field, index) => {
    const updated = [...persona[field]];
    updated.splice(index, 1);
    setPersona({ ...persona, [field]: updated });
  };

  if (loading) {
    return (
      <div className="page-container">
        <h1>Settings</h1>
        <div className="loading">Loading settings...</div>
      </div>
    );
  }

  return (
    <div className="page-container">
      <h1>Settings</h1>

      {message.text && (
        <div className={`message ${message.type}`}>
          {message.text}
        </div>
      )}

      {/* BRAND PERSONA SETTINGS */}
      <div className="settings-section persona-section">
        <h2>Brand Persona</h2>
        <p className="section-description">
          Define your brand identity. This tells the AI who you are, your tone, values, and what kind of content you will or won't create.
          The AI uses this to generate scripts that sound authentically like you.
        </p>

        <div className="settings-grid">
          {/* Identity Section */}
          <div className="persona-subsection">
            <h3>Identity</h3>

            <div className="setting-item">
              <label>Name</label>
              <input
                type="text"
                value={persona.name}
                onChange={(e) => setPersona({ ...persona, name: e.target.value })}
                className="text-input"
                placeholder="Your name or brand name"
              />
            </div>

            <div className="setting-item">
              <label>Title / Role</label>
              <input
                type="text"
                value={persona.title}
                onChange={(e) => setPersona({ ...persona, title: e.target.value })}
                className="text-input"
                placeholder="e.g., Real Estate Expert, Fitness Coach"
              />
            </div>

            <div className="setting-item">
              <label>Location</label>
              <input
                type="text"
                value={persona.location}
                onChange={(e) => setPersona({ ...persona, location: e.target.value })}
                className="text-input"
                placeholder="e.g., Austin, Texas"
              />
            </div>

            <div className="setting-item">
              <label>Bio / Background</label>
              <textarea
                value={persona.bio}
                onChange={(e) => setPersona({ ...persona, bio: e.target.value })}
                className="textarea-input"
                rows={3}
                placeholder="Brief background that helps the AI understand who you are..."
              />
              <span className="slider-hint">This context helps the AI write scripts that sound authentically like you</span>
            </div>
          </div>

          {/* Tone & Voice Section */}
          <div className="persona-subsection">
            <h3>Tone & Voice</h3>

            <div className="setting-item">
              <label>Communication Tone</label>
              <select
                value={persona.tone}
                onChange={(e) => setPersona({ ...persona, tone: e.target.value })}
                className="select-input"
              >
                <option value="professional">Professional - Business-like, polished</option>
                <option value="casual">Casual - Relaxed, conversational</option>
                <option value="warm">Warm - Friendly, approachable</option>
                <option value="energetic">Energetic - Enthusiastic, upbeat</option>
                <option value="authoritative">Authoritative - Expert, confident</option>
              </select>
            </div>

            <div className="setting-item">
              <label>Energy Level</label>
              <select
                value={persona.energy_level}
                onChange={(e) => setPersona({ ...persona, energy_level: e.target.value })}
                className="select-input"
              >
                <option value="calm">Calm - Relaxed, measured pace</option>
                <option value="warm">Warm - Comfortable, inviting energy</option>
                <option value="energetic">Energetic - Upbeat, positive</option>
                <option value="high-energy">High Energy - Exciting, dynamic</option>
              </select>
            </div>

            <div className="setting-item">
              <label>Humor Style</label>
              <select
                value={persona.humor_style}
                onChange={(e) => setPersona({ ...persona, humor_style: e.target.value })}
                className="select-input"
              >
                <option value="none">None - Serious, straightforward</option>
                <option value="light">Light - Occasional smile-worthy moments</option>
                <option value="playful">Playful - Fun, engaging humor</option>
                <option value="witty">Witty - Clever, quick remarks</option>
              </select>
            </div>
          </div>

          {/* Signature Elements */}
          <div className="persona-subsection">
            <h3>Signature Elements</h3>

            <div className="setting-item">
              <label>Opening Greeting</label>
              <input
                type="text"
                value={persona.signature_intro}
                onChange={(e) => setPersona({ ...persona, signature_intro: e.target.value })}
                className="text-input"
                placeholder="e.g., Hey neighbors!, What's up everyone?"
              />
              <span className="slider-hint">How you typically start your videos</span>
            </div>

            <div className="setting-item">
              <label>Call-to-Action</label>
              <input
                type="text"
                value={persona.signature_cta}
                onChange={(e) => setPersona({ ...persona, signature_cta: e.target.value })}
                className="text-input"
                placeholder="e.g., DM me to chat!, Follow for more tips!"
              />
              <span className="slider-hint">Use {'{location}'} to auto-insert your location</span>
            </div>

            {/* Hashtags */}
            <div className="setting-item">
              <label>Default Hashtags</label>
              <div className="tag-list">
                {(persona.hashtags || []).map((tag, i) => (
                  <span key={i} className="tag">
                    #{tag}
                    <button onClick={() => removeFromArray('hashtags', i)} className="tag-remove">×</button>
                  </span>
                ))}
              </div>
              <div className="add-tag-row">
                <input
                  type="text"
                  value={newHashtag}
                  onChange={(e) => setNewHashtag(e.target.value.replace('#', ''))}
                  onKeyPress={(e) => e.key === 'Enter' && addToArray('hashtags', newHashtag, setNewHashtag)}
                  className="text-input"
                  placeholder="Add hashtag (without #)"
                />
                <button
                  onClick={() => addToArray('hashtags', newHashtag, setNewHashtag)}
                  className="add-button"
                >
                  Add
                </button>
              </div>
            </div>
          </div>

          {/* Core Values */}
          <div className="persona-subsection">
            <h3>Core Values</h3>
            <p className="subsection-hint">What you stand for - the AI will reflect these in your content</p>

            <div className="tag-list vertical">
              {(persona.core_values || []).map((value, i) => (
                <div key={i} className="value-item">
                  <span>{value}</span>
                  <button onClick={() => removeFromArray('core_values', i)} className="remove-button">Remove</button>
                </div>
              ))}
            </div>
            <div className="add-tag-row">
              <input
                type="text"
                value={newValue}
                onChange={(e) => setNewValue(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && addToArray('core_values', newValue, setNewValue)}
                className="text-input flex-grow"
                placeholder="e.g., Honesty and transparency"
              />
              <button
                onClick={() => addToArray('core_values', newValue, setNewValue)}
                className="add-button"
              >
                Add Value
              </button>
            </div>
          </div>

          {/* Content Boundaries */}
          <div className="persona-subsection boundaries-section">
            <h3>Content Boundaries</h3>
            <p className="subsection-hint">
              What you will NEVER do - helps the AI avoid content that doesn't match your brand.
              For example, if a viral video shows unprofessional behavior, the AI will acknowledge it but pivot to your professional approach.
            </p>

            <div className="tag-list vertical">
              {(persona.content_boundaries || []).map((boundary, i) => (
                <div key={i} className="boundary-item">
                  <span>{boundary}</span>
                  <button onClick={() => removeFromArray('content_boundaries', i)} className="remove-button">Remove</button>
                </div>
              ))}
            </div>
            <div className="add-tag-row">
              <input
                type="text"
                value={newBoundary}
                onChange={(e) => setNewBoundary(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && addToArray('content_boundaries', newBoundary, setNewBoundary)}
                className="text-input flex-grow"
                placeholder="e.g., No dancing or inappropriate behavior"
              />
              <button
                onClick={() => addToArray('content_boundaries', newBoundary, setNewBoundary)}
                className="add-button"
              >
                Add Boundary
              </button>
            </div>
          </div>

          {/* Response Style */}
          <div className="persona-subsection">
            <h3>Response Style Guidelines</h3>
            <p className="subsection-hint">How should the AI handle different types of viral content?</p>

            <div className="setting-item">
              <textarea
                value={persona.response_style}
                onChange={(e) => setPersona({ ...persona, response_style: e.target.value })}
                className="textarea-input"
                rows={6}
                placeholder="When reviewing viral content:
- If professional: Praise it and add insights
- If unprofessional: Acknowledge entertainment but show professional approach
- If misinformation: Gently correct while being respectful..."
              />
            </div>
          </div>
        </div>

        <button
          onClick={savePersonaSettings}
          disabled={saving}
          className="save-button"
        >
          {saving ? 'Saving...' : 'Save Brand Persona'}
        </button>
      </div>

      {/* AUDIO SETTINGS */}
      <div className="settings-section">
        <h2>Audio Settings</h2>
        <p className="section-description">Configure how audio is mixed in the final video output.</p>

        <div className="settings-grid">
          {/* Original Volume */}
          <div className="setting-item">
            <label>
              Original Video Audio Volume: <strong>{Math.round(audio.original_volume * 100)}%</strong>
            </label>
            <input
              type="range"
              min="0"
              max="100"
              value={audio.original_volume * 100}
              onChange={(e) => setAudio({ ...audio, original_volume: e.target.value / 100 })}
              className="slider"
            />
            <span className="slider-hint">Volume of the source social media video's audio</span>
          </div>

          {/* Avatar Volume */}
          <div className="setting-item">
            <label>
              Avatar/Voiceover Volume: <strong>{Math.round(audio.avatar_volume * 100)}%</strong>
            </label>
            <input
              type="range"
              min="0"
              max="100"
              value={audio.avatar_volume * 100}
              onChange={(e) => setAudio({ ...audio, avatar_volume: e.target.value / 100 })}
              className="slider"
            />
            <span className="slider-hint">Volume of the AI avatar's voiceover</span>
          </div>

          {/* Music Volume */}
          <div className="setting-item">
            <label>
              Background Music Volume: <strong>{Math.round((audio.music_volume ?? 0.3) * 100)}%</strong>
            </label>
            <input
              type="range"
              min="0"
              max="100"
              value={(audio.music_volume ?? 0.3) * 100}
              onChange={(e) => setAudio({ ...audio, music_volume: e.target.value / 100 })}
              className="slider"
            />
            <span className="slider-hint">Volume of the background music track (if enabled)</span>
          </div>

          {/* Music Auto-Duck Toggle */}
          <div className="setting-item checkbox-item">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={audio.music_autoduck ?? true}
                onChange={(e) => setAudio({ ...audio, music_autoduck: e.target.checked })}
              />
              <span>Auto-Duck Music When Speaking</span>
            </label>
            <span className="slider-hint">Automatically lower music volume when the avatar is speaking (sidechain compression)</span>
          </div>

          {/* Ducking Toggle */}
          <div className="setting-item checkbox-item">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={audio.ducking_enabled}
                onChange={(e) => setAudio({ ...audio, ducking_enabled: e.target.checked })}
              />
              <span>Enable Audio Ducking</span>
            </label>
            <span className="slider-hint">When enabled, original audio lowers when avatar speaks</span>
          </div>

          {/* Ducking Settings (shown when enabled) */}
          {audio.ducking_enabled && (
            <div className="ducking-settings">
              <div className="setting-item">
                <label>
                  Avatar Appears After (seconds):
                </label>
                <input
                  type="number"
                  min="0"
                  max="30"
                  step="0.5"
                  value={audio.avatar_delay_seconds}
                  onChange={(e) => setAudio({ ...audio, avatar_delay_seconds: parseFloat(e.target.value) || 0 })}
                  className="number-input"
                />
                <span className="slider-hint">Time before avatar starts speaking (original audio plays full during this time)</span>
              </div>

              <div className="setting-item">
                <label>
                  Duck Original Audio To: <strong>{Math.round(audio.duck_to_percent * 100)}%</strong>
                </label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={audio.duck_to_percent * 100}
                  onChange={(e) => setAudio({ ...audio, duck_to_percent: e.target.value / 100 })}
                  className="slider"
                />
                <span className="slider-hint">Original audio volume when avatar is speaking</span>
              </div>
            </div>
          )}
        </div>

        <button
          onClick={saveAudioSettings}
          disabled={saving}
          className="save-button"
        >
          {saving ? 'Saving...' : 'Save Audio Settings'}
        </button>
      </div>

      {/* VIDEO SETTINGS */}
      <div className="settings-section">
        <h2>Video Settings</h2>
        <p className="section-description">Configure output video resolution and quality.</p>

        <div className="settings-grid">
          {/* Resolution */}
          <div className="setting-item">
            <label>Output Resolution</label>
            <div className="radio-group">
              <label className="radio-label">
                <input
                  type="radio"
                  name="resolution"
                  checked={video.output_height === 1280}
                  onChange={() => setVideo({ ...video, output_width: 720, output_height: 1280 })}
                />
                <span>720p (720x1280)</span>
              </label>
              <label className="radio-label">
                <input
                  type="radio"
                  name="resolution"
                  checked={video.output_height === 1920}
                  onChange={() => setVideo({ ...video, output_width: 1080, output_height: 1920 })}
                />
                <span>1080p (1080x1920) - Recommended</span>
              </label>
              <label className="radio-label">
                <input
                  type="radio"
                  name="resolution"
                  checked={video.output_height === 3840}
                  onChange={() => setVideo({ ...video, output_width: 2160, output_height: 3840 })}
                />
                <span>4K (2160x3840)</span>
              </label>
            </div>
          </div>

          {/* Quality */}
          <div className="setting-item">
            <label>
              Video Quality (CRF): <strong>{video.crf}</strong>
              <span className="quality-label">
                {video.crf <= 18 ? ' (High)' : video.crf <= 23 ? ' (Medium)' : ' (Low)'}
              </span>
            </label>
            <input
              type="range"
              min="15"
              max="28"
              value={video.crf}
              onChange={(e) => setVideo({ ...video, crf: parseInt(e.target.value) })}
              className="slider"
            />
            <span className="slider-hint">Lower = better quality, larger file size. 18 is recommended.</span>
          </div>

          {/* Preset */}
          <div className="setting-item">
            <label>Encoding Preset</label>
            <select
              value={video.preset}
              onChange={(e) => setVideo({ ...video, preset: e.target.value })}
              className="select-input"
            >
              <option value="ultrafast">Ultra Fast (lower quality)</option>
              <option value="fast">Fast</option>
              <option value="medium">Medium</option>
              <option value="slow">Slow (better quality)</option>
              <option value="veryslow">Very Slow (best quality)</option>
            </select>
            <span className="slider-hint">Slower presets produce better quality at the same file size</span>
          </div>

          {/* Greenscreen Toggle */}
          <div className="setting-item checkbox-item">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={video.greenscreen_enabled}
                onChange={(e) => setVideo({ ...video, greenscreen_enabled: e.target.checked })}
              />
              <span>Enable Greenscreen Background</span>
            </label>
            <span className="slider-hint">When enabled, HeyGen renders avatar with solid color background for chroma key compositing</span>
          </div>

          {/* Greenscreen Color (shown when enabled) */}
          {video.greenscreen_enabled && (
            <div className="greenscreen-settings">
              <div className="setting-item">
                <label>Greenscreen Color</label>
                <div className="color-picker-row">
                  <input
                    type="color"
                    value={video.greenscreen_color}
                    onChange={(e) => setVideo({ ...video, greenscreen_color: e.target.value })}
                    className="color-picker"
                  />
                  <input
                    type="text"
                    value={video.greenscreen_color}
                    onChange={(e) => setVideo({ ...video, greenscreen_color: e.target.value })}
                    className="color-input"
                    placeholder="#00FF00"
                  />
                </div>
                <span className="slider-hint">Background color for chroma key removal. Default green (#00FF00) works best with FFmpeg.</span>
              </div>
            </div>
          )}
        </div>

        <button
          onClick={saveVideoSettings}
          disabled={saving}
          className="save-button"
        >
          {saving ? 'Saving...' : 'Save Video Settings'}
        </button>
      </div>

      {/* AVATAR COMPOSITION SETTINGS */}
      <div className="settings-section">
        <h2>Avatar Composition</h2>
        <p className="section-description">Configure how the avatar is positioned and sized in the final video.</p>

        <div className="settings-grid">
          {/* Avatar Position */}
          <div className="setting-item">
            <label>Avatar Position</label>
            <div className="radio-group horizontal">
              <label className="radio-label">
                <input
                  type="radio"
                  name="avatar_position"
                  checked={video.avatar_position === 'bottom-left'}
                  onChange={() => setVideo({ ...video, avatar_position: 'bottom-left' })}
                />
                <span>Bottom Left</span>
              </label>
              <label className="radio-label">
                <input
                  type="radio"
                  name="avatar_position"
                  checked={video.avatar_position === 'bottom-center'}
                  onChange={() => setVideo({ ...video, avatar_position: 'bottom-center' })}
                />
                <span>Bottom Center</span>
              </label>
              <label className="radio-label">
                <input
                  type="radio"
                  name="avatar_position"
                  checked={video.avatar_position === 'bottom-right'}
                  onChange={() => setVideo({ ...video, avatar_position: 'bottom-right' })}
                />
                <span>Bottom Right</span>
              </label>
            </div>
            <span className="slider-hint">Where the avatar appears on screen</span>
          </div>

          {/* Avatar Scale */}
          <div className="setting-item">
            <label>
              Avatar Size: <strong>{Math.round((video.avatar_scale || 0.8) * 100)}%</strong>
            </label>
            <input
              type="range"
              min="30"
              max="100"
              value={(video.avatar_scale || 0.8) * 100}
              onChange={(e) => setVideo({ ...video, avatar_scale: e.target.value / 100 })}
              className="slider"
            />
            <span className="slider-hint">Size of the avatar relative to screen width (80% recommended for TikTok)</span>
          </div>

          {/* Avatar Offsets */}
          <div className="setting-item">
            <label>
              Horizontal Offset: <strong>{video.avatar_offset_x ?? -200}px</strong>
            </label>
            <input
              type="range"
              min="-500"
              max="500"
              value={video.avatar_offset_x ?? -200}
              onChange={(e) => setVideo({ ...video, avatar_offset_x: parseInt(e.target.value) })}
              className="slider"
            />
            <span className="slider-hint">Horizontal shift (negative = left, positive = right). Default: -200</span>
          </div>

          <div className="setting-item">
            <label>
              Vertical Offset: <strong>{video.avatar_offset_y ?? 500}px</strong>
            </label>
            <input
              type="range"
              min="0"
              max="1000"
              value={video.avatar_offset_y ?? 500}
              onChange={(e) => setVideo({ ...video, avatar_offset_y: parseInt(e.target.value) })}
              className="slider"
            />
            <span className="slider-hint">Push avatar down to hide transparent space above (higher = lower on screen). Default: 500</span>
          </div>
        </div>

        <button
          onClick={saveVideoSettings}
          disabled={saving}
          className="save-button"
        >
          {saving ? 'Saving...' : 'Save Avatar Settings'}
        </button>
      </div>

      {/* CAPTION SETTINGS */}
      <div className="settings-section">
        <h2>Caption Settings</h2>
        <p className="section-description">Configure TikTok-style karaoke captions with word-by-word highlighting.</p>

        <div className="settings-grid">
          {/* Caption Style */}
          <div className="setting-item">
            <label>Caption Style</label>
            <div className="radio-group horizontal">
              <label className="radio-label">
                <input
                  type="radio"
                  name="caption_style"
                  checked={video.caption_style === 'karaoke'}
                  onChange={() => setVideo({ ...video, caption_style: 'karaoke' })}
                />
                <span>Karaoke (word highlight)</span>
              </label>
              <label className="radio-label">
                <input
                  type="radio"
                  name="caption_style"
                  checked={video.caption_style === 'static'}
                  onChange={() => setVideo({ ...video, caption_style: 'static' })}
                />
                <span>Static (standard subtitles)</span>
              </label>
              <label className="radio-label">
                <input
                  type="radio"
                  name="caption_style"
                  checked={video.caption_style === 'none'}
                  onChange={() => setVideo({ ...video, caption_style: 'none' })}
                />
                <span>None</span>
              </label>
            </div>
          </div>

          {video.caption_style !== 'none' && (
            <>
              {/* Font Size */}
              <div className="setting-item">
                <label>
                  Font Size: <strong>{video.caption_font_size || 96}pt</strong>
                </label>
                <input
                  type="range"
                  min="48"
                  max="144"
                  value={video.caption_font_size || 96}
                  onChange={(e) => setVideo({ ...video, caption_font_size: parseInt(e.target.value) })}
                  className="slider"
                />
                <span className="slider-hint">Text size for captions (96pt recommended)</span>
              </div>

              {/* Caption Colors */}
              <div className="setting-item">
                <label>Caption Colors</label>
                <div className="color-grid">
                  <div className="color-item">
                    <span>Text Color</span>
                    <div className="color-picker-row">
                      <input
                        type="color"
                        value={video.caption_color || '#FFFFFF'}
                        onChange={(e) => setVideo({ ...video, caption_color: e.target.value })}
                        className="color-picker"
                      />
                      <input
                        type="text"
                        value={video.caption_color || '#FFFFFF'}
                        onChange={(e) => setVideo({ ...video, caption_color: e.target.value })}
                        className="color-input"
                      />
                    </div>
                  </div>

                  {video.caption_style === 'karaoke' && (
                    <div className="color-item">
                      <span>Highlight Color</span>
                      <div className="color-picker-row">
                        <input
                          type="color"
                          value={video.caption_highlight_color || '#FFFF00'}
                          onChange={(e) => setVideo({ ...video, caption_highlight_color: e.target.value })}
                          className="color-picker"
                        />
                        <input
                          type="text"
                          value={video.caption_highlight_color || '#FFFF00'}
                          onChange={(e) => setVideo({ ...video, caption_highlight_color: e.target.value })}
                          className="color-input"
                        />
                      </div>
                    </div>
                  )}

                  <div className="color-item">
                    <span>Outline Color</span>
                    <div className="color-picker-row">
                      <input
                        type="color"
                        value={video.caption_outline_color || '#000000'}
                        onChange={(e) => setVideo({ ...video, caption_outline_color: e.target.value })}
                        className="color-picker"
                      />
                      <input
                        type="text"
                        value={video.caption_outline_color || '#000000'}
                        onChange={(e) => setVideo({ ...video, caption_outline_color: e.target.value })}
                        className="color-input"
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Outline Width */}
              <div className="setting-item">
                <label>
                  Outline Width: <strong>{video.caption_outline_width || 5}px</strong>
                </label>
                <input
                  type="range"
                  min="1"
                  max="10"
                  value={video.caption_outline_width || 5}
                  onChange={(e) => setVideo({ ...video, caption_outline_width: parseInt(e.target.value) })}
                  className="slider"
                />
              </div>

              {/* Caption Position */}
              <div className="setting-item">
                <label>
                  Vertical Position: <strong>{video.caption_position_y || 850}</strong>
                </label>
                <input
                  type="range"
                  min="400"
                  max="1200"
                  value={video.caption_position_y || 850}
                  onChange={(e) => setVideo({ ...video, caption_position_y: parseInt(e.target.value) })}
                  className="slider"
                />
                <span className="slider-hint">Distance from bottom (higher = closer to center)</span>
              </div>
            </>
          )}
        </div>

        <button
          onClick={saveVideoSettings}
          disabled={saving}
          className="save-button"
        >
          {saving ? 'Saving...' : 'Save Caption Settings'}
        </button>
      </div>

      {/* LLM PROMPT SETTINGS */}
      <div className="settings-section">
        <h2>LLM Prompt Settings</h2>
        <p className="section-description">Customize the prompts used for AI script generation and analysis.</p>

        <div className="llm-prompts">
          {llmPrompts.filter(p => p.key !== 'character_persona').map(prompt => (
            <div key={prompt.key} className="prompt-item">
              <div className="prompt-header">
                <h3>{prompt.key.replace(/_/g, ' ').toUpperCase()}</h3>
                <span className="prompt-description">{prompt.description}</span>
              </div>

              {editingPrompt === prompt.key ? (
                <div className="prompt-edit">
                  <textarea
                    value={editValue}
                    onChange={(e) => setEditValue(e.target.value)}
                    rows={8}
                  />
                  <div className="prompt-actions">
                    <button
                      onClick={() => saveLLMPrompt(prompt.key)}
                      disabled={saving}
                      className="save-button small"
                    >
                      Save
                    </button>
                    <button
                      onClick={() => setEditingPrompt(null)}
                      className="cancel-button"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              ) : (
                <div className="prompt-view">
                  <pre>{prompt.value}</pre>
                  <button
                    onClick={() => { setEditingPrompt(prompt.key); setEditValue(prompt.value); }}
                    className="edit-button"
                  >
                    Edit
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      <style>{`
        .page-container {
          padding: 20px;
          max-width: 900px;
        }

        .page-container h1 {
          color: var(--text-primary);
          margin-bottom: 24px;
        }

        .message {
          padding: 12px 16px;
          border-radius: 6px;
          margin-bottom: 20px;
          font-weight: 500;
        }

        .message.success {
          background: rgba(16, 185, 129, 0.2);
          color: var(--success);
          border: 1px solid var(--success);
        }

        .message.error {
          background: rgba(239, 68, 68, 0.2);
          color: var(--error);
          border: 1px solid var(--error);
        }

        .settings-section {
          background: var(--bg-secondary);
          border: 1px solid var(--border);
          border-radius: 12px;
          padding: 24px;
          margin-bottom: 24px;
        }

        .settings-section h2 {
          margin: 0 0 8px 0;
          font-size: 1.25rem;
          color: var(--text-primary);
        }

        .section-description {
          color: var(--text-secondary);
          margin: 0 0 20px 0;
          font-size: 0.9rem;
        }

        .settings-grid {
          display: flex;
          flex-direction: column;
          gap: 20px;
        }

        .setting-item {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .setting-item label {
          font-weight: 500;
          color: var(--text-primary);
        }

        .slider {
          width: 100%;
          max-width: 400px;
          height: 8px;
          -webkit-appearance: none;
          background: var(--bg-tertiary);
          border-radius: 4px;
          outline: none;
        }

        .slider::-webkit-slider-thumb {
          -webkit-appearance: none;
          width: 20px;
          height: 20px;
          background: var(--accent-primary);
          border-radius: 50%;
          cursor: pointer;
        }

        .slider-hint {
          font-size: 0.8rem;
          color: var(--text-secondary);
        }

        .checkbox-item {
          flex-direction: row;
          align-items: flex-start;
          flex-wrap: wrap;
        }

        .checkbox-label {
          display: flex;
          align-items: center;
          gap: 8px;
          cursor: pointer;
          color: var(--text-primary);
        }

        .checkbox-label input[type="checkbox"] {
          width: 18px;
          height: 18px;
          cursor: pointer;
        }

        .ducking-settings {
          background: var(--bg-tertiary);
          padding: 16px;
          border-radius: 6px;
          border-left: 3px solid var(--accent-primary);
          display: flex;
          flex-direction: column;
          gap: 16px;
        }

        .greenscreen-settings {
          background: var(--bg-tertiary);
          padding: 16px;
          border-radius: 6px;
          border-left: 3px solid var(--success);
          display: flex;
          flex-direction: column;
          gap: 16px;
        }

        .color-picker-row {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .color-picker {
          width: 50px;
          height: 36px;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          padding: 0;
        }

        .color-input {
          width: 100px;
          padding: 8px 12px;
          border: 1px solid var(--border);
          border-radius: 4px;
          font-size: 0.9rem;
          font-family: monospace;
          background: var(--bg-tertiary);
          color: var(--text-primary);
        }

        .number-input {
          width: 80px;
          padding: 8px 12px;
          border: 1px solid var(--border);
          border-radius: 4px;
          font-size: 1rem;
          background: var(--bg-tertiary);
          color: var(--text-primary);
        }

        .radio-group {
          display: flex;
          flex-direction: column;
          gap: 10px;
        }

        .radio-group.horizontal {
          flex-direction: row;
          flex-wrap: wrap;
          gap: 16px;
        }

        .color-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 16px;
        }

        .color-item {
          display: flex;
          flex-direction: column;
          gap: 6px;
        }

        .color-item span {
          font-size: 0.85rem;
          color: var(--text-secondary);
        }

        .radio-label {
          display: flex;
          align-items: center;
          gap: 8px;
          cursor: pointer;
          color: var(--text-primary);
        }

        .radio-label input[type="radio"] {
          width: 16px;
          height: 16px;
        }

        .quality-label {
          font-weight: normal;
          color: var(--accent-primary);
        }

        .select-input {
          padding: 8px 12px;
          border: 1px solid var(--border);
          border-radius: 4px;
          font-size: 1rem;
          max-width: 300px;
          background: var(--bg-tertiary);
          color: var(--text-primary);
        }

        .save-button {
          margin-top: 16px;
          padding: 10px 20px;
          background: var(--accent-primary);
          color: #fff;
          border: none;
          border-radius: 6px;
          font-size: 1rem;
          cursor: pointer;
          transition: background 0.2s;
        }

        .save-button:hover {
          background: var(--accent-secondary);
        }

        .save-button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .save-button.small {
          padding: 6px 16px;
          font-size: 0.9rem;
          margin-top: 0;
        }

        .cancel-button {
          padding: 6px 16px;
          background: var(--bg-tertiary);
          color: var(--text-primary);
          border: 1px solid var(--border);
          border-radius: 6px;
          font-size: 0.9rem;
          cursor: pointer;
        }

        .cancel-button:hover {
          background: var(--border);
        }

        .edit-button {
          padding: 6px 16px;
          background: var(--bg-tertiary);
          color: var(--text-primary);
          border: 1px solid var(--border);
          border-radius: 4px;
          cursor: pointer;
        }

        .edit-button:hover {
          background: var(--border);
        }

        .llm-prompts {
          display: flex;
          flex-direction: column;
          gap: 20px;
        }

        .prompt-item {
          border: 1px solid var(--border);
          border-radius: 8px;
          padding: 16px;
          background: var(--bg-tertiary);
        }

        .prompt-header h3 {
          margin: 0 0 4px 0;
          font-size: 0.95rem;
          color: var(--text-primary);
        }

        .prompt-description {
          font-size: 0.8rem;
          color: var(--text-secondary);
        }

        .prompt-view pre {
          background: var(--bg-primary);
          padding: 12px;
          border-radius: 4px;
          font-size: 0.85rem;
          overflow-x: auto;
          white-space: pre-wrap;
          word-wrap: break-word;
          max-height: 150px;
          overflow-y: auto;
          margin: 12px 0;
          color: var(--text-primary);
          border: 1px solid var(--border);
        }

        .prompt-edit textarea {
          width: 100%;
          padding: 12px;
          border: 1px solid var(--border);
          border-radius: 4px;
          font-family: monospace;
          font-size: 0.85rem;
          resize: vertical;
          margin: 12px 0;
          background: var(--bg-primary);
          color: var(--text-primary);
        }

        .prompt-edit textarea:focus {
          outline: none;
          border-color: var(--accent-primary);
        }

        .prompt-actions {
          display: flex;
          gap: 8px;
        }

        .loading {
          padding: 40px;
          text-align: center;
          color: var(--text-secondary);
        }

        /* Brand Persona Styles */
        .persona-section {
          border-left: 4px solid var(--accent-primary);
        }

        .persona-subsection {
          background: var(--bg-tertiary);
          padding: 16px;
          border-radius: 8px;
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .persona-subsection h3 {
          margin: 0;
          font-size: 1rem;
          color: var(--text-primary);
          border-bottom: 1px solid var(--border);
          padding-bottom: 8px;
        }

        .subsection-hint {
          font-size: 0.85rem;
          color: var(--text-secondary);
          margin: 0;
        }

        .boundaries-section {
          border-left: 3px solid #ef4444;
        }

        .text-input {
          padding: 10px 12px;
          border: 1px solid var(--border);
          border-radius: 6px;
          font-size: 0.95rem;
          background: var(--bg-primary);
          color: var(--text-primary);
          width: 100%;
          max-width: 500px;
        }

        .text-input:focus {
          outline: none;
          border-color: var(--accent-primary);
        }

        .text-input.flex-grow {
          flex: 1;
          max-width: none;
        }

        .textarea-input {
          padding: 12px;
          border: 1px solid var(--border);
          border-radius: 6px;
          font-size: 0.95rem;
          background: var(--bg-primary);
          color: var(--text-primary);
          width: 100%;
          resize: vertical;
          font-family: inherit;
        }

        .textarea-input:focus {
          outline: none;
          border-color: var(--accent-primary);
        }

        .tag-list {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          min-height: 32px;
        }

        .tag-list.vertical {
          flex-direction: column;
          gap: 6px;
        }

        .tag {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          background: var(--accent-primary);
          color: white;
          padding: 4px 10px;
          border-radius: 16px;
          font-size: 0.85rem;
        }

        .tag-remove {
          background: none;
          border: none;
          color: white;
          cursor: pointer;
          padding: 0;
          font-size: 1rem;
          line-height: 1;
          opacity: 0.8;
        }

        .tag-remove:hover {
          opacity: 1;
        }

        .value-item, .boundary-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          background: var(--bg-primary);
          padding: 8px 12px;
          border-radius: 6px;
          border: 1px solid var(--border);
        }

        .value-item span, .boundary-item span {
          flex: 1;
          color: var(--text-primary);
        }

        .boundary-item {
          border-left: 3px solid #ef4444;
        }

        .remove-button {
          background: none;
          border: 1px solid var(--border);
          color: var(--text-secondary);
          padding: 4px 10px;
          border-radius: 4px;
          font-size: 0.8rem;
          cursor: pointer;
        }

        .remove-button:hover {
          background: #ef4444;
          color: white;
          border-color: #ef4444;
        }

        .add-tag-row {
          display: flex;
          gap: 8px;
          margin-top: 8px;
        }

        .add-button {
          padding: 10px 16px;
          background: var(--accent-primary);
          color: white;
          border: none;
          border-radius: 6px;
          font-size: 0.9rem;
          cursor: pointer;
          white-space: nowrap;
        }

        .add-button:hover {
          background: var(--accent-secondary);
        }
        
        /* Music Library */
        .music-list {
          display: flex;
          flex-direction: column;
          gap: 12px;
          margin-top: 16px;
        }
        .music-item {
          display: flex;
          align-items: center;
          justify-content: space-between;
          background: var(--bg-tertiary);
          padding: 12px;
          border-radius: 8px;
          border: 1px solid var(--border);
        }
        .music-info {
          display: flex;
          flex-direction: column;
        }
        .music-name {
          font-weight: 500;
          color: var(--text-primary);
        }
        .file-size {
          font-size: 0.8rem;
          color: var(--text-secondary);
        }
        .music-player {
          height: 32px;
        }
        .no-music {
          color: var(--text-secondary);
          font-style: italic;
        }
      `}</style>
      {/* VIRAL FACTORY SETTINGS */}
      <div className="settings-section">
        <h2>Viral Clip Factory</h2>
        <p className="section-description">Configure the AI behavior for the Viral Clip Factory pipeline.</p>

        <div className="settings-grid">
          <div className="setting-item">
            <label>Viral Strategy (System Prompt)</label>
            <textarea
              value={llmPrompts.find(p => p.key === 'VIRAL_SYSTEM_PROMPT')?.value || ''}
              onChange={(e) => {
                const newVal = e.target.value;
                setLlmPrompts(prev => {
                  const idx = prev.findIndex(p => p.key === 'VIRAL_SYSTEM_PROMPT');
                  if (idx >= 0) {
                    const copy = [...prev];
                    copy[idx] = { ...copy[idx], value: newVal };
                    return copy;
                  } else {
                    return [...prev, { key: 'VIRAL_SYSTEM_PROMPT', value: newVal }];
                  }
                });
              }}
              className="textarea-input"
              rows={8}
              placeholder="Instructions for Grok on how to identify viral clips..."
            />
            <button
              onClick={() => saveLLMPrompt('VIRAL_SYSTEM_PROMPT', llmPrompts.find(p => p.key === 'VIRAL_SYSTEM_PROMPT')?.value)}
              className="save-button"
              style={{ marginTop: '10px' }}
            >
              Save Prompt
            </button>
          </div>

          <div className="setting-item">
            <label>Channel Handle (@username)</label>
            <input
              type="text"
              value={llmPrompts.find(p => p.key === 'VIRAL_CHANNEL_HANDLE')?.value || ''}
              onChange={(e) => {
                const newVal = e.target.value;
                setLlmPrompts(prev => {
                  const idx = prev.findIndex(p => p.key === 'VIRAL_CHANNEL_HANDLE');
                  if (idx >= 0) {
                    const copy = [...prev];
                    copy[idx] = { ...copy[idx], value: newVal };
                    return copy;
                  } else {
                    return [...prev, { key: 'VIRAL_CHANNEL_HANDLE', value: newVal }];
                  }
                });
              }}
              className="text-input"
              placeholder="e.g. realDonaldTrump"
            />
            <button
              onClick={() => saveLLMPrompt('VIRAL_CHANNEL_HANDLE', llmPrompts.find(p => p.key === 'VIRAL_CHANNEL_HANDLE')?.value)}
              className="save-button"
              style={{ marginTop: '10px' }}
            >
              Save Handle
            </button>
          </div>

          {/* Caption Font Selection */}
          <div className="setting-item">
            <label>Caption Font</label>
            <div className="font-selector">
              <div className="checkbox-item" style={{ marginBottom: '12px' }}>
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={llmPrompts.find(p => p.key === 'VIRAL_FONT_RANDOM')?.value !== 'false'}
                    onChange={(e) => {
                      const newVal = e.target.checked ? 'true' : 'false';
                      setLlmPrompts(prev => {
                        const idx = prev.findIndex(p => p.key === 'VIRAL_FONT_RANDOM');
                        if (idx >= 0) {
                          const copy = [...prev];
                          copy[idx] = { ...copy[idx], value: newVal };
                          return copy;
                        } else {
                          return [...prev, { key: 'VIRAL_FONT_RANDOM', value: newVal }];
                        }
                      });
                      saveLLMPrompt('VIRAL_FONT_RANDOM', newVal);
                    }}
                  />
                  <span>Random Font (recommended for variety)</span>
                </label>
              </div>

              {llmPrompts.find(p => p.key === 'VIRAL_FONT_RANDOM')?.value === 'false' && (
                <>
                  <select
                    value={llmPrompts.find(p => p.key === 'VIRAL_CAPTION_FONT')?.value || 'Honk'}
                    onChange={(e) => {
                      const newVal = e.target.value;
                      setLlmPrompts(prev => {
                        const idx = prev.findIndex(p => p.key === 'VIRAL_CAPTION_FONT');
                        if (idx >= 0) {
                          const copy = [...prev];
                          copy[idx] = { ...copy[idx], value: newVal };
                          return copy;
                        } else {
                          return [...prev, { key: 'VIRAL_CAPTION_FONT', value: newVal }];
                        }
                      });
                    }}
                    className="select-input"
                    style={{ marginBottom: '12px' }}
                  >
                    <option value="Honk">Honk - Playful & Expressive</option>
                    <option value="Pirata One">Pirata One - Vintage Gothic</option>
                    <option value="Rubik Vinyl">Rubik Vinyl - Retro Groovy</option>
                    <option value="Rubik 80s Fade">Rubik 80s Fade - Neon Retro</option>
                    <option value="Rubik Dirt">Rubik Dirt - Grungy Distressed</option>
                  </select>
                  <button
                    onClick={() => saveLLMPrompt('VIRAL_CAPTION_FONT', llmPrompts.find(p => p.key === 'VIRAL_CAPTION_FONT')?.value)}
                    className="save-button"
                  >
                    Save Font
                  </button>
                </>
              )}
            </div>
            <span className="slider-hint">TikTok-style fonts for viral clip captions</span>
          </div>
        </div>
      </div>

      {/* VIRAL FONT LIBRARY */}
      <div className="settings-section">
        <h2>Viral Font Library</h2>
        <p className="section-description">TikTok-style fonts for viral clip captions. Random font is selected for each clip.</p>

        {/* Google Fonts Download */}
        <div className="setting-item" style={{ marginBottom: '20px' }}>
          <label>Add from Google Fonts</label>
          <div style={{ display: 'flex', gap: '10px' }}>
            <input
              type="text"
              placeholder="Font name (e.g., Bebas Neue, Oswald, Bangers)"
              value={newGoogleFont}
              onChange={(e) => setNewGoogleFont(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleDownloadGoogleFont()}
              style={{ flex: 1 }}
            />
            <button
              onClick={handleDownloadGoogleFont}
              disabled={downloadingFont || !newGoogleFont.trim()}
              className="save-button"
            >
              {downloadingFont ? 'Downloading...' : 'Download'}
            </button>
          </div>
          <span className="slider-hint">Enter exact font name from fonts.google.com</span>
        </div>

        {/* Font face declarations for previews */}
        <style>
          {fonts.map((font, i) => `
            @font-face {
              font-family: 'Preview-${font.name.replace(/\s+/g, '')}';
              src: url('${API_URL}/api/viral/fonts/${font.filename}/file') format('${font.filename.endsWith('.ttf') ? 'truetype' : 'opentype'}');
              font-display: swap;
            }
          `).join('\n')}
        </style>

        {/* Installed Fonts List */}
        <div className="music-list">
          {fonts.length === 0 ? (
            <p className="no-music">No custom fonts found.</p>
          ) : (
            fonts.map((font, i) => (
              <div key={i} className="music-item" style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                <div className="music-info" style={{ minWidth: '150px' }}>
                  <span className="music-name">{font.name}</span>
                  <span className="file-size">{font.filename}</span>
                </div>
                <div
                  className="font-preview"
                  style={{
                    fontFamily: `'Preview-${font.name.replace(/\s+/g, '')}', sans-serif`,
                    fontSize: '32px',
                    color: 'var(--text-primary)',
                    flex: 1,
                    textAlign: 'center',
                    letterSpacing: '1px'
                  }}
                >
                  America First
                </div>
                <button
                  onClick={() => handleDeleteFont(font.filename)}
                  className="delete-button"
                  style={{ background: '#dc3545', color: 'white', border: 'none', padding: '6px 12px', borderRadius: '4px', cursor: 'pointer' }}
                >
                  Delete
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      {/* VIRAL MUSIC LIBRARY */}
      <div className="settings-section">
        <h2>Viral Music Library</h2>
        <p className="section-description">Background music for viral clips (TradWest style). Files in /assets/music.</p>

        <div className="music-list">
          {musicFiles.length === 0 ? (
            <p className="no-music">No music files found. Add mp3/wav to /assets/music.</p>
          ) : (
            musicFiles.map((file, i) => (
              <div key={i} className="music-item">
                <div className="music-info">
                  <span className="music-name">{file.name}</span>
                  <span className="file-size">{(file.size / 1024 / 1024).toFixed(2)} MB</span>
                </div>
                <audio controls src={`/api/viral/music/${file.name}`} className="music-player" />
              </div>
            ))
          )}
        </div>
      </div>
    </div>

  );
}
