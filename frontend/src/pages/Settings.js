import React, { useState, useEffect } from 'react';
import {
  getAudioSettings, updateAudioSettings,
  getVideoSettings, updateVideoSettings,
  getLLMSettings, updateLLMSetting
} from '../api';

export default function Settings() {
  // Audio settings state
  const [audio, setAudio] = useState({
    original_volume: 0.7,
    avatar_volume: 1.0,
    ducking_enabled: true,
    avatar_delay_seconds: 3.0,
    duck_to_percent: 0.5
  });

  // Video settings state
  const [video, setVideo] = useState({
    output_width: 1080,
    output_height: 1920,
    crf: 18,
    preset: 'slow',
    greenscreen_enabled: true,
    greenscreen_color: '#00FF00'
  });

  // LLM settings state
  const [llmPrompts, setLlmPrompts] = useState([]);
  const [editingPrompt, setEditingPrompt] = useState(null);
  const [editValue, setEditValue] = useState('');

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
      const [audioData, videoData, llmData] = await Promise.all([
        getAudioSettings(),
        getVideoSettings(),
        getLLMSettings()
      ]);
      setAudio(audioData);
      setVideo(videoData);
      setLlmPrompts(llmData);
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

  const saveLLMPrompt = async (key) => {
    setSaving(true);
    try {
      await updateLLMSetting(key, { value: editValue });
      setEditingPrompt(null);
      loadAllSettings();
      showMessage('Prompt saved!', 'success');
    } catch (err) {
      showMessage('Error saving prompt', 'error');
    }
    setSaving(false);
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
              onChange={(e) => setAudio({...audio, original_volume: e.target.value / 100})}
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
              onChange={(e) => setAudio({...audio, avatar_volume: e.target.value / 100})}
              className="slider"
            />
            <span className="slider-hint">Volume of the AI avatar's voiceover</span>
          </div>

          {/* Ducking Toggle */}
          <div className="setting-item checkbox-item">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={audio.ducking_enabled}
                onChange={(e) => setAudio({...audio, ducking_enabled: e.target.checked})}
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
                  onChange={(e) => setAudio({...audio, avatar_delay_seconds: parseFloat(e.target.value) || 0})}
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
                  onChange={(e) => setAudio({...audio, duck_to_percent: e.target.value / 100})}
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
                  onChange={() => setVideo({...video, output_width: 720, output_height: 1280})}
                />
                <span>720p (720x1280)</span>
              </label>
              <label className="radio-label">
                <input
                  type="radio"
                  name="resolution"
                  checked={video.output_height === 1920}
                  onChange={() => setVideo({...video, output_width: 1080, output_height: 1920})}
                />
                <span>1080p (1080x1920) - Recommended</span>
              </label>
              <label className="radio-label">
                <input
                  type="radio"
                  name="resolution"
                  checked={video.output_height === 3840}
                  onChange={() => setVideo({...video, output_width: 2160, output_height: 3840})}
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
              onChange={(e) => setVideo({...video, crf: parseInt(e.target.value)})}
              className="slider"
            />
            <span className="slider-hint">Lower = better quality, larger file size. 18 is recommended.</span>
          </div>

          {/* Preset */}
          <div className="setting-item">
            <label>Encoding Preset</label>
            <select
              value={video.preset}
              onChange={(e) => setVideo({...video, preset: e.target.value})}
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
                onChange={(e) => setVideo({...video, greenscreen_enabled: e.target.checked})}
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
                    onChange={(e) => setVideo({...video, greenscreen_color: e.target.value})}
                    className="color-picker"
                  />
                  <input
                    type="text"
                    value={video.greenscreen_color}
                    onChange={(e) => setVideo({...video, greenscreen_color: e.target.value})}
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

      {/* LLM PROMPT SETTINGS */}
      <div className="settings-section">
        <h2>LLM Prompt Settings</h2>
        <p className="section-description">Customize the prompts used for AI script generation and analysis.</p>

        <div className="llm-prompts">
          {llmPrompts.map(prompt => (
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
      `}</style>
    </div>
  );
}
