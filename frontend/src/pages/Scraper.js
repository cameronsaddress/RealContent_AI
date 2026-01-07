import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { runScrape, getScrapeRuns, getNichePresets, createNichePreset, deleteNichePreset } from '../api';
import { format } from 'date-fns';

function Scraper() {
    const queryClient = useQueryClient();
    const [niche, setNiche] = useState('');
    const [platforms, setPlatforms] = useState({
        tiktok: true,
        instagram: true,
        youtube: false
    });
    const [newPresetName, setNewPresetName] = useState('');

    // Queries
    const { data: runs, isLoading: runsLoading } = useQuery({
        queryKey: ['scrapeRuns'],
        queryFn: () => getScrapeRuns({ limit: 20 }),
        refetchInterval: 5000 // Poll for updates
    });

    const { data: presets } = useQuery({
        queryKey: ['nichePresets'],
        queryFn: getNichePresets
    });

    // Mutations
    const scrapeMutation = useMutation({
        mutationFn: runScrape,
        onSuccess: () => {
            queryClient.invalidateQueries(['scrapeRuns']);
        }
    });

    const createPresetMutation = useMutation({
        mutationFn: createNichePreset,
        onSuccess: () => {
            queryClient.invalidateQueries(['nichePresets']);
            setNewPresetName('');
        }
    });

    const deletePresetMutation = useMutation({
        mutationFn: deleteNichePreset,
        onSuccess: () => {
            queryClient.invalidateQueries(['nichePresets']);
        }
    });

    const handleStartScrape = (e) => {
        e.preventDefault();
        if (!niche) return;

        const activePlatforms = Object.entries(platforms)
            .filter(([_, active]) => active)
            .map(([p]) => p);

        scrapeMutation.mutate({
            niche,
            platforms: activePlatforms
        });
    };

    const handlePlatformChange = (platform) => {
        setPlatforms(prev => ({
            ...prev,
            [platform]: !prev[platform]
        }));
    };

    const handleSavePreset = () => {
        if (!newPresetName || !niche) return;
        createPresetMutation.mutate({
            name: newPresetName,
            keywords: niche.split(' '),
            hashtags: []
        });
    };

    const loadPreset = (preset) => {
        setNiche(preset.keywords.join(' '));
    };

    return (
        <div className="scraper-page">
            <div className="page-header">
                <h1 className="page-title">Content Scraper</h1>
            </div>

            <div className="scraper-container">
                {/* Left Column: Controls */}
                <div className="scraper-controls">
                    <div className="card">
                        <h2>New Scrape</h2>
                        <form onSubmit={handleStartScrape}>
                            <div className="form-group">
                                <label className="form-label">Niche / Keywords</label>
                                <input
                                    type="text"
                                    className="form-input"
                                    placeholder="e.g. realtor marketing ideas"
                                    value={niche}
                                    onChange={(e) => setNiche(e.target.value)}
                                />
                            </div>

                            <div className="form-group">
                                <label className="form-label">Platforms</label>
                                <div className="checkbox-group">
                                    <label>
                                        <input
                                            type="checkbox"
                                            checked={platforms.tiktok}
                                            onChange={() => handlePlatformChange('tiktok')}
                                        />
                                        TikTok
                                    </label>
                                    <label>
                                        <input
                                            type="checkbox"
                                            checked={platforms.instagram}
                                            onChange={() => handlePlatformChange('instagram')}
                                        />
                                        Instagram
                                    </label>
                                    <label>
                                        <input
                                            type="checkbox"
                                            checked={platforms.youtube}
                                            onChange={() => handlePlatformChange('youtube')}
                                        />
                                        YouTube
                                    </label>
                                </div>
                            </div>

                            <button
                                type="submit"
                                className="btn btn-primary btn-block"
                                disabled={scrapeMutation.isPending || !niche}
                            >
                                {scrapeMutation.isPending ? 'Starting...' : 'Start Scrape'}
                            </button>

                            {scrapeMutation.isError && (
                                <div className="error-message">
                                    {scrapeMutation.error?.response?.data?.detail || 'Failed to start scrape'}
                                </div>
                            )}
                        </form>
                    </div>

                    <div className="card">
                        <div className="card-header-row">
                            <h2>Presets</h2>
                            <div className="preset-save-row">
                                <input
                                    type="text"
                                    placeholder="Save current niche as..."
                                    value={newPresetName}
                                    onChange={(e) => setNewPresetName(e.target.value)}
                                    className="form-input small"
                                />
                                <button
                                    className="btn btn-secondary small"
                                    onClick={handleSavePreset}
                                    disabled={!newPresetName || !niche}
                                >
                                    Save
                                </button>
                            </div>
                        </div>

                        <ul className="preset-list">
                            {presets?.map(preset => (
                                <li key={preset.id} className="preset-item">
                                    <span onClick={() => loadPreset(preset)}>{preset.name}</span>
                                    <button
                                        className="btn-icon"
                                        onClick={() => deletePresetMutation.mutate(preset.id)}
                                    >
                                        &times;
                                    </button>
                                </li>
                            ))}
                            {presets?.length === 0 && <li className="empty-text">No presets saved</li>}
                        </ul>
                    </div>
                </div>

                {/* Right Column: History */}
                <div className="scraper-history">
                    <div className="card">
                        <h2>Recent Runs</h2>
                        {runsLoading ? (
                            <div>Loading history...</div>
                        ) : (
                            <table className="history-table">
                                <thead>
                                    <tr>
                                        <th>ID</th>
                                        <th>Niche</th>
                                        <th>Platforms</th>
                                        <th>Status</th>
                                        <th>Results</th>
                                        <th>Started</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {runs?.map(run => (
                                        <tr key={run.id}>
                                            <td>#{run.id}</td>
                                            <td>{run.niche}</td>
                                            <td>
                                                <div className="platform-icons">
                                                    {run.platforms?.map(p => (
                                                        <span key={p} className={`platform-dot ${p}`} title={p} />
                                                    ))}
                                                </div>
                                            </td>
                                            <td>
                                                <span className={`status-badge status-${run.status}`}>
                                                    {run.status}
                                                </span>
                                            </td>
                                            <td>{run.results_count > 0 ? `${run.results_count} found` : '-'}</td>
                                            <td>{format(new Date(run.started_at), 'MMM d, HH:mm')}</td>
                                        </tr>
                                    ))}
                                    {runs?.length === 0 && (
                                        <tr>
                                            <td colSpan="6" className="text-center">No scrape runs yet</td>
                                        </tr>
                                    )}
                                </tbody>
                            </table>
                        )}
                    </div>
                </div>
            </div>

            <style>{`
        .scraper-container {
          display: grid;
          grid-template-columns: 350px 1fr;
          gap: 20px;
        }
        .form-input {
          width: 100%;
          padding: 8px 12px;
          border: 1px solid #ddd;
          border-radius: 4px;
        }
        .form-input.small {
          padding: 4px 8px;
          font-size: 13px;
        }
        .checkbox-group {
          display: flex;
          gap: 15px;
          margin-top: 8px;
        }
        .checkbox-group label {
          display: flex;
          align-items: center;
          gap: 5px;
          cursor: pointer;
        }
        .btn-block {
          width: 100%;
          margin-top: 15px;
        }
        .card {
          background: var(--bg-secondary);
          padding: 24px;
          border-radius: 12px;
          border: 1px solid var(--border);
          box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
          margin-bottom: 24px;
        }
        .card h2 {
          font-size: 1.1rem;
          margin-bottom: 20px;
          color: var(--text-primary);
          font-weight: 600;
        }
        .form-label {
          color: var(--text-secondary);
          font-size: 0.9rem;
          margin-bottom: 8px;
          display: block;
        }
        .form-input {
          width: 100%;
          padding: 10px 14px;
          background: var(--bg-tertiary);
          border: 1px solid var(--border);
          border-radius: 8px;
          color: var(--text-primary);
          font-size: 14px;
          transition: all 0.2s;
        }
        .form-input:focus {
          outline: none;
          border-color: var(--accent-primary);
          box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
        }
        .checkbox-group {
          display: flex;
          gap: 20px;
          margin-top: 10px;
        }
        .checkbox-group label {
          display: flex;
          align-items: center;
          gap: 8px;
          cursor: pointer;
          color: var(--text-primary);
          padding: 6px 10px;
          background: var(--bg-tertiary);
          border-radius: 6px;
          border: 1px solid var(--border);
          transition: all 0.2s;
        }
        .checkbox-group label:hover {
          border-color: var(--accent-primary);
        }
        .preset-save-row {
          display: flex;
          gap: 10px;
          margin-bottom: 15px;
        }
        .preset-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 12px 16px;
          background: var(--bg-tertiary);
          border: 1px solid var(--border);
          border-radius: 8px;
          margin-bottom: 8px;
          cursor: pointer;
          transition: all 0.2s;
          color: var(--text-primary);
        }
        .preset-item:hover {
          border-color: var(--accent-primary);
          transform: translateY(-1px);
        }
        .btn-icon {
          background: none;
          border: none;
          color: var(--text-secondary);
          cursor: pointer;
          padding: 4px;
          border-radius: 4px;
          display: flex;
          align-items: center;
          justify-content: center;
          width: 24px;
          height: 24px;
        }
        .btn-icon:hover {
          color: var(--error);
          background: rgba(239, 68, 68, 0.1);
        }
        .history-table {
          width: 100%;
          border-collapse: separate;
          border-spacing: 0;
        }
        .history-table th {
          background: var(--bg-tertiary);
          padding: 12px 16px;
          font-weight: 600;
          text-transform: uppercase;
          font-size: 12px;
          letter-spacing: 0.05em;
          color: var(--text-secondary);
          text-align: left;
        }
        .history-table th:first-child { border-top-left-radius: 8px; border-bottom-left-radius: 8px; }
        .history-table th:last-child { border-top-right-radius: 8px; border-bottom-right-radius: 8px; }
        .history-table td {
          padding: 14px 16px;
          border-bottom: 1px solid var(--border);
          color: var(--text-primary);
        }
        .platform-dot {
          display: inline-block;
          width: 8px;
          height: 8px;
          border-radius: 50%;
          margin-right: 4px;
        }
        .platform-dot.tiktok { background-color: #000; }
        .platform-dot.instagram { background-color: #E1306C; }
        .platform-dot.youtube { background-color: #FF0000; }
        .error-message {
          color: red;
          margin-top: 10px;
          font-size: 14px;
        }
      `}</style>
        </div>
    );
}

export default Scraper;
