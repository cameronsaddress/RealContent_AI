import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { getAssets } from '../api';
import { format } from 'date-fns';

function Assets() {
  const [statusFilter, setStatusFilter] = useState('');
  const [viewingAsset, setViewingAsset] = useState(null);

  const { data: assets, isLoading } = useQuery({
    queryKey: ['assets', statusFilter],
    queryFn: () => getAssets({
      status: statusFilter || undefined,
    }),
  });

  return (
    <div className="assets">
      <div className="page-header">
        <h1 className="page-title">Assets</h1>
      </div>

      <div className="filters">
        <select
          className="filter-select"
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
        >
          <option value="">All Status</option>
          <option value="pending">Pending</option>
          <option value="voice_generating">Voice Generating</option>
          <option value="voice_ready">Voice Ready</option>
          <option value="avatar_generating">Avatar Generating</option>
          <option value="avatar_ready">Avatar Ready</option>
          <option value="assembling">Assembling</option>
          <option value="captioning">Captioning</option>
          <option value="ready_to_publish">Ready to Publish</option>
          <option value="error">Error</option>
        </select>
      </div>

      {isLoading ? (
        <div className="loading">Loading assets...</div>
      ) : assets?.length === 0 ? (
        <div className="empty-state">
          <h3>No assets found</h3>
          <p>Assets will be created during video generation</p>
        </div>
      ) : (
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Script ID</th>
                <th>Voiceover</th>
                <th>Avatar Video</th>
                <th>Final Video</th>
                <th>Status</th>
                <th>Created</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {assets?.map((asset) => (
                <tr key={asset.id}>
                  <td>#{asset.id}</td>
                  <td>{asset.script_id ? `#${asset.script_id}` : <span className="status-badge status-ready_to_publish" style={{ background: 'var(--accent-primary)', color: 'white' }}>Global Identity</span>}</td>
                  <td>
                    {asset.voiceover_path ? (
                      <span style={{ color: 'var(--success)' }}>Ready</span>
                    ) : (
                      <span style={{ color: 'var(--text-secondary)' }}>-</span>
                    )}
                    {asset.voiceover_duration && ` (${asset.voiceover_duration.toFixed(1)}s)`}
                  </td>
                  <td>
                    {asset.avatar_video_path ? (
                      <span style={{ color: 'var(--success)' }}>Ready</span>
                    ) : (
                      <span style={{ color: 'var(--text-secondary)' }}>-</span>
                    )}
                  </td>
                  <td>
                    {asset.final_video_path ? (
                      <span style={{ color: 'var(--success)' }}>Ready</span>
                    ) : (
                      <span style={{ color: 'var(--text-secondary)' }}>-</span>
                    )}
                  </td>
                  <td>
                    <span className={`status-badge status-${asset.status}`}>
                      {asset.status.replace('_', ' ')}
                    </span>
                  </td>
                  <td>{format(new Date(asset.created_at), 'MMM d, HH:mm')}</td>
                  <td>
                    <div className="actions">
                      <button
                        className="action-btn view"
                        onClick={() => setViewingAsset(asset)}
                      >
                        View
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {viewingAsset && (
        <div className="modal-overlay" onClick={() => setViewingAsset(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2 className="modal-title">Asset #{viewingAsset.id}</h2>
              <button className="modal-close" onClick={() => setViewingAsset(null)}>
                &times;
              </button>
            </div>
            <div className="modal-body">
              <div className="asset-status-grid">
                <div className="asset-status-item">
                  <h4>Status</h4>
                  <span className={`status-badge status-${viewingAsset.status}`}>
                    {viewingAsset.status.replace('_', ' ')}
                  </span>
                </div>
                <div className="asset-status-item">
                  <h4>Script ID</h4>
                  <p>{viewingAsset.script_id ? `#${viewingAsset.script_id}` : 'Global Identity'}</p>
                </div>
              </div>

              <h3 style={{ marginTop: '24px', marginBottom: '16px' }}>File Paths</h3>

              <div className="form-group">
                <label className="form-label">Voiceover</label>
                <p className="file-path">{viewingAsset.voiceover_path || 'Not generated'}</p>
                {viewingAsset.voiceover_duration && (
                  <p className="file-meta">Duration: {viewingAsset.voiceover_duration.toFixed(2)}s</p>
                )}
              </div>

              <div className="form-group">
                <label className="form-label">SRT Captions</label>
                <p className="file-path">{viewingAsset.srt_path || 'Not generated'}</p>
              </div>

              <div className="form-group">
                <label className="form-label">ASS Captions</label>
                <p className="file-path">{viewingAsset.ass_path || 'Not generated'}</p>
              </div>

              <div className="form-group">
                <label className="form-label">Avatar Video</label>
                <p className="file-path">{viewingAsset.avatar_video_path || 'Not generated'}</p>
              </div>

              <div className="form-group">
                <label className="form-label">Background Video</label>
                <p className="file-path">{viewingAsset.background_video_path || 'Not assigned'}</p>
              </div>

              <div className="form-group">
                <label className="form-label">Combined Video</label>
                <p className="file-path">{viewingAsset.combined_video_path || 'Not generated'}</p>
              </div>

              <div className="form-group">
                <label className="form-label">Final Video</label>
                <p className="file-path">{viewingAsset.final_video_path || 'Not generated'}</p>
              </div>

              {viewingAsset.error_message && (
                <div className="form-group">
                  <label className="form-label" style={{ color: 'var(--error)' }}>Error</label>
                  <p style={{ color: 'var(--error)' }}>{viewingAsset.error_message}</p>
                </div>
              )}
            </div>
            <div className="modal-footer">
              <button className="btn btn-secondary" onClick={() => setViewingAsset(null)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      <style>{`
        .asset-status-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 16px;
        }
        .asset-status-item h4 {
          color: var(--text-secondary);
          font-size: 13px;
          margin-bottom: 8px;
        }
        .file-path {
          font-family: monospace;
          background: var(--bg-tertiary);
          padding: 8px 12px;
          border-radius: 6px;
          font-size: 13px;
          word-break: break-all;
        }
        .file-meta {
          font-size: 12px;
          color: var(--text-secondary);
          margin-top: 4px;
        }
      `}</style>
    </div>
  );
}

export default Assets;
