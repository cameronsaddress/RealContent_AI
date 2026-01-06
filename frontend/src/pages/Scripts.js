import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getScripts, updateScript } from '../api';
import { format } from 'date-fns';

function Scripts() {
  const queryClient = useQueryClient();
  const [statusFilter, setStatusFilter] = useState('');
  const [viewingScript, setViewingScript] = useState(null);
  const [editingScript, setEditingScript] = useState(null);

  const { data: scripts, isLoading } = useQuery({
    queryKey: ['scripts', statusFilter],
    queryFn: () => getScripts({
      status: statusFilter || undefined,
    }),
  });

  const updateMutation = useMutation({
    mutationFn: ({ id, data }) => updateScript(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries(['scripts']);
      setEditingScript(null);
    },
  });

  const handleSave = () => {
    if (editingScript) {
      updateMutation.mutate({
        id: editingScript.id,
        data: {
          hook: editingScript.hook,
          body: editingScript.body,
          cta: editingScript.cta,
          tiktok_caption: editingScript.tiktok_caption,
          ig_caption: editingScript.ig_caption,
          yt_title: editingScript.yt_title,
          yt_description: editingScript.yt_description,
          linkedin_text: editingScript.linkedin_text,
          x_text: editingScript.x_text,
        },
      });
    }
  };

  return (
    <div className="scripts">
      <div className="page-header">
        <h1 className="page-title">Scripts</h1>
      </div>

      <div className="filters">
        <select
          className="filter-select"
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
        >
          <option value="">All Status</option>
          <option value="pending">Pending</option>
          <option value="script_ready">Script Ready</option>
          <option value="voice_generating">Voice Generating</option>
          <option value="voice_ready">Voice Ready</option>
          <option value="error">Error</option>
        </select>
      </div>

      {isLoading ? (
        <div className="loading">Loading scripts...</div>
      ) : scripts?.length === 0 ? (
        <div className="empty-state">
          <h3>No scripts found</h3>
          <p>Scripts will be generated when ideas are approved</p>
        </div>
      ) : (
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Idea ID</th>
                <th>Hook Preview</th>
                <th>Duration</th>
                <th>Status</th>
                <th>Created</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {scripts?.map((script) => (
                <tr key={script.id}>
                  <td>#{script.id}</td>
                  <td>#{script.content_idea_id}</td>
                  <td className="hook-cell" title={script.hook}>
                    {script.hook?.substring(0, 60)}
                    {script.hook?.length > 60 && '...'}
                  </td>
                  <td>{script.duration_estimate ? `${script.duration_estimate}s` : '-'}</td>
                  <td>
                    <span className={`status-badge status-${script.status}`}>
                      {script.status.replace('_', ' ')}
                    </span>
                  </td>
                  <td>{format(new Date(script.created_at), 'MMM d, HH:mm')}</td>
                  <td>
                    <div className="actions">
                      <button
                        className="action-btn view"
                        onClick={() => setViewingScript(script)}
                      >
                        View
                      </button>
                      <button
                        className="action-btn view"
                        onClick={() => setEditingScript({ ...script })}
                      >
                        Edit
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {viewingScript && (
        <div className="modal-overlay" onClick={() => setViewingScript(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2 className="modal-title">Script #{viewingScript.id}</h2>
              <button className="modal-close" onClick={() => setViewingScript(null)}>
                &times;
              </button>
            </div>
            <div className="modal-body">
              <div className="form-group">
                <label className="form-label">Hook (First 3 seconds)</label>
                <p>{viewingScript.hook}</p>
              </div>
              <div className="form-group">
                <label className="form-label">Body</label>
                <p style={{ whiteSpace: 'pre-wrap' }}>{viewingScript.body}</p>
              </div>
              <div className="form-group">
                <label className="form-label">CTA (Call to Action)</label>
                <p>{viewingScript.cta}</p>
              </div>
              <div className="form-group">
                <label className="form-label">Duration Estimate</label>
                <p>{viewingScript.duration_estimate} seconds</p>
              </div>
              <hr style={{ margin: '20px 0', borderColor: 'var(--border)' }} />
              <h3 style={{ marginBottom: '16px' }}>Platform Captions</h3>
              <div className="form-group">
                <label className="form-label">TikTok</label>
                <p>{viewingScript.tiktok_caption || '-'}</p>
              </div>
              <div className="form-group">
                <label className="form-label">Instagram</label>
                <p>{viewingScript.ig_caption || '-'}</p>
              </div>
              <div className="form-group">
                <label className="form-label">YouTube Title</label>
                <p>{viewingScript.yt_title || '-'}</p>
              </div>
              <div className="form-group">
                <label className="form-label">LinkedIn</label>
                <p>{viewingScript.linkedin_text || '-'}</p>
              </div>
              <div className="form-group">
                <label className="form-label">X (Twitter)</label>
                <p>{viewingScript.x_text || '-'}</p>
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn btn-secondary" onClick={() => setViewingScript(null)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {editingScript && (
        <div className="modal-overlay" onClick={() => setEditingScript(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2 className="modal-title">Edit Script #{editingScript.id}</h2>
              <button className="modal-close" onClick={() => setEditingScript(null)}>
                &times;
              </button>
            </div>
            <div className="modal-body">
              <div className="form-group">
                <label className="form-label">Hook</label>
                <textarea
                  className="form-textarea"
                  value={editingScript.hook || ''}
                  onChange={(e) => setEditingScript({ ...editingScript, hook: e.target.value })}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Body</label>
                <textarea
                  className="form-textarea"
                  style={{ minHeight: '200px' }}
                  value={editingScript.body || ''}
                  onChange={(e) => setEditingScript({ ...editingScript, body: e.target.value })}
                />
              </div>
              <div className="form-group">
                <label className="form-label">CTA</label>
                <textarea
                  className="form-textarea"
                  value={editingScript.cta || ''}
                  onChange={(e) => setEditingScript({ ...editingScript, cta: e.target.value })}
                />
              </div>
              <hr style={{ margin: '20px 0', borderColor: 'var(--border)' }} />
              <div className="form-group">
                <label className="form-label">TikTok Caption</label>
                <textarea
                  className="form-textarea"
                  value={editingScript.tiktok_caption || ''}
                  onChange={(e) => setEditingScript({ ...editingScript, tiktok_caption: e.target.value })}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Instagram Caption</label>
                <textarea
                  className="form-textarea"
                  value={editingScript.ig_caption || ''}
                  onChange={(e) => setEditingScript({ ...editingScript, ig_caption: e.target.value })}
                />
              </div>
              <div className="form-group">
                <label className="form-label">YouTube Title</label>
                <input
                  className="form-input"
                  value={editingScript.yt_title || ''}
                  onChange={(e) => setEditingScript({ ...editingScript, yt_title: e.target.value })}
                />
              </div>
              <div className="form-group">
                <label className="form-label">LinkedIn Text</label>
                <textarea
                  className="form-textarea"
                  value={editingScript.linkedin_text || ''}
                  onChange={(e) => setEditingScript({ ...editingScript, linkedin_text: e.target.value })}
                />
              </div>
              <div className="form-group">
                <label className="form-label">X (Twitter) Text</label>
                <textarea
                  className="form-textarea"
                  value={editingScript.x_text || ''}
                  onChange={(e) => setEditingScript({ ...editingScript, x_text: e.target.value })}
                />
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn btn-secondary" onClick={() => setEditingScript(null)}>
                Cancel
              </button>
              <button className="btn btn-primary" onClick={handleSave}>
                Save Changes
              </button>
            </div>
          </div>
        </div>
      )}

      <style>{`
        .hook-cell {
          max-width: 300px;
        }
      `}</style>
    </div>
  );
}

export default Scripts;
