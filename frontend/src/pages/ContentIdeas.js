import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getContentIdeas, updateContentIdea, bulkApproveIdeas, bulkRejectIdeas } from '../api';
import { format } from 'date-fns';

function ContentIdeas() {
  const queryClient = useQueryClient();
  const [statusFilter, setStatusFilter] = useState('');
  const [pillarFilter, setPillarFilter] = useState('');
  const [selectedIds, setSelectedIds] = useState([]);
  const [viewingIdea, setViewingIdea] = useState(null);

  const { data: ideas, isLoading } = useQuery({
    queryKey: ['contentIdeas', statusFilter, pillarFilter],
    queryFn: () => getContentIdeas({
      status: statusFilter || undefined,
      pillar: pillarFilter || undefined,
    }),
  });

  const updateMutation = useMutation({
    mutationFn: ({ id, data }) => updateContentIdea(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries(['contentIdeas']);
      queryClient.invalidateQueries(['pipelineStats']);
    },
  });

  const bulkApproveMutation = useMutation({
    mutationFn: bulkApproveIdeas,
    onSuccess: () => {
      queryClient.invalidateQueries(['contentIdeas']);
      queryClient.invalidateQueries(['pipelineStats']);
      setSelectedIds([]);
    },
  });

  const bulkRejectMutation = useMutation({
    mutationFn: bulkRejectIdeas,
    onSuccess: () => {
      queryClient.invalidateQueries(['contentIdeas']);
      queryClient.invalidateQueries(['pipelineStats']);
      setSelectedIds([]);
    },
  });

  const handleSelectAll = (e) => {
    if (e.target.checked) {
      setSelectedIds(ideas?.map(i => i.id) || []);
    } else {
      setSelectedIds([]);
    }
  };

  const handleSelect = (id) => {
    setSelectedIds(prev =>
      prev.includes(id) ? prev.filter(i => i !== id) : [...prev, id]
    );
  };

  const handleApprove = (id) => {
    updateMutation.mutate({ id, data: { status: 'approved' } });
  };

  const handleReject = (id) => {
    updateMutation.mutate({ id, data: { status: 'rejected' } });
  };

  return (
    <div className="content-ideas">
      <div className="page-header">
        <h1 className="page-title">Content Ideas</h1>
      </div>

      <div className="filters">
        <select
          className="filter-select"
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
        >
          <option value="">All Status</option>
          <option value="pending">Pending</option>
          <option value="approved">Approved</option>
          <option value="rejected">Rejected</option>
          <option value="error">Error</option>
        </select>
        <select
          className="filter-select"
          value={pillarFilter}
          onChange={(e) => setPillarFilter(e.target.value)}
        >
          <option value="">All Pillars</option>
          <option value="market_intelligence">Market Intelligence</option>
          <option value="educational_tips">Educational Tips</option>
          <option value="lifestyle_local">Lifestyle & Local</option>
          <option value="brand_humanization">Brand Humanization</option>
        </select>
      </div>

      {selectedIds.length > 0 && (
        <div className="bulk-actions">
          <span>{selectedIds.length} selected</span>
          <button
            className="btn btn-success"
            onClick={() => bulkApproveMutation.mutate(selectedIds)}
          >
            Approve Selected
          </button>
          <button
            className="btn btn-danger"
            onClick={() => bulkRejectMutation.mutate(selectedIds)}
          >
            Reject Selected
          </button>
          <button
            className="btn btn-secondary"
            onClick={() => setSelectedIds([])}
          >
            Clear
          </button>
        </div>
      )}

      {isLoading ? (
        <div className="loading">Loading content ideas...</div>
      ) : ideas?.length === 0 ? (
        <div className="empty-state">
          <h3>No content ideas found</h3>
          <p>Run the content discovery workflow to scrape new ideas</p>
        </div>
      ) : (
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th className="checkbox-cell">
                  <input
                    type="checkbox"
                    onChange={handleSelectAll}
                    checked={selectedIds.length === ideas?.length}
                  />
                </th>
                <th>ID</th>
                <th>Platform</th>
                <th>Pillar</th>
                <th>Hook</th>
                <th>Score</th>
                <th>Status</th>
                <th>Created</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {ideas?.map((idea) => (
                <tr key={idea.id}>
                  <td className="checkbox-cell">
                    <input
                      type="checkbox"
                      checked={selectedIds.includes(idea.id)}
                      onChange={() => handleSelect(idea.id)}
                    />
                  </td>
                  <td>#{idea.id}</td>
                  <td>
                    {idea.source_platform && (
                      <span className={`platform-badge platform-${idea.source_platform}`}>
                        {idea.source_platform}
                      </span>
                    )}
                  </td>
                  <td>
                    {idea.pillar && (
                      <span className={`pillar-badge pillar-${idea.pillar}`}>
                        {idea.pillar.replace('_', ' ')}
                      </span>
                    )}
                  </td>
                  <td className="hook-cell" title={idea.suggested_hook}>
                    {idea.suggested_hook?.substring(0, 50)}
                    {idea.suggested_hook?.length > 50 && '...'}
                  </td>
                  <td>{idea.viral_score || '-'}/10</td>
                  <td>
                    <span className={`status-badge status-${idea.status}`}>
                      {idea.status.replace('_', ' ')}
                    </span>
                  </td>
                  <td>{format(new Date(idea.created_at), 'MMM d, HH:mm')}</td>
                  <td>
                    <div className="actions">
                      <button
                        className="action-btn view"
                        onClick={() => setViewingIdea(idea)}
                      >
                        View
                      </button>
                      {idea.status === 'pending' && (
                        <>
                          <button
                            className="action-btn approve"
                            onClick={() => handleApprove(idea.id)}
                          >
                            Approve
                          </button>
                          <button
                            className="action-btn reject"
                            onClick={() => handleReject(idea.id)}
                          >
                            Reject
                          </button>
                        </>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {viewingIdea && (
        <div className="modal-overlay" onClick={() => setViewingIdea(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2 className="modal-title">Content Idea #{viewingIdea.id}</h2>
              <button className="modal-close" onClick={() => setViewingIdea(null)}>
                &times;
              </button>
            </div>
            <div className="modal-body">
              <div className="form-group">
                <label className="form-label">Source Platform</label>
                <span className={`platform-badge platform-${viewingIdea.source_platform}`}>
                  {viewingIdea.source_platform}
                </span>
              </div>
              <div className="form-group">
                <label className="form-label">Source URL</label>
                <a href={viewingIdea.source_url} target="_blank" rel="noopener noreferrer">
                  {viewingIdea.source_url}
                </a>
              </div>
              <div className="form-group">
                <label className="form-label">Content Pillar</label>
                <span className={`pillar-badge pillar-${viewingIdea.pillar}`}>
                  {viewingIdea.pillar?.replace('_', ' ')}
                </span>
              </div>
              <div className="form-group">
                <label className="form-label">Viral Score</label>
                <strong>{viewingIdea.viral_score}/10</strong>
              </div>
              <div className="form-group">
                <label className="form-label">Suggested Hook</label>
                <p>{viewingIdea.suggested_hook}</p>
              </div>
              <div className="form-group">
                <label className="form-label">Original Text</label>
                <p style={{ whiteSpace: 'pre-wrap' }}>{viewingIdea.original_text}</p>
              </div>
              {viewingIdea.error_message && (
                <div className="form-group">
                  <label className="form-label" style={{ color: 'var(--error)' }}>Error</label>
                  <p style={{ color: 'var(--error)' }}>{viewingIdea.error_message}</p>
                </div>
              )}
            </div>
            <div className="modal-footer">
              {viewingIdea.status === 'pending' && (
                <>
                  <button
                    className="btn btn-success"
                    onClick={() => {
                      handleApprove(viewingIdea.id);
                      setViewingIdea(null);
                    }}
                  >
                    Approve
                  </button>
                  <button
                    className="btn btn-danger"
                    onClick={() => {
                      handleReject(viewingIdea.id);
                      setViewingIdea(null);
                    }}
                  >
                    Reject
                  </button>
                </>
              )}
              <button className="btn btn-secondary" onClick={() => setViewingIdea(null)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      <style>{`
        .hook-cell {
          max-width: 200px;
        }
      `}</style>
    </div>
  );
}

export default ContentIdeas;
