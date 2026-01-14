import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getContentIdeas, updateContentIdea, bulkApproveIdeas, bulkRejectIdeas, triggerTestPipeline } from '../api';
import { format } from 'date-fns';

function ContentIdeas() {
  const queryClient = useQueryClient();
  const [statusFilter, setStatusFilter] = useState('');
  const [pillarFilter, setPillarFilter] = useState('');
  const [selectedIds, setSelectedIds] = useState([]);
  const [viewingIdea, setViewingIdea] = useState(null);
  const [sortField, setSortField] = useState('created_at');
  const [sortDirection, setSortDirection] = useState('desc'); // desc = newest first

  // Fetch data first
  const { data: ideas, isLoading, error } = useQuery({
    queryKey: ['contentIdeas', statusFilter, pillarFilter],
    queryFn: () => getContentIdeas({
      status: statusFilter || undefined,
      pillar: pillarFilter || undefined,
    }),
  });

  const handleSort = (field) => {
    if (sortField === field) {
      setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  const sortedIdeas = React.useMemo(() => {
    if (!ideas) return [];
    return [...ideas].sort((a, b) => {
      let aVal = a[sortField];
      let bVal = b[sortField];

      // Handle nulls
      if (aVal == null) aVal = sortField === 'created_at' ? '' : 0;
      if (bVal == null) bVal = sortField === 'created_at' ? '' : 0;

      // Date comparison
      if (sortField === 'created_at') {
        aVal = new Date(aVal).getTime();
        bVal = new Date(bVal).getTime();
      }

      // String comparison
      if (typeof aVal === 'string') {
        aVal = aVal.toLowerCase();
        bVal = bVal.toLowerCase();
      }

      if (sortDirection === 'asc') {
        return aVal > bVal ? 1 : aVal < bVal ? -1 : 0;
      } else {
        return aVal < bVal ? 1 : aVal > bVal ? -1 : 0;
      }
    });
  }, [ideas, sortField, sortDirection]);

  const SortHeader = ({ field, children, width }) => (
    <th
      style={{ width, cursor: 'pointer', userSelect: 'none' }}
      onClick={() => handleSort(field)}
    >
      {children}
      {sortField === field && (
        <span style={{ marginLeft: '4px', opacity: 0.7 }}>
          {sortDirection === 'asc' ? '▲' : '▼'}
        </span>
      )}
    </th>
  );

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
      setSelectedIds(sortedIdeas?.map(i => i.id) || []);
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

  const getDetailedStatus = (idea) => {
    // If completed or explicit error, just show status
    if (['published', 'ready_to_publish', 'publishing'].includes(idea.status)) return idea.status;

    // If pending/approved, waiting for pipeline
    if (['pending', 'approved'].includes(idea.status)) return idea.status;

    // Check depth
    if (!idea.scripts || idea.scripts.length === 0) {
      if (idea.status === 'script_generating') return 'Generating Script...';
      return idea.status;
    }

    // Has script, check assets
    const latestScript = idea.scripts[idea.scripts.length - 1]; // items from query are ordered desc but let's be safe. wait, query was desc. so [0] is latest?
    // Actually backend returns .unique().all() but ordered by created_at desc.
    // We should assume data integrity for now or sort client side. 
    // Let's assume the backend serializer preserves order if we did eager loading correctly?
    // Actually, SQL order_by might not strictly apply to the nested collection unless specified in relationship or joinedload strategy.
    // Let's safe check client side.
    const scripts = [...idea.scripts].sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
    const script = scripts[0];

    if (!script.assets || script.assets.length === 0) return 'Script Ready (No Assets)';

    // Check assets
    // Find asset with most progress
    const assets = [...script.assets].sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
    const asset = assets[0];

    if (!asset.voiceover_path) return 'Missing Voice';
    if (!asset.avatar_video_path) return 'Missing Avatar';
    if (!asset.final_video_path) return 'Assembling...';

    return idea.status;
  };

  if (isLoading) return <div className="loading">Loading ideas...</div>;
  if (error) return <div className="error">Error: {error.message}</div>;

  return (
    <div className="content-ideas-page">
      <div className="page-header">
        <h1>Content Ideas</h1>
        <div className="filters">
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="filter-select"
          >
            <option value="">All Status</option>
            <option value="pending">Pending</option>
            <option value="approved">Approved</option>
            <option value="script_ready">Script Ready</option>
            <option value="voice_ready">Voice Ready</option>
            <option value="avatar_ready">Avatar Ready</option>
            <option value="published">Published</option>
            <option value="failed">Failed</option>
          </select>
          <select
            value={pillarFilter}
            onChange={(e) => setPillarFilter(e.target.value)}
            className="filter-select"
          >
            <option value="">All Pillars</option>
            <option value="market_intelligence">Market Intel</option>
            <option value="educational_tips">Educational</option>
            <option value="lifestyle_local">Lifestyle</option>
            <option value="brand_humanization">Brand</option>
          </select>
        </div>
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

      {!sortedIdeas?.length ? (
        <div className="empty-state">
          <p>No content ideas found. Check back later!</p>
        </div>
      ) : (
        <div className="table-container">
          <table className="ideas-table">
            <thead>
              <tr>
                <th style={{ width: '40px' }}>
                  <input
                    type="checkbox"
                    onChange={handleSelectAll}
                    checked={selectedIds.length === sortedIdeas?.length && sortedIdeas.length > 0}
                  />
                </th>
                <SortHeader field="id" width="60px">ID</SortHeader>
                <SortHeader field="source_platform" width="80px">PLATFORM</SortHeader>
                <SortHeader field="views" width="80px">VIEWS</SortHeader>
                <SortHeader field="likes" width="80px">LIKES</SortHeader>
                <SortHeader field="original_text" width={undefined}>CAPTION</SortHeader>
                <SortHeader field="author" width="100px">AUTHOR</SortHeader>
                <SortHeader field="status" width="100px">STATUS</SortHeader>
                <SortHeader field="created_at" width="100px">CREATED</SortHeader>
                <th style={{ width: '120px' }}>ACTIONS</th>
              </tr>
            </thead>
            <tbody>
              {sortedIdeas.map((idea) => (
                <tr key={idea.id} className={`row-${idea.status}`}>
                  <td>
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
                    {idea.views ? (
                      <span style={{ fontWeight: 500, color: '#0284c7' }}>
                        {idea.views >= 1000000
                          ? `${(idea.views / 1000000).toFixed(1)}M`
                          : idea.views >= 1000
                            ? `${(idea.views / 1000).toFixed(0)}K`
                            : idea.views}
                      </span>
                    ) : '-'}
                  </td>
                  <td>
                    {idea.likes ? (
                      <span style={{ fontWeight: 500, color: '#dc2626' }}>
                        {idea.likes >= 1000000
                          ? `${(idea.likes / 1000000).toFixed(1)}M`
                          : idea.likes >= 1000
                            ? `${(idea.likes / 1000).toFixed(0)}K`
                            : idea.likes}
                      </span>
                    ) : '-'}
                  </td>
                  <td className="caption-cell" title={idea.original_text}>
                    {idea.original_text?.substring(0, 60)}
                    {idea.original_text?.length > 60 && '...'}
                  </td>
                  <td style={{ fontSize: '0.85em' }}>
                    {idea.author ? `@${idea.author.substring(0, 12)}${idea.author.length > 12 ? '...' : ''}` : '-'}
                  </td>
                  <td>
                    <span className={`status-badge status-${idea.status}`}>
                      {idea.status.replace('_', ' ')}
                    </span>
                  </td>
                  <td style={{ fontSize: '0.85em' }}>{new Date(idea.created_at).toLocaleString('en-US', { month: 'short', day: 'numeric' })}</td>
                  <td>
                    <div className="actions">
                      <button
                        className="action-btn view"
                        onClick={() => setViewingIdea(idea)}
                      >
                        View
                      </button>
                      {['error', 'failed'].includes(idea.status) && (
                        <button
                          className="action-btn retry"
                          onClick={() => handleApprove(idea.id)}
                        >
                          Retry
                        </button>
                      )}
                      {['script_ready', 'voice_ready', 'avatar_ready'].includes(idea.status) && (
                        <button
                          className="action-btn retry"
                          style={{ backgroundColor: '#2196F3' }}
                          onClick={() => handleApprove(idea.id)}
                          title="Resume Pipeline (Skip generated steps)"
                        >
                          Resume
                        </button>
                      )}
                      {idea.status === 'pending' && (
                        <>
                          <button
                            className="action-btn approve"
                            onClick={() => handleApprove(idea.id)}
                          >
                            Approve
                          </button>
                          <button
                            className="action-btn test-approve"
                            style={{ backgroundColor: '#9333ea', color: 'white', marginLeft: '4px' }}
                            onClick={async (e) => {
                              e.stopPropagation(); // Prevent row click
                              if (confirm('Trigger TEST pipeline for this idea?')) {
                                try {
                                  await triggerTestPipeline({ content_idea_id: idea.id });
                                  alert('Test pipeline triggered!');
                                } catch (err) {
                                  alert('Failed to trigger test pipeline: ' + err.message);
                                }
                              }
                            }}
                            title="Trigger Test Pipeline (Immediate)"
                          >
                            Test
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
        </div >
      )
      }

      {
        viewingIdea && (
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
                {(viewingIdea.views > 0 || viewingIdea.likes > 0 || viewingIdea.author) && (
                  <div className="form-group">
                    <label className="form-label">Engagement Metrics</label>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px', marginTop: '8px' }}>
                      {viewingIdea.views > 0 && (
                        <div style={{ textAlign: 'center', padding: '8px', background: '#f0f9ff', borderRadius: '6px' }}>
                          <div style={{ fontSize: '1.2em', fontWeight: 'bold', color: '#0284c7' }}>
                            {viewingIdea.views >= 1000000
                              ? `${(viewingIdea.views / 1000000).toFixed(1)}M`
                              : viewingIdea.views >= 1000
                                ? `${(viewingIdea.views / 1000).toFixed(0)}K`
                                : viewingIdea.views}
                          </div>
                          <div style={{ fontSize: '0.8em', color: '#666' }}>Views</div>
                        </div>
                      )}
                      {viewingIdea.likes > 0 && (
                        <div style={{ textAlign: 'center', padding: '8px', background: '#fef2f2', borderRadius: '6px' }}>
                          <div style={{ fontSize: '1.2em', fontWeight: 'bold', color: '#dc2626' }}>
                            {viewingIdea.likes >= 1000000
                              ? `${(viewingIdea.likes / 1000000).toFixed(1)}M`
                              : viewingIdea.likes >= 1000
                                ? `${(viewingIdea.likes / 1000).toFixed(0)}K`
                                : viewingIdea.likes}
                          </div>
                          <div style={{ fontSize: '0.8em', color: '#666' }}>Likes</div>
                        </div>
                      )}
                      {viewingIdea.shares > 0 && (
                        <div style={{ textAlign: 'center', padding: '8px', background: '#f0fdf4', borderRadius: '6px' }}>
                          <div style={{ fontSize: '1.2em', fontWeight: 'bold', color: '#16a34a' }}>
                            {viewingIdea.shares >= 1000
                              ? `${(viewingIdea.shares / 1000).toFixed(0)}K`
                              : viewingIdea.shares}
                          </div>
                          <div style={{ fontSize: '0.8em', color: '#666' }}>Shares</div>
                        </div>
                      )}
                      {viewingIdea.comments > 0 && (
                        <div style={{ textAlign: 'center', padding: '8px', background: '#fefce8', borderRadius: '6px' }}>
                          <div style={{ fontSize: '1.2em', fontWeight: 'bold', color: '#ca8a04' }}>
                            {viewingIdea.comments >= 1000
                              ? `${(viewingIdea.comments / 1000).toFixed(0)}K`
                              : viewingIdea.comments}
                          </div>
                          <div style={{ fontSize: '0.8em', color: '#666' }}>Comments</div>
                        </div>
                      )}
                    </div>
                    {viewingIdea.author && (
                      <div style={{ marginTop: '12px', padding: '8px', background: '#f5f5f5', borderRadius: '6px' }}>
                        <strong>@{viewingIdea.author}</strong>
                        {viewingIdea.author_followers > 0 && (
                          <span style={{ marginLeft: '8px', color: '#666' }}>
                            ({viewingIdea.author_followers >= 1000000
                              ? `${(viewingIdea.author_followers / 1000000).toFixed(1)}M`
                              : viewingIdea.author_followers >= 1000
                                ? `${(viewingIdea.author_followers / 1000).toFixed(0)}K`
                                : viewingIdea.author_followers} followers)
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                )}
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
                {['error', 'failed'].includes(viewingIdea.status) && (
                  <button
                    className="btn btn-warning"
                    onClick={() => {
                      handleApprove(viewingIdea.id);
                      setViewingIdea(null);
                    }}
                  >
                    Retry Pipeline
                  </button>
                )}
                {['script_ready', 'voice_ready', 'avatar_ready'].includes(viewingIdea.status) && (
                  <button
                    className="btn btn-primary"
                    style={{ backgroundColor: '#2196F3', borderColor: '#2196F3' }}
                    onClick={() => {
                      handleApprove(viewingIdea.id);
                      setViewingIdea(null);
                    }}
                  >
                    Resume Pipeline
                  </button>
                )}
                <button className="btn btn-secondary" onClick={() => setViewingIdea(null)}>
                  Close
                </button>
              </div>
            </div>
          </div>
        )
      }

      <style>{`
        .ideas-table th {
          transition: background 0.2s;
        }
        .ideas-table th:hover {
          background: var(--bg-tertiary);
        }
        .caption-cell {
          max-width: 250px;
          font-size: 0.9em;
          color: #666;
        }
        .action-btn.retry {
          background: #f59e0b;
          color: white;
          border: none;
        }
        .action-btn.retry:hover {
          background: #d97706;
        }
        .btn-warning {
          background: #f59e0b;
          color: white;
          border: none;
        }
        .btn-warning:hover {
          background: #d97706;
        }
      `}</style>
    </div >
  );
}

export default ContentIdeas;
