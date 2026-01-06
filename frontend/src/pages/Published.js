import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { getPublished, getAnalytics } from '../api';
import { format } from 'date-fns';

function Published() {
  const [viewingPost, setViewingPost] = useState(null);

  const { data: published, isLoading } = useQuery({
    queryKey: ['published'],
    queryFn: () => getPublished({ limit: 50 }),
  });

  const { data: analytics } = useQuery({
    queryKey: ['analytics', viewingPost?.id],
    queryFn: () => getAnalytics({ published_id: viewingPost?.id }),
    enabled: !!viewingPost,
  });

  const platforms = [
    { key: 'tiktok', name: 'TikTok' },
    { key: 'ig', name: 'Instagram' },
    { key: 'yt', name: 'YouTube' },
    { key: 'linkedin', name: 'LinkedIn' },
    { key: 'x', name: 'X (Twitter)' },
    { key: 'facebook', name: 'Facebook' },
    { key: 'threads', name: 'Threads' },
    { key: 'pinterest', name: 'Pinterest' },
  ];

  const countPublishedPlatforms = (post) => {
    return platforms.filter(p => post[`${p.key}_url`]).length;
  };

  return (
    <div className="published">
      <div className="page-header">
        <h1 className="page-title">Published Content</h1>
      </div>

      {isLoading ? (
        <div className="loading">Loading published content...</div>
      ) : published?.length === 0 ? (
        <div className="empty-state">
          <h3>No published content yet</h3>
          <p>Videos will appear here once they are published</p>
        </div>
      ) : (
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Asset ID</th>
                <th>Platforms</th>
                <th>Published At</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {published?.map((post) => (
                <tr key={post.id}>
                  <td>#{post.id}</td>
                  <td>#{post.asset_id}</td>
                  <td>
                    <div className="platform-icons">
                      {platforms.map(p => (
                        post[`${p.key}_url`] && (
                          <span
                            key={p.key}
                            className={`platform-badge platform-${p.key === 'ig' ? 'instagram' : p.key === 'yt' ? 'youtube' : p.key}`}
                            title={p.name}
                          >
                            {p.key.toUpperCase()}
                          </span>
                        )
                      ))}
                    </div>
                    <span className="platform-count">
                      {countPublishedPlatforms(post)} platforms
                    </span>
                  </td>
                  <td>{format(new Date(post.published_at), 'MMM d, yyyy HH:mm')}</td>
                  <td>
                    <div className="actions">
                      <button
                        className="action-btn view"
                        onClick={() => setViewingPost(post)}
                      >
                        View Links
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {viewingPost && (
        <div className="modal-overlay" onClick={() => setViewingPost(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2 className="modal-title">Published Post #{viewingPost.id}</h2>
              <button className="modal-close" onClick={() => setViewingPost(null)}>
                &times;
              </button>
            </div>
            <div className="modal-body">
              <div className="form-group">
                <label className="form-label">Published At</label>
                <p>{format(new Date(viewingPost.published_at), 'MMMM d, yyyy HH:mm:ss')}</p>
              </div>

              <h3 style={{ marginTop: '24px', marginBottom: '16px' }}>Platform Links</h3>

              {platforms.map(p => {
                const url = viewingPost[`${p.key}_url`];
                const id = viewingPost[`${p.key}_id`];
                return (
                  <div key={p.key} className="platform-link-row">
                    <span className={`platform-badge platform-${p.key === 'ig' ? 'instagram' : p.key === 'yt' ? 'youtube' : p.key}`}>
                      {p.name}
                    </span>
                    {url ? (
                      <a href={url} target="_blank" rel="noopener noreferrer">
                        {url}
                      </a>
                    ) : (
                      <span className="not-published">Not published</span>
                    )}
                  </div>
                );
              })}

              {analytics && analytics.length > 0 && (
                <>
                  <h3 style={{ marginTop: '24px', marginBottom: '16px' }}>Analytics</h3>
                  <div className="analytics-grid">
                    {analytics.map((stat) => (
                      <div key={stat.id} className="analytics-card">
                        <span className={`platform-badge platform-${stat.platform}`}>
                          {stat.platform}
                        </span>
                        <div className="analytics-stats">
                          <div className="analytics-stat">
                            <span className="stat-value">{stat.views?.toLocaleString() || 0}</span>
                            <span className="stat-label">Views</span>
                          </div>
                          <div className="analytics-stat">
                            <span className="stat-value">{stat.likes?.toLocaleString() || 0}</span>
                            <span className="stat-label">Likes</span>
                          </div>
                          <div className="analytics-stat">
                            <span className="stat-value">{stat.comments?.toLocaleString() || 0}</span>
                            <span className="stat-label">Comments</span>
                          </div>
                          <div className="analytics-stat">
                            <span className="stat-value">{stat.shares?.toLocaleString() || 0}</span>
                            <span className="stat-label">Shares</span>
                          </div>
                        </div>
                        {stat.engagement_rate && (
                          <div className="engagement-rate">
                            {(stat.engagement_rate * 100).toFixed(2)}% engagement
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
            <div className="modal-footer">
              <button className="btn btn-secondary" onClick={() => setViewingPost(null)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      <style>{`
        .platform-icons {
          display: flex;
          flex-wrap: wrap;
          gap: 4px;
          margin-bottom: 4px;
        }
        .platform-count {
          font-size: 12px;
          color: var(--text-secondary);
        }
        .platform-link-row {
          display: flex;
          align-items: center;
          gap: 16px;
          padding: 12px 0;
          border-bottom: 1px solid var(--border);
        }
        .platform-link-row:last-child {
          border-bottom: none;
        }
        .platform-link-row a {
          flex: 1;
          word-break: break-all;
        }
        .not-published {
          color: var(--text-secondary);
          font-style: italic;
        }
        .analytics-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 16px;
        }
        .analytics-card {
          background: var(--bg-tertiary);
          border-radius: 8px;
          padding: 16px;
        }
        .analytics-stats {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 8px;
          margin-top: 12px;
        }
        .analytics-stat {
          text-align: center;
        }
        .analytics-stat .stat-value {
          display: block;
          font-size: 18px;
          font-weight: 600;
        }
        .analytics-stat .stat-label {
          font-size: 11px;
          color: var(--text-secondary);
        }
        .engagement-rate {
          margin-top: 12px;
          text-align: center;
          font-size: 13px;
          color: var(--success);
        }
      `}</style>
    </div>
  );
}

export default Published;
