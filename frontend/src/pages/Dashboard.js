import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { getPipelineStats, getPipelineOverview } from '../api';
import { format } from 'date-fns';

function Dashboard() {
  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['pipelineStats'],
    queryFn: getPipelineStats,
  });

  const { data: overview, isLoading: overviewLoading } = useQuery({
    queryKey: ['pipelineOverview'],
    queryFn: () => getPipelineOverview({ limit: 10 }),
  });

  if (statsLoading) {
    return <div className="loading">Loading dashboard...</div>;
  }

  return (
    <div className="dashboard">
      <div className="page-header">
        <h1 className="page-title">Dashboard</h1>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-label">Total Ideas</div>
          <div className="stat-value">{stats?.total_ideas || 0}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Pending Review</div>
          <div className="stat-value warning">{stats?.pending_ideas || 0}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Approved</div>
          <div className="stat-value primary">{stats?.approved_ideas || 0}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Scripts Ready</div>
          <div className="stat-value">{stats?.scripts_ready || 0}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Ready to Publish</div>
          <div className="stat-value success">{stats?.assets_ready || 0}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Published</div>
          <div className="stat-value success">{stats?.published || 0}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Errors</div>
          <div className="stat-value error">{stats?.errors || 0}</div>
        </div>
      </div>

      <div className="dashboard-grid">
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">By Content Pillar</h3>
          </div>
          <div className="pillar-stats">
            {stats?.by_pillar && Object.entries(stats.by_pillar).map(([pillar, count]) => (
              <div key={pillar} className="pillar-stat-row">
                <span className={`pillar-badge pillar-${pillar}`}>
                  {pillar.replace('_', ' ')}
                </span>
                <span className="pillar-count">{count}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h3 className="card-title">By Source Platform</h3>
          </div>
          <div className="platform-stats">
            {stats?.by_platform && Object.entries(stats.by_platform).map(([platform, count]) => (
              <div key={platform} className="platform-stat-row">
                <span className={`platform-badge platform-${platform}`}>
                  {platform}
                </span>
                <span className="platform-count">{count}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: '24px' }}>
        <div className="card-header">
          <h3 className="card-title">Recent Pipeline Activity</h3>
        </div>
        {overviewLoading ? (
          <div className="loading">Loading...</div>
        ) : overview?.length === 0 ? (
          <div className="empty-state">
            <h3>No content yet</h3>
            <p>Content will appear here once ideas are scraped</p>
          </div>
        ) : (
          <div className="table-container" style={{ border: 'none' }}>
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Platform</th>
                  <th>Pillar</th>
                  <th>Viral Score</th>
                  <th>Status</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {overview?.map((item) => (
                  <tr key={item.content_id}>
                    <td>#{item.content_id}</td>
                    <td>
                      {item.source_platform && (
                        <span className={`platform-badge platform-${item.source_platform}`}>
                          {item.source_platform}
                        </span>
                      )}
                    </td>
                    <td>
                      {item.pillar && (
                        <span className={`pillar-badge pillar-${item.pillar}`}>
                          {item.pillar.replace('_', ' ')}
                        </span>
                      )}
                    </td>
                    <td>{item.viral_score || '-'}/10</td>
                    <td>
                      <span className={`status-badge status-${item.content_status}`}>
                        {item.content_status.replace('_', ' ')}
                      </span>
                    </td>
                    <td>{format(new Date(item.created_at), 'MMM d, yyyy')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <style>{`
        .dashboard-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 24px;
        }
        .pillar-stats, .platform-stats {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }
        .pillar-stat-row, .platform-stat-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }
        .pillar-count, .platform-count {
          font-weight: 600;
          font-size: 18px;
        }
      `}</style>
    </div>
  );
}

export default Dashboard;
