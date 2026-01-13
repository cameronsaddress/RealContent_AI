import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { getPipelineStats, getPipelineOverview, getApiCredits } from '../api';
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

  const { data: creditsData, isLoading: creditsLoading } = useQuery({
    queryKey: ['apiCredits'],
    queryFn: getApiCredits,
    refetchInterval: 300000, // Refresh every 5 minutes
    staleTime: 60000, // Consider data stale after 1 minute
  });

  // Calculate totals from new stats format
  const totalIdeas = stats?.content_ideas
    ? Object.values(stats.content_ideas).reduce((a, b) => a + b, 0)
    : 0;

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
          <div className="stat-value">{totalIdeas}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Pending Review</div>
          <div className="stat-value warning">{stats?.content_ideas?.pending || overview?.counts?.pending || 0}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Approved</div>
          <div className="stat-value primary">{overview?.counts?.approved || 0}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Scripts Ready</div>
          <div className="stat-value">{stats?.content_ideas?.script_ready || stats?.scripts_total || 0}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Ready to Publish</div>
          <div className="stat-value success">{stats?.assets?.ready_to_publish || 0}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Published</div>
          <div className="stat-value success">{stats?.published_total || overview?.counts?.published || 0}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Processing</div>
          <div className="stat-value primary">{overview?.counts?.processing || 0}</div>
        </div>
      </div>

      {/* API Credits Section */}
      <div className="card" style={{ marginTop: '24px', marginBottom: '24px' }}>
        <div className="card-header">
          <h3 className="card-title">API Credits & Usage</h3>
          {creditsData?.fetched_at && (
            <span className="credits-updated">
              Updated: {format(new Date(creditsData.fetched_at), 'h:mm a')}
            </span>
          )}
        </div>
        {creditsLoading ? (
          <div className="loading">Loading credits...</div>
        ) : (
          <div className="credits-grid">
            {/* ElevenLabs */}
            <div className={`credit-card ${creditsData?.credits?.elevenlabs?.status === 'ok' ? '' : 'credit-card-error'}`}>
              <div className="credit-header">
                <span className="credit-icon">🎙️</span>
                <span className="credit-service">ElevenLabs</span>
              </div>
              {creditsData?.credits?.elevenlabs?.status === 'ok' ? (
                <>
                  <div className="credit-value">
                    {creditsData.credits.elevenlabs.remaining?.toLocaleString()}
                  </div>
                  <div className="credit-label">characters remaining</div>
                  <div className="credit-progress">
                    <div
                      className="credit-progress-bar"
                      style={{
                        width: `${Math.min(100, (creditsData.credits.elevenlabs.remaining / creditsData.credits.elevenlabs.character_limit) * 100)}%`
                      }}
                    />
                  </div>
                  <div className="credit-detail">
                    {creditsData.credits.elevenlabs.character_count?.toLocaleString()} / {creditsData.credits.elevenlabs.character_limit?.toLocaleString()} used
                  </div>
                </>
              ) : (
                <div className="credit-error">
                  {creditsData?.credits?.elevenlabs?.status === 'not_configured'
                    ? 'Not configured'
                    : creditsData?.credits?.elevenlabs?.message || 'Error'}
                </div>
              )}
            </div>

            {/* HeyGen */}
            <div className={`credit-card ${creditsData?.credits?.heygen?.status === 'ok' ? '' : 'credit-card-error'}`}>
              <div className="credit-header">
                <span className="credit-icon">🎬</span>
                <span className="credit-service">HeyGen</span>
              </div>
              {creditsData?.credits?.heygen?.status === 'ok' ? (
                <>
                  <div className="credit-value">
                    {creditsData.credits.heygen.remaining_credits}
                  </div>
                  <div className="credit-label">credits remaining</div>
                  <div className="credit-detail">
                    ({Math.floor(creditsData.credits.heygen.remaining_quota_seconds / 60)}m {creditsData.credits.heygen.remaining_quota_seconds % 60}s of video)
                  </div>
                </>
              ) : (
                <div className="credit-error">
                  {creditsData?.credits?.heygen?.status === 'not_configured'
                    ? 'Not configured'
                    : creditsData?.credits?.heygen?.message || 'Error'}
                </div>
              )}
            </div>

            {/* OpenRouter */}
            <div className={`credit-card ${creditsData?.credits?.openrouter?.status === 'ok' ? '' : 'credit-card-error'}`}>
              <div className="credit-header">
                <span className="credit-icon">🤖</span>
                <span className="credit-service">OpenRouter</span>
              </div>
              {creditsData?.credits?.openrouter?.status === 'ok' ? (
                <>
                  <div className="credit-value">
                    ${creditsData.credits.openrouter.remaining_usd?.toFixed(2) ?? 'Unlimited'}
                  </div>
                  <div className="credit-label">
                    {creditsData.credits.openrouter.limit_usd ? 'remaining balance' : 'no limit set'}
                  </div>
                  {creditsData.credits.openrouter.limit_usd && (
                    <>
                      <div className="credit-progress">
                        <div
                          className="credit-progress-bar"
                          style={{
                            width: `${Math.min(100, (creditsData.credits.openrouter.remaining_usd / creditsData.credits.openrouter.limit_usd) * 100)}%`
                          }}
                        />
                      </div>
                      <div className="credit-detail">
                        ${creditsData.credits.openrouter.usage_usd?.toFixed(2)} / ${creditsData.credits.openrouter.limit_usd?.toFixed(2)} used
                      </div>
                    </>
                  )}
                </>
              ) : (
                <div className="credit-error">
                  {creditsData?.credits?.openrouter?.status === 'not_configured'
                    ? 'Not configured'
                    : creditsData?.credits?.openrouter?.message || 'Error'}
                </div>
              )}
            </div>

            {/* OpenAI (Whisper) */}
            <div className={`credit-card ${creditsData?.credits?.openai?.status === 'ok' ? '' : 'credit-card-error'}`}>
              <div className="credit-header">
                <span className="credit-icon">💬</span>
                <span className="credit-service">OpenAI</span>
                <span className="credit-sublabel">(Whisper)</span>
              </div>
              {creditsData?.credits?.openai?.status === 'ok' ? (
                <>
                  <div className="credit-value credit-value-small">Configured</div>
                  <div className="credit-label">pay-as-you-go</div>
                  <div className="credit-detail">
                    <a href="https://platform.openai.com/usage" target="_blank" rel="noopener noreferrer">
                      View usage →
                    </a>
                  </div>
                </>
              ) : (
                <div className="credit-error">
                  {creditsData?.credits?.openai?.status === 'not_configured'
                    ? 'Not configured'
                    : creditsData?.credits?.openai?.message || 'Error'}
                </div>
              )}
            </div>

            {/* Blotato */}
            <div className={`credit-card ${creditsData?.credits?.blotato?.status === 'ok' ? '' : 'credit-card-error'}`}>
              <div className="credit-header">
                <span className="credit-icon">📤</span>
                <span className="credit-service">Blotato</span>
              </div>
              {creditsData?.credits?.blotato?.status === 'ok' ? (
                <>
                  <div className="credit-value credit-value-small">
                    {creditsData.credits.blotato.subscription_status === 'active' ? 'Active' : creditsData.credits.blotato.subscription_status}
                  </div>
                  <div className="credit-label">subscription status</div>
                  <div className="credit-detail">
                    <a href="https://app.blotato.com/settings/billing" target="_blank" rel="noopener noreferrer">
                      View credits →
                    </a>
                  </div>
                </>
              ) : (
                <div className="credit-error">
                  {creditsData?.credits?.blotato?.status === 'not_configured'
                    ? 'Not configured'
                    : creditsData?.credits?.blotato?.message || 'Error'}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      <div className="dashboard-grid">
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Content Pipeline Status</h3>
          </div>
          <div className="pillar-stats">
            {stats?.content_ideas && Object.entries(stats.content_ideas).map(([status, count]) => (
              <div key={status} className="pillar-stat-row">
                <span className={`status-badge status-${status}`}>
                  {status.replace('_', ' ')}
                </span>
                <span className="pillar-count">{count}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Asset Status</h3>
          </div>
          <div className="platform-stats">
            <div className="platform-stat-row">
              <span className="platform-badge">Voice Ready</span>
              <span className="platform-count">{stats?.assets?.voice_ready || 0}</span>
            </div>
            <div className="platform-stat-row">
              <span className="platform-badge platform-success">Ready to Publish</span>
              <span className="platform-count">{stats?.assets?.ready_to_publish || 0}</span>
            </div>
            <div className="platform-stat-row">
              <span className="platform-badge">Total Scripts</span>
              <span className="platform-count">{stats?.scripts_total || 0}</span>
            </div>
            <div className="platform-stat-row">
              <span className="platform-badge platform-success">Published</span>
              <span className="platform-count">{stats?.published_total || 0}</span>
            </div>
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: '24px' }}>
        <div className="card-header">
          <h3 className="card-title">Recent Published Content</h3>
        </div>
        {overviewLoading ? (
          <div className="loading">Loading...</div>
        ) : !overview?.recent_published || overview.recent_published.length === 0 ? (
          <div className="empty-state">
            <h3>No published content yet</h3>
            <p>Published videos will appear here</p>
          </div>
        ) : (
          <div className="table-container" style={{ border: 'none' }}>
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Script ID</th>
                  <th>TikTok</th>
                  <th>Instagram</th>
                  <th>Published</th>
                </tr>
              </thead>
              <tbody>
                {overview.recent_published.map((item) => (
                  <tr key={item.id}>
                    <td>#{item.id}</td>
                    <td>Script #{item.script_id}</td>
                    <td>
                      {item.tiktok_url ? (
                        <a href={item.tiktok_url} target="_blank" rel="noopener noreferrer" className="link-badge">
                          View
                        </a>
                      ) : '-'}
                    </td>
                    <td>
                      {item.ig_url ? (
                        <a href={item.ig_url} target="_blank" rel="noopener noreferrer" className="link-badge">
                          View
                        </a>
                      ) : '-'}
                    </td>
                    <td>{item.published_at ? format(new Date(item.published_at), 'MMM d, yyyy h:mm a') : '-'}</td>
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

        /* Credits Grid */
        .credits-grid {
          display: grid;
          grid-template-columns: repeat(5, 1fr);
          gap: 20px;
          padding: 8px 0;
        }
        @media (max-width: 1400px) {
          .credits-grid {
            grid-template-columns: repeat(3, 1fr);
          }
        }
        @media (max-width: 900px) {
          .credits-grid {
            grid-template-columns: repeat(2, 1fr);
          }
        }
        @media (max-width: 600px) {
          .credits-grid {
            grid-template-columns: 1fr;
          }
        }
        .credit-card {
          background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
          border: 1px solid #2d3748;
          border-radius: 12px;
          padding: 20px;
          transition: transform 0.2s, box-shadow 0.2s;
          min-height: 140px;
          display: flex;
          flex-direction: column;
        }
        .credit-card:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        }
        .credit-card-error {
          background: linear-gradient(135deg, #2d1f1f 0%, #1a1a2e 100%);
          border-color: #4a2c2c;
        }
        .credit-header {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 12px;
        }
        .credit-icon {
          font-size: 24px;
        }
        .credit-service {
          font-weight: 600;
          font-size: 16px;
          color: #e2e8f0;
        }
        .credit-sublabel {
          font-size: 12px;
          color: #718096;
          margin-left: 4px;
        }
        .credit-value {
          font-size: 32px;
          font-weight: 700;
          color: #68d391;
          line-height: 1.2;
        }
        .credit-value-small {
          font-size: 24px;
        }
        .credit-label {
          font-size: 13px;
          color: #a0aec0;
          margin-bottom: 8px;
          flex-shrink: 0;
        }
        .credit-progress {
          height: 6px;
          background: #2d3748;
          border-radius: 3px;
          overflow: hidden;
          margin-bottom: 8px;
        }
        .credit-progress-bar {
          height: 100%;
          background: linear-gradient(90deg, #68d391 0%, #48bb78 100%);
          border-radius: 3px;
          transition: width 0.3s ease;
        }
        .credit-detail {
          font-size: 12px;
          color: #718096;
          margin-top: auto;
        }
        .credit-detail a {
          color: #63b3ed;
          text-decoration: none;
        }
        .credit-detail a:hover {
          text-decoration: underline;
        }
        .credit-error {
          color: #fc8181;
          font-size: 14px;
          padding: 8px 0;
        }
        .credits-updated {
          font-size: 12px;
          color: #718096;
          margin-left: auto;
        }
        .card-header {
          display: flex;
          align-items: center;
        }
        .link-badge {
          color: #63b3ed;
          text-decoration: none;
          padding: 4px 8px;
          background: rgba(99, 179, 237, 0.1);
          border-radius: 4px;
        }
        .link-badge:hover {
          background: rgba(99, 179, 237, 0.2);
          text-decoration: none;
        }
        .platform-success {
          background: #276749 !important;
          color: #68d391 !important;
        }
      `}</style>
    </div>
  );
}

export default Dashboard;
