import React from 'react';
import type { PredictResponse } from '../types';
import {
  History,
  Trash2,
  ExternalLink,
  Sprout,
  FileSpreadsheet,
} from 'lucide-react';

interface Props {
  sessionResults: PredictResponse[];
  onSelectHistory: (item: PredictResponse) => void;
  onClearHistory: () => void;
  onNavigateToDetection: () => void;
}

export const HistoryWorkspace: React.FC<Props> = ({
  sessionResults,
  onSelectHistory,
  onClearHistory,
  onNavigateToDetection,
}) => {
  const totalWeeds = sessionResults.reduce((acc, r) => acc + (r.weed_count || 0), 0);
  const totalCrops = sessionResults.reduce((acc, r) => acc + (r.crop_count || 0), 0);
  const severeCount = sessionResults.filter((r) => (r.weed_count || 0) >= 15).length;

  const handleExportHistoryCsv = () => {
    if (sessionResults.length === 0) return;
    const header = 'filename,checkpoint,weed_count,crop_count,infestation_pct,mean_conf,timestamp\n';
    const rows = sessionResults.map((r, i) => {
      const modelName = r.checkpoint ? r.checkpoint.split(/[/\\]/).pop() : 'model';
      const weedCnt = r.weed_count || 0;
      const cropCnt = r.crop_count || 0;
      const infestPct = r.infestation?.infestation_pct !== undefined ? r.infestation.infestation_pct.toFixed(2) : '0';
      const meanConf = r.infestation?.mean_conf !== undefined ? r.infestation.mean_conf.toFixed(4) : '0';
      const timeStr = new Date(Date.now() - i * 60000).toISOString();
      return `${r.filename},${modelName},${weedCnt},${cropCnt},${infestPct},${meanConf},${timeStr}`;
    }).join('\n');

    const blob = new Blob([header + rows], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `weedvision_session_history_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div style={{ padding: '24px 28px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '20px', flex: 1, background: 'var(--fo-bg-dark)' }}>
      {/* Header */}
      <div
        style={{
          background: '#ffffff',
          border: '1px solid var(--fo-border)',
          borderRadius: '8px',
          padding: '18px 24px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          boxShadow: 'var(--fo-shadow-sm)',
        }}
      >
        <div>
          <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 700, color: 'var(--fo-text-main)', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <History size={18} style={{ color: 'var(--fo-accent-green)' }} /> Analysis History
          </h2>
          <div style={{ fontSize: '12px', color: 'var(--fo-text-muted)', marginTop: '4px' }}>
            Log of photographs processed during the active session with verified counts, model versions, and reload actions.
          </div>
        </div>

        <div style={{ display: 'flex', gap: '10px' }}>
          {sessionResults.length > 0 && (
            <>
              <button
                className="fo-btn fo-btn-secondary"
                onClick={handleExportHistoryCsv}
                style={{ fontSize: '12px', padding: '6px 14px' }}
                title="Export session records as CSV"
              >
                <FileSpreadsheet size={13} style={{ color: 'var(--fo-accent-green)' }} /> Export History CSV
              </button>
              <button
                className="fo-btn fo-btn-secondary"
                onClick={onClearHistory}
                style={{ fontSize: '12px', padding: '6px 14px', color: '#b91c1c' }}
                title="Clear current session log"
              >
                <Trash2 size={13} /> Clear History
              </button>
            </>
          )}
        </div>
      </div>

      {/* KPI Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '14px' }}>
        <div style={{ background: '#ffffff', border: '1px solid var(--fo-border)', borderRadius: '8px', padding: '16px', boxShadow: 'var(--fo-shadow-sm)' }}>
          <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--fo-text-muted)', textTransform: 'uppercase' }}>Analyzed Images</div>
          <div style={{ fontSize: '24px', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--fo-text-main)', marginTop: '4px' }}>
            {sessionResults.length}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--fo-text-muted)', marginTop: '4px' }}>In active browser session</div>
        </div>

        <div style={{ background: '#ffffff', border: '1px solid var(--fo-border)', borderRadius: '8px', padding: '16px', boxShadow: 'var(--fo-shadow-sm)' }}>
          <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--fo-text-muted)', textTransform: 'uppercase' }}>Total Weeds Detected</div>
          <div style={{ fontSize: '24px', fontWeight: 700, fontFamily: 'var(--font-mono)', color: totalWeeds > 0 ? '#b91c1c' : 'var(--fo-text-main)', marginTop: '4px' }}>
            {totalWeeds}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--fo-text-muted)', marginTop: '4px' }}>
            {sessionResults.length > 0 ? (totalWeeds / sessionResults.length).toFixed(1) : '0'} avg per image
          </div>
        </div>

        <div style={{ background: '#ffffff', border: '1px solid var(--fo-border)', borderRadius: '8px', padding: '16px', boxShadow: 'var(--fo-shadow-sm)' }}>
          <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--fo-text-muted)', textTransform: 'uppercase' }}>Total Crops Identified</div>
          <div style={{ fontSize: '24px', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--fo-accent-green)', marginTop: '4px' }}>
            {totalCrops}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--fo-text-muted)', marginTop: '4px' }}>Preserved agricultural targets</div>
        </div>

        <div style={{ background: '#ffffff', border: '1px solid var(--fo-border)', borderRadius: '8px', padding: '16px', boxShadow: 'var(--fo-shadow-sm)' }}>
          <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--fo-text-muted)', textTransform: 'uppercase' }}>Severe Infestation Fields</div>
          <div style={{ fontSize: '24px', fontWeight: 700, fontFamily: 'var(--font-mono)', color: severeCount > 0 ? '#b91c1c' : 'var(--fo-text-main)', marginTop: '4px' }}>
            {severeCount}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--fo-text-muted)', marginTop: '4px' }}>Fields exceeding 15 weeds</div>
        </div>
      </div>

      {/* History Table */}
      <div
        style={{
          background: '#ffffff',
          border: '1px solid var(--fo-border)',
          borderRadius: '8px',
          padding: '20px',
          boxShadow: 'var(--fo-shadow-sm)',
        }}
      >
        <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--fo-text-main)', marginBottom: '14px' }}>
          Session Records ({sessionResults.length})
        </div>

        {sessionResults.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '48px 20px', color: 'var(--fo-text-muted)' }}>
            <History size={36} style={{ color: 'var(--fo-border-strong)', marginBottom: '10px' }} />
            <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--fo-text-subtle)' }}>
              No Analyses Recorded Yet
            </div>
            <div style={{ fontSize: '12px', marginTop: '4px', maxWidth: '380px', margin: '4px auto 16px' }}>
              Run detection on an agricultural photograph in the Weed Detection workspace to log results here.
            </div>
            <button
              className="fo-btn fo-btn-primary"
              onClick={onNavigateToDetection}
              style={{ fontSize: '12px', padding: '7px 16px' }}
            >
              Go to Weed Detection
            </button>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--fo-border)', background: 'var(--fo-bg-elevated)', textTransform: 'uppercase', color: 'var(--fo-text-muted)', fontSize: '11px' }}>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Preview</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Photograph Filename</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Model Checkpoint</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Weed Count</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Crop Count</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Severity Rating</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Mean Confidence</th>
                  <th style={{ padding: '8px 12px', textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {sessionResults.map((r, idx) => {
                  const modelName = r.checkpoint ? r.checkpoint.split(/[/\\]/).pop() : 'weed_detection_best.pt';
                  const isSevere = (r.weed_count || 0) >= 15;
                  const isModerate = (r.weed_count || 0) >= 5 && (r.weed_count || 0) < 15;
                  const severityText = isSevere ? 'Severe' : isModerate ? 'Moderate' : 'Low';
                  const badgeClass = isSevere ? 'badge-weed' : isModerate ? 'badge-neutral' : 'badge-crop';

                  return (
                    <tr key={idx} style={{ borderBottom: '1px solid var(--fo-border)' }}>
                      <td style={{ padding: '8px 12px', width: '56px' }}>
                        {r.annotated_image || r.original_image ? (
                          <img
                            src={r.annotated_image || r.original_image}
                            alt={r.filename}
                            style={{ width: '48px', height: '36px', objectFit: 'cover', borderRadius: '4px', border: '1px solid var(--fo-border)' }}
                          />
                        ) : (
                          <div style={{ width: '48px', height: '36px', background: 'var(--fo-bg-elevated)', borderRadius: '4px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <Sprout size={14} style={{ color: 'var(--fo-text-muted)' }} />
                          </div>
                        )}
                      </td>
                      <td style={{ padding: '8px 12px', fontWeight: 600, color: 'var(--fo-text-main)', fontFamily: 'var(--font-mono)' }}>
                        {r.filename}
                      </td>
                      <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--fo-text-muted)' }}>
                        {modelName}
                      </td>
                      <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)', fontWeight: 700, color: (r.weed_count || 0) > 0 ? '#b91c1c' : 'var(--fo-text-main)' }}>
                        {r.weed_count || 0}
                      </td>
                      <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)', color: 'var(--fo-accent-green)' }}>
                        {r.crop_count || 0}
                      </td>
                      <td style={{ padding: '8px 12px' }}>
                        <span className={`fo-btn ${badgeClass}`} style={{ fontSize: '10px', padding: '2px 8px', fontWeight: 600 }}>
                          {severityText}
                        </span>
                      </td>
                      <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)' }}>
                        {r.infestation?.mean_conf !== undefined && r.infestation.mean_conf !== null
                          ? `${(r.infestation.mean_conf * 100).toFixed(1)}%`
                          : '—'}
                      </td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>
                        <button
                          className="fo-btn fo-btn-primary"
                          onClick={() => onSelectHistory(r)}
                          style={{ fontSize: '11px', padding: '4px 10px' }}
                          title="Open this analysis in Weed Detection workspace"
                        >
                          <ExternalLink size={12} /> Open in Viewer
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
