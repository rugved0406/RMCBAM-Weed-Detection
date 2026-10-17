import React, { useEffect, useState } from 'react';
import type { OverviewData, PredictResponse } from '../types';
import { getOverview } from '../services/api';
import {
  CheckCircle2,
  XCircle,
  Database,
  Cpu,
  Layers,
  Activity,
  Scan,
} from 'lucide-react';

interface Props {
  sessionResults: PredictResponse[];
  onNavigate: (workspaceId: any) => void;
  onSelectSample?: (sampleName: string) => void;
}

const CLASS_BADGES: { name: string; color: string }[] = [
  { name: 'Weed', color: '#e53935' },
  { name: 'Tomato', color: '#3987e5' },
  { name: 'Rose', color: '#8e24aa' },
  { name: 'Marigold', color: '#f57c00' },
  { name: 'Cauliflower', color: '#43a047' },
];

export const OverviewWorkspace: React.FC<Props> = ({ sessionResults, onNavigate }) => {
  const [data, setData] = useState<OverviewData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    fetchOverview();
  }, []);

  const fetchOverview = async () => {
    try {
      const res = await getOverview();
      setData(res);
    } catch (err) {
      console.error('Failed to load overview data:', err);
    } finally {
      setLoading(false);
    }
  };

  const totalSessionWeeds = sessionResults.reduce((acc, r) => acc + (r.weed_count || 0), 0);

  if (loading && !data) {
    return (
      <div style={{ padding: '40px', textAlign: 'center', color: 'var(--fo-text-subtle)' }}>
        Loading Overview Dashboard...
      </div>
    );
  }

  return (
    <div style={{ padding: '20px 24px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '20px', flex: 1 }}>
      {/* Hero Header */}
      <div
        style={{
          background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%)',
          border: '1px solid var(--fo-border)',
          borderRadius: '8px',
          padding: '20px 24px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.25)',
        }}
      >
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: '24px' }}>🌿</span>
            <h1 style={{ margin: 0, fontSize: '20px', fontWeight: 700, color: 'var(--fo-text-main)', letterSpacing: '-0.01em' }}>
              RMCBAM Weed &amp; Crop Detection
            </h1>
          </div>
          <div style={{ marginTop: '6px', fontSize: '13px', color: 'var(--fo-text-subtle)' }}>
            YOLOv8s with Residual Multi-Scale CBAM attention · Weed, Tomato, Rose, Marigold, Cauliflower
          </div>
        </div>

        <div style={{ display: 'flex', gap: '10px' }}>
          <button
            className="fo-btn fo-btn-primary"
            style={{ padding: '8px 16px', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}
            onClick={() => onNavigate('inspection')}
          >
            <Scan size={14} /> Open Image Detection
          </button>
        </div>
      </div>

      {/* 4 Metric KPI Cards matching 01_overview.jpg */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '14px' }}>
        {/* Card 1: Checkpoints */}
        <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontSize: '11px', fontWeight: 700, color: 'var(--fo-text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Checkpoints
            </span>
            <Layers size={14} style={{ color: 'var(--fo-accent-blue)' }} />
          </div>
          <div style={{ fontSize: '26px', fontWeight: 700, color: 'var(--fo-text-main)', fontFamily: 'var(--font-mono)', marginTop: '4px' }}>
            {data ? data.checkpoints.total : '—'}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--fo-text-subtle)', marginTop: '4px' }}>
            {data
              ? `${data.checkpoints.with_weed} with Weed class · ${data.checkpoints.rmcbam} RMCBAM`
              : 'Scanning models/...'}
          </div>
        </div>

        {/* Card 2: Dataset Images */}
        <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontSize: '11px', fontWeight: 700, color: 'var(--fo-text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Dataset Images
            </span>
            <Database size={14} style={{ color: '#10b981' }} />
          </div>
          <div style={{ fontSize: '26px', fontWeight: 700, color: 'var(--fo-text-main)', fontFamily: 'var(--font-mono)', marginTop: '4px' }}>
            {data && data.dataset.exists ? data.dataset.n_images : '—'}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--fo-text-subtle)', marginTop: '4px' }}>
            {data && data.dataset.exists
              ? `${data.dataset.n_labelme_json} LabelMe JSON · ${data.dataset.yaml_files.length} yaml`
              : data?.dataset.path || 'Folder scanning'}
          </div>
        </div>

        {/* Card 3: Compute */}
        <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontSize: '11px', fontWeight: 700, color: 'var(--fo-text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Compute
            </span>
            <Cpu size={14} style={{ color: '#f59e0b' }} />
          </div>
          <div style={{ fontSize: '26px', fontWeight: 700, color: 'var(--fo-text-main)', fontFamily: 'var(--font-mono)', marginTop: '4px' }}>
            {data ? data.compute.device : 'CPU'}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--fo-text-subtle)', marginTop: '4px' }}>
            {data
              ? `torch ${data.compute.torch_version} · CUDA ${data.compute.cuda_available ? 'yes' : 'no'}`
              : 'Detecting PyTorch...'}
          </div>
        </div>

        {/* Card 4: This session */}
        <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontSize: '11px', fontWeight: 700, color: 'var(--fo-text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              This Session
            </span>
            <Activity size={14} style={{ color: '#ec4899' }} />
          </div>
          <div style={{ fontSize: '26px', fontWeight: 700, color: 'var(--fo-accent-blue)', fontFamily: 'var(--font-mono)', marginTop: '4px' }}>
            {sessionResults.length} {sessionResults.length === 1 ? 'image' : 'images'}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--fo-text-subtle)', marginTop: '4px' }}>
            {totalSessionWeeds} weeds detected
          </div>
        </div>
      </div>

      {/* Two Columns: Readiness & Pipeline */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
        {/* Left: Readiness Checks */}
        <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '18px' }}>
          <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--fo-text-main)', marginBottom: '14px', letterSpacing: '0.02em' }}>
            Readiness Checklist
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
              {data?.readiness.base_ok ? (
                <CheckCircle2 size={16} style={{ color: '#10b981', flexShrink: 0, marginTop: '2px' }} />
              ) : (
                <XCircle size={16} style={{ color: '#ef4444', flexShrink: 0, marginTop: '2px' }} />
              )}
              <div style={{ fontSize: '12px', lineHeight: 1.5 }}>
                <b>Trained baseline YOLOv8s</b> checkpoint with a Weed class
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
              {data?.readiness.rmcbam_ok ? (
                <CheckCircle2 size={16} style={{ color: '#10b981', flexShrink: 0, marginTop: '2px' }} />
              ) : (
                <XCircle size={16} style={{ color: '#ef4444', flexShrink: 0, marginTop: '2px' }} />
              )}
              <div style={{ fontSize: '12px', lineHeight: 1.5 }}>
                <b>Trained YOLOv8s + RMCBAM</b> checkpoint with a Weed class
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
              {data?.readiness.labelme_ok ? (
                <CheckCircle2 size={16} style={{ color: '#10b981', flexShrink: 0, marginTop: '2px' }} />
              ) : (
                <XCircle size={16} style={{ color: '#f59e0b', flexShrink: 0, marginTop: '2px' }} />
              )}
              <div style={{ fontSize: '12px', lineHeight: 1.5 }}>
                LabelMe annotations in dataset folder ({data?.dataset.n_labelme_json || 0} found)
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
              {data?.readiness.yaml_ok ? (
                <CheckCircle2 size={16} style={{ color: '#10b981', flexShrink: 0, marginTop: '2px' }} />
              ) : (
                <XCircle size={16} style={{ color: '#f59e0b', flexShrink: 0, marginTop: '2px' }} />
              )}
              <div style={{ fontSize: '12px', lineHeight: 1.5 }}>
                <code style={{ background: 'var(--fo-bg-elevated)', padding: '1px 5px', borderRadius: '3px' }}>data.yaml</code> present (needed for live validation)
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
              {data?.readiness.classes_declared ? (
                <CheckCircle2 size={16} style={{ color: '#10b981', flexShrink: 0, marginTop: '2px' }} />
              ) : (
                <XCircle size={16} style={{ color: '#ef4444', flexShrink: 0, marginTop: '2px' }} />
              )}
              <div style={{ fontSize: '12px', lineHeight: 1.5 }}>
                All 5 classes declared in model
                {data?.readiness.missing_classes && data.readiness.missing_classes.length > 0 && (
                  <span style={{ color: '#ef4444' }}> — missing: {data.readiness.missing_classes.join(', ')}</span>
                )}
              </div>
            </div>
          </div>

          {!data?.readiness.has_project_model && (
            <div
              style={{
                marginTop: '16px',
                padding: '10px 14px',
                borderRadius: '6px',
                background: 'rgba(59, 130, 246, 0.08)',
                border: '1px solid rgba(59, 130, 246, 0.25)',
                fontSize: '11px',
                color: 'var(--fo-text-subtle)',
                lineHeight: 1.5,
              }}
            >
              No checkpoint with a <b>Weed</b> class was found in models/. Train with <code style={{ color: 'var(--fo-accent-blue)' }}>python tools/train.py --data data/yolo/data.yaml</code> or copy existing best.pt files into models/.
            </div>
          )}
        </div>

        {/* Right: Pipeline Steps & Class Legend */}
        <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '18px' }}>
          <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--fo-text-main)', marginBottom: '14px', letterSpacing: '0.02em' }}>
            Workflow Pipeline
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '12px', lineHeight: 1.5 }}>
            <div>
              <b style={{ color: 'var(--fo-accent-blue)' }}>1. Dataset Analytics</b> — scan images, convert LabelMe → YOLO, validate labels.
            </div>
            <div>
              <b style={{ color: 'var(--fo-accent-blue)' }}>2. Training</b> — <code style={{ background: 'var(--fo-bg-elevated)', padding: '1px 5px', borderRadius: '3px' }}>tools/train.py</code> trains baseline and RMCBAM with identical settings.
            </div>
            <div>
              <b style={{ color: 'var(--fo-accent-blue)' }}>3. Image Detection</b> — boxes, confidence, unique IDs (W-001…), downloads.
            </div>
            <div>
              <b style={{ color: 'var(--fo-accent-blue)' }}>4. Explainability</b> — Grad-CAM / HiResCAM from real gradients; RMCBAM attention maps.
            </div>
            <div>
              <b style={{ color: 'var(--fo-accent-blue)' }}>5. Weed Analysis &amp; Localization</b> — count, density, infestation %, severity, crop association, coordinates.
            </div>
            <div>
              <b style={{ color: 'var(--fo-accent-blue)' }}>6. Model Comparison</b> — results.csv, plots, or live validation on your data.yaml.
            </div>
          </div>

          {/* Class Legend Chips */}
          <div style={{ marginTop: '16px', paddingTop: '14px', borderTop: '1px solid var(--fo-border)' }}>
            <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--fo-text-muted)', textTransform: 'uppercase', marginBottom: '8px' }}>
              Model Class Color Palette
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
              {CLASS_BADGES.map((b) => (
                <div
                  key={b.name}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    padding: '3px 10px',
                    borderRadius: '4px',
                    background: 'var(--fo-bg-elevated)',
                    border: '1px solid var(--fo-border)',
                    fontSize: '11px',
                    color: 'var(--fo-text-main)',
                  }}
                >
                  <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: b.color }} />
                  <span>{b.name}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Session Summary Table matching 01_overview.jpg line 76-85 */}
      {sessionResults.length > 0 && (
        <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '18px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
            <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--fo-text-main)' }}>
              Session Summary ({sessionResults.length} Analysed Images)
            </div>
            <span style={{ fontSize: '11px', color: 'var(--fo-text-muted)' }}>
              Latest results from your active session
            </span>
          </div>

          <div style={{ overflowX: 'auto' }}>
            <table className="fo-json-tree" style={{ width: '100%', color: 'var(--fo-text-main)', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--fo-border)', textTransform: 'uppercase', color: 'var(--fo-text-muted)', fontSize: '11px' }}>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Image</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Model</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Weeds</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Crops</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Infestation %</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Severity</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Mean Conf</th>
                  <th style={{ padding: '8px 12px', textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {sessionResults.map((r, i) => {
                  const sev = r.infestation?.severity || 'Low';
                  const sevColor =
                    sev === 'Low' ? '#10b981' : sev === 'Moderate' ? '#f59e0b' : sev === 'High' ? '#f97316' : '#ef4444';
                  const modelName = r.checkpoint ? r.checkpoint.split(/[\\/]/).pop() || r.checkpoint : 'yolov8s';

                  return (
                    <tr key={i} style={{ borderBottom: '1px solid var(--fo-border)' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600, color: 'var(--fo-text-main)' }}>{r.filename}</td>
                      <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--fo-text-subtle)' }}>
                        {modelName}
                      </td>
                      <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--fo-accent-blue)' }}>
                        {r.weed_count}
                      </td>
                      <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)' }}>
                        {r.infestation?.infestation_pct !== undefined ? r.infestation.infestation_pct.toFixed(2) : 0}%
                      </td>
                      <td style={{ padding: '8px 12px' }}>
                        <span
                          style={{
                            padding: '2px 8px',
                            borderRadius: '3px',
                            fontSize: '10px',
                            fontWeight: 700,
                            background: `${sevColor}22`,
                            color: sevColor,
                            border: `1px solid ${sevColor}44`,
                          }}
                        >
                          {sev.toUpperCase()}
                        </span>
                      </td>
                      <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)' }}>
                        {r.infestation?.mean_conf !== undefined && r.infestation.mean_conf !== null
                          ? r.infestation.mean_conf.toFixed(3)
                          : '—'}
                      </td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>
                        <button
                          className="fo-btn fo-btn-secondary"
                          style={{ fontSize: '10px', padding: '3px 8px' }}
                          onClick={() => onNavigate('inspection')}
                        >
                          View
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
