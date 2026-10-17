import React, { useEffect, useState } from 'react';
import { getModelInfo } from '../services/api';
import {
  Layers,
  Cpu,
  CheckCircle2,
  AlertTriangle,
  ChevronDown,
  ChevronRight,
} from 'lucide-react';

interface Props {
  selectedCheckpoint: string;
}

export const ModelInfoWorkspace: React.FC<Props> = ({ selectedCheckpoint }) => {
  const [modelData, setModelData] = useState<any | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [showTrainArgs, setShowTrainArgs] = useState<boolean>(false);
  const [showArchTable, setShowArchTable] = useState<boolean>(true);

  useEffect(() => {
    fetchModelInfo();
  }, [selectedCheckpoint]);

  const fetchModelInfo = async () => {
    setLoading(true);
    try {
      const res = await getModelInfo(selectedCheckpoint);
      setModelData(res);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  if (loading && !modelData) {
    return (
      <div className="fo-canvas-workspace" style={{ padding: '48px', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ fontSize: '13px', color: 'var(--fo-text-subtle)' }}>Inspecting Architecture &amp; RMCBAM Gates...</div>
      </div>
    );
  }

  const cp = modelData?.checkpoint_info || {};
  const refArch = modelData?.reference_architectures || [];
  const arch = modelData?.architecture_table || [];

  return (
    <div style={{ padding: '20px 24px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '20px', flex: 1 }}>
      {/* Hero Header */}
      <div
        style={{
          background: 'var(--fo-bg-panel)',
          border: '1px solid var(--fo-border)',
          borderRadius: '8px',
          padding: '16px 20px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <div>
          <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 700, color: 'var(--fo-text-main)', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Layers size={18} style={{ color: 'var(--fo-accent-blue)' }} /> Model Information
          </h2>
          <div style={{ fontSize: '12px', color: 'var(--fo-text-subtle)', marginTop: '4px' }}>
            RMCBAM design, insertion points, and the structure of the selected checkpoint.
          </div>
        </div>
      </div>

      {/* RMCBAM Design Flowchart & Specification matching 08_model_information.jpg lines 32-42 */}
      <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '18px' }}>
        <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--fo-text-main)', marginBottom: '12px' }}>
          RMCBAM — Residual Multi-Scale CBAM
        </div>

        {/* Visual Graph Diagram SVG */}
        <div
          style={{
            background: '#090a0f',
            border: '1px solid var(--fo-border)',
            borderRadius: '6px',
            padding: '16px',
            overflowX: 'auto',
            marginBottom: '16px',
          }}
        >
          <svg viewBox="0 0 860 140" style={{ width: '100%', minWidth: '700px', height: '140px' }}>
            {/* Input Node */}
            <rect x="10" y="50" width="100" height="40" rx="4" fill="#1e293b" stroke="#3b82f6" strokeWidth="1.5" />
            <text x="60" y="68" fill="#ffffff" fontSize="11" fontWeight="bold" textAnchor="middle">C2f output x</text>
            <text x="60" y="82" fill="#94a3b8" fontSize="9" fontFamily="monospace" textAnchor="middle">C × H × W</text>

            {/* Arrows */}
            <path d="M 110 70 L 140 70" stroke="#64748b" strokeWidth="1.5" markerEnd="url(#arrow)" />

            {/* GAP & GMP */}
            <rect x="140" y="50" width="90" height="40" rx="4" fill="#1e293b" stroke="#475569" strokeWidth="1" />
            <text x="185" y="74" fill="#ffffff" fontSize="10" fontWeight="bold" textAnchor="middle">GAP + GMP</text>

            <path d="M 230 70 L 260 70" stroke="#64748b" strokeWidth="1.5" />

            {/* Shared Conv1D */}
            <rect x="260" y="50" width="105" height="40" rx="4" fill="#1e293b" stroke="#3b82f6" strokeWidth="1" />
            <text x="312" y="68" fill="#60a5fa" fontSize="10" fontWeight="bold" textAnchor="middle">Conv1d(k) ECA</text>
            <text x="312" y="82" fill="#94a3b8" fontSize="9" textAnchor="middle">σ → Mc (C×1×1)</text>

            <path d="M 365 70 L 395 70" stroke="#64748b" strokeWidth="1.5" />

            {/* Mul 1 */}
            <circle cx="410" cy="70" r="14" fill="#0f172a" stroke="#3b82f6" strokeWidth="1.5" />
            <text x="410" y="74" fill="#ffffff" fontSize="12" fontWeight="bold" textAnchor="middle">⊙</text>

            <path d="M 424 70 L 454 70" stroke="#64748b" strokeWidth="1.5" />

            {/* Spatial Multi-Scale Pool */}
            <rect x="454" y="50" width="150" height="40" rx="4" fill="#1e293b" stroke="#10b981" strokeWidth="1.5" />
            <text x="529" y="67" fill="#34d399" fontSize="10" fontWeight="bold" textAnchor="middle">Multi-Scale Spatial</text>
            <text x="529" y="81" fill="#94a3b8" fontSize="9" textAnchor="middle">{`3×3 ‖ 7×7 ‖ 5×5 dil 2`}</text>

            <path d="M 604 70 L 634 70" stroke="#64748b" strokeWidth="1.5" />

            {/* Mul 2 */}
            <circle cx="648" cy="70" r="14" fill="#0f172a" stroke="#10b981" strokeWidth="1.5" />
            <text x="648" y="74" fill="#ffffff" fontSize="12" fontWeight="bold" textAnchor="middle">⊙</text>

            <path d="M 662 70 L 690 70" stroke="#64748b" strokeWidth="1.5" />

            {/* Residual Gate × γ */}
            <rect x="690" y="50" width="60" height="40" rx="4" fill="#1e293b" stroke="#f59e0b" strokeWidth="1" />
            <text x="720" y="68" fill="#fbbf24" fontSize="10" fontWeight="bold" textAnchor="middle">× γ</text>
            <text x="720" y="81" fill="#94a3b8" fontSize="8" textAnchor="middle">init 0</text>

            <path d="M 750 70 L 775 70" stroke="#64748b" strokeWidth="1.5" />

            {/* Add Node */}
            <circle cx="790" cy="70" r="14" fill="#0f172a" stroke="#f59e0b" strokeWidth="1.5" />
            <text x="790" y="74" fill="#ffffff" fontSize="13" fontWeight="bold" textAnchor="middle">+</text>

            <path d="M 60 50 L 60 22 L 790 22 L 790 56" stroke="#475569" strokeWidth="1.5" strokeDasharray="3 3" fill="none" />
            <text x="420" y="16" fill="#64748b" fontSize="9" fontFamily="monospace" textAnchor="middle">identity residual shortcut x</text>

            <path d="M 804 70 L 830 70" stroke="#64748b" strokeWidth="1.5" />

            {/* Output */}
            <rect x="830" y="50" width="24" height="40" rx="4" fill="#10b981" />
            <text x="842" y="74" fill="#ffffff" fontSize="9" fontWeight="bold" textAnchor="middle">out</text>
          </svg>
        </div>

        {/* Technical Bullets matching line 35-42 */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '12px', lineHeight: 1.6, color: 'var(--fo-text-subtle)' }}>
          <div>
            <b style={{ color: 'var(--fo-text-main)' }}>• Channel attention (ECA-style):</b> Global average and max pooling share one 1-D convolution across channels. No fully-connected bottleneck, so no dimensionality reduction.
          </div>
          <div>
            <b style={{ color: 'var(--fo-text-main)' }}>• Multi-scale spatial attention:</b> Channel-wise mean and max maps go through <b>3×3</b>, <b>7×7</b> and <b>dilated 5×5 (rate 2, 9×9 receptive field)</b> convolutions in parallel, then a 1×1 fusion and a sigmoid. Small kernels capture thin grass-type weeds; large and dilated kernels capture broad-leaf clumps and their context.
          </div>
          <div>
            <b style={{ color: 'var(--fo-text-main)' }}>• Residual attention gating:</b> <code style={{ background: 'var(--fo-bg-elevated)', padding: '1px 5px', borderRadius: '3px' }}>out = x + γ ⊙ (x ⊙ Mc ⊙ Ms)</code> with a per-channel γ initialised to <b>0</b>. A newly inserted block is an exact identity, so COCO-pretrained weights load unchanged and attention is learned gradually.
          </div>
          <div>
            <b style={{ color: 'var(--fo-text-main)' }}>• Placement:</b> Wraps C2f blocks at layers <b>[4, 6, 8, 15, 18, 21]</b>: backbone P3/P4/P5 stages and neck outputs feeding the Detect head.
          </div>
          <div>
            <b style={{ color: 'var(--fo-text-main)' }}>• Checkpoint compatibility:</b> Converted blocks keep C2f parameter names (cv1, cv2, m) and add att.*, so baseline weights transfer 1:1.
          </div>
        </div>
      </div>

      {/* Reference Architectures Table matching 08_model_information.jpg line 44 */}
      {refArch.length > 0 && (
        <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '18px' }}>
          <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--fo-text-main)', marginBottom: '4px' }}>
            Reference architectures (built live, nc=5, untrained)
          </div>
          <div style={{ fontSize: '11px', color: 'var(--fo-text-subtle)', marginBottom: '12px' }}>
            Parameter and GFLOP cost of adding RMCBAM to YOLOv8s. These describe architecture only, not accuracy.
          </div>

          <table className="fo-json-tree" style={{ width: '100%', color: 'var(--fo-text-main)', borderCollapse: 'collapse', fontSize: '11px' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--fo-border)', textTransform: 'uppercase', color: 'var(--fo-text-muted)' }}>
                <th style={{ padding: '6px 10px', textAlign: 'left' }}>model</th>
                <th style={{ padding: '6px 10px', textAlign: 'left' }}>params (M)</th>
                <th style={{ padding: '6px 10px', textAlign: 'left' }}>GFLOPs@640</th>
                <th style={{ padding: '6px 10px', textAlign: 'left' }}>RMCBAM blocks</th>
                <th style={{ padding: '6px 10px', textAlign: 'left' }}>added params</th>
              </tr>
            </thead>
            <tbody>
              {refArch.map((r: any) => (
                <tr key={r.model} style={{ borderBottom: '1px solid var(--fo-border)' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 700, color: r.model.includes('RMCBAM') ? 'var(--fo-accent-blue)' : 'var(--fo-text-main)' }}>
                    {r.model}
                  </td>
                  <td style={{ padding: '6px 10px', fontFamily: 'var(--font-mono)' }}>{r.params_m}</td>
                  <td style={{ padding: '6px 10px', fontFamily: 'var(--font-mono)' }}>{r.gflops_640}</td>
                  <td style={{ padding: '6px 10px', fontFamily: 'var(--font-mono)' }}>{r.rmcbam_blocks}</td>
                  <td style={{ padding: '6px 10px', fontFamily: 'var(--font-mono)' }}>{r.added_params ? `+${r.added_params.toLocaleString()}` : '0'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Selected Checkpoint Section matching 08b_checkpoint_details.jpg */}
      <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '18px' }}>
        <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--fo-text-main)', marginBottom: '14px' }}>
          Selected Checkpoint: {cp.filename || selectedCheckpoint}
        </div>

        {/* Compatibility alerts matching line 54 */}
        {cp.compatibility_notes && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '16px' }}>
            {cp.compatibility_notes.map(([lvl, msg]: [string, string], idx: number) => {
              const isOk = lvl === 'ok';
              return (
                <div
                  key={idx}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '8px 12px',
                    borderRadius: '4px',
                    fontSize: '12px',
                    background: isOk ? 'rgba(16, 185, 129, 0.1)' : 'rgba(245, 158, 11, 0.1)',
                    border: `1px solid ${isOk ? 'rgba(16, 185, 129, 0.3)' : 'rgba(245, 158, 11, 0.3)'}`,
                    color: isOk ? '#10b981' : '#f59e0b',
                  }}
                >
                  {isOk ? <CheckCircle2 size={14} /> : <AlertTriangle size={14} />}
                  <span>{msg}</span>
                </div>
              );
            })}
          </div>
        )}

        {/* 4 Cards matching line 58 */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px', marginBottom: '16px' }}>
          <div style={{ background: 'var(--fo-bg-elevated)', borderRadius: '4px', padding: '12px' }}>
            <div style={{ fontSize: '10px', color: 'var(--fo-text-muted)', textTransform: 'uppercase' }}>Parameters</div>
            <div style={{ fontSize: '20px', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
              {cp.n_params ? `${(cp.n_params / 1e6).toFixed(2)} M` : '—'}
            </div>
            <div style={{ fontSize: '11px', color: 'var(--fo-text-subtle)' }}>{cp.size_mb} MB on disk</div>
          </div>

          <div style={{ background: 'var(--fo-bg-elevated)', borderRadius: '4px', padding: '12px' }}>
            <div style={{ fontSize: '10px', color: 'var(--fo-text-muted)', textTransform: 'uppercase' }}>GFLOPs @640</div>
            <div style={{ fontSize: '20px', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
              {cp.gflops ? cp.gflops.toFixed(1) : '—'}
            </div>
            <div style={{ fontSize: '11px', color: 'var(--fo-text-subtle)' }}>640×640 input</div>
          </div>

          <div style={{ background: 'var(--fo-bg-elevated)', borderRadius: '4px', padding: '12px' }}>
            <div style={{ fontSize: '10px', color: 'var(--fo-text-muted)', textTransform: 'uppercase' }}>Classes</div>
            <div style={{ fontSize: '20px', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
              {cp.nc || 0}
            </div>
            <div style={{ fontSize: '11px', color: 'var(--fo-text-subtle)' }}>
              {cp.names ? Object.values(cp.names).slice(0, 3).join(', ') : ''}
            </div>
          </div>

          <div style={{ background: 'var(--fo-bg-elevated)', borderRadius: '4px', padding: '12px' }}>
            <div style={{ fontSize: '10px', color: 'var(--fo-text-muted)', textTransform: 'uppercase' }}>Saved</div>
            <div style={{ fontSize: '20px', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
              {(cp.date || '—').slice(0, 10)}
            </div>
            <div style={{ fontSize: '11px', color: 'var(--fo-text-subtle)' }}>
              Ultralytics {cp.ultralytics_version || '?'}
            </div>
          </div>
        </div>

        {/* Learned RMCBAM Residual Gates Table matching line 68 */}
        {cp.rmcbam_gate_mean_abs && Object.keys(cp.rmcbam_gate_mean_abs).length > 0 && (
          <div style={{ marginBottom: '16px' }}>
            <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--fo-text-main)', marginBottom: '8px' }}>
              RMCBAM residual gates (mean |γ| per block, learned values)
            </div>
            <table className="fo-json-tree" style={{ width: '100%', color: 'var(--fo-text-main)', borderCollapse: 'collapse', fontSize: '11px' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--fo-border)', textTransform: 'uppercase', color: 'var(--fo-text-muted)' }}>
                  <th style={{ padding: '6px 10px', textAlign: 'left' }}>block</th>
                  <th style={{ padding: '6px 10px', textAlign: 'left' }}>mean |gamma|</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(cp.rmcbam_gate_mean_abs).map(([block, val]: [string, any]) => (
                  <tr key={block} style={{ borderBottom: '1px solid var(--fo-border)' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{block}</td>
                    <td style={{ padding: '6px 10px', fontFamily: 'var(--font-mono)' }}>{typeof val === 'number' ? val.toExponential(3) : val}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Training Arguments Collapsible JSON */}
        {cp.train_args && (
          <div style={{ marginBottom: '14px' }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                fontSize: '12px',
                fontWeight: 600,
                color: 'var(--fo-text-muted)',
                cursor: 'pointer',
              }}
              onClick={() => setShowTrainArgs(!showTrainArgs)}
            >
              {showTrainArgs ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
              <span>Training arguments stored in checkpoint</span>
            </div>
            {showTrainArgs && (
              <pre className="fo-json-tree" style={{ maxHeight: '160px', overflowY: 'auto', marginTop: '6px' }}>
                {JSON.stringify(cp.train_args, null, 2)}
              </pre>
            )}
          </div>
        )}

        {/* Per-layer Module Inspector Table */}
        {arch.length > 0 && (
          <div style={{ borderTop: '1px solid var(--fo-border)', paddingTop: '14px' }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                cursor: 'pointer',
                marginBottom: '10px',
              }}
              onClick={() => setShowArchTable(!showArchTable)}
            >
              <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--fo-text-main)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Cpu size={14} style={{ color: 'var(--fo-accent-blue)' }} /> DetectionModel Layer Architecture ({arch.length} Layers)
              </div>
              {showArchTable ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            </div>

            {showArchTable && (
              <div style={{ maxHeight: '280px', overflowY: 'auto' }}>
                <table className="fo-json-tree" style={{ width: '100%', color: 'var(--fo-text-main)', borderCollapse: 'collapse', fontSize: '11px' }}>
                  <thead>
                    <tr style={{ position: 'sticky', top: 0, background: 'var(--fo-bg-panel)', borderBottom: '1px solid var(--fo-border)', textTransform: 'uppercase', color: 'var(--fo-text-muted)', zIndex: 1 }}>
                      <th style={{ padding: '6px 8px', textAlign: 'left' }}>layer</th>
                      <th style={{ padding: '6px 8px', textAlign: 'left' }}>from</th>
                      <th style={{ padding: '6px 8px', textAlign: 'left' }}>module</th>
                      <th style={{ padding: '6px 8px', textAlign: 'left' }}>params</th>
                      <th style={{ padding: '6px 8px', textAlign: 'left' }}>status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {arch.map((row: any) => (
                      <tr key={row.layer} style={{ borderBottom: '1px solid var(--fo-border)' }}>
                        <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)', fontWeight: 700 }}>Layer {row.layer}</td>
                        <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)' }}>{row.from}</td>
                        <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)' }}>{row.module}</td>
                        <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)' }}>{row.params?.toLocaleString()}</td>
                        <td style={{ padding: '6px 8px' }}>
                          <span style={{ color: row.rmcbam ? 'var(--fo-accent-blue)' : 'var(--fo-text-muted)', fontWeight: 600 }}>
                            {row.rmcbam ? 'C2f_RMCBAM' : 'Standard'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
