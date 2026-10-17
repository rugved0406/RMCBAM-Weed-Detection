import React, { useState, useMemo } from 'react';
import type { ExplainResponse, PredictResponse } from '../types';
import { runExplain } from '../services/api';
import {
  Flame,
  Zap,
  AlertCircle,
  Download,
  BarChart2,
} from 'lucide-react';

interface Props {
  selectedFile: File | null;
  selectedCheckpoint: string;
  predictData: PredictResponse | null;
  onLoadSample?: (name?: string) => Promise<void>;
  onNavigate?: (ws: any) => void;
}

export const ExplainabilityWorkspace: React.FC<Props> = ({
  selectedFile,
  selectedCheckpoint,
  predictData,
  onLoadSample,
  onNavigate,
}) => {
  const [method, setMethod] = useState<'hirescam' | 'gradcam'>('hirescam');
  const [selectedLayers, setSelectedLayers] = useState<number[]>([15, 18, 21]);
  const [alpha, setAlpha] = useState<number>(0.45);
  const [iouMatch, setIouMatch] = useState<number>(0.5);
  const [selectedUids, setSelectedUids] = useState<string[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [explainData, setExplainData] = useState<ExplainResponse | null>(null);

  // Layer selected for channel attention weights bar chart
  const [selectedChannelLayer, setSelectedChannelLayer] = useState<number | null>(null);

  const weeds = useMemo(() => {
    if (!predictData) return [];
    return predictData.detections.filter(
      (d) => d.cls_name.toLowerCase().includes('weed') || d.cls_id === 0
    );
  }, [predictData]);

  // Set default selected weeds (top 6 by confidence)
  React.useEffect(() => {
    if (weeds.length > 0 && selectedUids.length === 0) {
      setSelectedUids(weeds.slice(0, 6).map((w) => w.uid));
    }
  }, [weeds]);

  // Set default channel layer when attention maps load
  React.useEffect(() => {
    if (explainData?.rmcbam_maps && explainData.rmcbam_maps.length > 0 && selectedChannelLayer === null) {
      setSelectedChannelLayer(explainData.rmcbam_maps[0].layer);
    }
  }, [explainData, selectedChannelLayer]);

  const resolveTargetFile = (): File | null => {
    if (selectedFile) return selectedFile;
    if (predictData?.original_image) {
      try {
        const arr = predictData.original_image.split(',');
        const mime = arr[0].match(/:(.*?);/)?.[1] || 'image/png';
        const bstr = atob(arr[1]);
        let n = bstr.length;
        const u8arr = new Uint8Array(n);
        while (n--) {
          u8arr[n] = bstr.charCodeAt(n);
        }
        return new File([u8arr], predictData.filename || 'image.png', { type: mime });
      } catch (e) {
        console.error('Failed to convert base64 image to File:', e);
      }
    }
    return null;
  };

  const handleRunExplain = async () => {
    const file = resolveTargetFile();
    if (!file) {
      setError('No image available to explain.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const layersStr = selectedLayers.join(',');
      const uidsStr = selectedUids.join(',');
      const res = await runExplain(
        file,
        selectedCheckpoint,
        method,
        layersStr,
        alpha,
        iouMatch,
        uidsStr,
        predictData?.params?.conf ?? 0.15,
        predictData?.params?.iou ?? 0.45,
        predictData?.params?.imgsz ?? 640,
        predictData?.params?.device ?? 'cpu'
      );
      setExplainData(res);
      if (res.rmcbam_maps && res.rmcbam_maps.length > 0) {
        setSelectedChannelLayer(res.rmcbam_maps[0].layer);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Explainability computation failed.');
    } finally {
      setLoading(false);
    }
  };

  const toggleLayer = (layer: number) => {
    setSelectedLayers((prev) =>
      prev.includes(layer) ? prev.filter((l) => l !== layer) : [...prev, layer].sort((a, b) => a - b)
    );
  };

  const toggleUid = (uid: string) => {
    setSelectedUids((prev) =>
      prev.includes(uid) ? prev.filter((u) => u !== uid) : [...prev, uid]
    );
  };

  const handleDownloadCombined = () => {
    if (!explainData?.combined_overlay) return;
    const a = document.createElement('a');
    a.href = explainData.combined_overlay;
    a.download = `${predictData?.filename.replace(/\.[^/.]+$/, '') || 'weed'}_gradcam_combined.png`;
    a.click();
  };

  if (!predictData) {
    return (
      <div className="fo-canvas-workspace" style={{ padding: '48px', alignItems: 'center', justifyContent: 'center' }}>
        <Flame size={40} style={{ color: 'var(--fo-text-muted)', marginBottom: '12px' }} />
        <div style={{ fontSize: '15px', fontWeight: 600 }}>No Image Selected for Explainability</div>
        <div style={{ fontSize: '12px', color: 'var(--fo-text-subtle)', marginTop: '4px', textAlign: 'center', maxWidth: '420px' }}>
          Select or upload an image in <b>Image Detection</b> workspace first, then compute gradient heatmaps.
        </div>
        <div style={{ display: 'flex', gap: '10px', marginTop: '16px' }}>
          {onLoadSample && (
            <button
              className="fo-btn fo-btn-primary"
              style={{ fontSize: '12px', padding: '6px 14px' }}
              onClick={() => onLoadSample('img112.jpg')}
            >
              Analyze Sample img112.jpg
            </button>
          )}
          {onNavigate && (
            <button
              className="fo-btn fo-btn-secondary"
              style={{ fontSize: '12px', padding: '6px 14px' }}
              onClick={() => onNavigate('inspection')}
            >
              Go to Image Detection
            </button>
          )}
        </div>
      </div>
    );
  }

  // Active channel attention map
  const activeChannelMap = explainData?.rmcbam_maps.find((m) => m.layer === selectedChannelLayer);
  const sortedChannels = activeChannelMap
    ? activeChannelMap.channel_weights
        .map((w, idx) => ({ channel: idx, weight: w }))
        .sort((a, b) => b.weight - a.weight)
    : [];

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
            <Flame size={18} style={{ color: '#f97316' }} /> Weed Explainability
          </h2>
          <div style={{ fontSize: '12px', color: 'var(--fo-text-subtle)', marginTop: '4px' }}>
            Grad-CAM / HiResCAM from the model's real gradients for each detected weed, plus RMCBAM attention maps.
          </div>
        </div>

        <button
          className="fo-btn fo-btn-primary"
          onClick={handleRunExplain}
          disabled={loading || selectedLayers.length === 0}
          style={{ padding: '8px 18px', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}
        >
          <Zap size={14} /> {loading ? 'Backpropagating Gradients...' : 'Generate Explanations'}
        </button>
      </div>

      {/* Interactive Controls Bar matching 03_explainability_hirescam.jpg */}
      <div
        style={{
          background: 'var(--fo-bg-panel)',
          border: '1px solid var(--fo-border)',
          borderRadius: '6px',
          padding: '14px 18px',
          display: 'grid',
          gridTemplateColumns: '1.2fr 2fr 1.2fr 1.2fr',
          gap: '16px',
          alignItems: 'center',
        }}
      >
        {/* Method */}
        <div>
          <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--fo-text-muted)', marginBottom: '6px', textTransform: 'uppercase' }}>
            Method
          </div>
          <div style={{ display: 'flex', gap: '10px' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', cursor: 'pointer' }}>
              <input
                type="radio"
                name="method"
                checked={method === 'hirescam'}
                onChange={() => setMethod('hirescam')}
              />
              HiResCAM
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', cursor: 'pointer' }}>
              <input
                type="radio"
                name="method"
                checked={method === 'gradcam'}
                onChange={() => setMethod('gradcam')}
              />
              Grad-CAM
            </label>
          </div>
        </div>

        {/* Target Layers */}
        <div>
          <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--fo-text-muted)', marginBottom: '6px', textTransform: 'uppercase' }}>
            Target Layers (neck outputs 15, 18, 21 feed head)
          </div>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            {[15, 18, 21].map((l) => (
              <button
                key={l}
                onClick={() => toggleLayer(l)}
                style={{
                  padding: '2px 8px',
                  borderRadius: '4px',
                  fontSize: '11px',
                  fontFamily: 'var(--font-mono)',
                  cursor: 'pointer',
                  background: selectedLayers.includes(l) ? 'rgba(59, 130, 246, 0.2)' : 'var(--fo-bg-elevated)',
                  color: selectedLayers.includes(l) ? 'var(--fo-accent-blue)' : 'var(--fo-text-muted)',
                  border: `1px solid ${selectedLayers.includes(l) ? 'var(--fo-accent-blue)' : 'var(--fo-border)'}`,
                }}
              >
                Layer {l}
              </button>
            ))}
          </div>
        </div>

        {/* Overlay Opacity */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', fontWeight: 700, color: 'var(--fo-text-muted)', marginBottom: '6px', textTransform: 'uppercase' }}>
            <span>Opacity</span>
            <span style={{ fontFamily: 'var(--font-mono)' }}>{alpha.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="0.1"
            max="0.9"
            step="0.05"
            value={alpha}
            onChange={(e) => setAlpha(parseFloat(e.target.value))}
            style={{ width: '100%', accentColor: 'var(--fo-accent-blue)', cursor: 'pointer' }}
          />
        </div>

        {/* Anchor IoU Match */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', fontWeight: 700, color: 'var(--fo-text-muted)', marginBottom: '6px', textTransform: 'uppercase' }}>
            <span>Anchor IoU</span>
            <span style={{ fontFamily: 'var(--font-mono)' }}>{iouMatch.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="0.3"
            max="0.9"
            step="0.05"
            value={iouMatch}
            onChange={(e) => setIouMatch(parseFloat(e.target.value))}
            style={{ width: '100%', accentColor: 'var(--fo-accent-blue)', cursor: 'pointer' }}
          />
        </div>
      </div>

      {/* Target Weeds Multi-Select */}
      {weeds.length > 0 && (
        <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '12px 16px' }}>
          <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--fo-text-muted)', textTransform: 'uppercase', marginBottom: '8px' }}>
            Weed Detections to Explain ({selectedUids.length} of {weeds.length} selected)
          </div>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            {weeds.map((w) => {
              const isChecked = selectedUids.includes(w.uid);
              return (
                <button
                  key={w.uid}
                  onClick={() => toggleUid(w.uid)}
                  style={{
                    padding: '3px 10px',
                    borderRadius: '4px',
                    fontSize: '11px',
                    cursor: 'pointer',
                    background: isChecked ? 'rgba(59, 130, 246, 0.15)' : 'var(--fo-bg-elevated)',
                    color: isChecked ? 'var(--fo-accent-blue)' : 'var(--fo-text-subtle)',
                    border: `1px solid ${isChecked ? 'var(--fo-accent-blue)' : 'var(--fo-border)'}`,
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px',
                    fontWeight: 600,
                  }}
                >
                  <input type="checkbox" checked={isChecked} readOnly style={{ pointerEvents: 'none' }} />
                  {w.uid} ({(w.conf * 100).toFixed(0)}%)
                </button>
              );
            })}
          </div>
        </div>
      )}

      {error && (
        <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', color: '#ef4444', padding: '10px 14px', borderRadius: '4px', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <AlertCircle size={14} /> {error}
        </div>
      )}

      {/* Combined Heatmap Side-by-Side View matching 03_explainability_hirescam.jpg */}
      {explainData?.combined_overlay && (
        <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '18px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
            <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--fo-text-main)' }}>
              Target Heatmap Comparison
            </div>
            <button className="fo-btn fo-btn-secondary" onClick={handleDownloadCombined} style={{ fontSize: '11px', padding: '4px 10px' }}>
              <Download size={13} /> Download Combined Heatmap (PNG)
            </button>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
            <div>
              <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--fo-text-muted)', marginBottom: '8px' }}>
                Explained Detections
              </div>
              <img
                src={predictData.annotated_image}
                alt="Explained detections"
                style={{ width: '100%', borderRadius: '4px', border: '1px solid var(--fo-border)', objectFit: 'contain' }}
              />
            </div>

            <div>
              <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--fo-accent-blue)', marginBottom: '8px' }}>
                {explainData.method === 'gradcam' ? 'Grad-CAM' : 'HiResCAM'} — combined, layers [{explainData.layers.join(', ')}]
              </div>
              <img
                src={explainData.combined_overlay}
                alt="CAM Overlay"
                style={{ width: '100%', borderRadius: '4px', border: '1px solid var(--fo-border)', objectFit: 'contain' }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Per-Weed Heatmaps matching 03b_per_weed_heatmaps.jpg */}
      {explainData?.per_target_crops && explainData.per_target_crops.length > 0 && (
        <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '18px' }}>
          <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--fo-text-main)' }}>
            Per-Weed Heatmaps
          </div>
          <div style={{ fontSize: '12px', color: 'var(--fo-text-subtle)', margin: '4px 0 14px' }}>
            Energy-in-box = share of the heatmap's total mass that falls inside that weed's box (higher → the evidence is localised on the weed).
          </div>

          {/* 3-Column Crops Gallery */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '14px', marginBottom: '16px' }}>
            {explainData.per_target_crops.map((c) => (
              <div
                key={c.uid}
                style={{
                  background: 'var(--fo-bg-elevated)',
                  border: '1px solid var(--fo-border)',
                  borderRadius: '6px',
                  padding: '10px',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                }}
              >
                <img
                  src={c.crop_image}
                  alt={c.uid}
                  style={{ width: '100%', maxHeight: '180px', objectFit: 'contain', borderRadius: '4px', marginBottom: '8px' }}
                />
                <div style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--fo-text-main)', textAlign: 'center' }}>
                  <b>{c.uid}</b> · conf {c.conf.toFixed(2)} · energy-in-box {c.energy_in_box.toFixed(2)}
                </div>
              </div>
            ))}
          </div>

          {/* Per-Weed Metrics Table */}
          <table className="fo-json-tree" style={{ width: '100%', color: 'var(--fo-text-main)', borderCollapse: 'collapse', fontSize: '11px' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--fo-border)', textTransform: 'uppercase', color: 'var(--fo-text-muted)' }}>
                <th style={{ padding: '6px 10px', textAlign: 'left' }}>weed_id</th>
                <th style={{ padding: '6px 10px', textAlign: 'left' }}>conf</th>
                <th style={{ padding: '6px 10px', textAlign: 'left' }}>energy_in_box</th>
                <th style={{ padding: '6px 10px', textAlign: 'left' }}>anchors_matched</th>
              </tr>
            </thead>
            <tbody>
              {explainData.per_target_crops.map((c) => (
                <tr key={c.uid} style={{ borderBottom: '1px solid var(--fo-border)' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 700, color: 'var(--fo-accent-blue)' }}>{c.uid}</td>
                  <td style={{ padding: '6px 10px', fontFamily: 'var(--font-mono)' }}>{c.conf.toFixed(3)}</td>
                  <td style={{ padding: '6px 10px', fontFamily: 'var(--font-mono)' }}>{c.energy_in_box.toFixed(3)}</td>
                  <td style={{ padding: '6px 10px', fontFamily: 'var(--font-mono)' }}>{c.anchors_matched || 1}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* RMCBAM Attention Maps matching 03c_rmcbam_attention.jpg */}
      {explainData?.rmcbam_maps && explainData.rmcbam_maps.length > 0 && (
        <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '18px' }}>
          <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--fo-text-main)' }}>
            RMCBAM Attention Maps
          </div>
          <div style={{ fontSize: '12px', color: 'var(--fo-text-subtle)', margin: '4px 0 14px', lineHeight: 1.5 }}>
            Spatial maps are the real sigmoid outputs of each block's multi-scale spatial attention (min–max normalised for display). γ is the learned residual gate: the block's output is x + γ·(attended x), so |γ|≈0 means that block barely changes the features.
          </div>

          {/* Spatial Overlays Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '14px', marginBottom: '20px' }}>
            {explainData.rmcbam_maps.map((m) => (
              <div
                key={m.layer}
                style={{
                  background: 'var(--fo-bg-elevated)',
                  border: '1px solid var(--fo-border)',
                  borderRadius: '6px',
                  padding: '10px',
                }}
              >
                <img
                  src={m.spatial_overlay}
                  alt={`Layer ${m.layer}`}
                  style={{ width: '100%', borderRadius: '4px', marginBottom: '6px' }}
                />
                <div style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--fo-text-subtle)', lineHeight: 1.4 }}>
                  layer {m.layer} · {m.feat_hw[0]}×{m.feat_hw[1]} · range {m.spatial_min.toFixed(2)}–{m.spatial_max.toFixed(2)} · mean|γ| {m.gamma_mean_abs.toExponential(2)}
                </div>
              </div>
            ))}
          </div>

          {/* Channel Attention Weights Interactive Bar Chart */}
          <div style={{ borderTop: '1px solid var(--fo-border)', paddingTop: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
              <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--fo-text-main)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <BarChart2 size={14} style={{ color: 'var(--fo-accent-blue)' }} /> Channel attention of layer:
                <select
                  style={{
                    background: 'var(--fo-bg-elevated)',
                    border: '1px solid var(--fo-border)',
                    color: 'var(--fo-text-main)',
                    padding: '3px 8px',
                    borderRadius: '4px',
                    fontSize: '11px',
                    marginLeft: '4px',
                  }}
                  value={selectedChannelLayer || ''}
                  onChange={(e) => setSelectedChannelLayer(parseInt(e.target.value))}
                >
                  {explainData.rmcbam_maps.map((m) => (
                    <option key={m.layer} value={m.layer}>
                      Layer {m.layer} ({m.name})
                    </option>
                  ))}
                </select>
              </div>

              <span style={{ fontSize: '11px', color: 'var(--fo-text-muted)' }}>
                ECA channel weights (sorted by rank)
              </span>
            </div>

            {/* Sorted Bar Chart Visualizer */}
            {sortedChannels.length > 0 && (
              <div
                style={{
                  height: '160px',
                  display: 'flex',
                  alignItems: 'flex-end',
                  gap: '1px',
                  padding: '10px 0',
                  borderBottom: '1px solid var(--fo-border)',
                }}
              >
                {sortedChannels.map((c, i) => (
                  <div
                    key={i}
                    title={`Channel ${c.channel}: Weight ${c.weight.toFixed(4)}`}
                    style={{
                      flex: 1,
                      height: `${Math.max(4, c.weight * 100)}%`,
                      backgroundColor: '#3987e5',
                      borderRadius: '1px 1px 0 0',
                      opacity: 0.85,
                      cursor: 'pointer',
                    }}
                  />
                ))}
              </div>
            )}
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--fo-text-muted)', marginTop: '4px' }}>
              <span>Rank 1 (highest weight)</span>
              <span>{sortedChannels.length} total channels</span>
              <span>Rank {sortedChannels.length} (lowest weight)</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
