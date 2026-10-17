import React, { useEffect, useState } from 'react';
import { getSettings, updateSettings, clearBackendCaches } from '../services/api';
import type { SystemSettings } from '../types';
import {
  Settings as SettingsIcon,
  Save,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  Trash2,
  Terminal,
} from 'lucide-react';

export const SettingsWorkspace: React.FC = () => {
  const [settings, setSettingsData] = useState<SystemSettings | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [saving, setSaving] = useState<boolean>(false);
  const [statusMsg, setStatusMsg] = useState<{ text: string; type: 'success' | 'error' } | null>(null);

  // Form local state
  const [modelsDir, setModelsDir] = useState<string>('models');
  const [runsDir, setRunsDir] = useState<string>('runs');
  const [datasetDir, setDatasetDir] = useState<string>('data');
  const [device, setDevice] = useState<string>('auto');
  const [imgsz, setImgsz] = useState<number>(640);
  const [conf, setConf] = useState<number>(0.25);
  const [iou, setIou] = useState<number>(0.45);
  const [maxDet, setMaxDet] = useState<number>(300);
  const [weedClass, setWeedClass] = useState<string>('Weed');
  const [cropClasses, setCropClasses] = useState<string[]>(['Tomato', 'Rose', 'Marigold', 'Cauliflower']);
  const [sevLow, setSevLow] = useState<number>(5.0);
  const [sevMod, setSevMod] = useState<number>(15.0);
  const [sevHigh, setSevHigh] = useState<number>(30.0);
  const [proximityFrac, setProximityFrac] = useState<number>(0.15);
  const [gsd, setGsd] = useState<number>(0.0);
  const [gradcamLayers, setGradcamLayers] = useState<string>('15,18,21');

  useEffect(() => {
    fetchSettings();
  }, []);

  const fetchSettings = async () => {
    try {
      const data = await getSettings();
      setSettingsData(data);
      setModelsDir(data.models_dir || 'models');
      setRunsDir(data.runs_dir || 'runs');
      setDatasetDir(data.dataset_dir || 'data');
      setDevice(data.device || 'auto');
      setImgsz(data.imgsz || 640);
      setConf(data.conf || 0.25);
      setIou(data.iou || 0.45);
      setMaxDet(data.max_det || 300);
      setWeedClass(data.weed_class || 'Weed');
      setCropClasses(data.crop_classes || ['Tomato', 'Rose', 'Marigold', 'Cauliflower']);
      setSevLow(data.severity_low !== undefined ? data.severity_low : 5.0);
      setSevMod(data.severity_moderate !== undefined ? data.severity_moderate : 15.0);
      setSevHigh(data.severity_high !== undefined ? data.severity_high : 30.0);
      setProximityFrac(data.proximity_frac !== undefined ? data.proximity_frac : 0.15);
      setGsd(data.gsd_cm_per_px !== undefined ? data.gsd_cm_per_px : 0.0);
      setGradcamLayers((data.gradcam_layers || [15, 18, 21]).join(','));
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const handleApplySave = async (persist: boolean) => {
    if (!(sevLow <= sevMod && sevMod <= sevHigh)) {
      setStatusMsg({ text: 'Severity thresholds must be increasing (Low ≤ Moderate ≤ High).', type: 'error' });
      return;
    }

    setSaving(true);
    setStatusMsg(null);
    try {
      const parsedLayers = gradcamLayers
        .split(',')
        .map((x) => parseInt(x.trim()))
        .filter((x) => !isNaN(x));

      const payload: Partial<SystemSettings> = {
        models_dir: modelsDir,
        runs_dir: runsDir,
        dataset_dir: datasetDir,
        device,
        imgsz,
        conf,
        iou,
        max_det: maxDet,
        weed_class: weedClass,
        crop_classes: cropClasses,
        severity_low: sevLow,
        severity_moderate: sevMod,
        severity_high: sevHigh,
        proximity_frac: proximityFrac,
        gsd_cm_per_px: gsd,
        gradcam_layers: parsedLayers,
      };

      await updateSettings(payload);
      setStatusMsg({
        text: persist
          ? 'Settings applied and saved to settings.json successfully.'
          : 'Settings applied for this session.',
        type: 'success',
      });
      setTimeout(() => setStatusMsg(null), 4000);
    } catch (e: any) {
      setStatusMsg({ text: e.message || 'Failed to save settings.', type: 'error' });
    } finally {
      setSaving(false);
    }
  };

  const handleClearCache = async () => {
    try {
      await clearBackendCaches();
      setStatusMsg({ text: 'Model memory and GPU caches cleared.', type: 'success' });
      setTimeout(() => setStatusMsg(null), 3000);
    } catch (e: any) {
      setStatusMsg({ text: 'Failed to clear cache.', type: 'error' });
    }
  };

  const handleClearSession = () => {
    sessionStorage.clear();
    localStorage.removeItem('wv_predict_data');
    setStatusMsg({ text: 'Session results and temporary data cleared.', type: 'success' });
    setTimeout(() => setStatusMsg(null), 3000);
  };

  if (loading || !settings) {
    return (
      <div className="fo-canvas-workspace" style={{ padding: '48px', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ fontSize: '13px', color: 'var(--fo-text-subtle)' }}>Loading Settings...</div>
      </div>
    );
  }

  const env = settings.environment || {
    python: '3.10+',
    torch: '2.0+',
    ultralytics: '8.3+',
    cuda: false,
    cuda_device_name: '',
    device_used: 'cpu',
  };

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
            <SettingsIcon size={18} style={{ color: 'var(--fo-accent-green)' }} /> Settings
          </h2>
          <div style={{ fontSize: '12px', color: 'var(--fo-text-subtle)', marginTop: '4px' }}>
            Paths, inference defaults, analysis thresholds and compute device.
          </div>
        </div>

        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            className="fo-btn fo-btn-secondary"
            onClick={() => handleApplySave(false)}
            disabled={saving}
            style={{ fontSize: '12px', padding: '6px 14px' }}
          >
            Apply
          </button>
          <button
            className="fo-btn fo-btn-primary"
            onClick={() => handleApplySave(true)}
            disabled={saving}
            style={{ fontSize: '12px', padding: '6px 14px', display: 'flex', alignItems: 'center', gap: '6px' }}
          >
            <Save size={13} /> {saving ? 'Saving...' : 'Apply & save as default'}
          </button>
        </div>
      </div>

      {statusMsg && (
        <div
          style={{
            background: statusMsg.type === 'success' ? '#f0fdf4' : '#fef2f2',
            border: `1px solid ${statusMsg.type === 'success' ? '#bbf7d0' : '#fecaca'}`,
            color: statusMsg.type === 'success' ? '#166534' : '#991b1b',
            padding: '10px 14px',
            borderRadius: '6px',
            fontSize: '12px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}
        >
          {statusMsg.type === 'success' ? <CheckCircle2 size={14} /> : <AlertCircle size={14} />}
          <span>{statusMsg.text}</span>
        </div>
      )}

      {/* SECTION 1: Paths */}
      <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '18px' }}>
        <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--fo-text-main)', marginBottom: '14px' }}>
          Paths
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '14px' }}>
          <div>
            <label style={{ fontSize: '11px', fontWeight: 600, color: 'var(--fo-text-subtle)', marginBottom: '4px', display: 'block' }}>
              Models directory
            </label>
            <input
              type="text"
              value={modelsDir}
              onChange={(e) => setModelsDir(e.target.value)}
              style={{ width: '100%', background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', color: 'var(--fo-text-main)', padding: '6px 10px', borderRadius: '4px', fontSize: '12px' }}
            />
          </div>
          <div>
            <label style={{ fontSize: '11px', fontWeight: 600, color: 'var(--fo-text-subtle)', marginBottom: '4px', display: 'block' }}>
              Runs directory (Ultralytics)
            </label>
            <input
              type="text"
              value={runsDir}
              onChange={(e) => setRunsDir(e.target.value)}
              style={{ width: '100%', background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', color: 'var(--fo-text-main)', padding: '6px 10px', borderRadius: '4px', fontSize: '12px' }}
            />
          </div>
          <div>
            <label style={{ fontSize: '11px', fontWeight: 600, color: 'var(--fo-text-subtle)', marginBottom: '4px', display: 'block' }}>
              Dataset directory
            </label>
            <input
              type="text"
              value={datasetDir}
              onChange={(e) => setDatasetDir(e.target.value)}
              style={{ width: '100%', background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', color: 'var(--fo-text-main)', padding: '6px 10px', borderRadius: '4px', fontSize: '12px' }}
            />
          </div>
        </div>
      </div>

      {/* SECTION 2: Inference */}
      <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '18px' }}>
        <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--fo-text-main)', marginBottom: '14px' }}>
          Inference Defaults
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '14px' }}>
          <div>
            <label style={{ fontSize: '11px', fontWeight: 600, color: 'var(--fo-text-subtle)', marginBottom: '4px', display: 'block' }}>
              Device
            </label>
            <select
              value={device}
              onChange={(e) => setDevice(e.target.value)}
              style={{ width: '100%', background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', color: 'var(--fo-text-main)', padding: '6px 10px', borderRadius: '4px', fontSize: '12px' }}
            >
              <option value="auto">auto</option>
              <option value="cpu">cpu</option>
              <option value="cuda:0">cuda:0</option>
            </select>
          </div>

          <div>
            <label style={{ fontSize: '11px', fontWeight: 600, color: 'var(--fo-text-subtle)', marginBottom: '4px', display: 'block' }}>
              Image size
            </label>
            <select
              value={imgsz}
              onChange={(e) => setImgsz(parseInt(e.target.value))}
              style={{ width: '100%', background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', color: 'var(--fo-text-main)', padding: '6px 10px', borderRadius: '4px', fontSize: '12px' }}
            >
              {[320, 416, 512, 640, 768, 960, 1280].map((sz) => (
                <option key={sz} value={sz}>{sz}px</option>
              ))}
            </select>
          </div>

          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', fontWeight: 600, color: 'var(--fo-text-subtle)', marginBottom: '4px' }}>
              <span>Confidence</span>
              <span style={{ fontFamily: 'var(--font-mono)' }}>{conf.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.01"
              max="0.95"
              step="0.01"
              value={conf}
              onChange={(e) => setConf(parseFloat(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--fo-accent-green)', cursor: 'pointer' }}
            />
          </div>

          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', fontWeight: 600, color: 'var(--fo-text-subtle)', marginBottom: '4px' }}>
              <span>NMS IoU</span>
              <span style={{ fontFamily: 'var(--font-mono)' }}>{iou.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.10"
              max="0.95"
              step="0.01"
              value={iou}
              onChange={(e) => setIou(parseFloat(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--fo-accent-green)', cursor: 'pointer' }}
            />
          </div>

          <div>
            <label style={{ fontSize: '11px', fontWeight: 600, color: 'var(--fo-text-subtle)', marginBottom: '4px', display: 'block' }}>
              Max detections
            </label>
            <input
              type="number"
              min="1"
              max="3000"
              value={maxDet}
              onChange={(e) => setMaxDet(parseInt(e.target.value) || 300)}
              style={{ width: '100%', background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', color: 'var(--fo-text-main)', padding: '6px 10px', borderRadius: '4px', fontSize: '12px' }}
            />
          </div>
        </div>
      </div>

      {/* SECTION 3: Classes */}
      <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '18px' }}>
        <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--fo-text-main)', marginBottom: '14px' }}>
          Classes
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '14px' }}>
          <div>
            <label style={{ fontSize: '11px', fontWeight: 600, color: 'var(--fo-text-subtle)', marginBottom: '4px', display: 'block' }}>
              Weed class name
            </label>
            <input
              type="text"
              value={weedClass}
              onChange={(e) => setWeedClass(e.target.value)}
              style={{ width: '100%', background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', color: 'var(--fo-text-main)', padding: '6px 10px', borderRadius: '4px', fontSize: '12px' }}
            />
          </div>
          <div>
            <label style={{ fontSize: '11px', fontWeight: 600, color: 'var(--fo-text-subtle)', marginBottom: '4px', display: 'block' }}>
              Crop classes (comma-separated)
            </label>
            <input
              type="text"
              value={cropClasses.join(', ')}
              onChange={(e) => setCropClasses(e.target.value.split(',').map((x) => x.trim()).filter(Boolean))}
              style={{ width: '100%', background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', color: 'var(--fo-text-main)', padding: '6px 10px', borderRadius: '4px', fontSize: '12px' }}
            />
          </div>
        </div>
      </div>

      {/* SECTION 4: Severity Thresholds */}
      <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '18px' }}>
        <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--fo-text-main)', marginBottom: '4px' }}>
          Severity thresholds (estimated infestation %)
        </div>
        <div style={{ fontSize: '11px', color: 'var(--fo-text-subtle)', marginBottom: '14px' }}>
          Defines the infestation boundaries for Low, Moderate, High and Severe warnings.
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '14px' }}>
          <div>
            <label style={{ fontSize: '11px', fontWeight: 600, color: 'var(--fo-text-subtle)', marginBottom: '4px', display: 'block' }}>
              Low → Moderate at (%)
            </label>
            <input
              type="number"
              step="0.5"
              value={sevLow}
              onChange={(e) => setSevLow(parseFloat(e.target.value) || 0)}
              style={{ width: '100%', background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', color: 'var(--fo-text-main)', padding: '6px 10px', borderRadius: '4px', fontSize: '12px' }}
            />
          </div>
          <div>
            <label style={{ fontSize: '11px', fontWeight: 600, color: 'var(--fo-text-subtle)', marginBottom: '4px', display: 'block' }}>
              Moderate → High at (%)
            </label>
            <input
              type="number"
              step="0.5"
              value={sevMod}
              onChange={(e) => setSevMod(parseFloat(e.target.value) || 0)}
              style={{ width: '100%', background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', color: 'var(--fo-text-main)', padding: '6px 10px', borderRadius: '4px', fontSize: '12px' }}
            />
          </div>
          <div>
            <label style={{ fontSize: '11px', fontWeight: 600, color: 'var(--fo-text-subtle)', marginBottom: '4px', display: 'block' }}>
              High → Severe at (%)
            </label>
            <input
              type="number"
              step="0.5"
              value={sevHigh}
              onChange={(e) => setSevHigh(parseFloat(e.target.value) || 0)}
              style={{ width: '100%', background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', color: 'var(--fo-text-main)', padding: '6px 10px', borderRadius: '4px', fontSize: '12px' }}
            />
          </div>
        </div>
      </div>

      {/* SECTION 5: Analysis */}
      <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '18px' }}>
        <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--fo-text-main)', marginBottom: '14px' }}>
          Analysis Parameters
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '14px' }}>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', fontWeight: 600, color: 'var(--fo-text-subtle)', marginBottom: '4px' }}>
              <span>Crop association radius</span>
              <span style={{ fontFamily: 'var(--font-mono)' }}>{(proximityFrac * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0.01"
              max="0.5"
              step="0.01"
              value={proximityFrac}
              onChange={(e) => setProximityFrac(parseFloat(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--fo-accent-green)', cursor: 'pointer' }}
            />
          </div>

          <div>
            <label style={{ fontSize: '11px', fontWeight: 600, color: 'var(--fo-text-subtle)', marginBottom: '4px', display: 'block' }}>
              Ground sampling distance (cm/px, 0 = unknown)
            </label>
            <input
              type="number"
              step="0.01"
              value={gsd}
              onChange={(e) => setGsd(parseFloat(e.target.value) || 0)}
              style={{ width: '100%', background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', color: 'var(--fo-text-main)', padding: '6px 10px', borderRadius: '4px', fontSize: '12px' }}
            />
          </div>

          <div>
            <label style={{ fontSize: '11px', fontWeight: 600, color: 'var(--fo-text-subtle)', marginBottom: '4px', display: 'block' }}>
              Grad-CAM default layers
            </label>
            <input
              type="text"
              value={gradcamLayers}
              onChange={(e) => setGradcamLayers(e.target.value)}
              style={{ width: '100%', background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', color: 'var(--fo-text-main)', padding: '6px 10px', borderRadius: '4px', fontSize: '12px' }}
            />
          </div>
        </div>
      </div>

      {/* SECTION 6: Environment Readout */}
      <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '18px' }}>
        <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--fo-text-main)', marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Terminal size={14} /> Execution Environment
        </div>

        <div style={{ background: 'var(--fo-bg-canvas)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '14px 18px', fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--fo-text-main)', lineHeight: 1.8 }}>
          <div><span style={{ color: 'var(--fo-text-muted)' }}>python:</span> &nbsp; &nbsp; &nbsp; {env.python}</div>
          <div><span style={{ color: 'var(--fo-text-muted)' }}>torch:</span> &nbsp; &nbsp; &nbsp; &nbsp;{env.torch}</div>
          <div><span style={{ color: 'var(--fo-text-muted)' }}>ultralytics:</span> &nbsp;{env.ultralytics}</div>
          <div><span style={{ color: 'var(--fo-text-muted)' }}>CUDA:</span> &nbsp; &nbsp; &nbsp; &nbsp; {env.cuda ? `True (${env.cuda_device_name})` : 'False (CPU)'}</div>
          <div><span style={{ color: 'var(--fo-text-muted)' }}>device used:</span> &nbsp;{env.device_used}</div>
        </div>

        {/* Clear Caches and Session buttons */}
        <div style={{ display: 'flex', gap: '10px', marginTop: '16px' }}>
          <button className="fo-btn fo-btn-secondary" onClick={handleClearCache} style={{ fontSize: '11px', padding: '6px 14px', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <RefreshCw size={13} /> Clear cached models &amp; data
          </button>
          <button className="fo-btn fo-btn-secondary" onClick={handleClearSession} style={{ fontSize: '11px', padding: '6px 14px', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Trash2 size={13} /> Clear session results
          </button>
        </div>
      </div>
    </div>
  );
};
