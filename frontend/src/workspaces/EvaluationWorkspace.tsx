import React, { useEffect, useState, useMemo } from 'react';
import type { CheckpointInfo } from '../types';
import {
  getComparison,
  runLiveValidation,
  benchmarkCheckpoints,
} from '../services/api';
import {
  BarChart2,
  RefreshCw,
  Activity,
  Play,
} from 'lucide-react';

interface Props {
  checkpoints: CheckpointInfo[];
}

type TabType = 'benchmarks' | 'validation' | 'latency' | 'architecture';

export const EvaluationWorkspace: React.FC<Props> = ({ checkpoints }) => {
  const [activeTab, setActiveTab] = useState<TabType>('benchmarks');
  const [comparisonData, setComparisonData] = useState<any | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  // Tab 1: Run selection
  const [selectedBaselineRun, setSelectedBaselineRun] = useState<string>('');
  const [selectedRmcbamRun, setSelectedRmcbamRun] = useState<string>('');
  const [selectedPlot, setSelectedPlot] = useState<string>('results.png');

  // Tab 2: Live Validation State
  const [valYaml, setValYaml] = useState<string>('data/yolo_weed_partial/data.yaml');
  const [valSplit, setValSplit] = useState<string>('val');
  const [valImgsz] = useState<number>(640);
  const [valRunning, setValRunning] = useState<boolean>(false);
  const [valResults, setValResults] = useState<any | null>(null);
  const [valError, setValError] = useState<string | null>(null);

  // Tab 3: Latency Benchmark State
  const [benchRunning, setBenchRunning] = useState<boolean>(false);
  const [benchResults, setBenchResults] = useState<any[]>([]);

  useEffect(() => {
    fetchComparison();
  }, []);

  const fetchComparison = async () => {
    setLoading(true);
    try {
      const data = await getComparison();
      setComparisonData(data);
      if (data.runs && data.runs.length > 0) {
        const base = data.runs.find((r: any) => r.is_baseline);
        const rm = data.runs.find((r: any) => !r.is_baseline);
        if (base) setSelectedBaselineRun(base.run_dir);
        if (rm) setSelectedRmcbamRun(rm.run_dir);
      }
      if (data.yamls && data.yamls.length > 0) {
        setValYaml(data.yamls[0]);
      }
    } catch (e) {
      console.error('Failed to load model comparison:', e);
    } finally {
      setLoading(false);
    }
  };

  const runsData: any[] = comparisonData?.runs || [];
  const currentBaselineRun = runsData.find((r) => r.run_dir === selectedBaselineRun);
  const currentRmcbamRun = runsData.find((r) => r.run_dir === selectedRmcbamRun);

  const bestEpochRows = useMemo(() => {
    const rows = [];
    if (currentBaselineRun?.best_epoch_metrics) {
      rows.push({
        model: 'YOLOv8s Baseline',
        run: currentBaselineRun.run_name,
        color: '#3b82f6',
        ...currentBaselineRun.best_epoch_metrics,
        best_epoch: currentBaselineRun.best_epoch,
      });
    }
    if (currentRmcbamRun?.best_epoch_metrics) {
      rows.push({
        model: 'YOLOv8s + RMCBAM',
        run: currentRmcbamRun.run_name,
        color: 'var(--fo-accent-green)',
        ...currentRmcbamRun.best_epoch_metrics,
        best_epoch: currentRmcbamRun.best_epoch,
      });
    }
    return rows;
  }, [currentBaselineRun, currentRmcbamRun]);

  const availablePlots = useMemo(() => {
    const titles = new Set<string>();
    if (currentBaselineRun?.plots) Object.keys(currentBaselineRun.plots).forEach((t) => titles.add(t));
    if (currentRmcbamRun?.plots) Object.keys(currentRmcbamRun.plots).forEach((t) => titles.add(t));
    return Array.from(titles);
  }, [currentBaselineRun, currentRmcbamRun]);

  const handleRunLiveValidation = async () => {
    setValRunning(true);
    setValError(null);
    setValResults(null);
    try {
      const baseCkpt = checkpoints.find((c) => !c.is_rmcbam)?.path;
      const rmCkpt = checkpoints.find((c) => c.is_rmcbam)?.path;
      const res = await runLiveValidation(undefined, baseCkpt, rmCkpt, valYaml, valSplit, valImgsz);
      setValResults(res);
    } catch (err: any) {
      setValError(err.response?.data?.detail || err.message || 'Validation failed.');
    } finally {
      setValRunning(false);
    }
  };

  const handleRunBenchmark = async () => {
    setBenchRunning(true);
    try {
      const paths = checkpoints.slice(0, 4).map((c) => c.path);
      const res = await benchmarkCheckpoints(paths, 10);
      setBenchResults(res.results || []);
    } catch (e) {
      console.error(e);
    } finally {
      setBenchRunning(false);
    }
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
            <BarChart2 size={18} style={{ color: 'var(--fo-accent-green)' }} /> Model Evaluation & Benchmarks
          </h2>
          <div style={{ fontSize: '12px', color: 'var(--fo-text-muted)', marginTop: '4px' }}>
            Verified evaluation metrics (Precision, Recall, F1, mAP@0.50, mAP@0.50:0.95), Ultralytics training curves, live validation, and latency benchmarking.
          </div>
        </div>

        <button
          className="fo-btn fo-btn-secondary"
          onClick={fetchComparison}
          disabled={loading}
          style={{ fontSize: '12px', padding: '6px 14px' }}
        >
          <RefreshCw size={13} style={{ animation: loading ? 'spin 1s linear infinite' : 'none' }} /> Refresh Runs
        </button>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: '8px', borderBottom: '1px solid var(--fo-border)', paddingBottom: '2px' }}>
        {[
          { id: 'benchmarks', label: 'Training Metrics & Curves' },
          { id: 'validation', label: 'Live Dataset Validation' },
          { id: 'latency', label: 'Hardware Latency & FPS' },
          { id: 'architecture', label: 'Model Architecture Specs' },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as TabType)}
            className="fo-btn"
            style={{
              padding: '8px 16px',
              fontSize: '12px',
              fontWeight: 600,
              background: activeTab === tab.id ? '#ffffff' : 'transparent',
              color: activeTab === tab.id ? 'var(--fo-accent-green)' : 'var(--fo-text-muted)',
              borderBottom: activeTab === tab.id ? '2px solid var(--fo-accent-green)' : '2px solid transparent',
              borderRadius: '6px 6px 0 0',
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* TAB 1: Training Metrics & Curves */}
      {activeTab === 'benchmarks' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
          {/* Best Epoch Comparison Table */}
          <div style={{ background: '#ffffff', border: '1px solid var(--fo-border)', borderRadius: '8px', padding: '20px', boxShadow: 'var(--fo-shadow-sm)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
              <div>
                <div style={{ fontSize: '14px', fontWeight: 700, color: 'var(--fo-text-main)' }}>
                  Metrics at Best Epoch (Fitness = 0.1·mAP50 + 0.9·mAP50-95)
                </div>
                <div style={{ fontSize: '11px', color: 'var(--fo-text-muted)', marginTop: '2px' }}>
                  Extracted from Ultralytics training run records saved in runs/detect/
                </div>
              </div>
            </div>

            {bestEpochRows.length > 0 ? (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--fo-border)', background: 'var(--fo-bg-elevated)', textTransform: 'uppercase', color: 'var(--fo-text-muted)', fontSize: '11px' }}>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>Architecture</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>Run Directory</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>Precision</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>Recall</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>F1-Score</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>mAP@0.50</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>mAP@0.50:0.95</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>Best Epoch</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bestEpochRows.map((r) => (
                      <tr key={r.model} style={{ borderBottom: '1px solid var(--fo-border)' }}>
                        <td style={{ padding: '10px 12px', fontWeight: 700, color: r.model.includes('RMCBAM') ? 'var(--fo-accent-green)' : 'var(--fo-text-main)' }}>
                          {r.model}
                        </td>
                        <td style={{ padding: '10px 12px', fontFamily: 'var(--font-mono)', fontSize: '11px' }}>{r.run}</td>
                        <td style={{ padding: '10px 12px', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{((r.precision || 0) * 100).toFixed(1)}%</td>
                        <td style={{ padding: '10px 12px', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{((r.recall || 0) * 100).toFixed(1)}%</td>
                        <td style={{ padding: '10px 12px', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{((r.F1 || 0) * 100).toFixed(1)}%</td>
                        <td style={{ padding: '10px 12px', fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--fo-accent-green)' }}>
                          {((r.mAP50 || 0) * 100).toFixed(1)}%
                        </td>
                        <td style={{ padding: '10px 12px', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                          {((r['mAP50-95'] || 0) * 100).toFixed(1)}%
                        </td>
                        <td style={{ padding: '10px 12px', fontFamily: 'var(--font-mono)' }}>{r.best_epoch || 0}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ padding: '24px', textAlign: 'center', color: 'var(--fo-text-muted)' }}>
                No training run data discovered in runs/detect/.
              </div>
            )}
          </div>

          {/* Ultralytics Training Plots Viewer */}
          {availablePlots.length > 0 && (
            <div style={{ background: '#ffffff', border: '1px solid var(--fo-border)', borderRadius: '8px', padding: '20px', boxShadow: 'var(--fo-shadow-sm)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                <div>
                  <div style={{ fontSize: '14px', fontWeight: 700, color: 'var(--fo-text-main)' }}>
                    Ultralytics Evaluation Artifacts
                  </div>
                  <div style={{ fontSize: '11px', color: 'var(--fo-text-muted)' }}>
                    High-resolution validation curves and confusion matrices saved directly by the YOLO training engine.
                  </div>
                </div>

                <div style={{ display: 'flex', gap: '6px' }}>
                  {availablePlots.slice(0, 6).map((plot) => (
                    <button
                      key={plot}
                      className={`fo-btn ${selectedPlot === plot ? 'fo-btn-primary' : 'fo-btn-secondary'}`}
                      style={{ fontSize: '11px', padding: '3px 8px' }}
                      onClick={() => setSelectedPlot(plot)}
                    >
                      {plot}
                    </button>
                  ))}
                </div>
              </div>

              {/* Side by side display of the selected plot */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                <div style={{ border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '12px', background: 'var(--fo-bg-elevated)' }}>
                  <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--fo-text-main)', marginBottom: '8px' }}>
                    YOLOv8s Baseline ({selectedPlot})
                  </div>
                  {currentBaselineRun?.plots?.[selectedPlot] ? (
                    <img
                      src={currentBaselineRun.plots[selectedPlot]}
                      alt="Baseline Plot"
                      style={{ width: '100%', height: 'auto', borderRadius: '4px', border: '1px solid var(--fo-border)' }}
                    />
                  ) : (
                    <div style={{ padding: '40px', textAlign: 'center', color: 'var(--fo-text-muted)' }}>Plot not available for baseline</div>
                  )}
                </div>

                <div style={{ border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '12px', background: 'var(--fo-bg-elevated)' }}>
                  <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--fo-accent-green)', marginBottom: '8px' }}>
                    YOLOv8s + RMCBAM ({selectedPlot})
                  </div>
                  {currentRmcbamRun?.plots?.[selectedPlot] ? (
                    <img
                      src={currentRmcbamRun.plots[selectedPlot]}
                      alt="RMCBAM Plot"
                      style={{ width: '100%', height: 'auto', borderRadius: '4px', border: '1px solid var(--fo-border)' }}
                    />
                  ) : (
                    <div style={{ padding: '40px', textAlign: 'center', color: 'var(--fo-text-muted)' }}>Plot not available for RMCBAM</div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 2: Live Dataset Validation */}
      {activeTab === 'validation' && (
        <div style={{ background: '#ffffff', border: '1px solid var(--fo-border)', borderRadius: '8px', padding: '20px', boxShadow: 'var(--fo-shadow-sm)' }}>
          <div style={{ fontSize: '14px', fontWeight: 700, color: 'var(--fo-text-main)', marginBottom: '4px' }}>
            Live Dataset Validation
          </div>
          <div style={{ fontSize: '12px', color: 'var(--fo-text-muted)', marginBottom: '16px' }}>
            Run real validation evaluation on your annotated dataset splits using Ultralytics validator.
          </div>

          <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-end', marginBottom: '16px' }}>
            <div>
              <label style={{ fontSize: '11px', fontWeight: 600, color: 'var(--fo-text-muted)', display: 'block', marginBottom: '4px' }}>
                Dataset Configuration
              </label>
              <input
                type="text"
                value={valYaml}
                onChange={(e) => setValYaml(e.target.value)}
                style={{ padding: '6px 10px', fontSize: '12px', borderRadius: '4px', border: '1px solid var(--fo-border)', width: '280px' }}
              />
            </div>

            <div>
              <label style={{ fontSize: '11px', fontWeight: 600, color: 'var(--fo-text-muted)', display: 'block', marginBottom: '4px' }}>
                Split
              </label>
              <select
                value={valSplit}
                onChange={(e) => setValSplit(e.target.value)}
                style={{ padding: '6px 10px', fontSize: '12px', borderRadius: '4px', border: '1px solid var(--fo-border)' }}
              >
                <option value="val">Validation (val)</option>
                <option value="test">Test (test)</option>
                <option value="train">Train (train)</option>
              </select>
            </div>

            <button
              className="fo-btn fo-btn-primary"
              onClick={handleRunLiveValidation}
              disabled={valRunning}
              style={{ fontSize: '12px', padding: '7px 18px' }}
            >
              <Play size={13} /> {valRunning ? 'Running Validation...' : 'Execute Live Validation'}
            </button>
          </div>

          {valError && (
            <div style={{ padding: '10px 14px', background: '#fee2e2', border: '1px solid #fecaca', borderRadius: '6px', color: '#991b1b', fontSize: '12px', marginBottom: '16px' }}>
              {valError}
            </div>
          )}

          {valResults && (
            <div style={{ marginTop: '16px' }}>
              <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--fo-text-main)', marginBottom: '8px' }}>
                Validation Results
              </div>
              <pre className="fo-json-tree" style={{ maxHeight: '300px' }}>
                {JSON.stringify(valResults, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}

      {/* TAB 3: Latency & Hardware Benchmarks */}
      {activeTab === 'latency' && (
        <div style={{ background: '#ffffff', border: '1px solid var(--fo-border)', borderRadius: '8px', padding: '20px', boxShadow: 'var(--fo-shadow-sm)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
            <div>
              <div style={{ fontSize: '14px', fontWeight: 700, color: 'var(--fo-text-main)' }}>
                Hardware Inference Latency & FPS Benchmark
              </div>
              <div style={{ fontSize: '12px', color: 'var(--fo-text-muted)', marginTop: '2px' }}>
                Measures actual wall-clock inference time on 10 repeated warmup runs on the active compute engine.
              </div>
            </div>

            <button
              className="fo-btn fo-btn-primary"
              onClick={handleRunBenchmark}
              disabled={benchRunning}
              style={{ fontSize: '12px', padding: '7px 16px' }}
            >
              <Activity size={13} /> {benchRunning ? 'Benchmarking Hardware...' : 'Run Latency Benchmark'}
            </button>
          </div>

          {benchResults.length > 0 ? (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--fo-border)', background: 'var(--fo-bg-elevated)', textTransform: 'uppercase', color: 'var(--fo-text-muted)', fontSize: '11px' }}>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Model Checkpoint</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Mean Latency</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Median Latency</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>95th Percentile</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Real-time FPS</th>
                </tr>
              </thead>
              <tbody>
                {benchResults.map((b) => (
                  <tr key={b.path} style={{ borderBottom: '1px solid var(--fo-border)' }}>
                    <td style={{ padding: '10px 12px', fontWeight: 600, color: b.is_rmcbam ? 'var(--fo-accent-green)' : 'var(--fo-text-main)' }}>
                      {b.checkpoint}
                    </td>
                    <td style={{ padding: '10px 12px', fontFamily: 'var(--font-mono)' }}>
                      {typeof b.mean_ms === 'number' ? `${b.mean_ms.toFixed(2)} ms` : b.mean_ms}
                    </td>
                    <td style={{ padding: '10px 12px', fontFamily: 'var(--font-mono)' }}>
                      {typeof b.median_ms === 'number' ? `${b.median_ms.toFixed(2)} ms` : b.median_ms}
                    </td>
                    <td style={{ padding: '10px 12px', fontFamily: 'var(--font-mono)' }}>
                      {typeof b.p95_ms === 'number' ? `${b.p95_ms.toFixed(2)} ms` : b.p95_ms}
                    </td>
                    <td style={{ padding: '10px 12px', fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--fo-accent-green)' }}>
                      {typeof b.fps === 'number' ? `${b.fps.toFixed(1)} FPS` : b.fps}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div style={{ padding: '36px', textAlign: 'center', color: 'var(--fo-text-muted)' }}>
              Click "Run Latency Benchmark" to measure inference speeds on this machine.
            </div>
          )}
        </div>
      )}

      {/* TAB 4: Architecture Specifications */}
      {activeTab === 'architecture' && (
        <div style={{ background: '#ffffff', border: '1px solid var(--fo-border)', borderRadius: '8px', padding: '20px', boxShadow: 'var(--fo-shadow-sm)' }}>
          <div style={{ fontSize: '14px', fontWeight: 700, color: 'var(--fo-text-main)', marginBottom: '6px' }}>
            Reference Architecture Specifications
          </div>
          <div style={{ fontSize: '12px', color: 'var(--fo-text-muted)', marginBottom: '16px' }}>
            Structural comparison between Standard YOLOv8s and YOLOv8s augmented with Residual Multiscale Channel-and-Spatial Attention (RMCBAM).
          </div>

          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--fo-border)', background: 'var(--fo-bg-elevated)', textTransform: 'uppercase', color: 'var(--fo-text-muted)', fontSize: '11px' }}>
                <th style={{ padding: '8px 12px', textAlign: 'left' }}>Specification</th>
                <th style={{ padding: '8px 12px', textAlign: 'left' }}>Standard YOLOv8s</th>
                <th style={{ padding: '8px 12px', textAlign: 'left' }}>YOLOv8s + RMCBAM</th>
              </tr>
            </thead>
            <tbody>
              <tr style={{ borderBottom: '1px solid var(--fo-border)' }}>
                <td style={{ padding: '10px 12px', fontWeight: 600 }}>Backbone Attention</td>
                <td style={{ padding: '10px 12px' }}>None (Standard C2f convolutions)</td>
                <td style={{ padding: '10px 12px', color: 'var(--fo-accent-green)', fontWeight: 600 }}>RMCBAM blocks on P3, P4, P5 stages</td>
              </tr>
              <tr style={{ borderBottom: '1px solid var(--fo-border)' }}>
                <td style={{ padding: '10px 12px', fontWeight: 600 }}>Channel Attention</td>
                <td style={{ padding: '10px 12px' }}>None</td>
                <td style={{ padding: '10px 12px', color: 'var(--fo-accent-green)' }}>ECA (Efficient 1D Convolutional Channel Attention)</td>
              </tr>
              <tr style={{ borderBottom: '1px solid var(--fo-border)' }}>
                <td style={{ padding: '10px 12px', fontWeight: 600 }}>Spatial Attention</td>
                <td style={{ padding: '10px 12px' }}>None</td>
                <td style={{ padding: '10px 12px', color: 'var(--fo-accent-green)' }}>Multi-scale depthwise dilated receptive fields (k=3, 5, 7)</td>
              </tr>
              <tr style={{ borderBottom: '1px solid var(--fo-border)' }}>
                <td style={{ padding: '10px 12px', fontWeight: 600 }}>Residual Gating</td>
                <td style={{ padding: '10px 12px' }}>Identity add</td>
                <td style={{ padding: '10px 12px', color: 'var(--fo-accent-green)' }}>Learnable scalar gate γ (initial 0.0)</td>
              </tr>
              <tr style={{ borderBottom: '1px solid var(--fo-border)' }}>
                <td style={{ padding: '10px 12px', fontWeight: 600 }}>Target Application</td>
                <td style={{ padding: '10px 12px' }}>General Object Detection</td>
                <td style={{ padding: '10px 12px', color: 'var(--fo-accent-green)', fontWeight: 600 }}>Small, Occluded Weeds in Dense Crop Foliage</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
