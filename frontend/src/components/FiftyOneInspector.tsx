import React, { useState } from 'react';
import type { PredictResponse, CheckpointInfo } from '../types';
import {
  ChevronDown,
  ChevronRight,
  Sliders,
  Filter,
  Download,
  FileSpreadsheet,
  Play,
  Loader2,
  Layers,
  Sparkles,
  Check,
  Copy,
  Upload,
  X,
  BarChart3,
  MapPin,
} from 'lucide-react';

interface Props {
  predictData: PredictResponse | null;
  checkpoints: CheckpointInfo[];
  selectedCheckpoint: string;
  onSelectCheckpoint: (path: string) => void;
  conf: number;
  onConfChange: (val: number) => void;
  iou?: number;
  onIouChange?: (val: number) => void;
  selectedDetectionUid: string | null;
  onSelectDetection: (uid: string | null) => void;
  hiddenClasses: string[];
  onToggleClassVisibility: (clsName: string) => void;
  onFileUpload: (file: File) => void;
  onClearImage?: () => void;
  onRunDetection?: () => void;
  isAnalyzing?: boolean;
  sampleImages?: string[];
  onSelectSample?: (filename: string) => void;
  hasPreview?: boolean;
}

export const FiftyOneInspector: React.FC<Props> = ({
  predictData,
  checkpoints,
  selectedCheckpoint,
  onSelectCheckpoint,
  conf,
  onConfChange,
  iou = 0.45,
  onIouChange,
  selectedDetectionUid,
  onSelectDetection,
  hiddenClasses,
  onToggleClassVisibility,
  onFileUpload,
  onClearImage,
  onRunDetection,
  isAnalyzing = false,
  sampleImages = [],
  onSelectSample,
  hasPreview = false,
}) => {
  const [copiedJson, setCopiedJson] = useState<boolean>(false);
  const [openSections, setOpenSections] = useState<Record<string, boolean>>({
    controls: true,
    results: true,
    detections: true,
    exports: true,
  });

  const toggleSection = (key: string) => {
    setOpenSections((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const handleCopyJson = () => {
    if (!predictData) return;
    navigator.clipboard.writeText(JSON.stringify(predictData, null, 2));
    setCopiedJson(true);
    setTimeout(() => setCopiedJson(false), 2000);
  };

  const handleDownloadCsv = () => {
    if (!predictData) return;
    const header = 'uid,cls_name,conf,x1,y1,x2,y2,cx,cy,area_px,area_pct\n';
    const rows = (predictData.detections_table || predictData.detections).map((d: any) =>
      `${d.uid},${d.cls_name},${(d.conf || 0).toFixed(4)},${(d.x1 || 0).toFixed(1)},${(d.y1 || 0).toFixed(1)},${(d.x2 || 0).toFixed(1)},${(d.y2 || 0).toFixed(1)},${(d.cx || 0).toFixed(1)},${(d.cy || 0).toFixed(1)},${(d.area_px || 0).toFixed(0)},${(d.area_pct || 0).toFixed(4)}`
    ).join('\n');
    const blob = new Blob([header + rows], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${predictData.filename.replace(/\.[^/.]+$/, '')}_detections.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadImage = () => {
    if (!predictData?.annotated_image) return;
    const a = document.createElement('a');
    a.href = predictData.annotated_image;
    a.download = `${predictData.filename.replace(/\.[^/.]+$/, '')}_annotated.png`;
    a.click();
  };

  const handleDownloadJson = () => {
    if (!predictData) return;
    const blob = new Blob([JSON.stringify(predictData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${predictData.filename.replace(/\.[^/.]+$/, '')}_analysis.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Extract class breakdown from real detections
  const classCounts: Record<string, number> = {};
  if (predictData) {
    for (const d of predictData.detections) {
      classCounts[d.cls_name] = (classCounts[d.cls_name] || 0) + 1;
    }
  }

  const selectedDet = predictData?.detections.find((d) => d.uid === selectedDetectionUid);

  // Real inference latency
  const speedInferenceMs = predictData?.speed?.inference;
  const totalSpeedMs = predictData?.speed
    ? ((predictData.speed.preprocess || 0) + (predictData.speed.inference || 0) + (predictData.speed.postprocess || 0))
    : undefined;

  // Real severity calculation based on weed count
  const weedCount = predictData?.weed_count || 0;
  let severityLabel = 'Low Infestation';
  let severityBadgeClass = 'badge-crop';
  if (weedCount >= 15) {
    severityLabel = 'Severe Infestation';
    severityBadgeClass = 'badge-weed';
  } else if (weedCount >= 5) {
    severityLabel = 'Moderate Infestation';
    severityBadgeClass = 'badge-neutral';
  }

  const activeCheckpointInfo = checkpoints.find((c) => c.path === selectedCheckpoint);

  return (
    <aside className="fo-inspector">
      {/* SECTION 1: INFERENCE ACTION & MODEL CONTROLS */}
      <div className="fo-inspector-section">
        <div
          className="fo-inspector-header"
          onClick={() => toggleSection('controls')}
        >
          <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Sliders size={13} style={{ color: 'var(--fo-accent-green)' }} /> Inference Setup
          </span>
          {openSections.controls ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </div>

        {openSections.controls && (
          <div className="fo-inspector-body">
            {/* Primary Action Button */}
            {onRunDetection && (
              <button
                className="fo-btn fo-btn-primary"
                onClick={onRunDetection}
                disabled={isAnalyzing || (!hasPreview && !predictData)}
                style={{
                  width: '100%',
                  padding: '10px 16px',
                  fontSize: '13px',
                  fontWeight: 600,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '8px',
                }}
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 size={15} style={{ animation: 'spin 1s linear infinite' }} />
                    Running Detection...
                  </>
                ) : (
                  <>
                    <Play size={15} />
                    Run Weed Detection
                  </>
                )}
              </button>
            )}

            {/* Model Checkpoint Selector */}
            <div>
              <label style={{ fontSize: '11px', fontWeight: 600, color: 'var(--fo-text-muted)', marginBottom: '5px', display: 'block' }}>
                Model Checkpoint
              </label>
              <select
                value={selectedCheckpoint}
                onChange={(e) => onSelectCheckpoint(e.target.value)}
                style={{
                  width: '100%',
                  background: '#ffffff',
                  border: '1px solid var(--fo-border)',
                  color: 'var(--fo-text-main)',
                  padding: '6px 10px',
                  borderRadius: '5px',
                  fontSize: '12px',
                  cursor: 'pointer',
                }}
              >
                {checkpoints.map((c) => (
                  <option key={c.path} value={c.path}>
                    {c.filename} {c.is_smoke ? '⚠️ (Smoke Run - 8 ep)' : (c.path === 'ensemble' ? '⭐ (Maximum Accuracy Ensemble)' : (c.is_rmcbam ? '⭐ (RMCBAM Tuned - 60 ep)' : '⭐ (Trained Baseline - 80 ep)'))}
                  </option>
                ))}
              </select>
            </div>

            {activeCheckpointInfo?.is_smoke && (
              <div
                style={{
                  background: '#fff1f2',
                  border: '1px solid #fecdd3',
                  padding: '8px 10px',
                  borderRadius: '6px',
                  fontSize: '11px',
                  color: '#9f1239',
                  lineHeight: 1.4,
                }}
              >
                <div style={{ fontWeight: 700, marginBottom: '2px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <span>⚠️ Smoke Run Selected</span>
                </div>
                <div>
                  This model was trained for only 8 smoke epochs and outputs false alarms. For highest accuracy, select <strong>weed_detection_best.pt</strong> or <strong>yolov8s_rmcbam_tuned.pt</strong>.
                </div>
              </div>
            )}

            {/* Confidence Threshold Slider */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', marginBottom: '4px' }}>
                <span style={{ fontWeight: 600, color: 'var(--fo-text-muted)' }}>Confidence Threshold</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--fo-accent-green)' }}>
                  {(conf * 100).toFixed(0)}%
                </span>
              </div>
              <input
                type="range"
                min="0.05"
                max="0.95"
                step="0.05"
                value={conf}
                onChange={(e) => onConfChange(parseFloat(e.target.value))}
                style={{ width: '100%', height: '5px' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--fo-text-muted)', marginTop: '2px' }}>
                <span>5% (High Recall)</span>
                <span>95% (High Precision)</span>
              </div>
            </div>

            {/* NMS IoU Threshold Slider */}
            {onIouChange && (
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', marginBottom: '4px' }}>
                  <span style={{ fontWeight: 600, color: 'var(--fo-text-muted)' }}>NMS IoU Threshold</span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--fo-text-main)' }}>
                    {iou.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min="0.10"
                  max="0.90"
                  step="0.05"
                  value={iou}
                  onChange={(e) => onIouChange(parseFloat(e.target.value))}
                  style={{ width: '100%', height: '5px' }}
                />
              </div>
            )}

            {/* Selected Detection Indicator */}
            {selectedDet && (
              <div
                style={{
                  background: 'var(--fo-accent-green-light)',
                  border: '1px solid var(--fo-accent-green-border)',
                  borderRadius: '5px',
                  padding: '6px 10px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  fontSize: '11px',
                }}
              >
                <div>
                  <span style={{ fontWeight: 700, color: 'var(--fo-text-main)' }}>Target #{selectedDet.uid}</span>
                  <span style={{ color: 'var(--fo-text-subtle)', marginLeft: '6px' }}>
                    {selectedDet.cls_name} ({((selectedDet.conf || 0) * 100).toFixed(1)}%)
                  </span>
                </div>
                <button
                  type="button"
                  className="fo-btn"
                  onClick={() => onSelectDetection(null)}
                  style={{ padding: '2px', background: 'transparent', color: 'var(--fo-text-muted)' }}
                  title="Deselect target"
                >
                  <X size={13} />
                </button>
              </div>
            )}

            {/* Image Upload & Clear Actions */}
            <div style={{ display: 'flex', gap: '8px', paddingTop: '6px', borderTop: '1px solid var(--fo-border)' }}>
              <label
                className="fo-btn fo-btn-secondary"
                style={{ flex: 1, cursor: 'pointer', fontSize: '11px', padding: '5px 8px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '5px' }}
              >
                <Upload size={12} style={{ color: 'var(--fo-accent-green)' }} /> Browse Image
                <input
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      onFileUpload(e.target.files[0]);
                    }
                  }}
                  style={{ display: 'none' }}
                />
              </label>

              {onClearImage && (hasPreview || predictData) && (
                <button
                  type="button"
                  className="fo-btn fo-btn-secondary"
                  onClick={onClearImage}
                  style={{ fontSize: '11px', padding: '5px 10px', color: '#b91c1c' }}
                  title="Clear active image"
                >
                  Clear
                </button>
              )}
            </div>

            {/* Quick Test Samples */}
            {sampleImages.length > 0 && onSelectSample && (
              <div style={{ paddingTop: '6px', borderTop: '1px solid var(--fo-border)' }}>
                <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--fo-text-muted)', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <Sparkles size={12} style={{ color: 'var(--fo-accent-green)' }} /> Quick Test Samples
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                  {sampleImages.slice(0, 4).map((sample) => (
                    <button
                      key={sample}
                      type="button"
                      className="fo-btn fo-btn-secondary"
                      onClick={() => onSelectSample(sample)}
                      style={{ fontSize: '11px', padding: '3px 8px', fontFamily: 'var(--font-mono)' }}
                      title={`Load test image ${sample}`}
                    >
                      {sample}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* SECTION 2: ESSENTIAL ANALYSIS RESULTS */}
      {predictData && (
        <div className="fo-inspector-section">
          <div
            className="fo-inspector-header"
            onClick={() => toggleSection('results')}
          >
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Layers size={13} style={{ color: 'var(--fo-accent-green)' }} /> Analysis Results
            </span>
            {openSections.results ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </div>

          {openSections.results && (
            <div className="fo-inspector-body">
              {/* Metric Cards Grid */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                <div style={{ background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '10px 12px' }}>
                  <div style={{ fontSize: '11px', color: 'var(--fo-text-muted)', marginBottom: '2px' }}>Weeds Detected</div>
                  <div style={{ fontSize: '20px', fontWeight: 700, fontFamily: 'var(--font-mono)', color: weedCount > 0 ? '#b91c1c' : 'var(--fo-text-main)' }}>
                    {weedCount}
                  </div>
                  <div style={{ marginTop: '4px' }}>
                    <span className={`fo-btn ${severityBadgeClass}`} style={{ fontSize: '10px', padding: '1px 6px', fontWeight: 600 }}>
                      {severityLabel}
                    </span>
                  </div>
                </div>

                <div style={{ background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '10px 12px' }}>
                  <div style={{ fontSize: '11px', color: 'var(--fo-text-muted)', marginBottom: '2px' }}>Crops Detected</div>
                  <div style={{ fontSize: '20px', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--fo-accent-green)' }}>
                    {predictData.crop_count}
                  </div>
                  <div style={{ fontSize: '11px', color: 'var(--fo-text-muted)', marginTop: '4px' }}>
                    {Object.keys(classCounts).filter((k) => k.toLowerCase() !== 'weed').length} crop species
                  </div>
                </div>

                <div style={{ background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '10px 12px' }}>
                  <div style={{ fontSize: '11px', color: 'var(--fo-text-muted)', marginBottom: '2px' }}>Inference Duration</div>
                  <div style={{ fontSize: '16px', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--fo-text-main)' }}>
                    {speedInferenceMs ? `${speedInferenceMs.toFixed(1)} ms` : totalSpeedMs ? `${totalSpeedMs.toFixed(1)} ms` : '—'}
                  </div>
                  <div style={{ fontSize: '10px', color: 'var(--fo-text-muted)', marginTop: '2px' }}>
                    {totalSpeedMs ? `Total: ${totalSpeedMs.toFixed(1)} ms` : 'Measured inference'}
                  </div>
                </div>

                <div style={{ background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '10px 12px' }}>
                  <div style={{ fontSize: '11px', color: 'var(--fo-text-muted)', marginBottom: '2px' }}>Mean Confidence</div>
                  <div style={{ fontSize: '16px', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--fo-text-main)' }}>
                    {predictData.infestation?.mean_conf !== undefined && predictData.infestation.mean_conf !== null
                      ? `${(predictData.infestation.mean_conf * 100).toFixed(1)}%`
                      : '—'}
                  </div>
                  <div style={{ fontSize: '10px', color: 'var(--fo-text-muted)', marginTop: '2px' }}>
                    Instance certainty
                  </div>
                </div>
              </div>

              {/* Quick Links to Weed Analysis & Weed Localization */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                <a
                  href="#analysis"
                  className="fo-btn fo-btn-secondary"
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '6px',
                    fontSize: '11px',
                    padding: '7px 8px',
                    textDecoration: 'none',
                    color: 'var(--fo-accent-green)',
                    background: 'var(--fo-accent-green-light)',
                    border: '1px solid var(--fo-accent-green-border)',
                    fontWeight: 600,
                    borderRadius: '6px',
                  }}
                >
                  <BarChart3 size={13} /> Weed Analysis
                </a>
                <a
                  href="#localization"
                  className="fo-btn fo-btn-secondary"
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '6px',
                    fontSize: '11px',
                    padding: '7px 8px',
                    textDecoration: 'none',
                    color: 'var(--fo-accent-green)',
                    background: 'var(--fo-accent-green-light)',
                    border: '1px solid var(--fo-accent-green-border)',
                    fontWeight: 600,
                    borderRadius: '6px',
                  }}
                >
                  <MapPin size={13} /> Localization
                </a>
              </div>

              {/* Supported Classes Breakdown */}
              <div>
                <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--fo-text-muted)', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <Filter size={11} /> Detected Classes (Click to toggle visibility)
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                  {Object.entries(classCounts).map(([clsName, count]) => {
                    const isHidden = hiddenClasses.includes(clsName);
                    const isWeed = clsName.toLowerCase() === 'weed';
                    return (
                      <button
                        key={clsName}
                        type="button"
                        onClick={() => onToggleClassVisibility(clsName)}
                        style={{
                          fontSize: '11px',
                          padding: '3px 8px',
                          borderRadius: '4px',
                          border: '1px solid var(--fo-border)',
                          background: isHidden ? '#f1f5f9' : isWeed ? '#fee2e2' : '#dcfce7',
                          color: isHidden ? '#94a3b8' : isWeed ? '#991b1b' : '#166534',
                          fontWeight: 600,
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '6px',
                          textDecoration: isHidden ? 'line-through' : 'none',
                        }}
                        title={isHidden ? `Show ${clsName}` : `Hide ${clsName}`}
                      >
                        <span>{clsName}</span>
                        <span style={{ fontFamily: 'var(--font-mono)', opacity: 0.8 }}>({count})</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Image Dimensions & Metadata */}
              <div style={{ background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '10px 12px', display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '11px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--fo-text-muted)' }}>Resolution:</span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{predictData.width} × {predictData.height} px</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--fo-text-muted)' }}>Active Model:</span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{activeCheckpointInfo?.filename || 'YOLOv8s'}</span>
                </div>
                {predictData.file_size && (
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--fo-text-muted)' }}>File Size:</span>
                    <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{(predictData.file_size / 1024).toFixed(1)} KB</span>
                  </div>
                )}
              </div>

              <div style={{ fontSize: '10px', color: 'var(--fo-text-muted)', lineHeight: 1.4, borderLeft: '2px solid var(--fo-border-strong)', paddingLeft: '8px' }}>
                Note: Confidence percentages indicate individual detection certainty on this specific photograph, not validated model accuracy across a benchmark dataset.
              </div>
            </div>
          )}
        </div>
      )}

      {/* SECTION 3: EXPORT TOOLS */}
      {predictData && (
        <div className="fo-inspector-section">
          <div
            className="fo-inspector-header"
            onClick={() => toggleSection('exports')}
          >
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Download size={13} style={{ color: 'var(--fo-accent-green)' }} /> Export Inspection Data
            </span>
            {openSections.exports ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </div>

          {openSections.exports && (
            <div className="fo-inspector-body">
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <button
                  type="button"
                  className="fo-btn fo-btn-secondary"
                  onClick={handleDownloadImage}
                  style={{ justifyContent: 'flex-start' }}
                >
                  <Download size={13} style={{ color: 'var(--fo-accent-green)' }} /> Save Annotated Image (PNG)
                </button>
                <button
                  type="button"
                  className="fo-btn fo-btn-secondary"
                  onClick={handleDownloadCsv}
                  style={{ justifyContent: 'flex-start' }}
                >
                  <FileSpreadsheet size={13} style={{ color: 'var(--fo-accent-green)' }} /> Export Detection Coordinates (CSV)
                </button>
                <button
                  type="button"
                  className="fo-btn fo-btn-secondary"
                  onClick={handleDownloadJson}
                  style={{ justifyContent: 'flex-start' }}
                >
                  <Download size={13} style={{ color: 'var(--fo-accent-green)' }} /> Export Full Inspection Payload (JSON)
                </button>
                <button
                  type="button"
                  className="fo-btn fo-btn-secondary"
                  onClick={handleCopyJson}
                  style={{ justifyContent: 'flex-start' }}
                >
                  {copiedJson ? (
                    <>
                      <Check size={13} style={{ color: 'var(--fo-accent-green)' }} /> Copied to Clipboard!
                    </>
                  ) : (
                    <>
                      <Copy size={13} /> Copy Raw JSON
                    </>
                  )}
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </aside>
  );
};
