import React, { useState, useEffect } from 'react';
import type { CheckpointInfo, PredictResponse } from '../types';
import { runPredict, runPredictSample, getSampleImages } from '../services/api';
import { FiftyOneViewport } from '../components/FiftyOneViewport';
import { FiftyOneInspector } from '../components/FiftyOneInspector';
import {
  AlertCircle,
  Table as TableIcon,
  Download,
  FileSpreadsheet,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';

interface Props {
  checkpoints: CheckpointInfo[];
  selectedCheckpoint: string;
  onSelectCheckpoint: (path: string) => void;
  predictData: PredictResponse | null;
  onPredictSuccess: (data: PredictResponse, file?: File) => void;
  selectedFile: File | null;
  onFileChange: (file: File) => void;
  onClearImage: () => void;
}

export const InspectionWorkspace: React.FC<Props> = ({
  checkpoints,
  selectedCheckpoint,
  onSelectCheckpoint,
  predictData,
  onPredictSuccess,
  selectedFile,
  onFileChange,
  onClearImage,
}) => {
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [conf, setConf] = useState<number>(0.15);
  const [iou, setIou] = useState<number>(0.45);
  const [imgsz] = useState<number>(640);
  const [sampleImages, setSampleImages] = useState<string[]>([]);
  const [showTable, setShowTable] = useState<boolean>(true);

  // Display toggles
  const [showBoxes, setShowBoxes] = useState<boolean>(true);
  const [showLabels, setShowLabels] = useState<boolean>(true);
  const [showConf, setShowConf] = useState<boolean>(true);
  const [hiddenClasses, setHiddenClasses] = useState<string[]>([]);
  const [selectedDetectionUid, setSelectedDetectionUid] = useState<string | null>(null);

  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [pendingSample, setPendingSample] = useState<string | null>(null);
  const hasAutoLoadedRef = React.useRef(false);

  useEffect(() => {
    fetchSamples();
    const wasExplicitlyCleared = sessionStorage.getItem('wv_cleared') === 'true';
    if (!hasAutoLoadedRef.current && !predictData && selectedCheckpoint && !wasExplicitlyCleared) {
      hasAutoLoadedRef.current = true;
      const initialSample = localStorage.getItem('wv_last_sample') || 'img112.jpg';
      handleLoadSample(initialSample);
    }
  }, [selectedCheckpoint, predictData]);

  const fetchSamples = async () => {
    try {
      const data = await getSampleImages();
      setSampleImages(data.samples || []);
    } catch {
      // ignore
    }
  };

  const parseError = (err: any, fallback: string): string => {
    if (err.response?.data?.detail) return err.response.data.detail;
    if (err.message === 'Network Error' || err.code === 'ERR_NETWORK') {
      return 'Network Error: Cannot connect to WeedVision backend. Ensure the Python API service is running on http://127.0.0.1:8000.';
    }
    return err.message || fallback;
  };

  const handleFileUpload = (file: File) => {
    onFileChange(file);
    const objUrl = URL.createObjectURL(file);
    setPreviewUrl(objUrl);
    setPendingSample(null);
    setError(null);
    // Execute prediction directly upon drop / upload
    executePrediction(file);
  };

  const handleSelectSample = (sampleName: string) => {
    setPendingSample(sampleName);
    setError(null);
    handleLoadSample(sampleName);
  };

  const handleLoadSample = async (filename: string, ckptOverride?: string, newConf?: number, newIou?: number) => {
    if (loading) return;
    setLoading(true);
    setError(null);
    try {
      const ckpt = ckptOverride || selectedCheckpoint;
      const res = await runPredictSample(
        filename,
        ckpt,
        newConf !== undefined ? newConf : conf,
        newIou !== undefined ? newIou : iou,
        imgsz
      );
      onPredictSuccess(res);
      setPreviewUrl(null);
      setPendingSample(null);
    } catch (err: any) {
      setError(parseError(err, 'Sample prediction failed.'));
    } finally {
      setLoading(false);
    }
  };

  const executePrediction = async (file: File, ckptOverride?: string, newConf?: number, newIou?: number) => {
    if (loading) return;
    setLoading(true);
    setError(null);
    try {
      const ckpt = ckptOverride || selectedCheckpoint;
      const res = await runPredict(
        file,
        ckpt,
        newConf !== undefined ? newConf : conf,
        newIou !== undefined ? newIou : iou,
        imgsz
      );
      onPredictSuccess(res, file);
      setPreviewUrl(null);
    } catch (err: any) {
      setError(parseError(err, 'Inference failed.'));
    } finally {
      setLoading(false);
    }
  };

  const handleRunDetection = () => {
    if (loading) return;
    if (selectedFile) {
      executePrediction(selectedFile, selectedCheckpoint, conf, iou);
    } else if (pendingSample) {
      handleLoadSample(pendingSample, selectedCheckpoint, conf, iou);
    } else if (predictData?.filename) {
      handleLoadSample(predictData.filename, selectedCheckpoint, conf, iou);
    }
  };

  const handleSelectCheckpoint = (path: string) => {
    onSelectCheckpoint(path);
    if (selectedFile) {
      executePrediction(selectedFile, path, conf, iou);
    } else if (predictData?.filename) {
      handleLoadSample(predictData.filename, path, conf, iou);
    }
  };

  const toggleClassVisibility = (clsName: string) => {
    setHiddenClasses((prev) =>
      prev.includes(clsName) ? prev.filter((c) => c !== clsName) : [...prev, clsName]
    );
  };

  const handleClearImage = () => {
    sessionStorage.setItem('wv_cleared', 'true');
    setPreviewUrl(null);
    setPendingSample(null);
    setError(null);
    onClearImage();
  };

  const handleDownloadCsv = () => {
    if (!predictData) return;
    const header = 'uid,cls_name,conf,x1,y1,x2,y2,cx,cy,w,h,area,area_pct\n';
    const rows = (predictData.detections_table || predictData.detections).map((d: any) =>
      `${d.uid},${d.cls_name},${(d.conf || 0).toFixed(4)},${(d.x1 || 0).toFixed(1)},${(d.y1 || 0).toFixed(1)},${(d.x2 || 0).toFixed(1)},${(d.y2 || 0).toFixed(1)},${(d.cx || 0).toFixed(1)},${(d.cy || 0).toFixed(1)},${(d.w || (d.x2 - d.x1) || 0).toFixed(1)},${(d.h || (d.y2 - d.y1) || 0).toFixed(1)},${(d.area || d.area_px || 0).toFixed(0)},${(d.area_pct || 0).toFixed(4)}`
    ).join('\n');
    const blob = new Blob([header + rows], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${predictData.filename.replace(/\.[^/.]+$/, '')}_detections.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadJson = () => {
    if (!predictData) return;
    const blob = new Blob([JSON.stringify(predictData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${predictData.filename.replace(/\.[^/.]+$/, '')}_detections.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadPng = () => {
    if (!predictData?.annotated_image) return;
    const a = document.createElement('a');
    a.href = predictData.annotated_image;
    a.download = `${predictData.filename.replace(/\.[^/.]+$/, '')}_annotated.png`;
    a.click();
  };

  // Client-side filtering by confidence and hidden classes (instant 0ms response)
  const filteredDetections = (predictData?.detections || []).filter((d) => {
    const c = d.conf ?? 0;
    const name = d.cls_name ?? '';
    return c >= conf && !hiddenClasses.includes(name);
  });

  const activeCheckpointInfo = checkpoints.find((c) => c.path === selectedCheckpoint);

  return (
    <div style={{ display: 'flex', width: '100%', height: '100%', overflow: 'hidden' }}>
      {/* Center Main Viewport Canvas & Table */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* Model & Diagnostics Bar */}
        <div
          style={{
            background: '#ffffff',
            borderBottom: '1px solid var(--fo-border)',
            padding: '8px 16px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            fontSize: '12px',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontWeight: 700, color: 'var(--fo-text-main)' }}>
              {activeCheckpointInfo ? activeCheckpointInfo.label : 'YOLOv8s'}
            </span>
            <span
              style={{
                fontSize: '11px',
                padding: '2px 8px',
                borderRadius: '4px',
                fontWeight: 600,
                background: activeCheckpointInfo?.is_rmcbam ? 'var(--fo-accent-green-light)' : 'var(--fo-bg-elevated)',
                color: activeCheckpointInfo?.is_rmcbam ? 'var(--fo-accent-green)' : 'var(--fo-text-muted)',
                border: activeCheckpointInfo?.is_rmcbam ? '1px solid var(--fo-accent-green-border)' : '1px solid var(--fo-border)',
              }}
            >
              {activeCheckpointInfo?.is_rmcbam ? 'YOLOv8s + RMCBAM Attention' : 'Baseline YOLOv8s'}
            </span>
            {predictData && (
              <span style={{ fontSize: '11px', color: 'var(--fo-text-muted)' }}>
                • Detections matching conf ≥ {(conf * 100).toFixed(0)}%: <b>{filteredDetections.length}</b>
              </span>
            )}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ fontSize: '11px', color: 'var(--fo-text-muted)' }}>Confidence:</span>
              <input
                type="range"
                min="0.05"
                max="0.95"
                step="0.05"
                value={conf}
                onChange={(e) => setConf(parseFloat(e.target.value))}
                style={{ width: '80px', accentColor: 'var(--fo-accent-green)', cursor: 'pointer' }}
                title={`Confidence threshold: ${(conf * 100).toFixed(0)}%`}
              />
              <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', minWidth: '32px' }}>
                {(conf * 100).toFixed(0)}%
              </span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ fontSize: '11px', color: 'var(--fo-text-muted)' }}>IoU:</span>
              <input
                type="range"
                min="0.10"
                max="0.90"
                step="0.05"
                value={iou}
                onChange={(e) => setIou(parseFloat(e.target.value))}
                style={{ width: '80px', accentColor: 'var(--fo-accent-green)', cursor: 'pointer' }}
                title={`NMS IoU: ${iou.toFixed(2)}`}
              />
              <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', minWidth: '28px' }}>
                {iou.toFixed(2)}
              </span>
            </div>
          </div>
        </div>

        {/* Error Notification Alert */}
        {error && (
          <div
            style={{
              padding: '10px 16px',
              background: '#fee2e2',
              borderBottom: '1px solid #fecaca',
              color: '#991b1b',
              fontSize: '12px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <AlertCircle size={15} />
              <span>{error}</span>
            </div>
            <button
              className="fo-btn fo-btn-secondary"
              style={{ fontSize: '11px', padding: '2px 8px' }}
              onClick={() => setError(null)}
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Viewport Image Canvas */}
        <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <FiftyOneViewport
            predictData={predictData}
            filteredDetections={filteredDetections}
            selectedDetectionUid={selectedDetectionUid}
            onSelectDetection={setSelectedDetectionUid}
            showBoxes={showBoxes}
            onToggleBoxes={setShowBoxes}
            showLabels={showLabels}
            onToggleLabels={setShowLabels}
            showConf={showConf}
            onToggleConf={setShowConf}
            onFileUpload={handleFileUpload}
            onClearImage={handleClearImage}
            onRunDetection={handleRunDetection}
            loading={loading}
            previewUrl={previewUrl}
            confThreshold={conf}
          />
        </div>

        {/* Detections Table Section */}
        {predictData && (
          <div
            style={{
              background: '#ffffff',
              borderTop: '1px solid var(--fo-border)',
              display: 'flex',
              flexDirection: 'column',
              maxHeight: showTable ? '260px' : '38px',
              transition: 'max-height 0.2s ease',
              flexShrink: 0,
            }}
          >
            <div
              style={{
                padding: '8px 16px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                background: 'var(--fo-bg-elevated)',
                cursor: 'pointer',
                userSelect: 'none',
              }}
              onClick={() => setShowTable(!showTable)}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <TableIcon size={14} style={{ color: 'var(--fo-accent-green)' }} />
                <span style={{ fontSize: '12px', fontWeight: 700, color: 'var(--fo-text-main)' }}>
                  DETECTIONS TABLE ({filteredDetections.length} Identified Instances)
                </span>
                <span style={{ fontSize: '11px', color: 'var(--fo-text-muted)' }}>
                  Confidence ≥ {(conf * 100).toFixed(0)}%
                </span>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <div style={{ display: 'flex', gap: '6px' }} onClick={(e) => e.stopPropagation()}>
                  <button
                    className="fo-btn fo-btn-secondary"
                    style={{ fontSize: '11px', padding: '3px 8px' }}
                    onClick={handleDownloadPng}
                    title="Export annotated image"
                  >
                    <Download size={12} /> PNG
                  </button>
                  <button
                    className="fo-btn fo-btn-secondary"
                    style={{ fontSize: '11px', padding: '3px 8px' }}
                    onClick={handleDownloadCsv}
                    title="Export table as CSV"
                  >
                    <FileSpreadsheet size={12} /> CSV
                  </button>
                  <button
                    className="fo-btn fo-btn-secondary"
                    style={{ fontSize: '11px', padding: '3px 8px' }}
                    onClick={handleDownloadJson}
                    title="Export table as JSON"
                  >
                    <Download size={12} /> JSON
                  </button>
                </div>
                {showTable ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
              </div>
            </div>

            {showTable && (
              <div style={{ overflow: 'auto', flex: 1, padding: '0 8px 8px' }}>
                <table className="fo-json-tree" style={{ width: '100%', color: 'var(--fo-text-main)', borderCollapse: 'collapse', fontSize: '11px' }}>
                  <thead>
                    <tr style={{ position: 'sticky', top: 0, background: 'var(--fo-bg-elevated)', borderBottom: '1px solid var(--fo-border)', textTransform: 'uppercase', color: 'var(--fo-text-muted)', zIndex: 1 }}>
                      <th style={{ padding: '6px 8px', textAlign: 'left' }}>#</th>
                      <th style={{ padding: '6px 8px', textAlign: 'left' }}>Class</th>
                      <th style={{ padding: '6px 8px', textAlign: 'left' }}>Confidence</th>
                      <th style={{ padding: '6px 8px', textAlign: 'left' }}>x1</th>
                      <th style={{ padding: '6px 8px', textAlign: 'left' }}>y1</th>
                      <th style={{ padding: '6px 8px', textAlign: 'left' }}>x2</th>
                      <th style={{ padding: '6px 8px', textAlign: 'left' }}>y2</th>
                      <th style={{ padding: '6px 8px', textAlign: 'left' }}>Center (X, Y)</th>
                      <th style={{ padding: '6px 8px', textAlign: 'left' }}>Dimensions (W × H)</th>
                      <th style={{ padding: '6px 8px', textAlign: 'left' }}>Area (px²)</th>
                      <th style={{ padding: '6px 8px', textAlign: 'left' }}>Area %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredDetections.map((d: any, idx: number) => {
                      const isSelected = d.uid === selectedDetectionUid;
                      const isWeed = (d.cls_name || '').toLowerCase() === 'weed';
                      const confVal = typeof d.conf === 'number' ? d.conf : (d.confidence ?? 0);
                      const x1Val = d.x1 ?? d.bbox?.[0] ?? 0;
                      const y1Val = d.y1 ?? d.bbox?.[1] ?? 0;
                      const x2Val = d.x2 ?? d.bbox?.[2] ?? 0;
                      const y2Val = d.y2 ?? d.bbox?.[3] ?? 0;
                      const cxVal = d.cx ?? ((x1Val + x2Val) / 2);
                      const cyVal = d.cy ?? ((y1Val + y2Val) / 2);
                      const wVal = Math.round(d.w ?? (x2Val - x1Val));
                      const hVal = Math.round(d.h ?? (y2Val - y1Val));
                      const areaPx = d.area_px ?? d.area ?? (wVal * hVal);
                      const totalPixels = (predictData?.width || 1) * (predictData?.height || 1);
                      const areaPctVal = d.area_pct ?? (totalPixels > 0 ? (100 * areaPx / totalPixels) : 0);
                      return (
                        <tr
                          key={d.uid || idx}
                          style={{
                            borderBottom: '1px solid var(--fo-border)',
                            background: isSelected ? 'var(--fo-accent-green-light)' : 'transparent',
                            cursor: 'pointer',
                          }}
                          onClick={() => setSelectedDetectionUid(isSelected ? null : d.uid)}
                        >
                          <td style={{ padding: '5px 8px', color: isWeed ? '#b91c1c' : 'var(--fo-accent-green)', fontWeight: 700 }}>{d.uid || idx + 1}</td>
                          <td style={{ padding: '5px 8px' }}>
                            <span
                              style={{
                                padding: '1px 6px',
                                borderRadius: '3px',
                                fontSize: '10px',
                                fontWeight: 600,
                                background: isWeed ? '#fee2e2' : '#dcfce7',
                                color: isWeed ? '#991b1b' : '#166534',
                              }}
                            >
                              {d.cls_name || d.class_name}
                            </span>
                          </td>
                          <td style={{ padding: '5px 8px', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{((confVal || 0) * 100).toFixed(1)}%</td>
                          <td style={{ padding: '5px 8px', fontFamily: 'var(--font-mono)' }}>{Math.round(x1Val)}</td>
                          <td style={{ padding: '5px 8px', fontFamily: 'var(--font-mono)' }}>{Math.round(y1Val)}</td>
                          <td style={{ padding: '5px 8px', fontFamily: 'var(--font-mono)' }}>{Math.round(x2Val)}</td>
                          <td style={{ padding: '5px 8px', fontFamily: 'var(--font-mono)' }}>{Math.round(y2Val)}</td>
                          <td style={{ padding: '5px 8px', fontFamily: 'var(--font-mono)' }}>({Math.round(cxVal)}, {Math.round(cyVal)})</td>
                          <td style={{ padding: '5px 8px', fontFamily: 'var(--font-mono)' }}>{wVal} × {hVal}</td>
                          <td style={{ padding: '5px 8px', fontFamily: 'var(--font-mono)' }}>{Math.round(areaPx).toLocaleString()}</td>
                          <td style={{ padding: '5px 8px', fontFamily: 'var(--font-mono)' }}>{Number(areaPctVal || 0).toFixed(2)}%</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Right Inspector Panel */}
      <FiftyOneInspector
        predictData={predictData}
        checkpoints={checkpoints}
        selectedCheckpoint={selectedCheckpoint}
        onSelectCheckpoint={handleSelectCheckpoint}
        conf={conf}
        onConfChange={setConf}
        iou={iou}
        onIouChange={setIou}
        selectedDetectionUid={selectedDetectionUid}
        onSelectDetection={setSelectedDetectionUid}
        hiddenClasses={hiddenClasses}
        onToggleClassVisibility={toggleClassVisibility}
        onFileUpload={handleFileUpload}
        onClearImage={handleClearImage}
        onRunDetection={handleRunDetection}
        isAnalyzing={loading}
        sampleImages={sampleImages}
        onSelectSample={handleSelectSample}
        hasPreview={Boolean(previewUrl || predictData)}
      />
    </div>
  );
};
