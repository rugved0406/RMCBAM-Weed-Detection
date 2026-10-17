import React, { useState, useRef, useEffect } from 'react';
import type { Detection, PredictResponse } from '../types';
import {
  ZoomIn,
  ZoomOut,
  Maximize2,
  SlidersHorizontal,
  Upload,
  Trash2,
  Image as ImageIcon,
  AlertCircle,
} from 'lucide-react';

interface Props {
  predictData: PredictResponse | null;
  filteredDetections: Detection[];
  selectedDetectionUid: string | null;
  onSelectDetection: (uid: string | null) => void;
  showBoxes: boolean;
  onToggleBoxes?: (val: boolean) => void;
  showLabels: boolean;
  onToggleLabels?: (val: boolean) => void;
  showConf: boolean;
  onToggleConf?: (val: boolean) => void;
  onFileUpload?: (file: File) => void;
  onClearImage?: () => void;
  onRunDetection?: () => void;
  loading?: boolean;
  previewUrl?: string | null;
  confThreshold?: number;
}

export const FiftyOneViewport: React.FC<Props> = ({
  predictData,
  filteredDetections,
  selectedDetectionUid,
  onSelectDetection,
  showBoxes,
  onToggleBoxes,
  showLabels,
  onToggleLabels,
  showConf,
  onToggleConf,
  onFileUpload,
  onClearImage,
  onRunDetection,
  loading = false,
  previewUrl = null,
  confThreshold = 0.15,
}) => {
  const [zoom, setZoom] = useState<number>(100);
  const [viewMode, setViewMode] = useState<'annotated' | 'original'>('annotated');
  const [splitView, setSplitView] = useState<boolean>(false);
  const [splitPos, setSplitPos] = useState<number>(50);
  const [isDraggingSplit, setIsDraggingSplit] = useState<boolean>(false);
  const [isDraggingFile, setIsDraggingFile] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [containerSize, setContainerSize] = useState<{ width: number; height: number }>({ width: 0, height: 0 });

  const imgRef = useRef<HTMLImageElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const zoomIn = () => setZoom((z) => Math.min(z + 20, 300));
  const zoomOut = () => setZoom((z) => Math.max(z - 20, 50));
  const resetZoom = () => setZoom(100);

  // Measure actual viewport container size so image is NEVER cropped
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const update = () => {
      setContainerSize({
        width: el.clientWidth,
        height: el.clientHeight,
      });
    };
    update();
    const ro = new ResizeObserver(() => update());
    ro.observe(el);
    window.addEventListener('resize', update);
    return () => {
      ro.disconnect();
      window.removeEventListener('resize', update);
    };
  }, []);

  const validateAndUpload = (file: File) => {
    setUploadError(null);
    const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
    const validExts = ['.jpg', '.jpeg', '.png', '.webp'];
    const hasValidExt = validExts.some((ext) => file.name.toLowerCase().endsWith(ext));
    if (!validTypes.includes(file.type) && !hasValidExt) {
      setUploadError('Unsupported file format. Please upload a standard JPG, PNG, or WEBP image.');
      return;
    }
    const maxSize = 25 * 1024 * 1024; // 25 MB
    if (file.size > maxSize) {
      setUploadError('File exceeds 25 MB limit. Please select a smaller field image.');
      return;
    }
    if (onFileUpload) {
      onFileUpload(file);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingFile(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingFile(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingFile(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndUpload(e.dataTransfer.files[0]);
    }
  };

  // Split-screen drag handler
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDraggingSplit || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      let pos = ((e.clientX - rect.left) / rect.width) * 100;
      pos = Math.max(5, Math.min(95, pos));
      setSplitPos(pos);
    };

    const handleMouseUp = () => setIsDraggingSplit(false);

    if (isDraggingSplit) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
    }

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDraggingSplit]);

  const handleNativeUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      validateAndUpload(e.target.files[0]);
    }
  };

  if (!predictData && !previewUrl) {
    return (
      <div
        className="fo-canvas-workspace"
        style={{ alignItems: 'center', justifyContent: 'center', position: 'relative', background: 'var(--fo-bg-canvas)' }}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {isDraggingFile && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              background: 'rgba(240, 253, 244, 0.94)',
              border: '2px dashed var(--fo-accent-green)',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 50,
              pointerEvents: 'none',
            }}
          >
            <Upload size={40} style={{ color: 'var(--fo-accent-green)', marginBottom: '12px' }} />
            <span style={{ fontSize: '15px', fontWeight: 600, color: 'var(--fo-accent-green-dark)' }}>
              Drop field photograph here
            </span>
            <span style={{ fontSize: '12px', color: 'var(--fo-text-muted)', marginTop: '4px' }}>
              Valid formats: JPG, PNG, WEBP (up to 25 MB)
            </span>
          </div>
        )}

        <div style={{ textAlign: 'center', padding: '48px 32px', maxWidth: '520px' }}>
          <div
            style={{
              width: '64px',
              height: '64px',
              borderRadius: '12px',
              background: '#ffffff',
              border: '1px solid var(--fo-border)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 16px',
              boxShadow: 'var(--fo-shadow-sm)',
            }}
          >
            <ImageIcon size={30} style={{ color: 'var(--fo-accent-green)' }} />
          </div>

          <div style={{ fontSize: '16px', fontWeight: 700, color: 'var(--fo-text-main)', marginBottom: '6px' }}>
            No Field Image Selected
          </div>
          <div style={{ fontSize: '13px', color: 'var(--fo-text-muted)', lineHeight: 1.5, marginBottom: '20px' }}>
            Upload or drag and drop high-resolution agricultural photographs to perform automated weed detection, crop identification, and infestation density analysis.
          </div>

          {uploadError && (
            <div
              style={{
                marginBottom: '16px',
                padding: '8px 14px',
                borderRadius: '6px',
                background: '#fee2e2',
                border: '1px solid #fecaca',
                color: '#991b1b',
                fontSize: '12px',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                justifyContent: 'center',
              }}
            >
              <AlertCircle size={14} />
              <span>{uploadError}</span>
            </div>
          )}

          <div style={{ display: 'flex', gap: '12px', justifyContent: 'center' }}>
            <label
              className="fo-btn fo-btn-primary"
              style={{ cursor: 'pointer', padding: '8px 20px', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '6px' }}
            >
              <Upload size={14} /> Browse Field Image
              <input type="file" accept="image/jpeg,image/png,image/webp" onChange={handleNativeUpload} style={{ display: 'none' }} />
            </label>
          </div>
          <div style={{ fontSize: '11px', color: 'var(--fo-text-muted)', marginTop: '12px' }}>
            Supported formats: JPEG, PNG, WEBP • Max size: 25 MB
          </div>
        </div>
      </div>
    );
  }

  const width = predictData?.width || 640;
  const height = predictData?.height || 640;
  const filename = predictData?.filename || 'Uploaded Image';
  const file_size = predictData?.file_size;
  const original_image = predictData?.original_image || previewUrl || '';
  const annotated_image = predictData?.annotated_image || previewUrl || '';
  const isPreInference = Boolean(previewUrl && !predictData);

  // When interactive SVG boxes are active, use clean original image to prevent double-drawn box artifacts
  const activeBaseImage = (viewMode === 'annotated' && !showBoxes)
    ? annotated_image
    : (viewMode === 'annotated' && original_image)
    ? original_image
    : original_image || annotated_image;

  // Available container bounds (safe inside container with 16px padding)
  const availHeight = containerSize.height > 0 ? Math.max(160, containerSize.height - 24) : undefined;
  const availWidth = containerSize.width > 0 ? Math.max(200, containerSize.width - 24) : undefined;

  return (
    <div
      className="fo-canvas-workspace"
      style={{ position: 'relative' }}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {isDraggingFile && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            background: 'rgba(240, 253, 244, 0.94)',
            border: '2px dashed var(--fo-accent-green)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 50,
            pointerEvents: 'none',
          }}
        >
          <Upload size={40} style={{ color: 'var(--fo-accent-green)', marginBottom: '12px' }} />
          <span style={{ fontSize: '15px', fontWeight: 600, color: 'var(--fo-accent-green-dark)' }}>
            Drop to replace active image
          </span>
        </div>
      )}

      {/* Top Image Inspection Toolbar */}
      <div className="fo-image-toolbar">
        {/* Left: Image Info & Replace Action */}
        <div className="fo-toolbar-group">
          <span style={{ fontWeight: 600, color: 'var(--fo-text-main)', fontFamily: 'var(--font-mono)' }}>
            {filename}
          </span>
          {predictData && (
            <span style={{ color: 'var(--fo-text-muted)', fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
              {width} × {height} px
            </span>
          )}
          {file_size && (
            <span style={{ color: 'var(--fo-text-muted)', fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
              {(file_size / 1024).toFixed(1)} KB
            </span>
          )}

          <label
            className="fo-btn fo-btn-secondary"
            style={{ padding: '3px 10px', fontSize: '11px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px', marginLeft: '6px' }}
            title="Select another photograph to inspect"
          >
            <Upload size={12} /> Replace Image
            <input type="file" accept="image/jpeg,image/png,image/webp" onChange={handleNativeUpload} style={{ display: 'none' }} />
          </label>

          {onClearImage && (
            <button
              className="fo-btn fo-btn-secondary"
              onClick={onClearImage}
              style={{
                padding: '3px 9px',
                fontSize: '11px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                color: '#b91c1c',
              }}
              title="Clear active image"
            >
              <Trash2 size={12} /> Clear
            </button>
          )}
        </div>

        {/* Right: View Modes, Overlays, Zoom Controls */}
        <div className="fo-toolbar-group">
          {/* View Mode Toggle: Annotated vs Original */}
          <div style={{ display: 'flex', background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', borderRadius: '5px', padding: '2px', gap: '2px' }}>
            <button
              onClick={() => {
                setViewMode('annotated');
                setSplitView(false);
              }}
              className="fo-btn"
              style={{
                padding: '2px 9px',
                fontSize: '11px',
                background: viewMode === 'annotated' && !splitView ? 'var(--fo-accent-green)' : 'transparent',
                color: viewMode === 'annotated' && !splitView ? '#fff' : 'var(--fo-text-subtle)',
              }}
              title="View model detections with bounding boxes"
            >
              Annotated
            </button>
            <button
              onClick={() => {
                setViewMode('original');
                setSplitView(false);
              }}
              className="fo-btn"
              style={{
                padding: '2px 9px',
                fontSize: '11px',
                background: viewMode === 'original' && !splitView ? 'var(--fo-accent-green)' : 'transparent',
                color: viewMode === 'original' && !splitView ? '#fff' : 'var(--fo-text-subtle)',
              }}
              title="View unmodified field photograph"
            >
              Original
            </button>
          </div>

          {/* Split Screen Slider Toggle */}
          <button
            className={`fo-icon-button ${splitView ? 'active' : ''}`}
            onClick={() => setSplitView(!splitView)}
            title="Toggle Split A/B Comparison"
          >
            <SlidersHorizontal size={13} />
          </button>

          <div style={{ width: '1px', height: '16px', background: 'var(--fo-border)', margin: '0 2px' }} />

          {/* Overlay Toggles */}
          {onToggleBoxes && (
            <label style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '11px', cursor: 'pointer', color: 'var(--fo-text-muted)' }}>
              <input
                type="checkbox"
                checked={showBoxes}
                onChange={(e) => onToggleBoxes(e.target.checked)}
                className="fo-checkbox"
              />
              Boxes
            </label>
          )}

          {onToggleLabels && (
            <label style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '11px', cursor: 'pointer', color: 'var(--fo-text-muted)' }}>
              <input
                type="checkbox"
                checked={showLabels}
                onChange={(e) => onToggleLabels(e.target.checked)}
                className="fo-checkbox"
              />
              Labels
            </label>
          )}

          {onToggleConf && (
            <label style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '11px', cursor: 'pointer', color: 'var(--fo-text-muted)' }}>
              <input
                type="checkbox"
                checked={showConf}
                onChange={(e) => onToggleConf(e.target.checked)}
                className="fo-checkbox"
              />
              Conf
            </label>
          )}

          <div style={{ width: '1px', height: '16px', background: 'var(--fo-border)', margin: '0 2px' }} />

          {/* Zoom Controls */}
          <button className="fo-icon-button" onClick={zoomOut} title="Zoom Out">
            <ZoomOut size={13} />
          </button>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', minWidth: '38px', textAlign: 'center' }}>
            {zoom}%
          </span>
          <button className="fo-icon-button" onClick={zoomIn} title="Zoom In">
            <ZoomIn size={13} />
          </button>
          <button className="fo-icon-button" onClick={resetZoom} title="Reset to 100%">
            <Maximize2 size={13} />
          </button>
        </div>
      </div>

      {/* Main Viewport Container */}
      <div className="fo-viewport-container" ref={containerRef} style={{ position: 'relative' }}>
        {/* Loading Spinner during inference */}
        {loading && (
          <div
            style={{
              position: 'absolute',
              top: '16px',
              right: '16px',
              background: '#ffffff',
              border: '1px solid var(--fo-accent-green)',
              borderRadius: '6px',
              padding: '8px 16px',
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              fontSize: '12px',
              fontWeight: 600,
              color: 'var(--fo-text-main)',
              zIndex: 30,
              boxShadow: 'var(--fo-shadow-md)',
            }}
          >
            <div
              style={{
                width: '14px',
                height: '14px',
                borderRadius: '50%',
                border: '2px solid var(--fo-accent-green)',
                borderTopColor: 'transparent',
                animation: 'spin 0.8s linear infinite',
              }}
            />
            <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
            <span>Running YOLOv8s inference...</span>
          </div>
        )}

        {/* Pre-Inference Guidance Banner */}
        {isPreInference && !loading && (
          <div
            style={{
              position: 'absolute',
              bottom: '20px',
              left: '50%',
              transform: 'translateX(-50%)',
              background: '#ffffff',
              border: '1px solid var(--fo-accent-green-border)',
              borderRadius: '8px',
              padding: '10px 20px',
              display: 'flex',
              alignItems: 'center',
              gap: '14px',
              boxShadow: 'var(--fo-shadow-md)',
              zIndex: 25,
            }}
          >
            <div style={{ fontSize: '12px', color: 'var(--fo-text-subtle)' }}>
              Image preview ready. Adjust thresholds or click Run Detection to start analysis.
            </div>
            {onRunDetection && (
              <button
                className="fo-btn fo-btn-primary"
                onClick={onRunDetection}
                style={{ fontSize: '12px', padding: '5px 14px' }}
              >
                Run Detection
              </button>
            )}
          </div>
        )}

        {/* Empty Result Notification */}
        {predictData && filteredDetections.length === 0 && !loading && (
          <div
            style={{
              position: 'absolute',
              top: '16px',
              left: '50%',
              transform: 'translateX(-50%)',
              background: '#ffffff',
              border: '1px solid var(--fo-border)',
              borderRadius: '6px',
              padding: '6px 16px',
              fontSize: '12px',
              color: 'var(--fo-text-muted)',
              zIndex: 25,
              boxShadow: 'var(--fo-shadow-sm)',
            }}
          >
            No weeds or crops detected at confidence threshold ≥ {(confThreshold * 100).toFixed(0)}%. Lower the confidence slider to inspect weaker detections.
          </div>
        )}

        {!splitView ? (
          /* Single Image Canvas - Always fits 100% inside container without any cropping */
          <div
            className="fo-image-canvas"
            style={{
              display: 'inline-block',
              position: 'relative',
              lineHeight: 0,
              maxWidth: availWidth ? `${availWidth}px` : '100%',
              maxHeight: availHeight ? `${availHeight}px` : '100%',
              transform: `scale(${zoom / 100})`,
              transformOrigin: 'center center',
              transition: 'transform 0.15s ease-out',
            }}
          >
            <img
              ref={imgRef}
              src={activeBaseImage}
              alt={filename}
              style={{
                display: 'block',
                maxWidth: availWidth ? `${availWidth}px` : '100%',
                maxHeight: availHeight ? `${availHeight}px` : '100%',
                width: 'auto',
                height: 'auto',
                objectFit: 'contain',
              }}
            />

            {/* Interactive SVG Bounding Box Layer */}
            {viewMode === 'annotated' && showBoxes && (
              <svg
                className="fo-svg-overlay"
                viewBox={`0 0 ${width} ${height}`}
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '100%',
                  pointerEvents: 'none',
                }}
              >
                {filteredDetections.map((d) => {
                  const isSelected = selectedDetectionUid === d.uid;
                  const isWeed = d.cls_name.toLowerCase() === 'weed';
                  const strokeColor = isWeed ? '#e11d48' : (d.color || '#16a34a');
                  return (
                    <g key={d.uid} onClick={() => onSelectDetection(isSelected ? null : d.uid)}>
                      {/* Bounding Box Rect */}
                      <rect
                        x={d.x1}
                        y={d.y1}
                        width={d.w}
                        height={d.h}
                        fill={isSelected ? `${strokeColor}33` : 'none'}
                        stroke={strokeColor}
                        strokeWidth={isSelected ? 3.5 : 2}
                        className={`fo-bbox ${isSelected ? 'selected' : ''}`}
                      />

                      {/* Label Text Pill */}
                      {showLabels && (
                        <g>
                          <rect
                            x={d.x1}
                            y={Math.max(0, d.y1 - 18)}
                            width={(d.uid.length + d.cls_name.length + (showConf ? 6 : 0)) * 7 + 12}
                            height={16}
                            fill={strokeColor}
                            rx={3}
                          />
                          <text
                            x={d.x1 + 4}
                            y={Math.max(12, d.y1 - 5)}
                            fill="#ffffff"
                            fontSize="10"
                            fontWeight="600"
                            fontFamily="JetBrains Mono, monospace"
                          >
                            {d.uid} {d.cls_name} {showConf ? `${((d.conf || 0) * 100).toFixed(0)}%` : ''}
                          </text>
                        </g>
                      )}
                    </g>
                  );
                })}
              </svg>
            )}
          </div>
        ) : (
          /* Split Screen Interactive View - Locked Aspect Ratio */
          <div
            className="fo-image-canvas"
            style={{
              position: 'relative',
              lineHeight: 0,
              maxWidth: availWidth ? `${availWidth}px` : '100%',
              maxHeight: availHeight ? `${availHeight}px` : '100%',
              transform: `scale(${zoom / 100})`,
              transformOrigin: 'center center',
              transition: 'transform 0.15s ease-out',
              display: 'inline-block',
              overflow: 'hidden',
            }}
          >
            {/* Background Annotated Image */}
            <img
              ref={imgRef}
              src={annotated_image}
              alt="Annotated"
              style={{
                display: 'block',
                maxWidth: availWidth ? `${availWidth}px` : '100%',
                maxHeight: availHeight ? `${availHeight}px` : '100%',
                width: 'auto',
                height: 'auto',
                objectFit: 'contain',
              }}
            />

            {/* Clipped Original Image Overlay */}
            <div
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                bottom: 0,
                width: `${splitPos}%`,
                overflow: 'hidden',
                borderRight: '2px solid var(--fo-accent-green)',
                boxShadow: '0 0 10px rgba(21, 128, 61, 0.4)',
              }}
            >
              <img
                src={original_image}
                alt="Original"
                style={{
                  display: 'block',
                  maxWidth: 'none',
                  width: imgRef.current ? `${imgRef.current.clientWidth}px` : (availWidth ? `${availWidth}px` : '100%'),
                  height: imgRef.current ? `${imgRef.current.clientHeight}px` : (availHeight ? `${availHeight}px` : '100%'),
                  objectFit: 'contain',
                }}
              />
            </div>

            {/* Drag Handle */}
            <div
              style={{
                position: 'absolute',
                top: '50%',
                left: `${splitPos}%`,
                transform: 'translate(-50%, -50%)',
                width: '26px',
                height: '26px',
                borderRadius: '50%',
                backgroundColor: 'var(--fo-accent-green)',
                color: '#fff',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '12px',
                cursor: 'ew-resize',
                zIndex: 20,
                boxShadow: 'var(--fo-shadow-md)',
              }}
              onMouseDown={() => setIsDraggingSplit(true)}
              title="Drag to compare original and annotated image"
            >
              ↔
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
