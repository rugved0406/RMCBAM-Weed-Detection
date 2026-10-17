import React, { useState, useMemo } from 'react';
import type { PredictResponse, Detection } from '../types';
import {
  MapPin,
  Target,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Download,
  Grid,
  Sparkles,
  ArrowRight,
} from 'lucide-react';

interface Props {
  predictData: PredictResponse | null;
  sessionResults?: PredictResponse[];
  onLoadSample?: (name?: string) => Promise<void>;
  onNavigate?: (ws: any) => void;
  onSelectHistory?: (item: PredictResponse) => void;
}

export const LocalizationWorkspace: React.FC<Props> = ({
  predictData,
  sessionResults = [],
  onLoadSample,
  onNavigate,
  onSelectHistory,
}) => {
  const [zoom, setZoom] = useState<number>(100);
  const [showCrops, setShowCrops] = useState<boolean>(true);
  const [minConf, setMinConf] = useState<number>(0.0);
  const [highlightWeed, setHighlightWeed] = useState<string>('—');
  const [loadingSample, setLoadingSample] = useState<string | null>(null);

  const zoomIn = () => setZoom((z) => Math.min(z + 25, 250));
  const zoomOut = () => setZoom((z) => Math.max(z - 25, 60));
  const resetZoom = () => setZoom(100);

  const handleSampleClick = async (sampleName: string) => {
    if (!onLoadSample) return;
    setLoadingSample(sampleName);
    try {
      await onLoadSample(sampleName);
    } finally {
      setLoadingSample(null);
    }
  };

  if (!predictData) {
    return (
      <div className="fo-canvas-workspace" style={{ padding: '48px', alignItems: 'center', justifyContent: 'center' }}>
        <div
          style={{
            background: 'var(--fo-bg-panel)',
            border: '1px solid var(--fo-border)',
            borderRadius: '10px',
            padding: '36px 44px',
            maxWidth: '560px',
            textAlign: 'center',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.04)',
          }}
        >
          <div
            style={{
              width: '48px',
              height: '48px',
              borderRadius: '10px',
              background: 'var(--fo-accent-green-light)',
              color: 'var(--fo-accent-green)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 16px',
              border: '1px solid var(--fo-accent-green-border)',
            }}
          >
            <MapPin size={24} />
          </div>
          <div style={{ fontSize: '17px', fontWeight: 700, color: 'var(--fo-text-main)' }}>
            Weed Spatial Localization
          </div>
          <div style={{ fontSize: '13px', color: 'var(--fo-text-subtle)', marginTop: '8px', lineHeight: 1.6 }}>
            Inspect pixel-level bounding coordinates, centroid positions, box areas, and spatial distribution across field coordinates.
          </div>

          <div style={{ marginTop: '24px', paddingTop: '20px', borderTop: '1px solid var(--fo-border)' }}>
            <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--fo-text-muted)', textTransform: 'uppercase', marginBottom: '10px' }}>
              Load a sample photograph to inspect coordinates
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', justifyContent: 'center' }}>
              {['img112.jpg', 'img26.jpg', 'img481.jpg', 'img397.jpg'].map((sample) => (
                <button
                  key={sample}
                  className="fo-btn fo-btn-secondary"
                  disabled={!!loadingSample}
                  onClick={() => handleSampleClick(sample)}
                  style={{ fontSize: '12px', padding: '6px 14px' }}
                >
                  <Sparkles size={12} style={{ color: 'var(--fo-accent-green)', marginRight: '4px' }} />
                  {loadingSample === sample ? 'Analyzing...' : sample}
                </button>
              ))}
            </div>
          </div>

          {onNavigate && (
            <div style={{ marginTop: '20px' }}>
              <button
                className="fo-btn fo-btn-primary"
                style={{ fontSize: '12px', padding: '7px 18px' }}
                onClick={() => onNavigate('detection')}
              >
                Go to Weed Detection <ArrowRight size={13} style={{ marginLeft: '4px' }} />
              </button>
            </div>
          )}
        </div>
      </div>
    );
  }

  const { width, height, detections } = predictData;

  const weeds = useMemo(() => {
    return detections.filter((d) => d.cls_name.toLowerCase().includes('weed') || d.cls_id === 0);
  }, [detections]);

  const crops = useMemo(() => {
    return detections.filter((d) => !d.cls_name.toLowerCase().includes('weed') && d.cls_id !== 0);
  }, [detections]);

  const shownDetections: Detection[] = useMemo(() => {
    const list = [...weeds, ...(showCrops ? crops : [])];
    return list.filter((d) => d.conf >= minConf);
  }, [weeds, crops, showCrops, minConf]);

  const weedIds = useMemo(() => {
    return weeds.filter((d) => d.conf >= minConf).map((d) => d.uid);
  }, [weeds, minConf]);

  // Dimensions of the display frame (generous, full-sized image viewer)
  const displayWidth = 840;
  const aspectRatio = (height && width) ? height / width : 0.75;
  const displayHeight = Math.round(displayWidth * aspectRatio);

  // Generate coordinate axis tick intervals (every 200px up to width and height)
  const xTicks = useMemo(() => {
    const step = width > 2400 ? 400 : 200;
    const ticks: number[] = [];
    for (let x = 0; x <= width; x += step) {
      ticks.push(x);
    }
    return ticks;
  }, [width]);

  const yTicks = useMemo(() => {
    const step = height > 2400 ? 400 : 200;
    const ticks: number[] = [];
    for (let y = 0; y <= height; y += step) {
      ticks.push(y);
    }
    return ticks;
  }, [height]);

  // 2D Spatial Distribution Heatmap (8 columns, 6 rows matching 05_localization.jpg lines 58-63)
  const heatmap2D = useMemo(() => {
    const cols = 8;
    const rows = 6;
    const grid = Array.from({ length: rows }, () => Array(cols).fill(0));
    const cellW = width / cols;
    const cellH = height / rows;

    weeds.forEach((d) => {
      const c = Math.min(Math.floor(d.cx / cellW), cols - 1);
      const r = Math.min(Math.floor(d.cy / cellH), rows - 1);
      if (r >= 0 && r < rows && c >= 0 && c < cols) {
        grid[r][c] += 1;
      }
    });

    let max = 1;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (grid[r][c] > max) max = grid[r][c];
      }
    }

    return { grid, cols, rows, max, cellW, cellH };
  }, [weeds, width, height]);

  const handleExportCsv = () => {
    const header = "uid,cls_name,conf,x1,y1,x2,y2,cx,cy,w,h,area_px,area_pct\n";
    const rows = shownDetections.map((d: any) => {
      const areaPct = d.area_pct ?? (100 * (d.area_px || 0) / ((predictData.width || 1) * (predictData.height || 1)));
      return `${d.uid},${d.cls_name},${(d.conf || 0).toFixed(4)},${(d.x1 || 0).toFixed(1)},${(d.y1 || 0).toFixed(1)},${(d.x2 || 0).toFixed(1)},${(d.y2 || 0).toFixed(1)},${(d.cx || 0).toFixed(1)},${(d.cy || 0).toFixed(1)},${(d.w || d.x2 - d.x1 || 0).toFixed(1)},${(d.h || d.y2 - d.y1 || 0).toFixed(1)},${(d.area_px || 0).toFixed(0)},${areaPct.toFixed(4)}`;
    }).join("\n");
    const blob = new Blob([header + rows], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${predictData.filename.replace(/\.[^/.]+$/, "")}_weed_coordinates.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div
      style={{
        padding: '20px 24px',
        overflowY: 'auto',
        display: 'flex',
        flexDirection: 'column',
        gap: '18px',
        flex: 1,
        minWidth: 0,
      }}
    >
      {/* Hero Header matching 05_localization.jpg */}
      <div
        style={{
          background: 'var(--fo-bg-panel)',
          border: '1px solid var(--fo-border)',
          borderRadius: '8px',
          padding: '16px 20px',
          display: 'flex',
          flexWrap: 'wrap',
          gap: '16px',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexShrink: 0,
        }}
      >
        <div>
          <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 700, color: 'var(--fo-text-main)', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <MapPin size={18} style={{ color: 'var(--fo-accent-green)' }} /> Weed Localization
          </h2>
          <div style={{ fontSize: '12px', color: 'var(--fo-text-subtle)', marginTop: '4px' }}>
            Weed IDs, pixel coordinates, confidence and box area on an interactive image-coordinate plot.
          </div>
        </div>

        {/* Zoom Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <button className="fo-icon-button" onClick={zoomOut} title="Zoom Out" style={{ width: '28px', height: '28px' }}>
            <ZoomOut size={13} />
          </button>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', minWidth: '40px', textAlign: 'center', color: 'var(--fo-text-subtle)' }}>
            {zoom}%
          </span>
          <button className="fo-icon-button" onClick={zoomIn} title="Zoom In" style={{ width: '28px', height: '28px' }}>
            <ZoomIn size={13} />
          </button>
          <button className="fo-icon-button" onClick={resetZoom} title="Reset to 100%" style={{ width: '28px', height: '28px' }}>
            <RotateCcw size={13} />
          </button>
        </div>
      </div>

      {/* Image Context Bar matching 05_localization.jpg lines 8-18 */}
      <div
        style={{
          background: 'var(--fo-bg-panel)',
          border: '1px solid var(--fo-border)',
          borderRadius: '6px',
          padding: '12px 18px',
          display: 'flex',
          flexWrap: 'wrap',
          justifyContent: 'space-between',
          alignItems: 'center',
          gap: '12px',
          flexShrink: 0,
        }}
      >
        {/* Left: Analysed Image Selector */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--fo-text-subtle)' }}>
            Analysed image:
          </span>
          <select
            value={predictData.filename}
            onChange={(e) => {
              const selected = sessionResults.find((s) => s.filename === e.target.value);
              if (selected && onSelectHistory) {
                onSelectHistory(selected);
              }
            }}
            style={{
              background: 'var(--fo-bg-elevated)',
              border: '1px solid var(--fo-border)',
              borderRadius: '4px',
              color: 'var(--fo-text-main)',
              fontSize: '12px',
              padding: '4px 10px',
              fontFamily: 'var(--font-mono)',
              minWidth: '240px',
            }}
          >
            {sessionResults.map((s, idx) => (
              <option key={idx} value={s.filename}>
                {s.filename} — {s.checkpoint ? s.checkpoint.split(/[\\/]/).pop() : 'model'}
              </option>
            ))}
            {!sessionResults.some((s) => s.filename === predictData.filename) && (
              <option value={predictData.filename}>
                {predictData.filename}
              </option>
            )}
          </select>
        </div>

        {/* Right: Quick Samples */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '11px', color: 'var(--fo-text-muted)' }}>Analyse sample:</span>
          {['img112.jpg', 'img26.jpg', 'img481.jpg', 'img397.jpg'].map((sample) => (
            <button
              key={sample}
              className="fo-btn fo-btn-secondary"
              disabled={!!loadingSample}
              onClick={() => handleSampleClick(sample)}
              style={{
                fontSize: '11px',
                padding: '3px 8px',
                fontFamily: 'var(--font-mono)',
                borderColor: predictData.filename.includes(sample) ? 'var(--fo-accent-green)' : undefined,
                color: predictData.filename.includes(sample) ? 'var(--fo-accent-green)' : undefined,
              }}
            >
              {loadingSample === sample ? '...' : sample}
            </button>
          ))}
        </div>
      </div>

      {/* Controls Bar matching 05_localization.jpg lines 22-27 */}
      <div
        style={{
          background: 'var(--fo-bg-panel)',
          border: '1px solid var(--fo-border)',
          borderRadius: '6px',
          padding: '12px 18px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '16px',
          flexShrink: 0,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '24px', flexWrap: 'wrap' }}>
          {/* Toggle crops */}
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', cursor: 'pointer', userSelect: 'none' }}>
            <input
              type="checkbox"
              className="fo-checkbox"
              checked={showCrops}
              onChange={(e) => setShowCrops(e.target.checked)}
            />
            <span style={{ fontWeight: 500, color: 'var(--fo-text-main)' }}>Show crop detections ({crops.length})</span>
          </label>

          {/* Min confidence slider */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '11px', fontWeight: 600, color: 'var(--fo-text-subtle)' }}>Min confidence</span>
            <input
              type="range"
              min="0.0"
              max="1.0"
              step="0.01"
              value={minConf}
              onChange={(e) => setMinConf(parseFloat(e.target.value))}
              style={{ width: '110px', accentColor: 'var(--fo-accent-green)', cursor: 'pointer' }}
            />
            <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', fontWeight: 700, minWidth: '32px', color: 'var(--fo-accent-green)' }}>
              {minConf.toFixed(2)}
            </span>
          </div>

          {/* Highlight weed dropdown */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '11px', fontWeight: 600, color: 'var(--fo-text-subtle)' }}>Highlight weed</span>
            <select
              style={{
                background: 'var(--fo-bg-elevated)',
                border: '1px solid var(--fo-border)',
                color: 'var(--fo-text-main)',
                padding: '4px 8px',
                borderRadius: '4px',
                fontSize: '11px',
                fontFamily: 'var(--font-mono)',
              }}
              value={highlightWeed}
              onChange={(e) => setHighlightWeed(e.target.value)}
            >
              <option value="—">— None —</option>
              {weedIds.map((id) => (
                <option key={id} value={id}>
                  {id}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Image metadata readout matching 05_localization.jpg line 28 */}
        <div style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--fo-text-muted)' }}>
          <span style={{ fontWeight: 700, color: 'var(--fo-text-main)' }}>{weedIds.length} weeds</span> · image {width}×{height} px (origin top-left)
        </div>
      </div>

      {/* FULL-SIZED Interactive Coordinate Plot Canvas matching 05_localization.jpg */}
      <div
        style={{
          background: 'var(--fo-bg-panel)',
          border: '1px solid var(--fo-border)',
          borderRadius: '8px',
          padding: '24px 20px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: `${displayHeight + 110}px`,
          flexShrink: 0,
          overflow: 'auto',
          position: 'relative',
        }}
      >
        {/* Scaled Plot Container */}
        <div
          style={{
            position: 'relative',
            display: 'inline-flex',
            flexDirection: 'column',
            alignItems: 'flex-start',
            transform: `scale(${zoom / 100})`,
            transformOrigin: 'top center',
            transition: 'transform 0.15s ease-out',
            margin: '0 auto',
          }}
        >
          {/* Main Coordinate Row: Y-Axis + Image Canvas */}
          <div style={{ display: 'flex', alignItems: 'stretch' }}>
            {/* Y-Axis Column */}
            <div
              style={{
                width: '56px',
                position: 'relative',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
                paddingRight: '6px',
                userSelect: 'none',
              }}
            >
              {/* Y-Axis Label */}
              <div
                style={{
                  position: 'absolute',
                  left: '-4px',
                  top: '50%',
                  transform: 'translateY(-50%) rotate(-90deg)',
                  fontSize: '11px',
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--fo-text-muted)',
                }}
              >
                y (px)
              </div>

              {/* Y-Axis Ticks */}
              {yTicks.map((y) => {
                const ratio = (y / height) * 100;
                return (
                  <div
                    key={y}
                    style={{
                      position: 'absolute',
                      top: `${ratio}%`,
                      right: '6px',
                      transform: 'translateY(-50%)',
                      fontSize: '9px',
                      fontFamily: 'var(--font-mono)',
                      color: 'var(--fo-text-muted)',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '4px',
                    }}
                  >
                    <span>{y}</span>
                    <span style={{ width: '4px', height: '1px', background: 'var(--fo-text-subtle)' }} />
                  </div>
                );
              })}
            </div>

            {/* Field Image Canvas + SVG Bounding Boxes Overlay */}
            <div
              style={{
                position: 'relative',
                width: `${displayWidth}px`,
                height: `${displayHeight}px`,
                background: '#090a0f',
                border: '1px solid #334155',
                borderRadius: '2px',
                boxShadow: '0 4px 18px rgba(0, 0, 0, 0.12)',
                lineHeight: 0,
                overflow: 'hidden',
              }}
            >
              {/* Background Field Image */}
              <img
                src={predictData.original_image || predictData.annotated_image}
                alt="Field photograph for weed localization"
                style={{
                  width: '100%',
                  height: '100%',
                  objectFit: 'fill',
                  display: 'block',
                }}
              />

              {/* Interactive SVG Bounding Boxes Overlay */}
              <svg
                viewBox={`0 0 ${width} ${height}`}
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '100%',
                }}
              >
                {shownDetections.map((d) => {
                  const isHl = d.uid === highlightWeed;
                  const isWeed = d.cls_name.toLowerCase().includes('weed');
                  const strokeColor = isHl ? '#22c55e' : isWeed ? '#0284c7' : '#16a34a';
                  const strokeWidth = isHl ? 3.5 : 1.5;

                  return (
                    <g
                      key={d.uid}
                      style={{ cursor: 'pointer' }}
                      onClick={() => setHighlightWeed(isHl ? '—' : d.uid)}
                    >
                      {/* Bounding Box Rect matching 05_localization.jpg */}
                      <rect
                        x={d.x1}
                        y={d.y1}
                        width={Math.max(2, d.x2 - d.x1)}
                        height={Math.max(2, d.y2 - d.y1)}
                        fill={isHl ? strokeColor : 'none'}
                        fillOpacity={isHl ? 0.28 : 0}
                        stroke={strokeColor}
                        strokeWidth={strokeWidth}
                      >
                        <title>{`${d.uid} (${d.cls_name}): conf ${((d.conf || 0) * 100).toFixed(1)}% | (${Math.round(d.cx)}, ${Math.round(d.cy)})`}</title>
                      </rect>

                      {/* Centroid Bullet Dot matching 05_localization.jpg */}
                      <circle
                        cx={d.cx}
                        cy={d.cy}
                        r={isHl ? 5.5 : 3.5}
                        fill={strokeColor}
                        stroke="#ffffff"
                        strokeWidth={1}
                      />
                    </g>
                  );
                })}
              </svg>
            </div>
          </div>

          {/* Bottom X-Axis Row */}
          <div style={{ display: 'flex', width: '100%' }}>
            {/* Blank corner matching Y-axis width */}
            <div style={{ width: '56px' }} />

            {/* X-Axis Numbers & Ticks */}
            <div
              style={{
                position: 'relative',
                width: `${displayWidth}px`,
                height: '36px',
                marginTop: '4px',
                userSelect: 'none',
              }}
            >
              {xTicks.map((x) => {
                const ratio = (x / width) * 100;
                return (
                  <div
                    key={x}
                    style={{
                      position: 'absolute',
                      left: `${ratio}%`,
                      top: '0',
                      transform: 'translateX(-50%)',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                    }}
                  >
                    <span style={{ width: '1px', height: '4px', background: 'var(--fo-text-subtle)' }} />
                    <span style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', color: 'var(--fo-text-muted)', marginTop: '2px' }}>
                      {x}
                    </span>
                  </div>
                );
              })}

              {/* X-Axis Label */}
              <div
                style={{
                  position: 'absolute',
                  bottom: '0',
                  left: '50%',
                  transform: 'translateX(-50%)',
                  fontSize: '11px',
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--fo-text-muted)',
                }}
              >
                x (px)
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Detections Coordinate Directory Table matching 05_localization.jpg lines 50-60 */}
      <div
        style={{
          background: 'var(--fo-bg-panel)',
          border: '1px solid var(--fo-border)',
          borderRadius: '6px',
          padding: '18px',
          flexShrink: 0,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
          <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--fo-text-main)', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Target size={14} style={{ color: 'var(--fo-accent-green)' }} /> Weed Coordinates Directory ({shownDetections.length} Targets)
          </div>
          <button className="fo-btn fo-btn-secondary" onClick={handleExportCsv} style={{ fontSize: '11px', padding: '4px 10px', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Download size={13} /> Download Coordinates CSV
          </button>
        </div>

        <div style={{ maxHeight: '340px', overflowY: 'auto', border: '1px solid var(--fo-border)', borderRadius: '4px' }}>
          <table className="fo-json-tree" style={{ width: '100%', color: 'var(--fo-text-main)', borderCollapse: 'collapse', fontSize: '11px' }}>
            <thead>
              <tr style={{ position: 'sticky', top: 0, background: 'var(--fo-bg-elevated)', borderBottom: '1px solid var(--fo-border)', textTransform: 'uppercase', color: 'var(--fo-text-muted)', zIndex: 1 }}>
                <th style={{ padding: '6px 8px', textAlign: 'left' }}>uid</th>
                <th style={{ padding: '6px 8px', textAlign: 'left' }}>cls_name</th>
                <th style={{ padding: '6px 8px', textAlign: 'left' }}>conf</th>
                <th style={{ padding: '6px 8px', textAlign: 'left' }}>x1</th>
                <th style={{ padding: '6px 8px', textAlign: 'left' }}>y1</th>
                <th style={{ padding: '6px 8px', textAlign: 'left' }}>x2</th>
                <th style={{ padding: '6px 8px', textAlign: 'left' }}>y2</th>
                <th style={{ padding: '6px 8px', textAlign: 'left' }}>cx</th>
                <th style={{ padding: '6px 8px', textAlign: 'left' }}>cy</th>
                <th style={{ padding: '6px 8px', textAlign: 'left' }}>w</th>
                <th style={{ padding: '6px 8px', textAlign: 'left' }}>h</th>
                <th style={{ padding: '6px 8px', textAlign: 'left' }}>area_px</th>
                <th style={{ padding: '6px 8px', textAlign: 'left' }}>area_pct</th>
              </tr>
            </thead>
            <tbody>
              {shownDetections.map((d) => {
                const isHl = d.uid === highlightWeed;
                const isWeed = d.cls_name.toLowerCase().includes('weed');
                const wVal = (d.w || d.x2 - d.x1).toFixed(1);
                const hVal = (d.h || d.y2 - d.y1).toFixed(1);
                const areaPct = (d.area_pct ?? (100 * (d.area_px || 0) / ((predictData.width || 1) * (predictData.height || 1)))).toFixed(4);

                return (
                  <tr
                    key={d.uid}
                    style={{
                      borderBottom: '1px solid var(--fo-border)',
                      background: isHl ? 'var(--fo-accent-green-light)' : 'transparent',
                      cursor: 'pointer',
                    }}
                    onClick={() => setHighlightWeed(isHl ? '—' : d.uid)}
                  >
                    <td style={{ padding: '6px 8px', fontWeight: 700, color: isHl ? 'var(--fo-accent-green)' : isWeed ? '#0284c7' : '#16a34a' }}>
                      {d.uid}
                    </td>
                    <td style={{ padding: '6px 8px' }}>{d.cls_name}</td>
                    <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)' }}>{(d.conf || 0).toFixed(4)}</td>
                    <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)' }}>{d.x1.toFixed(1)}</td>
                    <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)' }}>{d.y1.toFixed(1)}</td>
                    <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)' }}>{d.x2.toFixed(1)}</td>
                    <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)' }}>{d.y2.toFixed(1)}</td>
                    <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)' }}>{Math.round(d.cx)}</td>
                    <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)' }}>{d.cy.toFixed(1)}</td>
                    <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)' }}>{wVal}</td>
                    <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)' }}>{hVal}</td>
                    <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)' }}>{Math.round(d.area_px || 0).toLocaleString()}</td>
                    <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)' }}>{areaPct}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Weed Spatial Distribution 2D Heatmap matching 05_localization.jpg line 62 */}
      {weeds.length >= 2 && (
        <div
          style={{
            background: 'var(--fo-bg-panel)',
            border: '1px solid var(--fo-border)',
            borderRadius: '6px',
            padding: '18px',
            flexShrink: 0,
          }}
        >
          <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--fo-text-main)', display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px' }}>
            <Grid size={14} style={{ color: 'var(--fo-accent-green)' }} /> Weed Spatial Distribution (count per grid cell)
          </div>
          <div style={{ fontSize: '12px', color: 'var(--fo-text-subtle)', marginBottom: '14px' }}>
            8×6 density binning across image coordinates (X: 0 to {width}px, Y: 0 to {height}px).
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: `repeat(${heatmap2D.cols}, 1fr)`, gap: '4px' }}>
            {heatmap2D.grid.map((row, rIdx) =>
              row.map((val, cIdx) => {
                const ratio = val / heatmap2D.max;
                const bg = val > 0 ? `rgba(2, 132, 199, ${0.15 + ratio * 0.75})` : 'var(--fo-bg-elevated)';
                return (
                  <div
                    key={`${rIdx}-${cIdx}`}
                    title={`Bin [${cIdx}, ${rIdx}]: ${val} weeds`}
                    style={{
                      aspectRatio: '1.4',
                      borderRadius: '4px',
                      background: bg,
                      border: '1px solid var(--fo-border)',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '11px',
                      fontFamily: 'var(--font-mono)',
                      color: val > 0 ? '#fff' : 'var(--fo-text-muted)',
                      fontWeight: 700,
                    }}
                  >
                    <span>{val}</span>
                  </div>
                );
              })
            )}
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--fo-text-muted)', marginTop: '6px' }}>
            <span>Top-Left (0, 0)</span>
            <span>X (0 → {width} px)</span>
            <span>Bottom-Right ({width}, {height} px)</span>
          </div>
        </div>
      )}
    </div>
  );
};
