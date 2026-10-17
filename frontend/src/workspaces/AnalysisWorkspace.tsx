import React, { useState, useMemo } from 'react';
import type { PredictResponse, Detection } from '../types';
import {
  BarChart3,
  Grid,
  DollarSign,
  Download,
  Sliders,
  Sparkles,
  ArrowRight,
  Info,
} from 'lucide-react';

interface Props {
  predictData: PredictResponse | null;
  sessionResults?: PredictResponse[];
  onLoadSample?: (name?: string) => Promise<void>;
  onNavigate?: (ws: any) => void;
  onSelectHistory?: (item: PredictResponse) => void;
}

type TabMode = 'analysis-overview' | 'spatial-grid' | 'spray-roi';

export const AnalysisWorkspace: React.FC<Props> = ({
  predictData,
  sessionResults = [],
  onLoadSample,
  onNavigate,
  onSelectHistory,
}) => {
  const [activeTab, setActiveTab] = useState<TabMode>('analysis-overview');
  const [loadingSample, setLoadingSample] = useState<string | null>(null);

  // Interactive Quadrant Inspector State
  const [selectedCell, setSelectedCell] = useState<{ row: number; col: number } | null>({ row: 2, col: 3 });
  const [gridMetric, setGridMetric] = useState<'count' | 'coverage'>('count');

  // Interactive Economic ROI Calculator State
  const [fieldHectares, setFieldHectares] = useState<number>(50);
  const [sprayRateLPerHa, setSprayRateLPerHa] = useState<number>(200);
  const [herbicideCostPerL, setHerbicideCostPerL] = useState<number>(22);

  const handleSampleClick = async (sampleName: string) => {
    if (!onLoadSample) return;
    setLoadingSample(sampleName);
    try {
      await onLoadSample(sampleName);
    } finally {
      setLoadingSample(null);
    }
  };

  // If no prediction data is available, render an informative call-to-action with instant sample triggers
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
            <BarChart3 size={24} />
          </div>
          <div style={{ fontSize: '17px', fontWeight: 700, color: 'var(--fo-text-main)' }}>
            Weed Statistical Analysis
          </div>
          <div style={{ fontSize: '13px', color: 'var(--fo-text-subtle)', marginTop: '8px', lineHeight: 1.6 }}>
            Comprehensive analytics on weed count, density per megapixel, ground coverage, confidence distributions,
            crop-wise proximity associations, and precision spraying ROI.
          </div>

          <div style={{ marginTop: '24px', paddingTop: '20px', borderTop: '1px solid var(--fo-border)' }}>
            <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--fo-text-muted)', textTransform: 'uppercase', marginBottom: '10px' }}>
              Load a sample photograph to inspect analytics
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

  const { infestation, crop_association, per_weed_association, crop_count, width, height, detections } = predictData;

  const weeds = useMemo(() => {
    return detections.filter(d => d.cls_name.toLowerCase().includes('weed') || d.cls_id === 0);
  }, [detections]);

  const severityColor =
    infestation.severity === 'Low' ? '#16a34a' :
    infestation.severity === 'Moderate' ? '#d97706' :
    infestation.severity === 'High' ? '#ea580c' : '#dc2626';

  const severityBg =
    infestation.severity === 'Low' ? '#f0fdf4' :
    infestation.severity === 'Moderate' ? '#fffbeb' :
    infestation.severity === 'High' ? '#fff7ed' : '#fef2f2';

  const severityBorder =
    infestation.severity === 'Low' ? '#bbf7d0' :
    infestation.severity === 'Moderate' ? '#fde68a' :
    infestation.severity === 'High' ? '#fed7aa' : '#fecaca';

  // 1. Confidence Distribution Bins (0.0 to 1.0 with 0.05 bins matching 04_weed_analysis.jpg)
  const confHistogram = useMemo(() => {
    const bins = Array.from({ length: 20 }, (_, i) => ({
      start: (i * 0.05).toFixed(2),
      end: ((i + 1) * 0.05).toFixed(2),
      label: `${(i * 0.05).toFixed(2)} - ${((i + 1) * 0.05).toFixed(2)}`,
      count: 0,
    }));

    weeds.forEach((d) => {
      const idx = Math.min(Math.floor(d.conf / 0.05), 19);
      if (idx >= 0 && idx < 20) {
        bins[idx].count++;
      }
    });

    const max = Math.max(...bins.map((b) => b.count), 1);
    return { bins, max };
  }, [weeds]);

  // 2. Box Area Percentage Bins matching 04_weed_analysis.jpg
  const areaHistogram = useMemo(() => {
    const totalArea = Math.max(width * height, 1);
    const areas = weeds.map((d) => (100 * d.area_px) / totalArea);
    const maxArea = areas.length ? Math.max(...areas, 1.5) : 1.5;
    const numBins = 14;
    const step = maxArea / numBins;

    const bins = Array.from({ length: numBins }, (_, i) => ({
      start: (i * step).toFixed(2),
      end: ((i + 1) * step).toFixed(2),
      label: `${(i * step).toFixed(2)}% - ${((i + 1) * step).toFixed(2)}%`,
      count: 0,
    }));

    areas.forEach((a) => {
      const idx = Math.min(Math.floor(a / step), numBins - 1);
      if (idx >= 0 && idx < numBins) {
        bins[idx].count++;
      }
    });

    const max = Math.max(...bins.map((b) => b.count), 1);
    return { bins, max, step, maxArea };
  }, [weeds, width, height]);

  // 3. Compute 6x6 Spatial Field Grid
  const NUM_ROWS = 6;
  const NUM_COLS = 6;
  const ROW_LABELS = ['A', 'B', 'C', 'D', 'E', 'F'];

  const gridData = useMemo(() => {
    const grid: {
      count: number;
      area: number;
      weeds: Detection[];
      avgConf: number;
    }[][] = Array.from({ length: NUM_ROWS }, () =>
      Array.from({ length: NUM_COLS }, () => ({
        count: 0,
        area: 0,
        weeds: [],
        avgConf: 0,
      }))
    );

    const cellW = width / NUM_COLS;
    const cellH = height / NUM_ROWS;

    weeds.forEach((d) => {
      const c = Math.min(Math.floor(d.cx / cellW), NUM_COLS - 1);
      const r = Math.min(Math.floor(d.cy / cellH), NUM_ROWS - 1);
      if (r >= 0 && r < NUM_ROWS && c >= 0 && c < NUM_COLS) {
        grid[r][c].count += 1;
        grid[r][c].area += d.area_px;
        grid[r][c].weeds.push(d);
      }
    });

    let maxCount = 1;
    let maxArea = 1;
    for (let r = 0; r < NUM_ROWS; r++) {
      for (let c = 0; c < NUM_COLS; c++) {
        if (grid[r][c].count > maxCount) maxCount = grid[r][c].count;
        if (grid[r][c].area > maxArea) maxArea = grid[r][c].area;
        if (grid[r][c].weeds.length > 0) {
          const sumConf = grid[r][c].weeds.reduce((acc, w) => acc + w.conf, 0);
          grid[r][c].avgConf = sumConf / grid[r][c].weeds.length;
        }
      }
    }

    return { grid, maxCount, maxArea, cellW, cellH };
  }, [weeds, width, height]);

  // 4. Herbicide Savings ROI Calculations
  const roiCalculations = useMemo(() => {
    const totalVolumeConventional = fieldHectares * sprayRateLPerHa;
    const totalCostConventional = totalVolumeConventional * herbicideCostPerL;
    const bufferFactor = 1.15;
    const effectiveSpotCoverage = Math.min(100, infestation.infestation_pct * bufferFactor);
    const volumePrecision = totalVolumeConventional * (effectiveSpotCoverage / 100);
    const volumeSaved = Math.max(0, totalVolumeConventional - volumePrecision);
    const costPrecision = volumePrecision * herbicideCostPerL;
    const costSaved = Math.max(0, totalCostConventional - costPrecision);
    const savingsPct = totalVolumeConventional > 0 ? (volumeSaved / totalVolumeConventional) * 100 : 0;
    const activeChemicalAvoidedKg = (volumeSaved * 0.0045).toFixed(1);

    return {
      totalVolumeConventional,
      totalCostConventional,
      volumePrecision,
      volumeSaved,
      costPrecision,
      costSaved,
      savingsPct,
      activeChemicalAvoidedKg,
    };
  }, [fieldHectares, sprayRateLPerHa, herbicideCostPerL, infestation.infestation_pct]);

  const handleDownloadAssociationCsv = () => {
    if (!per_weed_association || per_weed_association.length === 0) return;
    const header = "weed_uid,cls_name,conf,associated_class,associated_uid,distance_px,distance_frac,within_radius\n";
    const rows = per_weed_association.map((a: any) =>
      `${a.weed_uid},${a.cls_name},${a.conf},${a.associated_class},${a.associated_uid},${a.distance_px},${a.distance_frac},${a.within_radius}`
    ).join("\n");
    const blob = new Blob([header + rows], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const el = document.createElement("a");
    el.href = url;
    el.download = `${predictData.filename.replace(/\.[^/.]+$/, '')}_weed_crop_association.csv`;
    el.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadBatchSummaryCsv = () => {
    if (sessionResults.length === 0) return;
    const header = "image,model,weeds,crops,density_per_MP,infestation_pct,severity\n";
    const rows = sessionResults.map((r) =>
      `${r.filename},${r.checkpoint ? r.checkpoint.split(/[\\/]/).pop() : 'model'},${r.weed_count},${r.crop_count},${r.infestation.density_per_mpx.toFixed(2)},${r.infestation.infestation_pct.toFixed(2)},${r.infestation.severity}`
    ).join("\n");
    const blob = new Blob([header + rows], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const el = document.createElement("a");
    el.href = url;
    el.download = `weed_analysis_summary.csv`;
    el.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div style={{ padding: '20px 24px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '18px', flex: 1 }}>
      {/* Top Bar Header with Title, Mode Tabs, and Image Switcher matching 04_weed_analysis.jpg */}
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
        }}
      >
        <div>
          <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 700, color: 'var(--fo-text-main)', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <BarChart3 size={18} style={{ color: 'var(--fo-accent-green)' }} /> Weed Analysis
          </h2>
          <div style={{ fontSize: '12px', color: 'var(--fo-text-subtle)', marginTop: '4px' }}>
            Count, density, area, infestation, confidence distribution, severity and crop-wise association.
          </div>
        </div>

        {/* View Tabs */}
        <div style={{ display: 'flex', gap: '6px' }}>
          <button
            className={`fo-btn ${activeTab === 'analysis-overview' ? 'fo-btn-primary' : 'fo-btn-secondary'}`}
            style={{ fontSize: '11px', padding: '6px 14px' }}
            onClick={() => setActiveTab('analysis-overview')}
          >
            Statistical Analysis
          </button>
          <button
            className={`fo-btn ${activeTab === 'spatial-grid' ? 'fo-btn-primary' : 'fo-btn-secondary'}`}
            style={{ fontSize: '11px', padding: '6px 14px' }}
            onClick={() => setActiveTab('spatial-grid')}
          >
            Spatial Field Grid
          </button>
          <button
            className={`fo-btn ${activeTab === 'spray-roi' ? 'fo-btn-primary' : 'fo-btn-secondary'}`}
            style={{ fontSize: '11px', padding: '6px 14px' }}
            onClick={() => setActiveTab('spray-roi')}
          >
            Herbicide Savings ROI
          </button>
        </div>
      </div>

      {/* Image Context Bar: Selector & Quick Samples matching 04_weed_analysis.jpg lines 8-18 */}
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

      {/* 5 KPI Metric Cards matching 04_weed_analysis.jpg lines 23-34 */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '12px' }}>
        {/* Card 1: Weed count */}
        <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '14px' }}>
          <div style={{ fontSize: '10px', fontWeight: 700, color: 'var(--fo-text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Weed Count
          </div>
          <div style={{ fontSize: '24px', fontWeight: 700, color: 'var(--fo-accent-green)', fontFamily: 'var(--font-mono)', marginTop: '2px' }}>
            {infestation.weed_count}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--fo-text-subtle)', marginTop: '2px' }}>
            {crop_count} crops detected
          </div>
        </div>

        {/* Card 2: Density */}
        <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '14px' }}>
          <div style={{ fontSize: '10px', fontWeight: 700, color: 'var(--fo-text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Density
          </div>
          <div style={{ fontSize: '24px', fontWeight: 700, color: 'var(--fo-text-main)', fontFamily: 'var(--font-mono)', marginTop: '2px' }}>
            {infestation.density_per_m2 ? `${infestation.density_per_m2.toFixed(2)}/m²` : `${infestation.density_per_mpx.toFixed(2)}/MP`}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--fo-text-subtle)', marginTop: '2px' }}>
            {infestation.density_per_m2 ? `Ground area ${infestation.ground_area_m2?.toFixed(1)} m²` : 'per megapixel (set GSD in Settings)'}
          </div>
        </div>

        {/* Card 3: Infestation */}
        <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '14px' }}>
          <div style={{ fontSize: '10px', fontWeight: 700, color: 'var(--fo-text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Infestation
          </div>
          <div style={{ fontSize: '24px', fontWeight: 700, color: severityColor, fontFamily: 'var(--font-mono)', marginTop: '2px' }}>
            {infestation.infestation_pct.toFixed(2)}%
          </div>
          <div style={{ fontSize: '11px', color: 'var(--fo-text-subtle)', marginTop: '2px' }}>
            union of weed boxes / image area
          </div>
        </div>

        {/* Card 4: Weed Box Area */}
        <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '14px' }}>
          <div style={{ fontSize: '10px', fontWeight: 700, color: 'var(--fo-text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Weed Box Area
          </div>
          <div style={{ fontSize: '24px', fontWeight: 700, color: 'var(--fo-text-main)', fontFamily: 'var(--font-mono)', marginTop: '2px' }}>
            {(infestation.union_box_area_px / 1000).toFixed(1)}k px
          </div>
          <div style={{ fontSize: '11px', color: 'var(--fo-text-subtle)', marginTop: '2px' }}>
            sum {(infestation.sum_box_area_px / 1000).toFixed(1)}k · mean {(infestation.mean_box_area_px / 1000).toFixed(1)}k
          </div>
        </div>

        {/* Card 5: Mean confidence */}
        <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '14px' }}>
          <div style={{ fontSize: '10px', fontWeight: 700, color: 'var(--fo-text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Mean Confidence
          </div>
          <div style={{ fontSize: '24px', fontWeight: 700, color: 'var(--fo-text-main)', fontFamily: 'var(--font-mono)', marginTop: '2px' }}>
            {infestation.mean_conf !== undefined && infestation.mean_conf !== null ? infestation.mean_conf.toFixed(3) : '—'}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--fo-text-subtle)', marginTop: '2px' }}>
            {infestation.min_conf !== undefined ? `min ${infestation.min_conf.toFixed(2)} · max ${infestation.max_conf?.toFixed(2)}` : ''}
          </div>
        </div>
      </div>

      {/* Severity Guide Pill Banner matching 04_weed_analysis.jpg lines 35-40 */}
      <div
        style={{
          background: severityBg,
          border: `1px solid ${severityBorder}`,
          borderRadius: '6px',
          padding: '10px 16px',
          display: 'flex',
          flexWrap: 'wrap',
          alignItems: 'center',
          gap: '12px',
          fontSize: '12px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontWeight: 600, color: 'var(--fo-text-main)' }}>Severity:</span>
          <span
            style={{
              padding: '2px 10px',
              borderRadius: '3px',
              fontSize: '11px',
              fontWeight: 700,
              background: severityColor,
              color: '#ffffff',
            }}
          >
            {infestation.severity.toUpperCase()}
          </span>
        </div>
        <div style={{ color: 'var(--fo-text-subtle)', fontSize: '11px' }}>
          thresholds: Low &lt; 5% ≤ Moderate &lt; 15% ≤ High &lt; 30% ≤ Severe (change in Settings) · Infestation % is a box-based upper bound.
        </div>
      </div>

      {activeTab === 'analysis-overview' && (
        <>
          {/* Two High-Precision SVG Histograms Side-by-Side matching 04_weed_analysis.jpg lines 41-51 */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
            {/* Histogram 1: Weed Confidence Distribution */}
            <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '18px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--fo-text-main)' }}>
                  Weed confidence distribution
                </div>
                <div style={{ fontSize: '11px', color: 'var(--fo-text-muted)', fontFamily: 'var(--font-mono)' }}>
                  N = {weeds.length} weeds
                </div>
              </div>

              {/* High precision SVG Graph */}
              <div style={{ width: '100%', height: '210px' }}>
                <svg viewBox="0 0 500 200" style={{ width: '100%', height: '100%', overflow: 'visible' }}>
                  {/* Y-Axis Grid Lines & Ticks */}
                  {[0, 0.25, 0.5, 0.75, 1.0].map((ratio) => {
                    const y = 170 - ratio * 150;
                    const val = Math.round(ratio * confHistogram.max);
                    return (
                      <g key={ratio}>
                        <line x1="36" y1={y} x2="490" y2={y} stroke="var(--fo-border)" strokeDasharray="3 3" strokeWidth="1" />
                        <text x="28" y={y + 3} textAnchor="end" fontSize="10" fill="var(--fo-text-muted)" fontFamily="var(--font-mono)">
                          {val}
                        </text>
                      </g>
                    );
                  })}

                  {/* Y-Axis Label */}
                  <text
                    x="-90"
                    y="12"
                    transform="rotate(-90)"
                    textAnchor="middle"
                    fontSize="10"
                    fill="var(--fo-text-muted)"
                  >
                    weeds
                  </text>

                  {/* X-Axis Baseline */}
                  <line x1="36" y1="170" x2="490" y2="170" stroke="var(--fo-text-subtle)" strokeWidth="1" />

                  {/* Histogram Bars */}
                  {confHistogram.bins.map((b, i) => {
                    const barWidth = 430 / 20 - 3;
                    const x = 40 + i * (430 / 20);
                    const barHeight = confHistogram.max > 0 ? (b.count / confHistogram.max) * 150 : 0;
                    const y = 170 - barHeight;

                    return (
                      <g key={i}>
                        <rect
                          x={x}
                          y={y}
                          width={barWidth}
                          height={barHeight}
                          fill={b.count > 0 ? 'var(--fo-accent-green)' : '#cbd5e1'}
                          opacity={b.count > 0 ? 0.9 : 0.2}
                          rx="2"
                          style={{ transition: 'all 0.15s ease' }}
                        >
                          <title>{`Confidence ${b.label}: ${b.count} weeds`}</title>
                        </rect>
                      </g>
                    );
                  })}

                  {/* X-Axis Labels: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 */}
                  {[0, 0.2, 0.4, 0.6, 0.8, 1.0].map((c) => {
                    const x = 40 + (c / 1.0) * 430;
                    return (
                      <g key={c}>
                        <line x1={x} y1="170" x2={x} y2="174" stroke="var(--fo-text-subtle)" strokeWidth="1" />
                        <text x={x} y="186" textAnchor="middle" fontSize="10" fill="var(--fo-text-muted)" fontFamily="var(--font-mono)">
                          {c.toFixed(1)}
                        </text>
                      </g>
                    );
                  })}

                  {/* X-Axis Caption */}
                  <text x="260" y="198" textAnchor="middle" fontSize="10" fill="var(--fo-text-subtle)">
                    confidence
                  </text>
                </svg>
              </div>
            </div>

            {/* Histogram 2: Weed Box Size (% of image) */}
            <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '18px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--fo-text-main)' }}>
                  Weed box size (% of image)
                </div>
                <div style={{ fontSize: '11px', color: 'var(--fo-text-muted)', fontFamily: 'var(--font-mono)' }}>
                  Max: {areaHistogram.maxArea.toFixed(2)}%
                </div>
              </div>

              {/* High precision SVG Graph */}
              <div style={{ width: '100%', height: '210px' }}>
                <svg viewBox="0 0 500 200" style={{ width: '100%', height: '100%', overflow: 'visible' }}>
                  {/* Y-Axis Grid Lines & Ticks */}
                  {[0, 0.25, 0.5, 0.75, 1.0].map((ratio) => {
                    const y = 170 - ratio * 150;
                    const val = Math.round(ratio * areaHistogram.max);
                    return (
                      <g key={ratio}>
                        <line x1="36" y1={y} x2="490" y2={y} stroke="var(--fo-border)" strokeDasharray="3 3" strokeWidth="1" />
                        <text x="28" y={y + 3} textAnchor="end" fontSize="10" fill="var(--fo-text-muted)" fontFamily="var(--font-mono)">
                          {val}
                        </text>
                      </g>
                    );
                  })}

                  {/* Y-Axis Label */}
                  <text
                    x="-90"
                    y="12"
                    transform="rotate(-90)"
                    textAnchor="middle"
                    fontSize="10"
                    fill="var(--fo-text-muted)"
                  >
                    weeds
                  </text>

                  {/* X-Axis Baseline */}
                  <line x1="36" y1="170" x2="490" y2="170" stroke="var(--fo-text-subtle)" strokeWidth="1" />

                  {/* Histogram Bars */}
                  {areaHistogram.bins.map((b, i) => {
                    const numBins = areaHistogram.bins.length;
                    const barWidth = 430 / numBins - 3;
                    const x = 40 + i * (430 / numBins);
                    const barHeight = areaHistogram.max > 0 ? (b.count / areaHistogram.max) * 150 : 0;
                    const y = 170 - barHeight;

                    return (
                      <g key={i}>
                        <rect
                          x={x}
                          y={y}
                          width={barWidth}
                          height={barHeight}
                          fill="#0284c7"
                          opacity={b.count > 0 ? 0.85 : 0.2}
                          rx="2"
                          style={{ transition: 'all 0.15s ease' }}
                        >
                          <title>{`Box Size ${b.label}: ${b.count} weeds`}</title>
                        </rect>
                      </g>
                    );
                  })}

                  {/* X-Axis Ticks matching 04_weed_analysis.jpg (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4) */}
                  {[0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4].map((areaVal) => {
                    const ratio = Math.min(1.0, areaVal / areaHistogram.maxArea);
                    const x = 40 + ratio * 430;
                    return (
                      <g key={areaVal}>
                        <line x1={x} y1="170" x2={x} y2="174" stroke="var(--fo-text-subtle)" strokeWidth="1" />
                        <text x={x} y="186" textAnchor="middle" fontSize="10" fill="var(--fo-text-muted)" fontFamily="var(--font-mono)">
                          {areaVal.toFixed(1)}
                        </text>
                      </g>
                    );
                  })}

                  {/* X-Axis Caption */}
                  <text x="260" y="198" textAnchor="middle" fontSize="10" fill="var(--fo-text-subtle)">
                    box area (% of image)
                  </text>
                </svg>
              </div>
            </div>
          </div>

          {/* Crop-wise Weed Association matching 04_weed_analysis.jpg & 04b_crop_association.jpg */}
          <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '18px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
              <div style={{ fontSize: '14px', fontWeight: 700, color: 'var(--fo-text-main)' }}>
                Crop-wise weed association
              </div>
              {per_weed_association && per_weed_association.length > 0 && (
                <button className="fo-btn fo-btn-secondary" onClick={handleDownloadAssociationCsv} style={{ fontSize: '11px', padding: '4px 10px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <Download size={13} /> Download Association CSV
                </button>
              )}
            </div>

            <div style={{ fontSize: '12px', color: 'var(--fo-text-subtle)', marginBottom: '14px' }}>
              Each weed is linked to the nearest crop box (weed centre → crop box edge) within 15% of the image diagonal.
            </div>

            {crop_count === 0 ? (
              <div
                style={{
                  padding: '16px 20px',
                  background: 'var(--fo-bg-elevated)',
                  border: '1px solid var(--fo-border)',
                  borderRadius: '6px',
                  color: 'var(--fo-text-subtle)',
                  fontSize: '12px',
                  lineHeight: 1.6,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                }}
              >
                <Info size={18} style={{ color: 'var(--fo-text-muted)', flexShrink: 0 }} />
                <span>
                  No crop detections (Tomato / Rose / Marigold / Cauliflower) in this image, so weeds cannot be associated with crops.
                  This requires a model trained with crop classes.
                </span>
              </div>
            ) : (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '16px' }}>
                {/* Bar chart per crop matching 04b_crop_association.jpg */}
                <div style={{ background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '14px' }}>
                  <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--fo-text-muted)', marginBottom: '12px', textTransform: 'uppercase' }}>
                    Weeds associated per crop
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    {crop_association.map((c) => {
                      const maxW = Math.max(...crop_association.map((x) => x.associated_weeds), 1);
                      const pct = (c.associated_weeds / maxW) * 100;
                      return (
                        <div key={c.crop_class}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', marginBottom: '3px' }}>
                            <span style={{ fontWeight: 600, color: 'var(--fo-text-main)' }}>{c.crop_class}</span>
                            <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--fo-accent-green)' }}>
                              {c.associated_weeds} weeds
                            </span>
                          </div>
                          <div style={{ width: '100%', height: '8px', background: 'var(--fo-bg-panel)', borderRadius: '4px', overflow: 'hidden', border: '1px solid var(--fo-border)' }}>
                            <div style={{ width: `${pct}%`, height: '100%', background: 'var(--fo-accent-green)' }} />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Per-crop Table matching 04b_crop_association.jpg */}
                <div style={{ background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '14px', overflowX: 'auto' }}>
                  <table className="fo-json-tree" style={{ width: '100%', color: 'var(--fo-text-main)', borderCollapse: 'collapse', fontSize: '11px' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid var(--fo-border)', textTransform: 'uppercase', color: 'var(--fo-text-muted)' }}>
                        <th style={{ padding: '6px 8px', textAlign: 'left' }}>crop_class</th>
                        <th style={{ padding: '6px 8px', textAlign: 'left' }}>instances</th>
                        <th style={{ padding: '6px 8px', textAlign: 'left' }}>associated_weeds</th>
                        <th style={{ padding: '6px 8px', textAlign: 'left' }}>weeds/instance</th>
                      </tr>
                    </thead>
                    <tbody>
                      {crop_association.map((c) => (
                        <tr key={c.crop_class} style={{ borderBottom: '1px solid var(--fo-border)' }}>
                          <td style={{ padding: '6px 8px', fontWeight: 600 }}>{c.crop_class}</td>
                          <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)' }}>{c.crop_instances}</td>
                          <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)', color: 'var(--fo-accent-green)', fontWeight: 700 }}>{c.associated_weeds}</td>
                          <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)' }}>
                            {typeof c.weeds_per_crop_instance === 'number' ? c.weeds_per_crop_instance.toFixed(2) : '—'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Per-Weed Detailed Association Table */}
            {per_weed_association && per_weed_association.length > 0 && (
              <div style={{ marginTop: '12px', overflowX: 'auto', border: '1px solid var(--fo-border)', borderRadius: '4px' }}>
                <table className="fo-json-tree" style={{ width: '100%', color: 'var(--fo-text-main)', borderCollapse: 'collapse', fontSize: '11px' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--fo-border)', background: 'var(--fo-bg-elevated)', textTransform: 'uppercase', color: 'var(--fo-text-muted)' }}>
                      <th style={{ padding: '6px 8px', textAlign: 'left' }}>weed_uid</th>
                      <th style={{ padding: '6px 8px', textAlign: 'left' }}>cls_name</th>
                      <th style={{ padding: '6px 8px', textAlign: 'left' }}>conf</th>
                      <th style={{ padding: '6px 8px', textAlign: 'left' }}>associated_class</th>
                      <th style={{ padding: '6px 8px', textAlign: 'left' }}>associated_uid</th>
                      <th style={{ padding: '6px 8px', textAlign: 'left' }}>distance_px</th>
                      <th style={{ padding: '6px 8px', textAlign: 'left' }}>within_radius</th>
                    </tr>
                  </thead>
                  <tbody>
                    {per_weed_association.slice(0, 30).map((a: any, i: number) => (
                      <tr key={i} style={{ borderBottom: '1px solid var(--fo-border)' }}>
                        <td style={{ padding: '6px 8px', fontWeight: 700, color: 'var(--fo-accent-green)' }}>{a.weed_uid}</td>
                        <td style={{ padding: '6px 8px' }}>{a.cls_name}</td>
                        <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)' }}>{typeof a.conf === 'number' ? a.conf.toFixed(3) : a.conf}</td>
                        <td style={{ padding: '6px 8px', fontWeight: 600 }}>{a.associated_class}</td>
                        <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)' }}>{a.associated_uid || '—'}</td>
                        <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)' }}>
                          {typeof a.distance_px === 'number' ? Math.round(a.distance_px) : a.distance_px} px
                        </td>
                        <td style={{ padding: '6px 8px' }}>
                          <span style={{ color: a.within_radius ? '#16a34a' : '#d97706', fontWeight: 600 }}>
                            {a.within_radius ? 'Yes' : 'No'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* All Analysed Images Session Summary matching 04_weed_analysis.jpg line 73-88 */}
          {sessionResults.length > 1 && (
            <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '18px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
                <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--fo-text-main)' }}>
                  All Analysed Images ({sessionResults.length})
                </div>
                <button className="fo-btn fo-btn-secondary" onClick={handleDownloadBatchSummaryCsv} style={{ fontSize: '11px', padding: '4px 10px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <Download size={13} /> Download Batch Summary CSV
                </button>
              </div>

              {/* Bar chart of infestation by image */}
              <div style={{ height: '140px', display: 'flex', alignItems: 'flex-end', gap: '8px', padding: '10px 0', borderBottom: '1px solid var(--fo-border)', marginBottom: '14px' }}>
                {sessionResults.map((r, i) => {
                  const sColor =
                    r.infestation.severity === 'Low' ? '#16a34a' :
                    r.infestation.severity === 'Moderate' ? '#d97706' :
                    r.infestation.severity === 'High' ? '#ea580c' : '#dc2626';
                  const maxInfest = Math.max(...sessionResults.map((x) => x.infestation.infestation_pct), 1);
                  const hPct = (r.infestation.infestation_pct / maxInfest) * 100;
                  return (
                    <div
                      key={i}
                      title={`${r.filename}: ${r.infestation.infestation_pct.toFixed(1)}% (${r.infestation.severity})`}
                      style={{
                        flex: 1,
                        height: `${Math.max(12, hPct)}%`,
                        backgroundColor: sColor,
                        borderRadius: '3px 3px 0 0',
                        display: 'flex',
                        alignItems: 'flex-start',
                        justifyContent: 'center',
                        paddingTop: '2px',
                        fontSize: '9px',
                        fontFamily: 'var(--font-mono)',
                        color: '#fff',
                        fontWeight: 700,
                      }}
                    >
                      {r.infestation.infestation_pct.toFixed(0)}%
                    </div>
                  );
                })}
              </div>

              {/* Summary table */}
              <table className="fo-json-tree" style={{ width: '100%', color: 'var(--fo-text-main)', borderCollapse: 'collapse', fontSize: '11px' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--fo-border)', textTransform: 'uppercase', color: 'var(--fo-text-muted)' }}>
                    <th style={{ padding: '6px 8px', textAlign: 'left' }}>image</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left' }}>model</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left' }}>weeds</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left' }}>crops</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left' }}>density_per_MP</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left' }}>infestation_%</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left' }}>severity</th>
                  </tr>
                </thead>
                <tbody>
                  {sessionResults.map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid var(--fo-border)' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600 }}>{r.filename}</td>
                      <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)' }}>
                        {r.checkpoint ? r.checkpoint.split(/[\\/]/).pop() : 'model'}
                      </td>
                      <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--fo-accent-green)' }}>{r.weed_count}</td>
                      <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)' }}>{r.crop_count}</td>
                      <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)' }}>{r.infestation.density_per_mpx.toFixed(2)}</td>
                      <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)' }}>{r.infestation.infestation_pct.toFixed(2)}%</td>
                      <td style={{ padding: '6px 8px' }}>
                        <span style={{ fontWeight: 600, color: r.infestation.severity === 'Low' ? '#16a34a' : '#d97706' }}>
                          {r.infestation.severity}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}

      {/* Spatial Field Grid Tab */}
      {activeTab === 'spatial-grid' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '16px' }}>
          <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
              <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--fo-text-muted)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Grid size={14} /> 6×6 SPATIAL QUADRANT DENSITY MATRIX
              </div>
              <div style={{ display: 'flex', gap: '6px' }}>
                <button
                  className={`fo-btn ${gridMetric === 'count' ? 'fo-btn-primary' : 'fo-btn-secondary'}`}
                  style={{ fontSize: '10px', padding: '2px 8px' }}
                  onClick={() => setGridMetric('count')}
                >
                  Weed Count
                </button>
                <button
                  className={`fo-btn ${gridMetric === 'coverage' ? 'fo-btn-primary' : 'fo-btn-secondary'}`}
                  style={{ fontSize: '10px', padding: '2px 8px' }}
                  onClick={() => setGridMetric('coverage')}
                >
                  Area Coverage
                </button>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: '4px' }}>
              {gridData.grid.map((row, rIdx) =>
                row.map((cell, cIdx) => {
                  const isSelected = selectedCell?.row === rIdx && selectedCell?.col === cIdx;
                  const ratio = gridMetric === 'count' ? cell.count / gridData.maxCount : cell.area / gridData.maxArea;
                  const bgIntensity = Math.min(0.85, 0.1 + ratio * 0.75);

                  return (
                    <div
                      key={`${rIdx}-${cIdx}`}
                      onClick={() => setSelectedCell({ row: rIdx, col: cIdx })}
                      style={{
                        aspectRatio: '1',
                        borderRadius: '4px',
                        background: cell.count > 0 ? `rgba(220, 38, 38, ${bgIntensity})` : 'var(--fo-bg-elevated)',
                        border: isSelected ? '2px solid var(--fo-accent-green)' : '1px solid var(--fo-border)',
                        cursor: 'pointer',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '11px',
                        fontFamily: 'var(--font-mono)',
                        color: cell.count > 0 ? '#fff' : 'var(--fo-text-muted)',
                        fontWeight: 700,
                      }}
                    >
                      <span style={{ fontSize: '9px', opacity: 0.8 }}>{ROW_LABELS[rIdx]}{cIdx + 1}</span>
                      <span>{gridMetric === 'count' ? cell.count : `${Math.round(cell.area / 1000)}k`}</span>
                    </div>
                  );
                })
              )}
            </div>
          </div>

          {/* Quadrant Detail Inspector */}
          <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '16px' }}>
            <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--fo-text-muted)', marginBottom: '12px' }}>
              QUADRANT INSPECTOR {selectedCell ? `(${ROW_LABELS[selectedCell.row]}${selectedCell.col + 1})` : ''}
            </div>
            {selectedCell ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '12px' }}>
                <div className="fo-data-row">
                  <span className="fo-data-label">Weeds in Quadrant</span>
                  <span className="fo-data-value" style={{ fontWeight: 700, color: 'var(--fo-accent-green)' }}>
                    {gridData.grid[selectedCell.row][selectedCell.col].count}
                  </span>
                </div>
                <div className="fo-data-row">
                  <span className="fo-data-label">Total Weed Leaf Area</span>
                  <span className="fo-data-value">
                    {Math.round(gridData.grid[selectedCell.row][selectedCell.col].area).toLocaleString()} px²
                  </span>
                </div>
                <div className="fo-data-row">
                  <span className="fo-data-label">Average Confidence</span>
                  <span className="fo-data-value">
                    {(gridData.grid[selectedCell.row][selectedCell.col].avgConf * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ) : (
              <div style={{ color: 'var(--fo-text-muted)', fontSize: '12px' }}>Select any quadrant cell.</div>
            )}
          </div>
        </div>
      )}

      {/* Spray ROI Tab */}
      {activeTab === 'spray-roi' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
          <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '16px' }}>
            <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--fo-text-muted)', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Sliders size={14} /> FIELD APPLICATION PARAMETERS
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '12px' }}>
              <div>
                <label style={{ display: 'block', color: 'var(--fo-text-subtle)', marginBottom: '4px' }}>Field Surface Area (Hectares)</label>
                <input
                  type="number"
                  value={fieldHectares}
                  onChange={(e) => setFieldHectares(Math.max(1, parseFloat(e.target.value) || 1))}
                  style={{ width: '100%', background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', color: 'var(--fo-text-main)', padding: '6px 10px', borderRadius: '4px' }}
                />
              </div>
              <div>
                <label style={{ display: 'block', color: 'var(--fo-text-subtle)', marginBottom: '4px' }}>Application Rate (L/ha)</label>
                <input
                  type="number"
                  value={sprayRateLPerHa}
                  onChange={(e) => setSprayRateLPerHa(Math.max(1, parseFloat(e.target.value) || 1))}
                  style={{ width: '100%', background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', color: 'var(--fo-text-main)', padding: '6px 10px', borderRadius: '4px' }}
                />
              </div>
              <div>
                <label style={{ display: 'block', color: 'var(--fo-text-subtle)', marginBottom: '4px' }}>Chemical Cost ($/L)</label>
                <input
                  type="number"
                  value={herbicideCostPerL}
                  onChange={(e) => setHerbicideCostPerL(Math.max(0.1, parseFloat(e.target.value) || 0.1))}
                  style={{ width: '100%', background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', color: 'var(--fo-text-main)', padding: '6px 10px', borderRadius: '4px' }}
                />
              </div>
            </div>
          </div>

          <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '16px' }}>
            <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--fo-text-muted)', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <DollarSign size={14} style={{ color: 'var(--fo-accent-green)' }} /> FINANCIAL SAVINGS ESTIMATION
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '12px' }}>
              <div className="fo-data-row">
                <span className="fo-data-label">Broadcast Blanket Cost</span>
                <span className="fo-data-value">${Math.round(roiCalculations.totalCostConventional).toLocaleString()}</span>
              </div>
              <div className="fo-data-row">
                <span className="fo-data-label">Targeted Precision Cost</span>
                <span className="fo-data-value" style={{ color: 'var(--fo-accent-green)', fontWeight: 700 }}>
                  ${Math.round(roiCalculations.costPrecision).toLocaleString()}
                </span>
              </div>
              <div className="fo-data-row">
                <span className="fo-data-label">Estimated Chemical Cost Saved</span>
                <span className="fo-data-value" style={{ color: 'var(--fo-accent-green)', fontWeight: 700 }}>
                  ${Math.round(roiCalculations.costSaved).toLocaleString()} ({roiCalculations.savingsPct.toFixed(1)}%)
                </span>
              </div>
              <div className="fo-data-row">
                <span className="fo-data-label">Chemical Volume Saved</span>
                <span className="fo-data-value">{Math.round(roiCalculations.volumeSaved).toLocaleString()} L</span>
              </div>
              <div className="fo-data-row">
                <span className="fo-data-label">Active Chemical Avoided</span>
                <span className="fo-data-value">{roiCalculations.activeChemicalAvoidedKg} kg</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
