import React, { useEffect, useState, useMemo } from 'react';
import { getComparison, runLiveValidation, benchmarkCheckpoints } from '../services/api';
import {
  GitCompare,
  Zap,
  Cpu,
  BarChart2,
  TrendingUp,
  Image as ImageIcon,
  Maximize2,
  X,
  Layers,
  Activity,
} from 'lucide-react';

interface Props {
  checkpoints: any[];
}

type TabType = 'training-runs' | 'live-validation' | 'efficiency';

const VARIANT_COLORS: Record<string, string> = {
  Baseline: '#64748b',
  RMCBAM: '#16a34a',
};

// ---------------------------------------------------------------------------
// SVG Grouped Bar Chart Component (matches 06_comparison_training_runs.jpg)
// ---------------------------------------------------------------------------
interface GroupedBarChartProps {
  title?: string;
  metrics: string[];
  series: { name: string; color: string; values: Record<string, number> }[];
  yMax?: number;
}

const GroupedBarChart: React.FC<GroupedBarChartProps> = ({ title, metrics, series, yMax = 1.05 }) => {
  const [hoveredBar, setHoveredBar] = useState<{ series: string; metric: string; val: number } | null>(null);

  const width = 760;
  const height = 280;
  const padding = { top: 30, right: 30, bottom: 46, left: 50 };
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;

  const yTicks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
  const groupWidth = innerWidth / (metrics.length || 1);
  const barWidth = Math.min(32, (groupWidth * 0.65) / (series.length || 1));

  return (
    <div style={{ background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '16px 20px', position: 'relative' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
        <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--fo-text-main)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
          {title || 'Best-epoch metrics'}
        </div>
        <div style={{ display: 'flex', gap: '16px', fontSize: '11px' }}>
          {series.map((s) => (
            <div key={s.name} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ width: '10px', height: '10px', borderRadius: '2px', background: s.color }} />
              <span style={{ fontWeight: 600, color: 'var(--fo-text-main)' }}>{s.name}</span>
            </div>
          ))}
        </div>
      </div>

      <div style={{ width: '100%', overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
          {/* Y Axis Grid & Labels */}
          {yTicks.map((tick) => {
            if (tick > yMax) return null;
            const y = padding.top + innerHeight - (tick / yMax) * innerHeight;
            return (
              <g key={tick}>
                <line x1={padding.left} y1={y} x2={width - padding.right} y2={y} stroke="var(--fo-border)" strokeDasharray="3 3" opacity={0.6} />
                <text x={padding.left - 8} y={y + 3} textAnchor="end" fontSize="10" fill="var(--fo-text-muted)" fontFamily="var(--font-mono)">
                  {tick.toFixed(1)}
                </text>
              </g>
            );
          })}

          {/* Metric Groups */}
          {metrics.map((metric, gIdx) => {
            const groupCenterX = padding.left + gIdx * groupWidth + groupWidth / 2;
            const totalGroupBarsWidth = series.length * barWidth + (series.length - 1) * 4;
            const groupStartX = groupCenterX - totalGroupBarsWidth / 2;

            return (
              <g key={metric}>
                {series.map((s, sIdx) => {
                  const val = s.values[metric] ?? 0;
                  const barH = Math.max(3, (Math.min(val, yMax) / yMax) * innerHeight);
                  const barX = groupStartX + sIdx * (barWidth + 4);
                  const barY = padding.top + innerHeight - barH;
                  const isHovered = hoveredBar?.series === s.name && hoveredBar?.metric === metric;

                  return (
                    <g key={s.name}>
                      <rect
                        x={barX}
                        y={barY}
                        width={barWidth}
                        height={barH}
                        rx={3}
                        fill={s.color}
                        opacity={isHovered ? 1 : 0.88}
                        stroke={isHovered ? '#ffffff' : 'none'}
                        strokeWidth={1.5}
                        style={{ cursor: 'pointer', transition: 'all 0.15s ease' }}
                        onMouseEnter={() => setHoveredBar({ series: s.name, metric, val })}
                        onMouseLeave={() => setHoveredBar(null)}
                      />
                      {/* Value label on top of bar */}
                      <text
                        x={barX + barWidth / 2}
                        y={barY - 5}
                        textAnchor="middle"
                        fontSize="9.5"
                        fontWeight={700}
                        fill="var(--fo-text-main)"
                        fontFamily="var(--font-mono)"
                      >
                        {val.toFixed(3)}
                      </text>
                    </g>
                  );
                })}

                {/* X Axis Metric Label */}
                <text
                  x={groupCenterX}
                  y={height - padding.bottom + 18}
                  textAnchor="middle"
                  fontSize="11"
                  fontWeight={600}
                  fill="var(--fo-text-main)"
                >
                  {metric}
                </text>
              </g>
            );
          })}

          {/* Bottom baseline */}
          <line
            x1={padding.left}
            y1={padding.top + innerHeight}
            x2={width - padding.right}
            y2={padding.top + innerHeight}
            stroke="var(--fo-border)"
            strokeWidth={1.5}
          />
        </svg>
      </div>

      {hoveredBar && (
        <div
          style={{
            position: 'absolute',
            bottom: '12px',
            right: '20px',
            background: 'var(--fo-bg-panel)',
            border: '1px solid var(--fo-border)',
            borderRadius: '4px',
            padding: '4px 10px',
            fontSize: '11px',
            pointerEvents: 'none',
            display: 'flex',
            gap: '8px',
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
          }}
        >
          <span style={{ fontWeight: 700, color: series.find((s) => s.name === hoveredBar.series)?.color }}>
            {hoveredBar.series}
          </span>
          <span style={{ color: 'var(--fo-text-muted)' }}>{hoveredBar.metric}:</span>
          <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--fo-text-main)' }}>
            {(hoveredBar.val * 100).toFixed(2)}% ({hoveredBar.val.toFixed(4)})
          </span>
        </div>
      )}
    </div>
  );
};

// ---------------------------------------------------------------------------
// SVG Epoch Curves Component (matches 06b_comparison_curves.jpg)
// ---------------------------------------------------------------------------
interface EpochLineChartProps {
  metricName: string;
  series: { name: string; color: string; data: { epoch: number; value: number }[] }[];
}

const EpochLineChart: React.FC<EpochLineChartProps> = ({ metricName, series }) => {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  const width = 760;
  const height = 300;
  const padding = { top: 30, right: 30, bottom: 44, left: 55 };
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;

  // Compute all points to get min & max
  const allVals = series.flatMap((s) => s.data.map((d) => d.value)).filter((v) => !isNaN(v));
  const allEpochs = series.flatMap((s) => s.data.map((d) => d.epoch));

  const minEpoch = allEpochs.length > 0 ? Math.min(...allEpochs) : 0;
  const maxEpoch = allEpochs.length > 0 ? Math.max(...allEpochs) : 1;
  const epochSpan = Math.max(1, maxEpoch - minEpoch);

  const rawMinVal = allVals.length > 0 ? Math.min(...allVals) : 0;
  const rawMaxVal = allVals.length > 0 ? Math.max(...allVals) : 1;

  // Scale nicely: for [0, 1] bounded metrics (precision, recall, mAP)
  const isLoss = metricName.toLowerCase().includes('loss');
  const minVal = isLoss ? Math.max(0, Math.floor(rawMinVal * 10) / 10) : 0;
  const maxVal = isLoss
    ? Math.max(0.1, Math.ceil(rawMaxVal * 1.15 * 10) / 10)
    : Math.max(0.1, Math.min(1.05, Math.ceil(rawMaxVal * 1.1 * 10) / 10));
  const valSpan = Math.max(0.0001, maxVal - minVal);

  const getY = (val: number) => padding.top + innerHeight - ((val - minVal) / valSpan) * innerHeight;
  const getX = (epoch: number) => padding.left + ((epoch - minEpoch) / epochSpan) * innerWidth;

  const yTicks = [0, 0.25, 0.5, 0.75, 1.0].map((f) => minVal + f * valSpan);

  // Distinct epochs across all series
  const uniqueEpochs = Array.from(new Set(allEpochs)).sort((a, b) => a - b);

  return (
    <div style={{ background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '16px 20px', position: 'relative' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
        <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--fo-text-main)', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <TrendingUp size={14} style={{ color: 'var(--fo-accent-blue)' }} />
          <span>{metricName} per epoch</span>
        </div>
        <div style={{ display: 'flex', gap: '16px', fontSize: '11px' }}>
          {series.map((s) => (
            <div key={s.name} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ width: '16px', height: '3px', borderRadius: '2px', background: s.color }} />
              <span style={{ fontWeight: 600, color: 'var(--fo-text-main)' }}>{s.name}</span>
            </div>
          ))}
        </div>
      </div>

      <div style={{ width: '100%', overflowX: 'auto' }}>
        <svg
          viewBox={`0 0 ${width} ${height}`}
          style={{ width: '100%', height: 'auto', display: 'block', cursor: 'crosshair' }}
          onMouseLeave={() => setHoverIndex(null)}
          onMouseMove={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const mouseX = ((e.clientX - rect.left) / rect.width) * width;
            if (mouseX >= padding.left && mouseX <= width - padding.right && uniqueEpochs.length > 0) {
              const epochHover = minEpoch + ((mouseX - padding.left) / innerWidth) * epochSpan;
              // Find closest epoch
              let closest = uniqueEpochs[0];
              let minDist = Math.abs(uniqueEpochs[0] - epochHover);
              for (const ep of uniqueEpochs) {
                const dist = Math.abs(ep - epochHover);
                if (dist < minDist) {
                  minDist = dist;
                  closest = ep;
                }
              }
              setHoverIndex(closest);
            }
          }}
        >
          {/* Y Axis Gridlines */}
          {yTicks.map((tickVal, idx) => {
            const y = getY(tickVal);
            return (
              <g key={idx}>
                <line x1={padding.left} y1={y} x2={width - padding.right} y2={y} stroke="var(--fo-border)" strokeDasharray="3 3" opacity={0.6} />
                <text x={padding.left - 8} y={y + 3} textAnchor="end" fontSize="10" fill="var(--fo-text-muted)" fontFamily="var(--font-mono)">
                  {tickVal.toFixed(3)}
                </text>
              </g>
            );
          })}

          {/* X Axis Epoch Markers */}
          {uniqueEpochs.map((ep) => {
            const x = getX(ep);
            return (
              <g key={ep}>
                <line x1={x} y1={padding.top + innerHeight} x2={x} y2={padding.top + innerHeight + 4} stroke="var(--fo-border)" />
                <text x={x} y={padding.top + innerHeight + 16} textAnchor="middle" fontSize="10" fill="var(--fo-text-muted)" fontFamily="var(--font-mono)">
                  {ep}
                </text>
              </g>
            );
          })}

          {/* Epoch axis label */}
          <text x={padding.left + innerWidth / 2} y={height - 8} textAnchor="middle" fontSize="11" fill="var(--fo-text-muted)">
            Epoch
          </text>

          {/* Hover guideline */}
          {hoverIndex !== null && (
            <line
              x1={getX(hoverIndex)}
              y1={padding.top}
              x2={getX(hoverIndex)}
              y2={padding.top + innerHeight}
              stroke="var(--fo-text-main)"
              strokeDasharray="2 2"
              strokeWidth={1}
              opacity={0.8}
            />
          )}

          {/* Series lines and circles */}
          {series.map((s) => {
            if (s.data.length === 0) return null;
            const sortedData = [...s.data].sort((a, b) => a.epoch - b.epoch);
            const pathD = sortedData
              .map((pt, i) => `${i === 0 ? 'M' : 'L'} ${getX(pt.epoch).toFixed(1)} ${getY(pt.value).toFixed(1)}`)
              .join(' ');

            return (
              <g key={s.name}>
                <path d={pathD} fill="none" stroke={s.color} strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round" />
                {sortedData.map((pt) => {
                  const isHovered = hoverIndex === pt.epoch;
                  return (
                    <circle
                      key={pt.epoch}
                      cx={getX(pt.epoch)}
                      cy={getY(pt.value)}
                      r={isHovered ? 5.5 : 3}
                      fill={s.color}
                      stroke="#ffffff"
                      strokeWidth={isHovered ? 2 : 1}
                      style={{ transition: 'all 0.1s ease' }}
                    />
                  );
                })}
              </g>
            );
          })}
        </svg>
      </div>

      {hoverIndex !== null && (
        <div
          style={{
            position: 'absolute',
            top: '16px',
            right: '20px',
            background: 'var(--fo-bg-panel)',
            border: '1px solid var(--fo-border)',
            borderRadius: '6px',
            padding: '8px 12px',
            fontSize: '11px',
            pointerEvents: 'none',
            display: 'flex',
            flexDirection: 'column',
            gap: '4px',
            boxShadow: '0 4px 16px rgba(0,0,0,0.2)',
          }}
        >
          <div style={{ fontWeight: 700, color: 'var(--fo-text-main)', borderBottom: '1px solid var(--fo-border)', paddingBottom: '3px' }}>
            Epoch {hoverIndex}
          </div>
          {series.map((s) => {
            const pt = s.data.find((d) => d.epoch === hoverIndex);
            return (
              <div key={s.name} style={{ display: 'flex', justifyContent: 'space-between', gap: '12px' }}>
                <span style={{ color: s.color, fontWeight: 600 }}>{s.name}:</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>
                  {pt ? pt.value.toFixed(4) : '—'}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

// ---------------------------------------------------------------------------
// SVG Precision-Recall or F1 Curve Component (Tab 2)
// ---------------------------------------------------------------------------
interface CurveChartProps {
  title: string;
  xlabel: string;
  ylabel: string;
  series: { name: string; color: string; x: number[]; y: number[] }[];
}

const CurveChart: React.FC<CurveChartProps> = ({ title, xlabel, ylabel, series }) => {
  const width = 680;
  const height = 260;
  const padding = { top: 24, right: 24, bottom: 42, left: 50 };
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;

  const getX = (val: number) => padding.left + Math.max(0, Math.min(1, val)) * innerWidth;
  const getY = (val: number) => padding.top + innerHeight - Math.max(0, Math.min(1, val)) * innerHeight;

  return (
    <div style={{ background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '16px', position: 'relative' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
        <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--fo-text-main)' }}>{title}</div>
        <div style={{ display: 'flex', gap: '14px', fontSize: '11px' }}>
          {series.map((s) => (
            <div key={s.name} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ width: '12px', height: '3px', borderRadius: '2px', background: s.color }} />
              <span style={{ fontWeight: 600, color: 'var(--fo-text-main)' }}>{s.name}</span>
            </div>
          ))}
        </div>
      </div>

      <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
        {[0, 0.25, 0.5, 0.75, 1.0].map((t) => {
          const y = getY(t);
          const x = getX(t);
          return (
            <g key={t}>
              <line x1={padding.left} y1={y} x2={width - padding.right} y2={y} stroke="var(--fo-border)" strokeDasharray="3 3" opacity={0.6} />
              <text x={padding.left - 6} y={y + 3} textAnchor="end" fontSize="10" fill="var(--fo-text-muted)" fontFamily="var(--font-mono)">
                {t.toFixed(2)}
              </text>
              <line x1={x} y1={padding.top + innerHeight} x2={x} y2={padding.top + innerHeight + 4} stroke="var(--fo-border)" />
              <text x={x} y={padding.top + innerHeight + 16} textAnchor="middle" fontSize="10" fill="var(--fo-text-muted)" fontFamily="var(--font-mono)">
                {t.toFixed(2)}
              </text>
            </g>
          );
        })}

        <text x={padding.left + innerWidth / 2} y={height - 6} textAnchor="middle" fontSize="11" fill="var(--fo-text-muted)">
          {xlabel}
        </text>

        <text x={14} y={padding.top + innerHeight / 2} textAnchor="middle" fontSize="11" fill="var(--fo-text-muted)" transform={`rotate(-90 14 ${padding.top + innerHeight / 2})`}>
          {ylabel}
        </text>

        {series.map((s) => {
          if (!s.x || !s.y || s.x.length === 0) return null;
          const pts: string[] = [];
          for (let i = 0; i < Math.min(s.x.length, s.y.length); i++) {
            pts.push(`${i === 0 ? 'M' : 'L'} ${getX(s.x[i]).toFixed(1)} ${getY(s.y[i]).toFixed(1)}`);
          }
          return <path key={s.name} d={pts.join(' ')} fill="none" stroke={s.color} strokeWidth={2.2} strokeLinecap="round" />;
        })}
      </svg>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Confusion Matrix Heatmap Component (Tab 2)
// ---------------------------------------------------------------------------
interface ConfusionMatrixProps {
  modelName: string;
  matrix: number[][];
  names: Record<string, string> | string[];
}

const ConfusionMatrixHeatmap: React.FC<ConfusionMatrixProps> = ({ modelName, matrix, names }) => {
  const classNames = useMemo(() => {
    const list: string[] = [];
    if (Array.isArray(names)) {
      list.push(...names);
    } else if (names && typeof names === 'object') {
      const keys = Object.keys(names).sort((a, b) => Number(a) - Number(b));
      keys.forEach((k) => list.push(names[k]));
    }
    list.push('background');
    return list;
  }, [names]);

  const n = Math.min(classNames.length, matrix.length);

  // Column normalise
  const colSums = useMemo(() => {
    const sums: number[] = new Array(n).fill(0);
    for (let c = 0; c < n; c++) {
      for (let r = 0; r < n; r++) {
        sums[c] += matrix[r]?.[c] || 0;
      }
    }
    return sums;
  }, [matrix, n]);

  return (
    <div style={{ background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '16px', flex: 1 }}>
      <div style={{ fontSize: '12px', fontWeight: 700, color: VARIANT_COLORS[modelName] || 'var(--fo-text-main)', marginBottom: '4px' }}>
        {modelName}: Confusion Matrix
      </div>
      <div style={{ fontSize: '10px', color: 'var(--fo-text-subtle)', marginBottom: '12px' }}>
        Column-normalised with raw counts. True (horizontal) vs Predicted (vertical).
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table style={{ borderCollapse: 'collapse', fontSize: '11px', textAlign: 'center', margin: '0 auto' }}>
          <thead>
            <tr>
              <th style={{ padding: '4px 8px', fontSize: '10px', color: 'var(--fo-text-muted)' }}>Pred \ True</th>
              {classNames.slice(0, n).map((cName, idx) => (
                <th key={idx} style={{ padding: '6px 8px', fontWeight: 600, color: 'var(--fo-text-main)', maxWidth: '80px', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {cName}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {classNames.slice(0, n).map((rowName, rIdx) => (
              <tr key={rIdx}>
                <td style={{ padding: '6px 8px', fontWeight: 600, color: 'var(--fo-text-main)', textAlign: 'right', whiteSpace: 'nowrap' }}>
                  {rowName}
                </td>
                {classNames.slice(0, n).map((_, cIdx) => {
                  const rawCount = matrix[rIdx]?.[cIdx] || 0;
                  const cSum = colSums[cIdx];
                  const norm = cSum > 0 ? rawCount / cSum : 0;

                  return (
                    <td
                      key={cIdx}
                      style={{
                        padding: '8px 10px',
                        border: '1px solid var(--fo-border)',
                        background: `rgba(22, 163, 74, ${Math.max(0.04, norm * 0.85)})`,
                        color: norm > 0.5 ? '#ffffff' : 'var(--fo-text-main)',
                        fontFamily: 'var(--font-mono)',
                        fontWeight: norm > 0.3 ? 700 : 400,
                        minWidth: '60px',
                      }}
                      title={`True: ${classNames[cIdx]}, Pred: ${rowName}\nNorm: ${(norm * 100).toFixed(1)}%\nCount: ${rawCount}`}
                    >
                      <div>{(norm * 100).toFixed(0)}%</div>
                      <div style={{ fontSize: '9px', opacity: 0.7 }}>n={rawCount}</div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------
export const ComparisonWorkspace: React.FC<Props> = ({ checkpoints }) => {
  const [activeTab, setActiveTab] = useState<TabType>('training-runs');
  const [runsData, setRunsData] = useState<any[]>([]);
  const [yamls, setYamls] = useState<string[]>([]);
  const [metricColumns, setMetricColumns] = useState<string[]>([]);

  // Tab 1: Training runs
  const [selectedBaselineRun, setSelectedBaselineRun] = useState<string>('');
  const [selectedRmcbamRun, setSelectedRmcbamRun] = useState<string>('');
  const [selectedEpochMetric, setSelectedEpochMetric] = useState<string>('mAP50');
  const [previewPlot, setPreviewPlot] = useState<{ title: string; src: string; model: string } | null>(null);

  // Tab 2: Live Validation
  const [valBaselineCkpt, setValBaselineCkpt] = useState<string>('');
  const [valRmcbamCkpt, setValRmcbamCkpt] = useState<string>('');
  const [selectedYaml, setSelectedYaml] = useState<string>('data/yolo_weed_partial/data.yaml');
  const [selectedSplit, setSelectedSplit] = useState<string>('val');
  const [valLoading, setValLoading] = useState<boolean>(false);
  const [valResults, setValResults] = useState<Record<string, any> | null>(null);
  const [valError, setValError] = useState<string | null>(null);

  // Tab 3: Efficiency
  const [selectedEfficiencyCkpts, setSelectedEfficiencyCkpts] = useState<string[]>([]);
  const [timedRuns, setTimedRuns] = useState<number>(10);
  const [benchmarking, setBenchmarking] = useState<boolean>(false);
  const [benchmarkResults, setBenchmarkResults] = useState<any[]>([]);
  const [benchError, setBenchError] = useState<string | null>(null);

  useEffect(() => {
    fetchComparisonData();
  }, []);

  useEffect(() => {
    if (checkpoints.length > 0) {
      const base = checkpoints.find((c) => !c.is_rmcbam) || checkpoints[0];
      const rm = checkpoints.find((c) => c.is_rmcbam) || checkpoints[0];
      if (!valBaselineCkpt) setValBaselineCkpt(base.path);
      if (!valRmcbamCkpt) setValRmcbamCkpt(rm.path);
      if (selectedEfficiencyCkpts.length === 0) {
        setSelectedEfficiencyCkpts(checkpoints.slice(0, 3).map((c) => c.path));
      }
    }
  }, [checkpoints]);

  const fetchComparisonData = async () => {
    try {
      const data = await getComparison();
      const runs = data.runs || [];
      setRunsData(runs);
      setYamls(data.yamls || ['data/yolo_weed_partial/data.yaml']);
      const cols = data.metric_columns || ['precision', 'recall', 'mAP50', 'mAP50-95'];
      setMetricColumns(cols);

      if (runs.length > 0) {
        const baseRun = runs.find((r: any) => !r.run_name.toLowerCase().includes('rmcbam')) || runs[0];
        const rmRun = runs.find((r: any) => r.run_name.toLowerCase().includes('rmcbam')) || runs[0];
        setSelectedBaselineRun(baseRun.run_dir);
        setSelectedRmcbamRun(rmRun.run_dir);
      }
    } catch (e) {
      console.error('Failed to load comparison data:', e);
    }
  };

  const currentBaselineRun = runsData.find((r) => r.run_dir === selectedBaselineRun);
  const currentRmcbamRun = runsData.find((r) => r.run_dir === selectedRmcbamRun);

  // Best epoch metrics rows
  const bestEpochRows = useMemo(() => {
    const rows = [];
    if (currentBaselineRun?.metrics) {
      rows.push({
        model: 'Baseline',
        run: currentBaselineRun.run_name,
        ...currentBaselineRun.metrics,
      });
    }
    if (currentRmcbamRun?.metrics) {
      rows.push({
        model: 'RMCBAM',
        run: currentRmcbamRun.run_name,
        ...currentRmcbamRun.metrics,
      });
    }
    return rows;
  }, [currentBaselineRun, currentRmcbamRun]);

  // Format data for Epoch Line Chart
  const epochSeries = useMemo(() => {
    const mapMetricKey: Record<string, string> = {
      precision: 'metrics/precision(B)',
      recall: 'metrics/recall(B)',
      mAP50: 'metrics/mAP50(B)',
      'mAP50-95': 'metrics/mAP50-95(B)',
      'train/box_loss': 'train/box_loss',
      'train/cls_loss': 'train/cls_loss',
      'train/dfl_loss': 'train/dfl_loss',
      'val/box_loss': 'val/box_loss',
      'val/cls_loss': 'val/cls_loss',
      'val/dfl_loss': 'val/dfl_loss',
    };
    const key = mapMetricKey[selectedEpochMetric] || selectedEpochMetric;

    const extractPoints = (run: any) => {
      if (!run?.results_table) return [];
      return run.results_table
        .map((row: any) => {
          const val = row[key] !== undefined ? row[key] : row[selectedEpochMetric];
          return {
            epoch: Number(row.epoch || 0),
            value: typeof val === 'number' ? val : parseFloat(val),
          };
        })
        .filter((pt: any) => !isNaN(pt.value) && pt.epoch !== undefined);
    };

    return [
      {
        name: 'Baseline',
        color: VARIANT_COLORS.Baseline,
        data: extractPoints(currentBaselineRun),
      },
      {
        name: 'RMCBAM',
        color: VARIANT_COLORS.RMCBAM,
        data: extractPoints(currentRmcbamRun),
      },
    ];
  }, [currentBaselineRun, currentRmcbamRun, selectedEpochMetric]);

  const handleDualLiveValidation = async () => {
    if (!valBaselineCkpt || !valRmcbamCkpt) return;
    setValLoading(true);
    setValError(null);
    try {
      const res = await runLiveValidation(
        undefined,
        valBaselineCkpt,
        valRmcbamCkpt,
        selectedYaml,
        selectedSplit
      );
      setValResults(res.validation || null);
    } catch (e: any) {
      setValError(e.response?.data?.detail || e.message || 'Live dataset validation failed.');
    } finally {
      setValLoading(false);
    }
  };

  const handleRunBenchmark = async () => {
    if (selectedEfficiencyCkpts.length === 0) return;
    setBenchmarking(true);
    setBenchError(null);
    try {
      const res = await benchmarkCheckpoints(selectedEfficiencyCkpts, timedRuns);
      setBenchmarkResults(res.benchmark || []);
    } catch (e: any) {
      setBenchError(e.response?.data?.detail || e.message || 'Latency benchmark failed.');
    } finally {
      setBenchmarking(false);
    }
  };

  // Extract shared plot titles for side-by-side display
  const availablePlots = useMemo(() => {
    const titles = new Set<string>();
    if (currentBaselineRun?.plots) Object.keys(currentBaselineRun.plots).forEach((t) => titles.add(t));
    if (currentRmcbamRun?.plots) Object.keys(currentRmcbamRun.plots).forEach((t) => titles.add(t));
    return Array.from(titles);
  }, [currentBaselineRun, currentRmcbamRun]);

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
          flexWrap: 'wrap',
          gap: '12px',
        }}
      >
        <div>
          <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 700, color: 'var(--fo-text-main)', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <GitCompare size={18} style={{ color: 'var(--fo-accent-blue)' }} /> Model Comparison
          </h2>
          <div style={{ fontSize: '12px', color: 'var(--fo-text-subtle)', marginTop: '4px' }}>
            YOLOv8s baseline vs YOLOv8s + RMCBAM — from training logs, live validation, and measured efficiency.
          </div>
        </div>

        {/* 3 Tabs Header matching 06_comparison_training_runs.jpg line 22 */}
        <div style={{ display: 'flex', gap: '6px' }}>
          <button
            className={`fo-btn ${activeTab === 'training-runs' ? 'fo-btn-primary' : 'fo-btn-secondary'}`}
            style={{ fontSize: '11px', padding: '6px 14px' }}
            onClick={() => setActiveTab('training-runs')}
          >
            <BarChart2 size={13} /> Training runs (results.csv)
          </button>
          <button
            className={`fo-btn ${activeTab === 'live-validation' ? 'fo-btn-primary' : 'fo-btn-secondary'}`}
            style={{ fontSize: '11px', padding: '6px 14px' }}
            onClick={() => setActiveTab('live-validation')}
          >
            <Zap size={13} /> Live validation
          </button>
          <button
            className={`fo-btn ${activeTab === 'efficiency' ? 'fo-btn-primary' : 'fo-btn-secondary'}`}
            style={{ fontSize: '11px', padding: '6px 14px' }}
            onClick={() => setActiveTab('efficiency')}
          >
            <Cpu size={13} /> Efficiency
          </button>
        </div>
      </div>

      {/* =========================================================================
          TAB 1: Training Runs (results.csv) matching 06_comparison_training_runs.jpg & 06b_comparison_curves.jpg
          ========================================================================= */}
      {activeTab === 'training-runs' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
          {/* Run Selectors */}
          <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '14px 18px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
            <div>
              <label style={{ fontSize: '11px', fontWeight: 700, color: 'var(--fo-text-muted)', textTransform: 'uppercase', marginBottom: '6px', display: 'block' }}>
                Baseline Run
              </label>
              <select
                style={{ width: '100%', background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', color: 'var(--fo-text-main)', padding: '6px 10px', borderRadius: '4px', fontSize: '12px' }}
                value={selectedBaselineRun}
                onChange={(e) => setSelectedBaselineRun(e.target.value)}
              >
                {runsData.map((r) => (
                  <option key={r.run_dir} value={r.run_dir}>
                    {r.run_name}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label style={{ fontSize: '11px', fontWeight: 700, color: 'var(--fo-text-muted)', textTransform: 'uppercase', marginBottom: '6px', display: 'block' }}>
                RMCBAM Run
              </label>
              <select
                style={{ width: '100%', background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', color: 'var(--fo-text-main)', padding: '6px 10px', borderRadius: '4px', fontSize: '12px' }}
                value={selectedRmcbamRun}
                onChange={(e) => setSelectedRmcbamRun(e.target.value)}
              >
                {runsData.map((r) => (
                  <option key={r.run_dir} value={r.run_dir}>
                    {r.run_name}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {(currentBaselineRun?.is_smoke || currentRmcbamRun?.is_smoke) && (
            <div style={{ background: 'rgba(239, 68, 68, 0.08)', border: '1px solid rgba(239, 68, 68, 0.25)', color: '#ef4444', padding: '10px 14px', borderRadius: '4px', fontSize: '12px' }}>
              These are <b>PIPELINE SMOKE-TEST</b> runs (few epochs, partially-labelled Weed images). The numbers below are real outputs of those runs.
            </div>
          )}

          {/* Metrics at best epoch Table matching 06_comparison_training_runs.jpg line 44 */}
          {bestEpochRows.length > 0 && (
            <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '18px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div>
                <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--fo-text-main)', marginBottom: '4px' }}>
                  Metrics at best epoch
                </div>
                <div style={{ fontSize: '11px', color: 'var(--fo-text-subtle)' }}>
                  fitness = 0.1·mAP50 + 0.9·mAP50-95, as Ultralytics selects best.pt
                </div>
              </div>

              <div style={{ overflowX: 'auto' }}>
                <table className="fo-json-tree" style={{ width: '100%', color: 'var(--fo-text-main)', borderCollapse: 'collapse', fontSize: '11px' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--fo-border)', textTransform: 'uppercase', color: 'var(--fo-text-muted)' }}>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>model</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>run</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>precision</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>recall</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>F1</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>mAP50</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>mAP50-95</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>best_epoch</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bestEpochRows.map((row) => (
                      <tr key={row.model} style={{ borderBottom: '1px solid var(--fo-border)' }}>
                        <td style={{ padding: '8px 12px', fontWeight: 700, color: VARIANT_COLORS[row.model] || '#16a34a' }}>{row.model}</td>
                        <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)' }}>{row.run}</td>
                        <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)' }}>{row.precision ? (row.precision * 100).toFixed(1) + '%' : '—'}</td>
                        <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)' }}>{row.recall ? (row.recall * 100).toFixed(1) + '%' : '—'}</td>
                        <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)' }}>{row.F1 ? (row.F1 * 100).toFixed(1) + '%' : '—'}</td>
                        <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--fo-accent-blue)' }}>{row.mAP50 ? (row.mAP50 * 100).toFixed(1) + '%' : '—'}</td>
                        <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)' }}>{row['mAP50-95'] ? (row['mAP50-95'] * 100).toFixed(1) + '%' : '—'}</td>
                        <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)' }}>{row.best_epoch || 0}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Grouped Bar Chart comparing Baseline vs RMCBAM matching 06_comparison_training_runs.jpg line 46 */}
              <GroupedBarChart
                title="Best-epoch metrics"
                metrics={['precision', 'recall', 'F1', 'mAP50', 'mAP50-95']}
                series={[
                  {
                    name: 'Baseline',
                    color: VARIANT_COLORS.Baseline,
                    values: currentBaselineRun?.metrics || {},
                  },
                  {
                    name: 'RMCBAM',
                    color: VARIANT_COLORS.RMCBAM,
                    values: currentRmcbamRun?.metrics || {},
                  },
                ]}
              />

              {/* Metric Over Epochs Interactive Plot matching 06b_comparison_curves.jpg */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span style={{ fontSize: '12px', fontWeight: 700, color: 'var(--fo-text-main)' }}>Metric over epochs:</span>
                    <select
                      style={{
                        background: 'var(--fo-bg-elevated)',
                        border: '1px solid var(--fo-border)',
                        color: 'var(--fo-text-main)',
                        padding: '4px 10px',
                        borderRadius: '4px',
                        fontSize: '11px',
                        fontWeight: 600,
                      }}
                      value={selectedEpochMetric}
                      onChange={(e) => setSelectedEpochMetric(e.target.value)}
                    >
                      {(metricColumns.length > 0 ? metricColumns : ['mAP50', 'mAP50-95', 'precision', 'recall', 'train/box_loss', 'train/cls_loss', 'val/box_loss', 'val/cls_loss']).map((col) => (
                        <option key={col} value={col}>
                          {col}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div style={{ fontSize: '11px', color: 'var(--fo-text-subtle)' }}>
                    Hover over chart to inspect epoch-by-epoch trajectory
                  </div>
                </div>

                <EpochLineChart metricName={selectedEpochMetric} series={epochSeries} />
              </div>
            </div>
          )}

          {/* Plots saved by Ultralytics side-by-side matching 06_comparison_training_runs.jpg line 54 */}
          {availablePlots.length > 0 && (
            <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '18px' }}>
              <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--fo-text-main)', marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <ImageIcon size={15} style={{ color: 'var(--fo-accent-blue)' }} /> Plots saved by Ultralytics
              </div>
              <div style={{ fontSize: '11px', color: 'var(--fo-text-subtle)', marginBottom: '16px' }}>
                Side-by-side comparison of curves, confusion matrices, and validation batches. Click any plot to zoom.
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '22px' }}>
                {availablePlots.map((title) => {
                  const basePlot = currentBaselineRun?.plots?.[title];
                  const rmPlot = currentRmcbamRun?.plots?.[title];

                  return (
                    <div key={title} style={{ background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '14px' }}>
                      <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--fo-text-main)', marginBottom: '12px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span>{title}</span>
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                        <div>
                          <div style={{ fontSize: '11px', color: 'var(--fo-text-muted)', marginBottom: '6px', display: 'flex', justifyContent: 'space-between' }}>
                            <span>Baseline ({currentBaselineRun?.run_name})</span>
                            {basePlot && <span style={{ fontSize: '10px', color: 'var(--fo-accent-blue)', cursor: 'pointer' }} onClick={() => setPreviewPlot({ title, src: basePlot, model: 'Baseline' })}>Zoom</span>}
                          </div>
                          {basePlot ? (
                            <div
                              style={{ cursor: 'pointer', position: 'relative', overflow: 'hidden', borderRadius: '4px' }}
                              onClick={() => setPreviewPlot({ title, src: basePlot, model: 'Baseline' })}
                            >
                              <img src={basePlot} alt="Baseline plot" style={{ width: '100%', height: 'auto', display: 'block', borderRadius: '4px' }} />
                              <div style={{ position: 'absolute', top: '6px', right: '6px', background: 'rgba(0,0,0,0.6)', borderRadius: '3px', padding: '3px 6px', color: '#fff', fontSize: '10px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                                <Maximize2 size={11} />
                              </div>
                            </div>
                          ) : (
                            <div style={{ height: '180px', background: 'var(--fo-bg-panel)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--fo-text-muted)', fontSize: '11px', borderRadius: '4px' }}>
                              Baseline: not available
                            </div>
                          )}
                        </div>

                        <div>
                          <div style={{ fontSize: '11px', color: 'var(--fo-accent-blue)', marginBottom: '6px', display: 'flex', justifyContent: 'space-between' }}>
                            <span>RMCBAM ({currentRmcbamRun?.run_name})</span>
                            {rmPlot && <span style={{ fontSize: '10px', color: 'var(--fo-accent-blue)', cursor: 'pointer' }} onClick={() => setPreviewPlot({ title, src: rmPlot, model: 'RMCBAM' })}>Zoom</span>}
                          </div>
                          {rmPlot ? (
                            <div
                              style={{ cursor: 'pointer', position: 'relative', overflow: 'hidden', borderRadius: '4px' }}
                              onClick={() => setPreviewPlot({ title, src: rmPlot, model: 'RMCBAM' })}
                            >
                              <img src={rmPlot} alt="RMCBAM plot" style={{ width: '100%', height: 'auto', display: 'block', borderRadius: '4px' }} />
                              <div style={{ position: 'absolute', top: '6px', right: '6px', background: 'rgba(0,0,0,0.6)', borderRadius: '3px', padding: '3px 6px', color: '#fff', fontSize: '10px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                                <Maximize2 size={11} />
                              </div>
                            </div>
                          ) : (
                            <div style={{ height: '180px', background: 'var(--fo-bg-panel)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--fo-text-muted)', fontSize: '11px', borderRadius: '4px' }}>
                              RMCBAM: not available
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* =========================================================================
          TAB 2: Live Validation matching 06c_live_validation.jpg & 06d_live_validation_curves_cm.jpg
          ========================================================================= */}
      {activeTab === 'live-validation' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
          <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '16px' }}>
            <div style={{ fontSize: '12px', color: 'var(--fo-text-subtle)', marginBottom: '14px' }}>
              Runs Ultralytics validation on your labelled split. Results are computed live from your dataset on this machine.
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '14px', alignItems: 'flex-end' }}>
              <div>
                <label style={{ fontSize: '11px', fontWeight: 700, color: 'var(--fo-text-muted)', textTransform: 'uppercase', marginBottom: '4px', display: 'block' }}>
                  Baseline Checkpoint
                </label>
                <select
                  style={{ width: '100%', background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', color: 'var(--fo-text-main)', padding: '6px 8px', borderRadius: '4px', fontSize: '11px' }}
                  value={valBaselineCkpt}
                  onChange={(e) => setValBaselineCkpt(e.target.value)}
                >
                  {checkpoints.map((c) => (
                    <option key={c.path} value={c.path}>
                      {c.label || c.filename}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label style={{ fontSize: '11px', fontWeight: 700, color: 'var(--fo-text-muted)', textTransform: 'uppercase', marginBottom: '4px', display: 'block' }}>
                  RMCBAM Checkpoint
                </label>
                <select
                  style={{ width: '100%', background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', color: 'var(--fo-text-main)', padding: '6px 8px', borderRadius: '4px', fontSize: '11px' }}
                  value={valRmcbamCkpt}
                  onChange={(e) => setValRmcbamCkpt(e.target.value)}
                >
                  {checkpoints.map((c) => (
                    <option key={c.path} value={c.path}>
                      {c.label || c.filename}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label style={{ fontSize: '11px', fontWeight: 700, color: 'var(--fo-text-muted)', textTransform: 'uppercase', marginBottom: '4px', display: 'block' }}>
                  data.yaml
                </label>
                <select
                  style={{ width: '100%', background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', color: 'var(--fo-text-main)', padding: '6px 8px', borderRadius: '4px', fontSize: '11px' }}
                  value={selectedYaml}
                  onChange={(e) => setSelectedYaml(e.target.value)}
                >
                  {yamls.map((y) => (
                    <option key={y} value={y}>
                      {y}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label style={{ fontSize: '11px', fontWeight: 700, color: 'var(--fo-text-muted)', textTransform: 'uppercase', marginBottom: '4px', display: 'block' }}>
                  Split
                </label>
                <select
                  style={{ width: '100%', background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', color: 'var(--fo-text-main)', padding: '6px 8px', borderRadius: '4px', fontSize: '11px' }}
                  value={selectedSplit}
                  onChange={(e) => setSelectedSplit(e.target.value)}
                >
                  <option value="val">val</option>
                  <option value="test">test</option>
                  <option value="train">train</option>
                </select>
              </div>

              <div>
                <button
                  className="fo-btn fo-btn-primary"
                  onClick={handleDualLiveValidation}
                  disabled={valLoading}
                  style={{ width: '100%', padding: '7px 16px', fontSize: '12px', whiteSpace: 'nowrap' }}
                >
                  <Zap size={13} /> {valLoading ? 'Validating Both...' : 'Run validation on both'}
                </button>
              </div>
            </div>

            {valError && (
              <div style={{ marginTop: '12px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', color: '#ef4444', padding: '8px 12px', borderRadius: '4px', fontSize: '12px' }}>
                {valError}
              </div>
            )}
          </div>

          {/* Validation Metrics Results Table matching 06c_live_validation.jpg line 149 */}
          {valResults && Object.keys(valResults).length > 0 && (
            <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '18px', display: 'flex', flexDirection: 'column', gap: '18px' }}>
              <div>
                <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--fo-text-main)', marginBottom: '4px' }}>
                  Validation metrics (computed now)
                </div>
                <div style={{ fontSize: '11px', color: 'var(--fo-text-subtle)' }}>
                  Fresh validation evaluated across the {selectedSplit} partition
                </div>
              </div>

              <div style={{ overflowX: 'auto' }}>
                <table className="fo-json-tree" style={{ width: '100%', color: 'var(--fo-text-main)', borderCollapse: 'collapse', fontSize: '11px' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--fo-border)', textTransform: 'uppercase', color: 'var(--fo-text-muted)' }}>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>model</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>precision</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>recall</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>F1</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>mAP50</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>mAP50-95</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>inference ms/img</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(valResults).map(([mName, res]) => (
                      <tr key={mName} style={{ borderBottom: '1px solid var(--fo-border)' }}>
                        <td style={{ padding: '8px 12px', fontWeight: 700, color: VARIANT_COLORS[mName] || '#16a34a' }}>{mName}</td>
                        <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)' }}>{(res.precision * 100).toFixed(1)}%</td>
                        <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)' }}>{(res.recall * 100).toFixed(1)}%</td>
                        <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)' }}>{(res.F1 * 100).toFixed(1)}%</td>
                        <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--fo-accent-blue)' }}>{(res.mAP50 * 100).toFixed(1)}%</td>
                        <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)' }}>{(res['mAP50-95'] * 100).toFixed(1)}%</td>
                        <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)' }}>{res.speed_ms?.inference?.toFixed(2) || '—'} ms</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Validation Metrics Grouped Bar Chart matching line 151 */}
              <GroupedBarChart
                title="Validation metrics (computed now)"
                metrics={['precision', 'recall', 'F1', 'mAP50', 'mAP50-95']}
                series={Object.entries(valResults).map(([mName, res]) => ({
                  name: mName,
                  color: VARIANT_COLORS[mName] || '#16a34a',
                  values: {
                    precision: res.precision,
                    recall: res.recall,
                    F1: res.F1,
                    mAP50: res.mAP50,
                    'mAP50-95': res['mAP50-95'],
                  },
                }))}
              />

              {/* Per-class results table matching line 156 */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--fo-text-main)' }}>
                  Per-Class Results
                </div>
                <div style={{ overflowX: 'auto' }}>
                  <table className="fo-json-tree" style={{ width: '100%', color: 'var(--fo-text-main)', borderCollapse: 'collapse', fontSize: '11px' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid var(--fo-border)', textTransform: 'uppercase', color: 'var(--fo-text-muted)' }}>
                        <th style={{ padding: '8px 12px', textAlign: 'left' }}>model</th>
                        <th style={{ padding: '8px 12px', textAlign: 'left' }}>class</th>
                        <th style={{ padding: '8px 12px', textAlign: 'left' }}>precision</th>
                        <th style={{ padding: '8px 12px', textAlign: 'left' }}>recall</th>
                        <th style={{ padding: '8px 12px', textAlign: 'left' }}>AP50</th>
                        <th style={{ padding: '8px 12px', textAlign: 'left' }}>AP50-95</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(valResults).map(([mName, res]) =>
                        (res.per_class || []).map((pc: any, idx: number) => (
                          <tr key={`${mName}-${idx}`} style={{ borderBottom: '1px solid var(--fo-border)' }}>
                            <td style={{ padding: '8px 12px', fontWeight: 600, color: VARIANT_COLORS[mName] || '#16a34a' }}>{mName}</td>
                            <td style={{ padding: '8px 12px' }}>{pc.class}</td>
                            <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)' }}>{(pc.precision * 100).toFixed(1)}%</td>
                            <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)' }}>{(pc.recall * 100).toFixed(1)}%</td>
                            <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--fo-accent-blue)' }}>{(pc.AP50 * 100).toFixed(1)}%</td>
                            <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)' }}>{(pc['AP50-95'] * 100).toFixed(1)}%</td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Curves: Precision-Recall & F1-Confidence matching line 159 */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                {(() => {
                  const prSeries = Object.entries(valResults)
                    .map(([mName, res]) => {
                      const curve = res.curves?.['Precision-Recall(B)'];
                      if (!curve) return null;
                      const yData = Array.isArray(curve.y[0])
                        ? curve.y[0].map((_: any, colIdx: number) => curve.y.reduce((acc: number, row: any) => acc + row[colIdx], 0) / curve.y.length)
                        : curve.y;
                      return { name: mName, color: VARIANT_COLORS[mName] || '#16a34a', x: curve.x, y: yData };
                    })
                    .filter(Boolean) as any[];

                  return prSeries.length > 0 ? (
                    <CurveChart title="Precision–Recall (All Classes)" xlabel="Recall" ylabel="Precision" series={prSeries} />
                  ) : null;
                })()}

                {(() => {
                  const f1Series = Object.entries(valResults)
                    .map(([mName, res]) => {
                      const curve = res.curves?.['F1-Confidence(B)'];
                      if (!curve) return null;
                      const yData = Array.isArray(curve.y[0])
                        ? curve.y[0].map((_: any, colIdx: number) => curve.y.reduce((acc: number, row: any) => acc + row[colIdx], 0) / curve.y.length)
                        : curve.y;
                      return { name: mName, color: VARIANT_COLORS[mName] || '#16a34a', x: curve.x, y: yData };
                    })
                    .filter(Boolean) as any[];

                  return f1Series.length > 0 ? (
                    <CurveChart title="F1–Confidence (All Classes)" xlabel="Confidence" ylabel="F1" series={f1Series} />
                  ) : null;
                })()}
              </div>

              {/* Confusion Matrices matching line 174 */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '16px' }}>
                {Object.entries(valResults).map(([mName, res]) => {
                  if (!res.confusion_matrix) return null;
                  return (
                    <ConfusionMatrixHeatmap
                      key={mName}
                      modelName={mName}
                      matrix={res.confusion_matrix}
                      names={res.names}
                    />
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* =========================================================================
          TAB 3: Efficiency matching 06e_efficiency.jpg
          ========================================================================= */}
      {activeTab === 'efficiency' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
          <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '16px' }}>
            <div style={{ fontSize: '12px', color: 'var(--fo-text-subtle)', marginBottom: '14px' }}>
              Parameters and GFLOPs are read directly from model checkpoints; latency is measured live on this machine's device.
            </div>

            {/* Checkpoint Multi-Select & Benchmark Controls */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '14px', flexWrap: 'wrap' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <span style={{ fontSize: '11px', fontWeight: 700, color: 'var(--fo-text-muted)', textTransform: 'uppercase' }}>
                  Timed runs:
                </span>
                <input
                  type="range"
                  min="3"
                  max="50"
                  value={timedRuns}
                  onChange={(e) => setTimedRuns(parseInt(e.target.value))}
                  style={{ width: '120px', accentColor: 'var(--fo-accent-blue)', cursor: 'pointer' }}
                />
                <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', minWidth: '24px', fontWeight: 700 }}>
                  {timedRuns}
                </span>
              </div>

              <button
                className="fo-btn fo-btn-primary"
                onClick={handleRunBenchmark}
                disabled={benchmarking || selectedEfficiencyCkpts.length === 0}
                style={{ padding: '7px 16px', fontSize: '12px' }}
              >
                <Cpu size={13} /> {benchmarking ? `Timing ${timedRuns} runs on active device...` : 'Measure latency'}
              </button>
            </div>

            {benchError && (
              <div style={{ marginTop: '10px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', color: '#ef4444', padding: '8px 12px', borderRadius: '4px', fontSize: '12px' }}>
                {benchError}
              </div>
            )}
          </div>

          {/* Specifications Table matching 06e_efficiency.jpg line 105 */}
          <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '18px' }}>
            <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--fo-text-main)', marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Layers size={15} style={{ color: 'var(--fo-accent-blue)' }} /> Checkpoint Architectural Specifications
            </div>
            <div style={{ fontSize: '11px', color: 'var(--fo-text-subtle)', marginBottom: '14px' }}>
              Static characteristics extracted from PyTorch model weights
            </div>

            <div style={{ overflowX: 'auto' }}>
              <table className="fo-json-tree" style={{ width: '100%', color: 'var(--fo-text-main)', borderCollapse: 'collapse', fontSize: '11px' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--fo-border)', textTransform: 'uppercase', color: 'var(--fo-text-muted)' }}>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>checkpoint</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>RMCBAM</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>classes</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>params (M)</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>GFLOPs@640</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>size (MB)</th>
                  </tr>
                </thead>
                <tbody>
                  {checkpoints.map((c) => (
                    <tr key={c.path} style={{ borderBottom: '1px solid var(--fo-border)' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{c.label || c.filename}</td>
                      <td style={{ padding: '8px 12px' }}>
                        <span style={{ color: c.is_rmcbam ? '#16a34a' : 'var(--fo-text-muted)', fontWeight: 600 }}>
                          {c.is_rmcbam ? 'True' : 'False'}
                        </span>
                      </td>
                      <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)' }}>{c.nc}</td>
                      <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)' }}>{(c.n_params / 1e6).toFixed(3)}</td>
                      <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)' }}>{c.gflops ? c.gflops.toFixed(2) : '—'}</td>
                      <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)' }}>{c.size_mb} MB</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Benchmark Results Table & Visual Comparison matching line 114 */}
          {benchmarkResults.length > 0 && (
            <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '18px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div>
                <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--fo-text-main)', marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Activity size={15} style={{ color: 'var(--fo-accent-blue)' }} /> Measured Latency Benchmark ({timedRuns} runs on active device)
                </div>
                <div style={{ fontSize: '11px', color: 'var(--fo-text-subtle)' }}>
                  Inference timing across pre-processing, forward pass, and post-processing
                </div>
              </div>

              <div style={{ overflowX: 'auto' }}>
                <table className="fo-json-tree" style={{ width: '100%', color: 'var(--fo-text-main)', borderCollapse: 'collapse', fontSize: '11px' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--fo-border)', textTransform: 'uppercase', color: 'var(--fo-text-muted)' }}>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>checkpoint</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>mean (ms)</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>median (ms)</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>p95 (ms)</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>FPS</th>
                    </tr>
                  </thead>
                  <tbody>
                    {benchmarkResults.map((b) => (
                      <tr key={b.path} style={{ borderBottom: '1px solid var(--fo-border)' }}>
                        <td style={{ padding: '8px 12px', fontWeight: 600, color: b.is_rmcbam ? 'var(--fo-accent-blue)' : 'var(--fo-text-main)' }}>
                          {b.checkpoint}
                        </td>
                        <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)', fontWeight: 700 }}>
                          {typeof b.mean_ms === 'number' ? b.mean_ms.toFixed(2) : b.mean_ms} ms
                        </td>
                        <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)' }}>
                          {typeof b.median_ms === 'number' ? b.median_ms.toFixed(2) : b.median_ms} ms
                        </td>
                        <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)' }}>
                          {typeof b.p95_ms === 'number' ? b.p95_ms.toFixed(2) : b.p95_ms} ms
                        </td>
                        <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)', color: '#10b981', fontWeight: 700 }}>
                          {typeof b.fps === 'number' ? b.fps.toFixed(1) : b.fps}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Latency & FPS Visual Comparison */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginTop: '6px' }}>
                <div style={{ background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '14px' }}>
                  <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--fo-text-muted)', textTransform: 'uppercase', marginBottom: '10px' }}>
                    Latency (ms) — Lower is Better
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {benchmarkResults.map((b) => {
                      const mean = typeof b.mean_ms === 'number' ? b.mean_ms : parseFloat(b.mean_ms) || 0;
                      const maxMs = Math.max(...benchmarkResults.map((x) => typeof x.mean_ms === 'number' ? x.mean_ms : 1));
                      const pct = Math.min(100, Math.max(8, (mean / (maxMs || 1)) * 100));
                      return (
                        <div key={b.checkpoint}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', marginBottom: '2px' }}>
                            <span style={{ fontWeight: 600, color: b.is_rmcbam ? '#16a34a' : 'var(--fo-text-main)' }}>{b.checkpoint}</span>
                            <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>{mean.toFixed(2)} ms</span>
                          </div>
                          <div style={{ width: '100%', height: '8px', background: 'var(--fo-bg-panel)', borderRadius: '4px', overflow: 'hidden' }}>
                            <div style={{ width: `${pct}%`, height: '100%', background: b.is_rmcbam ? '#16a34a' : '#64748b' }} />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div style={{ background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '14px' }}>
                  <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--fo-text-muted)', textTransform: 'uppercase', marginBottom: '10px' }}>
                    Throughput (FPS) — Higher is Better
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {benchmarkResults.map((b) => {
                      const fps = typeof b.fps === 'number' ? b.fps : parseFloat(b.fps) || 0;
                      const maxFps = Math.max(...benchmarkResults.map((x) => typeof x.fps === 'number' ? x.fps : 1));
                      const pct = Math.min(100, Math.max(8, (fps / (maxFps || 1)) * 100));
                      return (
                        <div key={b.checkpoint}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', marginBottom: '2px' }}>
                            <span style={{ fontWeight: 600, color: b.is_rmcbam ? '#16a34a' : 'var(--fo-text-main)' }}>{b.checkpoint}</span>
                            <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: '#10b981' }}>{fps.toFixed(1)} FPS</span>
                          </div>
                          <div style={{ width: '100%', height: '8px', background: 'var(--fo-bg-panel)', borderRadius: '4px', overflow: 'hidden' }}>
                            <div style={{ width: `${pct}%`, height: '100%', background: '#10b981' }} />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Modal Zoom for Ultralytics Plots */}
      {previewPlot && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.85)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9999,
            padding: '24px',
          }}
          onClick={() => setPreviewPlot(null)}
        >
          <div
            style={{
              position: 'relative',
              maxWidth: '92vw',
              maxHeight: '92vh',
              background: 'var(--fo-bg-panel)',
              border: '1px solid var(--fo-border)',
              borderRadius: '8px',
              padding: '16px',
              display: 'flex',
              flexDirection: 'column',
              gap: '12px',
              boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ fontSize: '14px', fontWeight: 700, color: 'var(--fo-text-main)' }}>
                {previewPlot.title} — <span style={{ color: VARIANT_COLORS[previewPlot.model] || '#16a34a' }}>{previewPlot.model}</span>
              </div>
              <button
                className="fo-btn fo-btn-secondary"
                onClick={() => setPreviewPlot(null)}
                style={{ padding: '4px 8px', borderRadius: '4px' }}
              >
                <X size={16} />
              </button>
            </div>
            <div style={{ overflow: 'auto', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <img
                src={previewPlot.src}
                alt={previewPlot.title}
                style={{ maxWidth: '100%', maxHeight: '80vh', objectFit: 'contain', borderRadius: '4px' }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
