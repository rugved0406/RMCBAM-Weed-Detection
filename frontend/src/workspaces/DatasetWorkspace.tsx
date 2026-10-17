import React, { useEffect, useState } from 'react';
import { getDatasetInfo, computeQuality, convertDataset } from '../services/api';
import {
  Database,
  Download,
  RefreshCw,
  Zap,
} from 'lucide-react';

type TabType = 'annotations' | 'yolo-validation' | 'image-quality' | 'converter' | 'recovered-boxes';

const CLASS_COLORS: Record<string, string> = {
  Weed: '#e53935',
  Tomato: '#3987e5',
  Rose: '#8e24aa',
  Marigold: '#f57c00',
  Cauliflower: '#43a047',
};

export const DatasetWorkspace: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('annotations');
  const [data, setData] = useState<any | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  // Tab 3: Image Quality State
  const [qualityLimit, setQualityLimit] = useState<number>(50);
  const [qualityLoading, setQualityLoading] = useState<boolean>(false);
  const [qualityData, setQualityData] = useState<any | null>(null);

  // Tab 4: Converter State
  const [srcFolder, setSrcFolder] = useState<string>('data');
  const [jsonFolder, setJsonFolder] = useState<string>('');
  const [outFolder, setOutFolder] = useState<string>('data/yolo');
  const [valFraction, setValFraction] = useState<number>(0.2);
  const [includeUnlabelled, setIncludeUnlabelled] = useState<boolean>(false);
  const [converting, setConverting] = useState<boolean>(false);
  const [convertResult, setConvertResult] = useState<any | null>(null);
  const [convertError, setConvertError] = useState<string | null>(null);

  useEffect(() => {
    fetchDataset();
  }, []);

  const fetchDataset = async () => {
    setLoading(true);
    try {
      const res = await getDatasetInfo();
      setData(res);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const handleComputeQuality = async () => {
    setQualityLoading(true);
    try {
      const res = await computeQuality(qualityLimit);
      setQualityData(res);
    } catch (e) {
      console.error(e);
    } finally {
      setQualityLoading(false);
    }
  };

  const handleConvert = async () => {
    setConverting(true);
    setConvertError(null);
    setConvertResult(null);
    try {
      const res = await convertDataset({
        src_folder: srcFolder,
        out_folder: outFolder,
        json_folder: jsonFolder || undefined,
        val_fraction: valFraction,
        include_unlabelled: includeUnlabelled,
      });
      setConvertResult(res);
    } catch (e: any) {
      setConvertError(e.response?.data?.detail || e.message || 'Conversion failed.');
    } finally {
      setConverting(false);
    }
  };

  const handleDownloadQualityCsv = () => {
    if (!qualityData?.records || qualityData.records.length === 0) return;
    const records = qualityData.records;
    const keys = Object.keys(records[0]);
    const header = keys.join(',') + '\n';
    const rows = records.map((r: any) => keys.map((k) => JSON.stringify(r[k] ?? '')).join(',')).join('\n');
    const blob = new Blob([header + rows], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'image_quality.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  if (loading && !data) {
    return (
      <div className="fo-canvas-workspace" style={{ padding: '48px', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ fontSize: '13px', color: 'var(--fo-text-subtle)' }}>Scanning Dataset Inventory &amp; Annotations...</div>
      </div>
    );
  }

  const inv = data?.inventory || {};
  const ann = data?.annotations || {};
  const val = data?.validation || {};
  const rec = data?.recovered || null;

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
            <Database size={18} style={{ color: '#10b981' }} /> Dataset Analytics
          </h2>
          <div style={{ fontSize: '12px', color: 'var(--fo-text-subtle)', marginTop: '4px' }}>
            Inventory, annotation coverage, class distribution, label validation and image quality.
          </div>
        </div>

        {/* 5 Tabs matching 07_dataset_annotations.jpg line 73 */}
        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
          <button
            className={`fo-btn ${activeTab === 'annotations' ? 'fo-btn-primary' : 'fo-btn-secondary'}`}
            style={{ fontSize: '11px', padding: '5px 12px' }}
            onClick={() => setActiveTab('annotations')}
          >
            Annotations
          </button>
          <button
            className={`fo-btn ${activeTab === 'yolo-validation' ? 'fo-btn-primary' : 'fo-btn-secondary'}`}
            style={{ fontSize: '11px', padding: '5px 12px' }}
            onClick={() => setActiveTab('yolo-validation')}
          >
            YOLO validation
          </button>
          <button
            className={`fo-btn ${activeTab === 'image-quality' ? 'fo-btn-primary' : 'fo-btn-secondary'}`}
            style={{ fontSize: '11px', padding: '5px 12px' }}
            onClick={() => setActiveTab('image-quality')}
          >
            Image quality
          </button>
          <button
            className={`fo-btn ${activeTab === 'converter' ? 'fo-btn-primary' : 'fo-btn-secondary'}`}
            style={{ fontSize: '11px', padding: '5px 12px' }}
            onClick={() => setActiveTab('converter')}
          >
            Convert LabelMe → YOLO
          </button>
          <button
            className={`fo-btn ${activeTab === 'recovered-boxes' ? 'fo-btn-primary' : 'fo-btn-secondary'}`}
            style={{ fontSize: '11px', padding: '5px 12px' }}
            onClick={() => setActiveTab('recovered-boxes')}
          >
            Recovered patch boxes
          </button>
        </div>
      </div>

      {/* 4 Overview KPI Cards matching line 64 */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
        <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '14px' }}>
          <div style={{ fontSize: '10px', fontWeight: 700, color: 'var(--fo-text-muted)', textTransform: 'uppercase' }}>Images</div>
          <div style={{ fontSize: '24px', fontWeight: 700, color: 'var(--fo-accent-blue)', fontFamily: 'var(--font-mono)', marginTop: '2px' }}>
            {inv.n_images || 0}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--fo-text-subtle)', marginTop: '2px' }}>
            {inv.images_by_folder ? `${Object.keys(inv.images_by_folder).length} folder(s)` : ''}
          </div>
        </div>

        <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '14px' }}>
          <div style={{ fontSize: '10px', fontWeight: 700, color: 'var(--fo-text-muted)', textTransform: 'uppercase' }}>LabelMe JSON</div>
          <div style={{ fontSize: '24px', fontWeight: 700, color: 'var(--fo-text-main)', fontFamily: 'var(--font-mono)', marginTop: '2px' }}>
            {inv.n_labelme_json || 0}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--fo-text-subtle)', marginTop: '2px' }}>files with 'shapes'</div>
        </div>

        <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '14px' }}>
          <div style={{ fontSize: '10px', fontWeight: 700, color: 'var(--fo-text-muted)', textTransform: 'uppercase' }}>YOLO label files</div>
          <div style={{ fontSize: '24px', fontWeight: 700, color: 'var(--fo-text-main)', fontFamily: 'var(--font-mono)', marginTop: '2px' }}>
            {inv.n_yolo_txt || 0}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--fo-text-subtle)', marginTop: '2px' }}>under */labels/*</div>
        </div>

        <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '14px' }}>
          <div style={{ fontSize: '10px', fontWeight: 700, color: 'var(--fo-text-muted)', textTransform: 'uppercase' }}>YAML files</div>
          <div style={{ fontSize: '24px', fontWeight: 700, color: '#10b981', fontFamily: 'var(--font-mono)', marginTop: '2px' }}>
            {inv.yaml_files ? inv.yaml_files.length : 0}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--fo-text-subtle)', marginTop: '2px' }}>data.yaml configs</div>
        </div>
      </div>

      {/* TAB 1: Annotations matching 07_dataset_annotations.jpg */}
      {activeTab === 'annotations' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
          {ann.class_counts && Object.keys(ann.class_counts).length > 0 ? (
            <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '18px' }}>
              <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--fo-text-main)', marginBottom: '14px' }}>
                Annotated instances per class
              </div>

              {/* Bar Chart matching line 88 */}
              <div style={{ display: 'grid', gridTemplateColumns: `repeat(${Object.keys(ann.class_counts).length}, 1fr)`, gap: '12px', alignItems: 'flex-end', height: '180px', padding: '10px 0', borderBottom: '1px solid var(--fo-border)', marginBottom: '14px' }}>
                {Object.entries(ann.class_counts).map(([cls, cnt]: [string, any]) => {
                  const maxVal = Math.max(...(Object.values(ann.class_counts) as number[]), 1);
                  const hPct = (cnt / maxVal) * 100;
                  const color = CLASS_COLORS[cls] || '#898781';

                  return (
                    <div key={cls} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', height: '100%', justifyContent: 'flex-end' }}>
                      <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--fo-text-main)', marginBottom: '4px' }}>
                        {cnt}
                      </span>
                      <div style={{ width: '60%', height: `${Math.max(6, hPct)}%`, backgroundColor: color, borderRadius: '3px 3px 0 0' }} />
                      <span style={{ fontSize: '11px', fontWeight: 600, color: 'var(--fo-text-muted)', marginTop: '6px', textAlign: 'center' }}>
                        {cls}
                      </span>
                    </div>
                  );
                })}
              </div>

              {/* Stats Row */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px', marginTop: '14px' }}>
                <div style={{ background: 'var(--fo-bg-elevated)', borderRadius: '4px', padding: '12px' }}>
                  <div style={{ fontSize: '11px', color: 'var(--fo-text-muted)' }}>Images without a JSON</div>
                  <div style={{ fontSize: '18px', fontWeight: 700, fontFamily: 'var(--font-mono)', marginTop: '2px' }}>
                    {ann.missing_json_count} / {inv.n_images}
                  </div>
                </div>

                <div style={{ background: 'var(--fo-bg-elevated)', borderRadius: '4px', padding: '12px' }}>
                  <div style={{ fontSize: '11px', color: 'var(--fo-text-muted)' }}>Total Shapes</div>
                  <div style={{ fontSize: '18px', fontWeight: 700, fontFamily: 'var(--font-mono)', marginTop: '2px' }}>
                    {ann.total_shapes} ({Object.entries(ann.shapes || {}).map(([s, c]) => `${s}: ${c}`).join(', ')})
                  </div>
                </div>
              </div>

              {ann.missing_json_files && ann.missing_json_files.length > 0 && (
                <div style={{ marginTop: '14px', borderTop: '1px solid var(--fo-border)', paddingTop: '12px' }}>
                  <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--fo-text-muted)', marginBottom: '6px' }}>
                    Images without annotations ({ann.missing_json_count})
                  </div>
                  <div style={{ maxHeight: '120px', overflowY: 'auto', fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--fo-text-subtle)', background: 'var(--fo-bg-elevated)', padding: '8px', borderRadius: '4px' }}>
                    {ann.missing_json_files.slice(0, 30).join(', ')}
                    {ann.missing_json_files.length > 30 ? '...' : ''}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '24px', textAlign: 'center', color: 'var(--fo-text-muted)', fontSize: '13px' }}>
              No LabelMe annotations found in dataset directory.
            </div>
          )}
        </div>
      )}

      {/* TAB 2: YOLO Validation */}
      {activeTab === 'yolo-validation' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
          <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '18px' }}>
            <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--fo-text-main)', marginBottom: '14px' }}>
              YOLO Dataset Splits &amp; Classes
            </div>

            {val.splits ? (
              <>
                <table className="fo-json-tree" style={{ width: '100%', color: 'var(--fo-text-main)', borderCollapse: 'collapse', fontSize: '11px', marginBottom: '16px' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--fo-border)', textTransform: 'uppercase', color: 'var(--fo-text-muted)' }}>
                      <th style={{ padding: '6px 10px', textAlign: 'left' }}>split</th>
                      <th style={{ padding: '6px 10px', textAlign: 'left' }}>images</th>
                      <th style={{ padding: '6px 10px', textAlign: 'left' }}>labels</th>
                      <th style={{ padding: '6px 10px', textAlign: 'left' }}>empty</th>
                      <th style={{ padding: '6px 10px', textAlign: 'left' }}>corrupt</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(val.splits).map(([sp, info]: [string, any]) => (
                      <tr key={sp} style={{ borderBottom: '1px solid var(--fo-border)' }}>
                        <td style={{ padding: '6px 10px', fontWeight: 700, color: 'var(--fo-accent-blue)' }}>{sp}</td>
                        <td style={{ padding: '6px 10px', fontFamily: 'var(--font-mono)' }}>{info.images}</td>
                        <td style={{ padding: '6px 10px', fontFamily: 'var(--font-mono)' }}>{info.labels}</td>
                        <td style={{ padding: '6px 10px', fontFamily: 'var(--font-mono)' }}>{info.empty || 0}</td>
                        <td style={{ padding: '6px 10px', fontFamily: 'var(--font-mono)' }}>{info.corrupt || 0}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>

                {val.missing_classes && val.missing_classes.length > 0 && (
                  <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', color: '#ef4444', padding: '10px 14px', borderRadius: '4px', fontSize: '12px', marginBottom: '14px' }}>
                    No instances for: {val.missing_classes.join(', ')}
                  </div>
                )}

                <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--fo-text-muted)', marginBottom: '8px' }}>
                  Instances per class (all splits):
                </div>
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                  {val.instances &&
                    Object.entries(val.instances).map(([cls, cnt]) => (
                      <span key={cls} style={{ background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', padding: '4px 10px', borderRadius: '4px', fontSize: '11px', fontFamily: 'var(--font-mono)' }}>
                        <b>{cls}:</b> {cnt as number} boxes
                      </span>
                    ))}
                </div>
              </>
            ) : (
              <div style={{ color: 'var(--fo-text-muted)', fontSize: '12px' }}>
                No active data.yaml validation found in partial dataset.
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 3: Image Quality matching 07b_dataset_quality.jpg */}
      {activeTab === 'image-quality' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
          <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '14px', flexWrap: 'wrap' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <span style={{ fontSize: '11px', fontWeight: 700, color: 'var(--fo-text-muted)', textTransform: 'uppercase' }}>
                  Images to analyse:
                </span>
                <input
                  type="range"
                  min="10"
                  max={Math.max(10, inv.n_images || 100)}
                  value={qualityLimit}
                  onChange={(e) => setQualityLimit(parseInt(e.target.value))}
                  style={{ width: '120px', accentColor: 'var(--fo-accent-blue)', cursor: 'pointer' }}
                />
                <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', minWidth: '30px' }}>
                  {qualityLimit}
                </span>
              </div>

              <button
                className="fo-btn fo-btn-primary"
                onClick={handleComputeQuality}
                disabled={qualityLoading}
                style={{ padding: '6px 14px', fontSize: '12px' }}
              >
                <Zap size={13} /> {qualityLoading ? 'Computing Quality Stats...' : 'Compute image quality statistics'}
              </button>
            </div>
          </div>

          {qualityData && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
              {/* 4 Cards matching 07b line 144 */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
                <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '14px' }}>
                  <div style={{ fontSize: '10px', fontWeight: 700, color: 'var(--fo-text-muted)', textTransform: 'uppercase' }}>Readable</div>
                  <div style={{ fontSize: '24px', fontWeight: 700, color: 'var(--fo-accent-blue)', fontFamily: 'var(--font-mono)', marginTop: '2px' }}>
                    {qualityData.summary.readable}
                  </div>
                  <div style={{ fontSize: '11px', color: 'var(--fo-text-subtle)', marginTop: '2px' }}>images analysed</div>
                </div>

                <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '14px' }}>
                  <div style={{ fontSize: '10px', fontWeight: 700, color: 'var(--fo-text-muted)', textTransform: 'uppercase' }}>Likely Augmented</div>
                  <div style={{ fontSize: '24px', fontWeight: 700, color: 'var(--fo-text-main)', fontFamily: 'var(--font-mono)', marginTop: '2px' }}>
                    {qualityData.summary.likely_augmented}
                  </div>
                  <div style={{ fontSize: '11px', color: 'var(--fo-text-subtle)', marginTop: '2px' }}>black padding &gt; 2%</div>
                </div>

                <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '14px' }}>
                  <div style={{ fontSize: '10px', fontWeight: 700, color: 'var(--fo-text-muted)', textTransform: 'uppercase' }}>Near-Duplicate Groups</div>
                  <div style={{ fontSize: '24px', fontWeight: 700, color: 'var(--fo-text-main)', fontFamily: 'var(--font-mono)', marginTop: '2px' }}>
                    {qualityData.summary.near_duplicate_groups}
                  </div>
                  <div style={{ fontSize: '11px', color: 'var(--fo-text-subtle)', marginTop: '2px' }}>share 8×8 hash</div>
                </div>

                <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '14px' }}>
                  <div style={{ fontSize: '10px', fontWeight: 700, color: 'var(--fo-text-muted)', textTransform: 'uppercase' }}>Resolutions</div>
                  <div style={{ fontSize: '24px', fontWeight: 700, color: '#10b981', fontFamily: 'var(--font-mono)', marginTop: '2px' }}>
                    {qualityData.summary.resolutions_count}
                  </div>
                  <div style={{ fontSize: '11px', color: 'var(--fo-text-subtle)', marginTop: '2px' }}>
                    median {qualityData.summary.median_width}×{qualityData.summary.median_height}
                  </div>
                </div>
              </div>

              {/* Quality Records Table */}
              <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '18px' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                  <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--fo-text-main)' }}>
                    Image Quality Inspection Table ({qualityData.records.length} Images)
                  </div>
                  <button className="fo-btn fo-btn-secondary" onClick={handleDownloadQualityCsv} style={{ fontSize: '11px', padding: '4px 10px' }}>
                    <Download size={13} /> Download Quality Table CSV
                  </button>
                </div>

                <div style={{ maxHeight: '280px', overflowY: 'auto' }}>
                  <table className="fo-json-tree" style={{ width: '100%', color: 'var(--fo-text-main)', borderCollapse: 'collapse', fontSize: '11px' }}>
                    <thead>
                      <tr style={{ position: 'sticky', top: 0, background: 'var(--fo-bg-panel)', borderBottom: '1px solid var(--fo-border)', textTransform: 'uppercase', color: 'var(--fo-text-muted)' }}>
                        <th style={{ padding: '6px 8px', textAlign: 'left' }}>filename</th>
                        <th style={{ padding: '6px 8px', textAlign: 'left' }}>resolution</th>
                        <th style={{ padding: '6px 8px', textAlign: 'left' }}>sharpness</th>
                        <th style={{ padding: '6px 8px', textAlign: 'left' }}>green_%</th>
                        <th style={{ padding: '6px 8px', textAlign: 'left' }}>padding_%</th>
                        <th style={{ padding: '6px 8px', textAlign: 'left' }}>augmented</th>
                      </tr>
                    </thead>
                    <tbody>
                      {qualityData.records.map((r: any, idx: number) => (
                        <tr key={idx} style={{ borderBottom: '1px solid var(--fo-border)' }}>
                          <td style={{ padding: '6px 8px', fontWeight: 600 }}>{r.name || r.filename}</td>
                          <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)' }}>{r.width}×{r.height}</td>
                          <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)' }}>{r.sharpness_lapvar ? Math.round(r.sharpness_lapvar) : '—'}</td>
                          <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)' }}>{(r.green_frac * 100).toFixed(1)}%</td>
                          <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)' }}>{(r.black_border_frac * 100).toFixed(1)}%</td>
                          <td style={{ padding: '6px 8px' }}>
                            <span style={{ color: r.likely_augmented ? '#f59e0b' : '#10b981', fontWeight: 600 }}>
                              {r.likely_augmented ? 'Yes' : 'No'}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 4: Converter matching line 175 */}
      {activeTab === 'converter' && (
        <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '18px' }}>
          <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--fo-text-main)', marginBottom: '4px' }}>
            Convert LabelMe → YOLO
          </div>
          <div style={{ fontSize: '12px', color: 'var(--fo-text-subtle)', marginBottom: '16px' }}>
            Copies images into a new YOLO dataset folder and writes labels + data.yaml. Source files are never modified.
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px', marginBottom: '14px' }}>
            <div>
              <label style={{ fontSize: '11px', fontWeight: 600, color: 'var(--fo-text-subtle)', marginBottom: '4px', display: 'block' }}>
                Images folder
              </label>
              <input
                type="text"
                value={srcFolder}
                onChange={(e) => setSrcFolder(e.target.value)}
                style={{ width: '100%', background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', color: 'var(--fo-text-main)', padding: '6px 10px', borderRadius: '4px', fontSize: '12px' }}
              />
            </div>

            <div>
              <label style={{ fontSize: '11px', fontWeight: 600, color: 'var(--fo-text-subtle)', marginBottom: '4px', display: 'block' }}>
                JSON folder (blank = next to images)
              </label>
              <input
                type="text"
                value={jsonFolder}
                onChange={(e) => setJsonFolder(e.target.value)}
                style={{ width: '100%', background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', color: 'var(--fo-text-main)', padding: '6px 10px', borderRadius: '4px', fontSize: '12px' }}
              />
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr auto', gap: '14px', alignItems: 'flex-end', marginBottom: '16px' }}>
            <div>
              <label style={{ fontSize: '11px', fontWeight: 600, color: 'var(--fo-text-subtle)', marginBottom: '4px', display: 'block' }}>
                Output folder
              </label>
              <input
                type="text"
                value={outFolder}
                onChange={(e) => setOutFolder(e.target.value)}
                style={{ width: '100%', background: 'var(--fo-bg-elevated)', border: '1px solid var(--fo-border)', color: 'var(--fo-text-main)', padding: '6px 10px', borderRadius: '4px', fontSize: '12px' }}
              />
            </div>

            <div>
              <label style={{ fontSize: '11px', fontWeight: 600, color: 'var(--fo-text-subtle)', marginBottom: '4px', display: 'block' }}>
                Validation fraction: {valFraction}
              </label>
              <input
                type="range"
                min="0.05"
                max="0.5"
                step="0.05"
                value={valFraction}
                onChange={(e) => setValFraction(parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--fo-accent-blue)', cursor: 'pointer' }}
              />
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <input
                type="checkbox"
                id="includeUnlabelled"
                checked={includeUnlabelled}
                onChange={(e) => setIncludeUnlabelled(e.target.checked)}
                style={{ accentColor: 'var(--fo-accent-blue)', cursor: 'pointer' }}
              />
              <label htmlFor="includeUnlabelled" style={{ fontSize: '12px', color: 'var(--fo-text-subtle)', cursor: 'pointer' }}>
                Include unlabelled images
              </label>
            </div>

            <button
              className="fo-btn fo-btn-primary"
              onClick={handleConvert}
              disabled={converting}
              style={{ padding: '8px 18px', fontSize: '12px' }}
            >
              <RefreshCw size={13} /> {converting ? 'Converting...' : 'Convert'}
            </button>
          </div>

          {convertError && (
            <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', color: '#ef4444', padding: '10px 14px', borderRadius: '4px', fontSize: '12px' }}>
              {convertError}
            </div>
          )}

          {convertResult && (
            <div style={{ background: 'var(--fo-bg-elevated)', border: '1px solid #10b981', borderRadius: '6px', padding: '14px' }}>
              <div style={{ fontSize: '12px', fontWeight: 700, color: '#10b981', marginBottom: '6px' }}>
                Conversion Completed Successfully!
              </div>
              <pre className="fo-json-tree" style={{ maxHeight: '180px', overflowY: 'auto' }}>
                {JSON.stringify(convertResult, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}

      {/* TAB 5: Recovered Patch Boxes matching 07c_recovered_boxes.jpg line 197 */}
      {activeTab === 'recovered-boxes' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
          {rec ? (
            <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '18px' }}>
              <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--fo-text-main)', marginBottom: '4px' }}>
                Recovered Patch Boxes
              </div>
              <div style={{ fontSize: '12px', color: 'var(--fo-text-subtle)', marginBottom: '14px' }}>
                Boxes recovered by locating each patch from Crops/Weed_Crops and Crops/Crop_Flower_Crops inside its source image (normalised cross-correlation).
              </div>

              {/* 4 Cards matching line 206 */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px', marginBottom: '16px' }}>
                <div style={{ background: 'var(--fo-bg-elevated)', borderRadius: '4px', padding: '12px' }}>
                  <div style={{ fontSize: '10px', color: 'var(--fo-text-muted)', textTransform: 'uppercase' }}>Patches</div>
                  <div style={{ fontSize: '20px', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>{rec.total_patches}</div>
                  <div style={{ fontSize: '11px', color: 'var(--fo-text-subtle)' }}>
                    {Object.entries(rec.label_counts || {}).map(([l, c]) => `${l}: ${c}`).join(', ')}
                  </div>
                </div>

                <div style={{ background: 'var(--fo-bg-elevated)', borderRadius: '4px', padding: '12px' }}>
                  <div style={{ fontSize: '10px', color: 'var(--fo-text-muted)', textTransform: 'uppercase' }}>Matched</div>
                  <div style={{ fontSize: '20px', fontWeight: 700, fontFamily: 'var(--font-mono)', color: '#10b981' }}>{rec.matched_count}</div>
                  <div style={{ fontSize: '11px', color: 'var(--fo-text-subtle)' }}>NCC ≥ 0.9</div>
                </div>

                <div style={{ background: 'var(--fo-bg-elevated)', borderRadius: '4px', padding: '12px' }}>
                  <div style={{ fontSize: '10px', color: 'var(--fo-text-muted)', textTransform: 'uppercase' }}>Match Score</div>
                  <div style={{ fontSize: '20px', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>{rec.min_score}</div>
                  <div style={{ fontSize: '11px', color: 'var(--fo-text-subtle)' }}>minimum NCC (1.0 = exact)</div>
                </div>

                <div style={{ background: 'var(--fo-bg-elevated)', borderRadius: '4px', padding: '12px' }}>
                  <div style={{ fontSize: '10px', color: 'var(--fo-text-muted)', textTransform: 'uppercase' }}>Source Images</div>
                  <div style={{ fontSize: '20px', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>{rec.source_images_count}</div>
                  <div style={{ fontSize: '11px', color: 'var(--fo-text-subtle)' }}>with ≥1 recovered box</div>
                </div>
              </div>

              {/* Records Table */}
              <div style={{ maxHeight: '280px', overflowY: 'auto' }}>
                <table className="fo-json-tree" style={{ width: '100%', color: 'var(--fo-text-main)', borderCollapse: 'collapse', fontSize: '11px' }}>
                  <thead>
                    <tr style={{ position: 'sticky', top: 0, background: 'var(--fo-bg-panel)', borderBottom: '1px solid var(--fo-border)', textTransform: 'uppercase', color: 'var(--fo-text-muted)' }}>
                      <th style={{ padding: '6px 8px', textAlign: 'left' }}>patch</th>
                      <th style={{ padding: '6px 8px', textAlign: 'left' }}>label</th>
                      <th style={{ padding: '6px 8px', textAlign: 'left' }}>source</th>
                      <th style={{ padding: '6px 8px', textAlign: 'left' }}>score</th>
                      <th style={{ padding: '6px 8px', textAlign: 'left' }}>accepted</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rec.records.map((r: any, idx: number) => (
                      <tr key={idx} style={{ borderBottom: '1px solid var(--fo-border)' }}>
                        <td style={{ padding: '6px 8px', fontWeight: 600 }}>{r.patch}</td>
                        <td style={{ padding: '6px 8px' }}>{r.label}</td>
                        <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)' }}>{r.source}</td>
                        <td style={{ padding: '6px 8px', fontFamily: 'var(--font-mono)' }}>{typeof r.score === 'number' ? r.score.toFixed(3) : r.score}</td>
                        <td style={{ padding: '6px 8px' }}>
                          <span style={{ color: r.accepted ? '#10b981' : '#ef4444', fontWeight: 600 }}>
                            {r.accepted ? 'True' : 'False'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div style={{ background: 'var(--fo-bg-panel)', border: '1px solid var(--fo-border)', borderRadius: '6px', padding: '24px', textAlign: 'center', color: 'var(--fo-text-muted)', fontSize: '13px' }}>
              No recovered boxes found in data/recovered/recovered_boxes.csv.
            </div>
          )}
        </div>
      )}
    </div>
  );
};
