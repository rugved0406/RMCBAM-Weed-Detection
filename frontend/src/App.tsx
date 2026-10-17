import React, { useEffect, useState } from 'react';
import { TopBar } from './components/TopBar';
import { LeftSidebar } from './components/LeftSidebar';
import type { WorkspaceId } from './components/LeftSidebar';
import { InspectionWorkspace } from './workspaces/InspectionWorkspace';
import { AnalysisWorkspace } from './workspaces/AnalysisWorkspace';
import { LocalizationWorkspace } from './workspaces/LocalizationWorkspace';
import { ExplainabilityWorkspace } from './workspaces/ExplainabilityWorkspace';
import { DatasetWorkspace } from './workspaces/DatasetWorkspace';
import { ComparisonWorkspace } from './workspaces/ComparisonWorkspace';
import { HistoryWorkspace } from './workspaces/HistoryWorkspace';
import { SettingsWorkspace } from './workspaces/SettingsWorkspace';
import { ErrorBoundary } from './components/ErrorBoundary';

import { getHealth, getCheckpoints, runPredictSample } from './services/api';
import type { SystemHealth, CheckpointInfo, PredictResponse } from './types';

const VALID_WORKSPACES: WorkspaceId[] = ['detection', 'analysis', 'localization', 'explainability', 'dataset', 'comparison', 'history', 'settings'];

const getStoredWorkspace = (): WorkspaceId => {
  try {
    const hash = window.location.hash.replace('#', '') as any;
    if (VALID_WORKSPACES.includes(hash)) return hash;
    // Map previous legacy names
    if (hash === 'overview' || hash === 'inspection') return 'detection';
    if (hash === 'evaluation' || hash === 'model-info') return 'comparison';

    const stored = localStorage.getItem('wv_active_workspace') as any;
    if (VALID_WORKSPACES.includes(stored)) return stored;
    if (stored === 'overview' || stored === 'inspection') return 'detection';
    if (stored === 'evaluation' || stored === 'model-info') return 'comparison';
  } catch {
    // fallback
  }
  return 'detection'; // Weed Detection is the primary landing page!
};

const getStoredPredictData = (): PredictResponse | null => {
  try {
    const raw = sessionStorage.getItem('wv_predict_data');
    if (raw) return JSON.parse(raw);
  } catch {
    // fallback
  }
  return null;
};

export const App: React.FC = () => {
  const [activeWorkspace, setActiveWorkspace] = useState<WorkspaceId>(getStoredWorkspace);
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string>(() => {
    return localStorage.getItem('wv_selected_checkpoint') || '';
  });

  // Analyzed Prediction state shared across workspaces
  const [predictData, setPredictData] = useState<PredictResponse | null>(getStoredPredictData);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [sessionHistory, setSessionHistory] = useState<PredictResponse[]>(() => {
    const initial = getStoredPredictData();
    return initial ? [initial] : [];
  });

  // Sync workspace changes to URL hash and localStorage
  const handleSelectWorkspace = (ws: WorkspaceId) => {
    setActiveWorkspace(ws);
    try {
      localStorage.setItem('wv_active_workspace', ws);
      window.location.hash = ws;
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    initApp();

    const handleHashChange = () => {
      const hash = window.location.hash.replace('#', '') as WorkspaceId;
      if (VALID_WORKSPACES.includes(hash)) {
        setActiveWorkspace(hash);
      }
    };
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  const updatePredictData = (data: PredictResponse | null, file?: File | null) => {
    setPredictData(data);
    if (file !== undefined) setSelectedFile(file);
    try {
      if (data) {
        // Safe storage of lightweight metadata; base64 strings exceed 5MB quota
        try {
          const { original_image: _o, annotated_image: _a, ...meta } = data;
          sessionStorage.setItem('wv_predict_data_meta', JSON.stringify(meta));
        } catch {
          // ignore quota limits
        }
        sessionStorage.removeItem('wv_cleared');
        if (data.filename) {
          localStorage.setItem('wv_last_sample', data.filename);
        }
        setSessionHistory((prev) => {
          const filtered = prev.filter((p) => p.filename !== data.filename);
          return [data, ...filtered];
        });
      } else {
        sessionStorage.removeItem('wv_predict_data');
        sessionStorage.removeItem('wv_predict_data_meta');
      }
    } catch (e) {
      console.warn('Session storage notice:', e);
    }
  };

  const initApp = async () => {
    try {
      const h = await getHealth();
      setHealth(h);

      const ckptsData = await getCheckpoints();
      setCheckpoints(ckptsData.checkpoints || []);

      if (ckptsData.checkpoints && ckptsData.checkpoints.length > 0) {
        const storedCkpt = localStorage.getItem('wv_selected_checkpoint');
        const match = storedCkpt && ckptsData.checkpoints.find((c) => c.path === storedCkpt);
        const best = match ||
                     ckptsData.checkpoints.find((c) => c.filename.includes('weed_detection_best.pt')) ||
                     ckptsData.checkpoints.find((c) => c.filename.includes('yolov8s_rmcbam_tuned.pt')) ||
                     ckptsData.checkpoints.find((c) => !c.is_smoke && c.filename.endsWith('best.pt')) ||
                     ckptsData.checkpoints.find((c) => c.filename.includes('yolov8s_rmcbam_best.pt')) ||
                     ckptsData.checkpoints[0];
        setSelectedCheckpoint(best.path);
        localStorage.setItem('wv_selected_checkpoint', best.path);

        // Auto-load default sample only if no session data and user hasn't explicitly clicked Delete
        const wasCleared = sessionStorage.getItem('wv_cleared') === 'true';
        const hasSession = sessionStorage.getItem('wv_predict_data');
        if (!hasSession && !wasCleared && !predictData) {
          try {
            const initialSample = localStorage.getItem('wv_last_sample') || 'img112.jpg';
            const sampleRes = await runPredictSample(initialSample, best.path, 0.15, 0.45, 640);
            updatePredictData(sampleRes);
          } catch (e) {
            console.warn('Initial sample notice:', e);
          }
        }
      }
    } catch (err) {
      console.error('Error connecting to WeedVision API:', err);
    }
  };

  const handleClearImage = () => {
    sessionStorage.setItem('wv_cleared', 'true');
    updatePredictData(null, null);
  };

  const handleSelectHistoryItem = (item: PredictResponse) => {
    updatePredictData(item);
    handleSelectWorkspace('detection');
  };

  const handleClearHistory = () => {
    setSessionHistory([]);
  };

  const activeCheckpointInfo = checkpoints.find((c) => c.path === selectedCheckpoint);
  const isRmcbam = activeCheckpointInfo ? activeCheckpointInfo.is_rmcbam : false;

  return (
    <div className="fo-app">
      <TopBar
        health={health}
        activeCheckpoint={selectedCheckpoint}
        isRmcbam={isRmcbam}
      />
      <div className="fo-main-layout">
        <LeftSidebar
          activeWorkspace={activeWorkspace}
          onSelectWorkspace={handleSelectWorkspace}
          historyCount={sessionHistory.length}
        />

        <main style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
          <ErrorBoundary>
            {activeWorkspace === 'detection' && (
              <InspectionWorkspace
                checkpoints={checkpoints}
                selectedCheckpoint={selectedCheckpoint}
                onSelectCheckpoint={(path) => {
                  setSelectedCheckpoint(path);
                  localStorage.setItem('wv_selected_checkpoint', path);
                }}
                predictData={predictData}
                onPredictSuccess={(res, file) => {
                  updatePredictData(res, file);
                }}
                selectedFile={selectedFile}
                onFileChange={setSelectedFile}
                onClearImage={handleClearImage}
              />
            )}

            {activeWorkspace === 'analysis' && (
              <AnalysisWorkspace
                predictData={predictData}
                sessionResults={sessionHistory}
                onLoadSample={async (name?: string) => {
                  try {
                    const sampleName = name || 'img112.jpg';
                    const sampleRes = await runPredictSample(sampleName, selectedCheckpoint, 0.15, 0.45, 640);
                    updatePredictData(sampleRes);
                  } catch (e) {
                    console.error('Error running sample in Weed Analysis:', e);
                  }
                }}
                onNavigate={(ws) => handleSelectWorkspace(ws)}
                onSelectHistory={handleSelectHistoryItem}
              />
            )}

            {activeWorkspace === 'localization' && (
              <LocalizationWorkspace
                predictData={predictData}
                sessionResults={sessionHistory}
                onLoadSample={async (name?: string) => {
                  try {
                    const sampleName = name || 'img112.jpg';
                    const sampleRes = await runPredictSample(sampleName, selectedCheckpoint, 0.15, 0.45, 640);
                    updatePredictData(sampleRes);
                  } catch (e) {
                    console.error('Error running sample in Weed Localization:', e);
                  }
                }}
                onNavigate={(ws) => handleSelectWorkspace(ws)}
                onSelectHistory={handleSelectHistoryItem}
              />
            )}

            {activeWorkspace === 'history' && (
              <HistoryWorkspace
                sessionResults={sessionHistory}
                onSelectHistory={handleSelectHistoryItem}
                onClearHistory={handleClearHistory}
                onNavigateToDetection={() => handleSelectWorkspace('detection')}
              />
            )}

            {activeWorkspace === 'explainability' && (
              <ExplainabilityWorkspace
                selectedFile={selectedFile}
                selectedCheckpoint={selectedCheckpoint}
                predictData={predictData}
                onLoadSample={async (name?: string) => {
                  try {
                    const sampleName = name || 'img112.jpg';
                    const sampleRes = await runPredictSample(sampleName, selectedCheckpoint, 0.15, 0.45, 640);
                    updatePredictData(sampleRes);
                  } catch (e) {
                    console.error('Error running sample in Explainability:', e);
                  }
                }}
                onNavigate={(ws) => handleSelectWorkspace(ws)}
              />
            )}

            {activeWorkspace === 'dataset' && (
              <DatasetWorkspace />
            )}

            {activeWorkspace === 'comparison' && (
              <ComparisonWorkspace checkpoints={checkpoints} />
            )}

            {activeWorkspace === 'settings' && (
              <SettingsWorkspace />
            )}
          </ErrorBoundary>
        </main>
      </div>
    </div>
  );
};

export default App;
