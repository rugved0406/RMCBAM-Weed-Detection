import axios from 'axios';
import type {
  SystemHealth,
  CheckpointInfo,
  PredictResponse,
  ExplainResponse,
  SystemSettings,
  OverviewData,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 180000,
});

export const getHealth = async (): Promise<SystemHealth> => {
  const res = await api.get('/health');
  return res.data;
};

export const getOverview = async (): Promise<OverviewData> => {
  const res = await api.get('/overview');
  return res.data;
};

export const getCheckpoints = async (): Promise<{ checkpoints: CheckpointInfo[] }> => {
  const res = await api.get('/checkpoints');
  return res.data;
};

export const runPredict = async (
  file: File,
  checkpoint?: string,
  conf?: number,
  iou?: number,
  imgsz?: number,
  device?: string,
  augment?: boolean,
  agnostic_nms?: boolean,
  ensemble?: boolean
): Promise<PredictResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  if (checkpoint) formData.append('checkpoint', checkpoint);
  if (conf !== undefined) formData.append('conf', conf.toString());
  if (iou !== undefined) formData.append('iou', iou.toString());
  if (imgsz !== undefined) formData.append('imgsz', imgsz.toString());
  if (device) formData.append('device', device);
  if (augment !== undefined) formData.append('augment', augment ? 'true' : 'false');
  if (agnostic_nms !== undefined) formData.append('agnostic_nms', agnostic_nms ? 'true' : 'false');
  if (ensemble !== undefined) formData.append('ensemble', ensemble ? 'true' : 'false');

  const res = await api.post('/predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res.data;
};

export const getSampleImages = async (): Promise<{ samples: string[] }> => {
  const res = await api.get('/samples');
  return res.data;
};

export const runPredictSample = async (
  filename: string,
  checkpoint?: string,
  conf?: number,
  iou?: number,
  imgsz?: number,
  device?: string,
  augment?: boolean,
  agnostic_nms?: boolean,
  ensemble?: boolean
): Promise<PredictResponse> => {
  const formData = new FormData();
  formData.append('filename', filename);
  if (checkpoint) formData.append('checkpoint', checkpoint);
  if (conf !== undefined) formData.append('conf', conf.toString());
  if (iou !== undefined) formData.append('iou', iou.toString());
  if (imgsz !== undefined) formData.append('imgsz', imgsz.toString());
  if (device) formData.append('device', device);
  if (augment !== undefined) formData.append('augment', augment ? 'true' : 'false');
  if (agnostic_nms !== undefined) formData.append('agnostic_nms', agnostic_nms ? 'true' : 'false');
  if (ensemble !== undefined) formData.append('ensemble', ensemble ? 'true' : 'false');

  const res = await api.post('/predict-sample', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res.data;
};

export const runExplain = async (
  file: File,
  checkpoint?: string,
  method: 'hirescam' | 'gradcam' = 'hirescam',
  layers?: string,
  alpha?: number,
  iou_match?: number,
  selected_uids?: string,
  conf?: number,
  iou?: number,
  imgsz?: number,
  device?: string
): Promise<ExplainResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  if (checkpoint) formData.append('checkpoint', checkpoint);
  formData.append('method', method);
  if (layers) formData.append('layers', layers);
  if (alpha !== undefined) formData.append('alpha', alpha.toString());
  if (iou_match !== undefined) formData.append('iou_match', iou_match.toString());
  if (selected_uids) formData.append('selected_uids', selected_uids);
  if (conf !== undefined) formData.append('conf', conf.toString());
  if (iou !== undefined) formData.append('iou', iou.toString());
  if (imgsz !== undefined) formData.append('imgsz', imgsz.toString());
  if (device) formData.append('device', device);

  const res = await api.post('/explain', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res.data;
};

export const getComparison = async () => {
  const res = await api.get('/comparison');
  return res.data;
};

export const runLiveValidation = async (
  checkpoint?: string,
  checkpointBaseline?: string,
  checkpointRmcbam?: string,
  dataYaml = 'data/yolo_weed_partial/data.yaml',
  split = 'val',
  imgsz = 640
) => {
  const formData = new FormData();
  if (checkpoint) formData.append('checkpoint', checkpoint);
  if (checkpointBaseline) formData.append('checkpoint_baseline', checkpointBaseline);
  if (checkpointRmcbam) formData.append('checkpoint_rmcbam', checkpointRmcbam);
  formData.append('data_yaml', dataYaml);
  formData.append('split', split);
  formData.append('imgsz', imgsz.toString());
  const res = await api.post('/comparison/validate', formData);
  return res.data;
};

export const benchmarkCheckpoints = async (checkpoints: string[], runs = 10) => {
  const formData = new FormData();
  checkpoints.forEach((c) => formData.append('checkpoints', c));
  formData.append('runs', runs.toString());
  const res = await api.post('/comparison/benchmark', formData);
  return res.data;
};

export const getDatasetInfo = async () => {
  const res = await api.get('/dataset');
  return res.data;
};

export const computeQuality = async (limit = 100) => {
  const formData = new FormData();
  formData.append('limit', limit.toString());
  const res = await api.post('/dataset/quality', formData);
  return res.data;
};

export const convertDataset = async (params: {
  src_folder: string;
  out_folder: string;
  json_folder?: string;
  val_fraction?: number;
  include_unlabelled?: boolean;
}) => {
  const formData = new FormData();
  formData.append('src_folder', params.src_folder);
  formData.append('out_folder', params.out_folder);
  if (params.json_folder) formData.append('json_folder', params.json_folder);
  if (params.val_fraction !== undefined) formData.append('val_fraction', params.val_fraction.toString());
  if (params.include_unlabelled !== undefined) formData.append('include_unlabelled', params.include_unlabelled ? 'true' : 'false');
  const res = await api.post('/dataset/convert', formData);
  return res.data;
};

export const getModelInfo = async (checkpoint?: string) => {
  const res = await api.get('/model-info', { params: { checkpoint } });
  return res.data;
};

export const getSettings = async (): Promise<SystemSettings> => {
  const res = await api.get('/settings');
  return res.data;
};

export const updateSettings = async (settings: Partial<SystemSettings>): Promise<{ status: string; settings: SystemSettings }> => {
  const res = await api.post('/settings', settings);
  return res.data;
};

export const clearBackendCaches = async (): Promise<{ status: string; message: string }> => {
  const res = await api.post('/settings/clear-cache');
  return res.data;
};

export const getAccuracyRecommendations = async () => {
  const res = await api.get('/accuracy/recommendations');
  return res.data;
};

export const applyAccuracyRecommendations = async () => {
  const res = await api.post('/accuracy/apply-recommended');
  return res.data;
};
