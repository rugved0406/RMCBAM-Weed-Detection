export interface Detection {
  uid: string;
  cls_id?: number;
  cls_name: string;
  conf: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  cx: number;
  cy: number;
  w: number;
  h: number;
  area_px: number;
  area_pct: number;
  color?: string;
}

export interface CheckpointInfo {
  path: string;
  filename: string;
  label: string;
  ok: boolean;
  error?: string;
  size_mb: number;
  names: Record<number, string>;
  nc: number;
  is_rmcbam: boolean;
  rmcbam_layers: number[];
  n_params: number;
  gflops?: number;
  epoch?: number;
  best_fitness?: number;
  is_smoke: boolean;
  has_weed_class: boolean;
  compatibility_notes: [string, string][];
}

export interface InfestationMetrics {
  weed_count: number;
  image_px: number;
  density_per_mpx: number;
  sum_box_area_px: number;
  union_box_area_px: number;
  infestation_pct: number;
  mean_box_area_px: number;
  median_box_area_px: number;
  mean_conf?: number;
  min_conf?: number;
  max_conf?: number;
  severity: 'Low' | 'Moderate' | 'High' | 'Severe';
  density_per_m2?: number;
  ground_area_m2?: number;
}

export interface CropAssociationItem {
  crop_class: string;
  crop_instances: number;
  associated_weeds: number;
  weeds_per_crop_instance?: number;
  mean_distance_px?: number;
}

export interface PerWeedAssociationItem {
  weed_uid: string;
  cls_name: string;
  conf: number;
  associated_class: string;
  associated_uid: string;
  distance_px: number;
  distance_frac: number;
  within_radius: boolean;
}

export interface DetectionsTableRow {
  uid: string;
  cls_name: string;
  conf: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  cx: number;
  cy: number;
  w: number;
  h: number;
  area: number;
  area_pct: number;
}

export interface PredictResponse {
  filename: string;
  width: number;
  height: number;
  file_size?: number;
  checkpoint: string;
  params: {
    conf: number;
    iou: number;
    imgsz: number;
    device: string;
  };
  speed: {
    preprocess?: number;
    inference?: number;
    postprocess?: number;
  };
  detections: Detection[];
  detections_table?: DetectionsTableRow[];
  weed_count: number;
  crop_count: number;
  infestation: InfestationMetrics;
  crop_association: CropAssociationItem[];
  per_weed_association?: PerWeedAssociationItem[];
  original_image: string;
  annotated_image: string;
}

export interface RMCBAMAttentionMap {
  name: string;
  layer: number;
  feat_hw: [number, number];
  spatial_heatmap: string;
  spatial_overlay: string;
  spatial_min: number;
  spatial_max: number;
  gamma_mean_abs: number;
  channel_weights: number[];
}

export interface TargetCrop {
  uid: string;
  conf: number;
  energy_in_box: number;
  anchors_matched: number;
  crop_image: string;
}

export interface ExplainResponse {
  method: string;
  layers: number[];
  alpha?: number;
  iou_match?: number;
  combined_cam?: string;
  combined_overlay?: string;
  per_target_crops?: TargetCrop[];
  per_target_overlays?: Record<string, string>;
  energy_scores: Record<string, number>;
  rmcbam_maps: RMCBAMAttentionMap[];
}

export interface OverviewData {
  checkpoints: {
    total: number;
    with_weed: number;
    rmcbam: number;
    baseline: number;
  };
  dataset: {
    exists: boolean;
    path: string;
    n_images: number;
    n_labelme_json: number;
    yaml_files: string[];
  };
  compute: {
    device: string;
    torch_version: string;
    cuda_available: boolean;
    cuda_device_name: string;
  };
  readiness: {
    base_ok: boolean;
    rmcbam_ok: boolean;
    labelme_ok: boolean;
    yaml_ok: boolean;
    classes_declared: boolean;
    missing_classes: string[];
    has_project_model: boolean;
  };
  classes: string[];
  class_colors: Record<string, string>;
}

export interface SystemHealth {
  status: string;
  project: string;
  device: string;
  cuda_available: boolean;
  checkpoints_count: number;
  demo_mode: boolean;
  classes: string[];
}

export interface SystemSettings {
  models_dir: string;
  runs_dir: string;
  dataset_dir: string;
  device: string;
  imgsz: number;
  conf: number;
  iou: number;
  max_det: number;
  weed_class: string;
  crop_classes: string[];
  severity_low: number;
  severity_moderate: number;
  severity_high: number;
  proximity_frac: number;
  gsd_cm_per_px: number;
  gradcam_layers: number[];
  environment?: {
    python: string;
    torch: string;
    ultralytics: string;
    cuda: boolean;
    cuda_device_name: string;
    device_used: string;
  };
}
