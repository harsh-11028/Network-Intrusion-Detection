import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 60000,
})

export interface ModelInfo {
  key: string
  name: string
  short: string
  description: string
  color: string
  recommended?: boolean
  threshold_note?: string
  threshold_presets: ThresholdPreset[]
}

export interface ThresholdPreset {
  strategy: string
  threshold: number | null
  val_attack_recall: number | null
  val_attack_f1: number | null
  val_fpr: number | null
  feasibility_note: string
  infeasible: boolean
}

export interface TestMetricEntry {
  accuracy?: number
  attack_recall?: number
  attack_precision?: number
  attack_f1?: number
  macro_f1?: number
  roc_auc?: number
  pr_auc?: number
  tp?: number
  fp?: number
  fn?: number
  tn?: number
  threshold?: number
  strategy?: string
  [key: string]: unknown
}

export interface PredictionRow {
  row_index: number
  probability: number
  prediction: number
  label_text: string
  true_label?: number | null
  attack_cat?: string
}

export interface EvaluationMetrics {
  accuracy: number
  attack_recall: number
  attack_precision: number
  attack_f1: number
  macro_f1?: number
  roc_auc?: number
  pr_auc?: number
  tp: number; fp: number; fn: number; tn: number
  confusion_matrix: number[][]
  n_labeled: number
}

export interface PredictionResult {
  model_key: string
  threshold: number
  n_rows: number
  n_attacks: number
  n_normal: number
  attack_pct: number
  rows: PredictionRow[]
  has_ground_truth: boolean
  evaluation: EvaluationMetrics | null
}

export interface FeatureImportance {
  feature: string
  importance: number
}

// API calls
export const getHealth = () => api.get('/health')
export const getModels = () => api.get<{ models: ModelInfo[] }>('/models')
export const getOfficialMetrics = () => api.get<{ test_metrics: Record<string, Record<string, TestMetricEntry>> }>('/metrics/official')
export const getValidationMetrics = () => api.get('/metrics/validation')
export const getThresholdAnalysis = () => api.get<{ threshold_analysis: Record<string, unknown>[] }>('/metrics/thresholds')
export const getTestPredictions = () => api.get<{ test_predictions: Record<string, number[]> }>('/metrics/test-predictions')
export const getFeatureImportances = () => api.get<{ feature_importances: Record<string, FeatureImportance[]> }>('/insights/features')
export const getFeatureMetadata = () => api.get('/insights/feature-metadata')

export const predict = async (file: File, modelKey: string, threshold: number): Promise<PredictionResult> => {
  const form = new FormData()
  form.append('file', file)
  form.append('model_key', modelKey)
  form.append('threshold', String(threshold))
  const res = await api.post<PredictionResult>('/predict', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return res.data
}

export const exportPredictions = async (file: File, modelKey: string, threshold: number): Promise<Blob> => {
  const form = new FormData()
  form.append('file', file)
  form.append('model_key', modelKey)
  form.append('threshold', String(threshold))
  const res = await api.post('/predict/export', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    responseType: 'blob',
  })
  return res.data
}

export default api
