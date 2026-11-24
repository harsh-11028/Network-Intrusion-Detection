import { createContext, useContext, useState, ReactNode } from 'react'
import { PredictionResult, ModelInfo } from './api'

interface AnalysisState {
  result: PredictionResult | null
  file: File | null
  modelKey: string
  threshold: number
  models: ModelInfo[]
  setResult: (r: PredictionResult | null) => void
  setFile: (f: File | null) => void
  setModelKey: (k: string) => void
  setThreshold: (t: number) => void
  setModels: (m: ModelInfo[]) => void
}

const AnalysisContext = createContext<AnalysisState | null>(null)

export function AnalysisProvider({ children }: { children: ReactNode }) {
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [modelKey, setModelKey] = useState('xgboost')
  const [threshold, setThreshold] = useState(0.49)
  const [models, setModels] = useState<ModelInfo[]>([])

  return (
    <AnalysisContext.Provider value={{
      result, setResult, file, setFile,
      modelKey, setModelKey, threshold, setThreshold,
      models, setModels
    }}>
      {children}
    </AnalysisContext.Provider>
  )
}

export function useAnalysis() {
  const ctx = useContext(AnalysisContext)
  if (!ctx) throw new Error('useAnalysis must be inside AnalysisProvider')
  return ctx
}
