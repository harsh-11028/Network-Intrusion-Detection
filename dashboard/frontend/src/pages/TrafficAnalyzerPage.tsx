import { useState, useEffect, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Upload, AlertCircle, CheckCircle2, Download, Info,
  ChevronLeft, ChevronRight, Search, Sliders, Shield, ArrowRight
} from 'lucide-react'
import { getModels, predict, exportPredictions, ModelInfo, PredictionRow } from '../services/api'
import { useAnalysis } from '../services/AnalysisContext'

const MODEL_COLORS: Record<string, string> = {
  logistic_regression: '#6366f1',
  decision_tree: '#f59e0b',
  random_forest: '#10b981',
  xgboost: '#06b6d4',
}

const DEFAULT_THRESHOLDS: Record<string, number> = {
  logistic_regression: 0.13,
  decision_tree: 0.50,
  random_forest: 0.50,
  xgboost: 0.49,
}

const PAGE_SIZE = 50

function DropZone({ onFile, activeFile }: { onFile: (f: File) => void; activeFile: File | null }) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f?.name.endsWith('.csv')) onFile(f)
  }

  return (
    <div
      onDragOver={e => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${
        dragging
          ? 'border-cyan-400 bg-cyan-500/10'
          : activeFile
          ? 'border-emerald-500/40 bg-emerald-500/5 hover:border-emerald-500/60'
          : 'border-bg-border hover:border-cyan-500/40 hover:bg-bg-card/60'
      }`}
    >
      <div className="flex flex-col items-center justify-center gap-2">
        <div className={`w-10 h-10 rounded-full flex items-center justify-center ${activeFile ? 'bg-emerald-500/15 text-emerald-400' : 'bg-cyan-500/10 text-cyan-400'}`}>
          {activeFile ? <CheckCircle2 size={22} /> : <Upload size={20} />}
        </div>
        {activeFile ? (
          <div>
            <div className="text-sm font-semibold text-text-primary">{activeFile.name}</div>
            <div className="text-xs text-text-muted mt-0.5">{(activeFile.size / 1024).toFixed(1)} KB · Click or drag to replace</div>
          </div>
        ) : (
          <div>
            <p className="text-sm font-semibold text-text-primary">Click to select or drag & drop CSV file</p>
            <p className="text-xs text-text-muted mt-0.5">Formatted with 42 UNSW-NB15 raw feature attributes</p>
          </div>
        )}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept=".csv"
        className="hidden"
        onChange={e => {
          const f = e.target.files?.[0]
          if (f) onFile(f)
        }}
      />
    </div>
  )
}

type FilterType = 'all' | 'attack' | 'normal'

export default function TrafficAnalyzerPage() {
  const navigate = useNavigate()
  const {
    result, setResult, file, setFile,
    modelKey, setModelKey, threshold, setThreshold,
    models, setModels
  } = useAnalysis()

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [page, setPage] = useState(0)
  const [filter, setFilter] = useState<FilterType>('all')
  const [search, setSearch] = useState('')
  const [exporting, setExporting] = useState(false)

  useEffect(() => {
    if (models.length === 0) {
      getModels().then(r => setModels(r.data.models)).catch(() => {})
    }
  }, [models.length, setModels])

  const selectedModel = models.find(m => m.key === modelKey)
  const presets = selectedModel?.threshold_presets ?? []

  const handleFile = useCallback((f: File) => {
    setFile(f)
    setError(null)
    setResult(null)
  }, [setFile, setResult])

  const handleAnalyze = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setPage(0)
    try {
      const r = await predict(file, modelKey, threshold)
      setResult(r)
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
        ?? 'Inference error. Verify backend status and CSV schema.'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  const handleExport = async () => {
    if (!file) return
    setExporting(true)
    try {
      const blob = await exportPredictions(file, modelKey, threshold)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `predictions_${modelKey}_th${threshold.toFixed(2)}.csv`
      a.click()
      URL.revokeObjectURL(url)
    } catch {
      setError('Failed to export prediction CSV.')
    } finally {
      setExporting(false)
    }
  }

  // Filter and search
  const filteredRows: PredictionRow[] = (result?.rows ?? []).filter(row => {
    if (filter === 'attack' && row.prediction !== 1) return false
    if (filter === 'normal' && row.prediction !== 0) return false
    if (search.trim()) {
      const s = search.toLowerCase()
      if (!String(row.row_index).includes(s) && !(row.attack_cat ?? '').toLowerCase().includes(s)) {
        return false
      }
    }
    return true
  })

  const totalPages = Math.ceil(filteredRows.length / PAGE_SIZE)
  const pageRows = filteredRows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)

  return (
    <div className="space-y-6 max-w-6xl">
      {/* Step 1: Upload Panel */}
      <div className="card">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <span className="w-5 h-5 rounded-full bg-cyan-500/20 text-cyan-400 flex items-center justify-center text-xs font-bold font-mono">1</span>
            <h2 className="text-sm font-semibold text-text-primary uppercase tracking-wide">
              Select or Drop Offline Network Traffic CSV
            </h2>
          </div>
          <span className="text-[11px] text-text-muted">Requires 42 UNSW-NB15 features</span>
        </div>

        <DropZone onFile={handleFile} activeFile={file} />

        <div className="mt-2.5 flex items-center justify-between text-xs text-text-muted flex-wrap gap-2">
          <div className="flex items-center gap-1.5">
            <Info size={13} className="text-cyan-400" />
            <span>Pre-packaged test samples available in:</span>
            <code className="text-cyan-400 bg-bg-secondary px-1.5 py-0.5 rounded border border-bg-border">sample_traffic.csv</code>
            <span>or</span>
            <code className="text-cyan-400 bg-bg-secondary px-1.5 py-0.5 rounded border border-bg-border">sample_traffic_labeled.csv</code>
          </div>
          {file && (
            <button
              onClick={() => { setFile(null); setResult(null); setError(null) }}
              className="text-text-secondary hover:text-red-400 text-xs transition-colors"
            >
              Clear file
            </button>
          )}
        </div>
      </div>

      {/* Step 2: Model Selection Grid */}
      {models.length > 0 && (
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <span className="w-5 h-5 rounded-full bg-cyan-500/20 text-cyan-400 flex items-center justify-center text-xs font-bold font-mono">2</span>
              <h2 className="text-sm font-semibold text-text-primary uppercase tracking-wide">
                Select Detection Model
              </h2>
            </div>
            <span className="text-[11px] text-text-muted">Phase 7.1 Retrained Weights</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {models.map(m => {
              const isSelected = modelKey === m.key
              const color = MODEL_COLORS[m.key] || '#94a3b8'
              return (
                <div
                  key={m.key}
                  onClick={() => {
                    setModelKey(m.key)
                    setThreshold(DEFAULT_THRESHOLDS[m.key] ?? 0.5)
                  }}
                  className={`p-3.5 rounded-xl border transition-all cursor-pointer flex flex-col justify-between ${
                    isSelected
                      ? 'bg-bg-secondary shadow-md'
                      : 'bg-bg-card border-bg-border hover:border-cyan-500/40 hover:bg-bg-secondary/40'
                  }`}
                  style={{
                    borderColor: isSelected ? color : undefined,
                    boxShadow: isSelected ? `0 0 12px ${color}20` : undefined
                  }}
                >
                  <div>
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="font-bold text-xs text-text-primary flex items-center gap-1.5">
                        <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
                        {m.name}
                      </span>
                      {m.recommended && (
                        <span className="text-[10px] px-1.5 py-0.2 rounded bg-cyan-500/15 text-cyan-300 font-semibold border border-cyan-500/30">
                          Recommended
                        </span>
                      )}
                    </div>
                    <p className="text-[11px] text-text-secondary leading-relaxed line-clamp-3">
                      {m.description}
                    </p>
                  </div>
                  {m.threshold_note && (
                    <div className="mt-2 pt-2 border-t border-bg-border/60 text-[10px] text-amber-400/90 flex items-start gap-1">
                      <Info size={11} className="flex-shrink-0 mt-0.5" />
                      <span>{m.threshold_note}</span>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Step 3: Decision Threshold Configuration */}
      <div className="card">
        <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
          <div className="flex items-center gap-2">
            <span className="w-5 h-5 rounded-full bg-cyan-500/20 text-cyan-400 flex items-center justify-center text-xs font-bold font-mono">3</span>
            <h2 className="text-sm font-semibold text-text-primary uppercase tracking-wide">
              Classification Operating Point
            </h2>
          </div>
          <div className="flex items-center gap-2 font-mono text-xs">
            <span className="text-text-muted">Threshold:</span>
            <span className="px-2 py-0.5 rounded bg-cyan-500/15 border border-cyan-500/30 text-cyan-300 font-bold text-sm">
              {threshold.toFixed(2)}
            </span>
          </div>
        </div>

        {/* Slider */}
        <div className="flex items-center gap-4 py-2">
          <span className="text-xs font-mono text-text-muted">0.01</span>
          <input
            type="range"
            min={0.01}
            max={0.99}
            step={0.01}
            value={threshold}
            onChange={e => setThreshold(parseFloat(e.target.value))}
            className="flex-1 accent-cyan-400 cursor-pointer h-1.5 bg-bg-secondary rounded-lg"
          />
          <span className="text-xs font-mono text-text-muted">0.99</span>
        </div>

        {/* Validation Presets from Phase 7.1 */}
        {presets.length > 0 && (
          <div className="mt-3 pt-3 border-t border-bg-border/70">
            <div className="text-[11px] text-text-muted font-medium mb-2 uppercase tracking-wide">
              Phase 7.1 Validation Presets (Leakage-Free Internal Splits):
            </div>
            <div className="flex flex-wrap gap-2">
              {presets.map(p => {
                const isActive = !p.infeasible && p.threshold !== null && Math.abs(p.threshold - threshold) < 0.005
                return (
                  <button
                    key={p.strategy}
                    disabled={p.infeasible || p.threshold === null}
                    onClick={() => {
                      if (!p.infeasible && p.threshold !== null) {
                        setThreshold(p.threshold)
                      }
                    }}
                    title={p.feasibility_note}
                    className={`text-xs px-2.5 py-1.5 rounded-lg border transition-all text-left ${
                      p.infeasible || p.threshold === null
                        ? 'border-bg-border/40 text-text-muted/40 cursor-not-allowed bg-bg-secondary/30'
                        : isActive
                        ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-300 font-semibold shadow-sm'
                        : 'bg-bg-card border-bg-border text-text-secondary hover:border-cyan-500/40 hover:text-text-primary'
                    }`}
                  >
                    <div className="flex items-center gap-1.5">
                      <span>{p.strategy}</span>
                      {!p.infeasible && p.threshold !== null && (
                        <span className="font-mono text-cyan-400">({p.threshold.toFixed(2)})</span>
                      )}
                      {p.infeasible && <span className="text-red-400 text-[10px]">[Infeasible]</span>}
                    </div>
                  </button>
                )
              })}
            </div>
          </div>
        )}
      </div>

      {/* Action / Analyze Bar */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-3">
          <button
            onClick={handleAnalyze}
            disabled={!file || loading}
            className="btn-primary flex items-center gap-2 px-6 py-2.5 text-sm font-semibold shadow-lg shadow-cyan-500/20 disabled:opacity-50"
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-black/40 border-t-black rounded-full animate-spin" />
                Processing Inference Pipeline…
              </>
            ) : (
              <>
                <Shield size={16} />
                Run Intrusion Detection Inference
              </>
            )}
          </button>

          {result && (
            <button
              onClick={handleExport}
              disabled={exporting}
              className="btn-secondary flex items-center gap-2 text-xs px-3.5 py-2.5"
            >
              <Download size={14} />
              {exporting ? 'Generating CSV…' : 'Export Predictions CSV'}
            </button>
          )}
        </div>

        {result && (
          <button
            onClick={() => navigate('/')}
            className="text-xs text-cyan-400 hover:text-cyan-300 font-medium inline-flex items-center gap-1"
          >
            Switch to Visual Dashboard Overview <ArrowRight size={13} />
          </button>
        )}
      </div>

      {/* Error alert */}
      {error && (
        <div className="card border-red-500/30 bg-red-500/10 text-red-300 p-4 flex items-start gap-3">
          <AlertCircle size={18} className="text-red-400 flex-shrink-0 mt-0.5" />
          <div className="text-xs leading-relaxed">
            <span className="font-semibold block text-red-200 mb-0.5">Analysis Failed</span>
            {error}
          </div>
        </div>
      )}

      {/* Results Section */}
      {result && (
        <div className="space-y-4">
          {/* Quick Metrics Bar */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="p-3 rounded-xl bg-bg-card border border-bg-border text-center">
              <span className="text-[10px] text-text-muted uppercase font-semibold">Total Ingested</span>
              <div className="text-lg font-bold font-mono text-text-primary">{result.n_rows.toLocaleString()}</div>
            </div>
            <div className="p-3 rounded-xl bg-bg-card border border-bg-border text-center">
              <span className="text-[10px] text-text-muted uppercase font-semibold">Flagged Attacks</span>
              <div className="text-lg font-bold font-mono text-red-400">{result.n_attacks.toLocaleString()} ({result.attack_pct}%)</div>
            </div>
            <div className="p-3 rounded-xl bg-bg-card border border-bg-border text-center">
              <span className="text-[10px] text-text-muted uppercase font-semibold">Normal Classified</span>
              <div className="text-lg font-bold font-mono text-emerald-400">{result.n_normal.toLocaleString()}</div>
            </div>
            <div className="p-3 rounded-xl bg-bg-card border border-bg-border text-center">
              <span className="text-[10px] text-text-muted uppercase font-semibold">Ground Truth Verified</span>
              <div className="text-lg font-bold font-mono text-cyan-400">{result.has_ground_truth ? 'Yes (Labeled)' : 'No (Unlabeled)'}</div>
            </div>
          </div>

          {/* Results Table & Filtering */}
          <div className="card">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
              <h3 className="text-xs font-semibold text-text-primary uppercase tracking-wider">
                Inference Results Inspection ({filteredRows.length.toLocaleString()} Matched)
              </h3>
              <div className="flex items-center gap-2">
                <div className="relative">
                  <Search size={13} className="absolute left-2.5 top-2.5 text-text-muted" />
                  <input
                    type="text"
                    placeholder="Search row or category…"
                    value={search}
                    onChange={e => { setSearch(e.target.value); setPage(0) }}
                    className="pl-8 pr-3 py-1.5 bg-bg-secondary border border-bg-border rounded-lg text-xs text-text-primary focus:outline-none focus:border-cyan-500/50 w-44 font-mono"
                  />
                </div>
                {(['all', 'attack', 'normal'] as FilterType[]).map(f => (
                  <button
                    key={f}
                    onClick={() => { setFilter(f); setPage(0) }}
                    className={`text-xs px-2.5 py-1.5 rounded-lg border font-medium capitalize transition-colors ${
                      filter === f
                        ? 'bg-cyan-500/20 border-cyan-500/40 text-cyan-300'
                        : 'border-bg-border text-text-secondary hover:border-cyan-500/30'
                    }`}
                  >
                    {f}
                  </button>
                ))}
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left border-collapse">
                <thead>
                  <tr className="border-b border-bg-border text-text-muted uppercase font-medium">
                    <th className="py-2.5 px-3">Flow ID</th>
                    <th className="py-2.5 px-3">Attack Probability Score</th>
                    <th className="py-2.5 px-3">Classification</th>
                    {result.has_ground_truth && <th className="py-2.5 px-3">True Label</th>}
                    {result.rows[0]?.attack_cat !== undefined && <th className="py-2.5 px-3">Category</th>}
                  </tr>
                </thead>
                <tbody className="divide-y divide-bg-border/40 font-mono">
                  {pageRows.map(row => (
                    <tr key={row.row_index} className="hover:bg-bg-secondary/40 transition-colors">
                      <td className="py-2 px-3 text-text-muted">#{row.row_index}</td>
                      <td className="py-2 px-3">
                        <div className="flex items-center gap-2">
                          <div className="w-24 h-1.5 rounded-full bg-bg-border overflow-hidden">
                            <div
                              className="h-full rounded-full transition-all"
                              style={{
                                width: `${row.probability * 100}%`,
                                backgroundColor: row.prediction === 1 ? '#ef4444' : '#10b981'
                              }}
                            />
                          </div>
                          <span className="text-xs" style={{ color: row.prediction === 1 ? '#ef4444' : '#10b981' }}>
                            {(row.probability * 100).toFixed(1)}%
                          </span>
                        </div>
                      </td>
                      <td className="py-2 px-3">
                        {row.prediction === 1 ? (
                          <span className="badge-attack">Attack</span>
                        ) : (
                          <span className="badge-normal">Normal</span>
                        )}
                      </td>
                      {result.has_ground_truth && (
                        <td className="py-2 px-3">
                          {row.true_label === 1 ? (
                            <span className="badge-attack">Attack</span>
                          ) : row.true_label === 0 ? (
                            <span className="badge-normal">Normal</span>
                          ) : (
                            <span className="text-text-muted">—</span>
                          )}
                        </td>
                      )}
                      {row.attack_cat !== undefined && (
                        <td className="py-2 px-3 text-text-secondary">{row.attack_cat}</td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination Controls */}
            {totalPages > 1 && (
              <div className="flex items-center justify-between pt-3 mt-3 border-t border-bg-border/60 text-xs text-text-muted">
                <span>
                  Showing {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, filteredRows.length)} of {filteredRows.length.toLocaleString()}
                </span>
                <div className="flex items-center gap-1.5">
                  <button
                    disabled={page === 0}
                    onClick={() => setPage(p => p - 1)}
                    className="p-1.5 rounded-md border border-bg-border hover:border-cyan-500/40 disabled:opacity-30 disabled:cursor-not-allowed"
                  >
                    <ChevronLeft size={14} />
                  </button>
                  <span className="px-2 font-mono">
                    Page {page + 1} of {totalPages}
                  </span>
                  <button
                    disabled={page >= totalPages - 1}
                    onClick={() => setPage(p => p + 1)}
                    className="p-1.5 rounded-md border border-bg-border hover:border-cyan-500/40 disabled:opacity-30 disabled:cursor-not-allowed"
                  >
                    <ChevronRight size={14} />
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
