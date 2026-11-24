import { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, LineChart, Line
} from 'recharts'
import { getOfficialMetrics, getThresholdAnalysis } from '../services/api'
import { Info, CheckCircle2, ShieldCheck, AlertCircle } from 'lucide-react'

const MODEL_COLORS: Record<string, string> = {
  logistic_regression: '#6366f1',
  decision_tree: '#f59e0b',
  random_forest: '#10b981',
  xgboost: '#06b6d4',
}

const MODEL_NAMES: Record<string, string> = {
  logistic_regression: 'Logistic Regression',
  decision_tree: 'Decision Tree',
  random_forest: 'Random Forest',
  xgboost: 'XGBoost',
}

const STRATEGY_OPTIONS = [
  { key: 'default_0_5', label: 'Default (0.50)' },
  { key: 'strategy_a_max_attack_f1', label: 'Strategy A (Max Attack F1)' },
  { key: 'strategy_b_max_macro_f1', label: 'Strategy B (Max Macro F1)' },
  { key: 'strategy_c_security_oriented', label: 'Strategy C (Security ≥ 97%)' },
  { key: 'strategy_d_operational', label: 'Strategy D (Operational Target)' },
]

type MetricKey = 'accuracy' | 'attack_f1' | 'roc_auc' | 'pr_auc' | 'attack_recall' | 'attack_precision'
const METRIC_OPTIONS: { key: MetricKey; label: string }[] = [
  { key: 'accuracy', label: 'Accuracy' },
  { key: 'attack_recall', label: 'Attack Recall' },
  { key: 'attack_precision', label: 'Attack Precision' },
  { key: 'attack_f1', label: 'Attack F1' },
  { key: 'roc_auc', label: 'ROC-AUC' },
  { key: 'pr_auc', label: 'PR-AUC' },
]

const MODEL_KEYS = ['logistic_regression', 'decision_tree', 'random_forest', 'xgboost']

export default function ModelPerformancePage() {
  const [metrics, setMetrics] = useState<Record<string, Record<string, Record<string, unknown>>> | null>(null)
  const [thresholdData, setThresholdData] = useState<Record<string, unknown>[]>([])
  const [selectedStrategy, setSelectedStrategy] = useState('default_0_5')
  const [sweepMetric, setSweepMetric] = useState<'attack_f1' | 'attack_recall' | 'accuracy' | 'false_positive_rate'>('attack_f1')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([getOfficialMetrics(), getThresholdAnalysis()])
      .then(([m, t]) => {
        setMetrics(m.data.test_metrics as Record<string, Record<string, Record<string, unknown>>>)
        setThresholdData(t.data.threshold_analysis)
      })
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[50vh]">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-sm text-text-muted">Loading official Phase 7.1 benchmark metrics…</p>
        </div>
      </div>
    )
  }

  if (!metrics) {
    return (
      <div className="card border-red-500/30 bg-red-500/10 text-red-400 p-6">
        <p className="font-semibold">Failed to load official test metrics.</p>
        <p className="text-xs mt-1 text-red-300">Ensure the backend server is running and outputs/phase7_1/final_test/test_metrics.json exists.</p>
      </div>
    )
  }

  // 1. Multi-metric comparison bar data across models for the selected strategy
  const barData = METRIC_OPTIONS.map(({ key, label }) => {
    const entry: Record<string, unknown> = { metric: label }
    MODEL_KEYS.forEach(mk => {
      const modelStrategies = metrics[mk] || {}
      // Fallback to default_0_5 if strategy not present (e.g. Decision Tree Strategy D)
      const dataObj = (modelStrategies[selectedStrategy] || modelStrategies['default_0_5']) as Record<string, unknown> | undefined
      const val = dataObj?.[key] as number | undefined
      entry[MODEL_NAMES[mk]] = typeof val === 'number' ? Math.round(val * 10000) / 10000 : null
    })
    return entry
  })

  // 2. Threshold sweep line chart data from threshold_analysis.csv
  const sweepLines = (() => {
    if (!thresholdData || thresholdData.length === 0) return []
    const byModel: Record<string, { threshold: number; value: number }[]> = {}
    thresholdData.forEach(row => {
      const mk = (row['model_key'] || row['model']) as string
      const th = row['threshold'] as number
      const val = row[sweepMetric] as number
      if (mk && typeof th === 'number' && typeof val === 'number') {
        if (!byModel[mk]) byModel[mk] = []
        byModel[mk].push({ threshold: th, value: Math.round(val * 10000) / 10000 })
      }
    })

    const thresholds = [...new Set(thresholdData.map(r => r['threshold'] as number))].filter(t => typeof t === 'number').sort((a, b) => a - b)
    return thresholds.map(th => {
      const pt: Record<string, unknown> = { threshold: th }
      MODEL_KEYS.forEach(mk => {
        const found = byModel[mk]?.find(v => Math.abs(v.threshold - th) < 0.0001)
        if (found) pt[MODEL_NAMES[mk]] = found.value
      })
      return pt
    })
  })()

  // 3. Table data for all 4 models under the selected strategy
  const tableData: Array<{ model_key: string; name: string; isFallback: boolean; [key: string]: unknown }> = MODEL_KEYS.map(mk => {
    const modelStrategies = metrics[mk] || {}
    const hasStrategy = Boolean(modelStrategies[selectedStrategy])
    const dataObj = (hasStrategy ? modelStrategies[selectedStrategy] : modelStrategies['default_0_5']) as Record<string, unknown> | undefined
    return {
      model_key: mk,
      name: MODEL_NAMES[mk],
      isFallback: !hasStrategy,
      ...(dataObj || {})
    }
  })

  const currentStratLabel = STRATEGY_OPTIONS.find(s => s.key === selectedStrategy)?.label || selectedStrategy

  return (
    <div className="space-y-6 max-w-6xl">
      {/* Strategy selector banner */}
      <div className="card border-bg-border bg-bg-secondary/60">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <ShieldCheck className="text-cyan-400" size={18} />
              <h2 className="text-sm font-semibold text-text-primary uppercase tracking-wide">
                Official Benchmark Evaluation (Untouched Test Set: 55,945 Samples)
              </h2>
            </div>
            <p className="text-xs text-text-muted mt-1">
              Evaluated strictly once in Phase 7.1. Decision thresholds were optimized on leakage-free internal validation and verified here.
            </p>
          </div>
          <div className="flex items-center gap-1.5 flex-wrap">
            {STRATEGY_OPTIONS.map(s => {
              const active = selectedStrategy === s.key
              return (
                <button
                  key={s.key}
                  onClick={() => setSelectedStrategy(s.key)}
                  className={`text-xs px-3 py-1.5 rounded-lg font-medium transition-all ${
                    active
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                      : 'bg-bg-card text-text-secondary border border-bg-border hover:border-cyan-500/30 hover:text-text-primary'
                  }`}
                >
                  {s.label}
                </button>
              )
            })}
          </div>
        </div>
      </div>

      {/* Official metrics comparison table */}
      <div className="card">
        <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
          <div>
            <h3 className="text-sm font-semibold text-text-primary">
              Test Performance Summary — {currentStratLabel}
            </h3>
            <p className="text-xs text-text-muted">Standardized test metrics evaluated on 55,945 flows (34,206 Normal, 21,739 Attack).</p>
          </div>
          <span className="text-[11px] px-2.5 py-1 rounded-md bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 font-mono">
            Ground Truth: Official UNSW-NB15 Test
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-bg-border text-text-muted font-medium uppercase tracking-wider">
                <th className="py-2.5 px-3">Model</th>
                <th className="py-2.5 px-3">Operating Threshold</th>
                <th className="py-2.5 px-3">Accuracy</th>
                <th className="py-2.5 px-3">Attack Recall</th>
                <th className="py-2.5 px-3">Attack Precision</th>
                <th className="py-2.5 px-3">Attack F1</th>
                <th className="py-2.5 px-3">ROC-AUC</th>
                <th className="py-2.5 px-3">PR-AUC</th>
                <th className="py-2.5 px-3">FPR</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-bg-border/40">
              {tableData.map(row => {
                const color = MODEL_COLORS[row.model_key] || '#94a3b8'
                const th = typeof row.threshold === 'number' ? row.threshold.toFixed(2) : '—'
                const acc = typeof row.accuracy === 'number' ? (row.accuracy * 100).toFixed(2) + '%' : '—'
                const rec = typeof row.attack_recall === 'number' ? (row.attack_recall * 100).toFixed(2) + '%' : '—'
                const prec = typeof row.attack_precision === 'number' ? (row.attack_precision * 100).toFixed(2) + '%' : '—'
                const f1 = typeof row.attack_f1 === 'number' ? (row.attack_f1 * 100).toFixed(2) + '%' : '—'
                const rocAuc = typeof row.roc_auc === 'number' ? row.roc_auc.toFixed(4) : '—'
                const prAuc = typeof row.pr_auc === 'number' ? row.pr_auc.toFixed(4) : '—'
                const fpr = typeof row.false_positive_rate === 'number' ? (row.false_positive_rate * 100).toFixed(2) + '%' : '—'

                return (
                  <tr key={row.model_key} className="hover:bg-bg-secondary/50 transition-colors">
                    <td className="py-3 px-3 font-semibold flex items-center gap-2" style={{ color }}>
                      <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
                      {row.name}
                      {row.isFallback && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20 ml-1" title="Infeasible strategy; displaying Default (0.50)">
                          Infeasible (Default)
                        </span>
                      )}
                    </td>
                    <td className="py-3 px-3 font-mono text-text-primary">{th}</td>
                    <td className="py-3 px-3 font-mono text-text-primary font-medium">{acc}</td>
                    <td className="py-3 px-3 font-mono text-emerald-400 font-medium">{rec}</td>
                    <td className="py-3 px-3 font-mono text-cyan-400">{prec}</td>
                    <td className="py-3 px-3 font-mono text-text-primary font-semibold">{f1}</td>
                    <td className="py-3 px-3 font-mono text-text-secondary">{rocAuc}</td>
                    <td className="py-3 px-3 font-mono text-text-secondary">{prAuc}</td>
                    <td className="py-3 px-3 font-mono text-amber-400">{fpr}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Confusion Matrix breakdown per model */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {tableData.map(row => {
          const color = MODEL_COLORS[row.model_key] || '#94a3b8'
          const tp = typeof row.true_positives === 'number' ? row.true_positives.toLocaleString() : '—'
          const fp = typeof row.false_positives === 'number' ? row.false_positives.toLocaleString() : '—'
          const fn = typeof row.false_negatives === 'number' ? row.false_negatives.toLocaleString() : '—'
          const tn = typeof row.true_negatives === 'number' ? row.true_negatives.toLocaleString() : '—'

          return (
            <div key={row.model_key} className="card p-4 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-semibold uppercase tracking-wider" style={{ color }}>{row.name}</span>
                  <span className="text-[10px] text-text-muted font-mono">Th: {typeof row.threshold === 'number' ? row.threshold.toFixed(2) : '0.50'}</span>
                </div>
                <div className="grid grid-cols-2 gap-1.5 text-center text-[11px] my-2">
                  <div className="p-2 rounded bg-emerald-500/10 border border-emerald-500/20">
                    <div className="text-[10px] text-emerald-400/80 font-medium">TP (Detected)</div>
                    <div className="font-mono font-bold text-emerald-300">{tp}</div>
                  </div>
                  <div className="p-2 rounded bg-amber-500/10 border border-amber-500/20">
                    <div className="text-[10px] text-amber-400/80 font-medium">FP (False Alarm)</div>
                    <div className="font-mono font-bold text-amber-300">{fp}</div>
                  </div>
                  <div className="p-2 rounded bg-red-500/10 border border-red-500/20">
                    <div className="text-[10px] text-red-400/80 font-medium">FN (Missed)</div>
                    <div className="font-mono font-bold text-red-300">{fn}</div>
                  </div>
                  <div className="p-2 rounded bg-blue-500/10 border border-blue-500/20">
                    <div className="text-[10px] text-blue-400/80 font-medium">TN (Normal Ok)</div>
                    <div className="font-mono font-bold text-blue-300">{tn}</div>
                  </div>
                </div>
              </div>
              <div className="text-[10px] text-text-muted text-center pt-1 border-t border-bg-border/60">
                Total Test Flows: 55,945
              </div>
            </div>
          )
        })}
      </div>

      {/* Multi-metric bar comparison chart */}
      <div className="card">
        <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
          <div>
            <h3 className="text-sm font-semibold text-text-primary">Comparative Test Performance by Metric</h3>
            <p className="text-xs text-text-muted">Direct visual comparison of all 4 models under {currentStratLabel}.</p>
          </div>
          <div className="flex items-center gap-3 text-xs">
            {MODEL_KEYS.map(mk => (
              <span key={mk} className="flex items-center gap-1.5 font-medium" style={{ color: MODEL_COLORS[mk] }}>
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: MODEL_COLORS[mk] }} />
                {MODEL_NAMES[mk]}
              </span>
            ))}
          </div>
        </div>

        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={barData} barCategoryGap="20%">
            <CartesianGrid strokeDasharray="3 3" stroke="#1e2d3d" vertical={false} />
            <XAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 11 }} />
            <YAxis domain={[0, 1]} tick={{ fill: '#64748b', fontSize: 11 }} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
            <Tooltip
              contentStyle={{ background: '#111827', border: '1px solid #1e2d3d', borderRadius: 8 }}
              itemStyle={{ color: '#e2e8f0', fontSize: 12 }}
              formatter={(v: number) => typeof v === 'number' ? `${(v * 100).toFixed(2)}%` : '—'}
            />
            {MODEL_KEYS.map(mk => (
              <Bar key={mk} dataKey={MODEL_NAMES[mk]} fill={MODEL_COLORS[mk]} radius={[4, 4, 0, 0]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Validation threshold sweep curve */}
      {sweepLines.length > 0 && (
        <div className="card">
          <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
            <div>
              <h3 className="text-sm font-semibold text-text-primary">
                Leakage-Free Validation Threshold Sweep (21,548 Samples)
              </h3>
              <p className="text-xs text-text-muted">
                Observed trade-offs across decision thresholds [0.05 – 0.95] on the internal validation set.
              </p>
            </div>
            <div className="flex items-center gap-1.5 bg-bg-secondary p-1 rounded-lg border border-bg-border">
              {[
                { key: 'attack_f1', label: 'Attack F1' },
                { key: 'attack_recall', label: 'Attack Recall' },
                { key: 'accuracy', label: 'Accuracy' },
                { key: 'false_positive_rate', label: 'FPR' },
              ].map(opt => (
                <button
                  key={opt.key}
                  onClick={() => setSweepMetric(opt.key as any)}
                  className={`text-xs px-2.5 py-1 rounded transition-colors ${
                    sweepMetric === opt.key
                      ? 'bg-cyan-500/20 text-cyan-300 font-semibold'
                      : 'text-text-secondary hover:text-text-primary'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={sweepLines}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2d3d" vertical={false} />
              <XAxis dataKey="threshold" tick={{ fill: '#94a3b8', fontSize: 11 }} tickFormatter={v => typeof v === 'number' ? v.toFixed(2) : v} />
              <YAxis domain={[0, 1]} tick={{ fill: '#64748b', fontSize: 11 }} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
              <Tooltip
                contentStyle={{ background: '#111827', border: '1px solid #1e2d3d', borderRadius: 8 }}
                itemStyle={{ color: '#e2e8f0', fontSize: 12 }}
                formatter={(v: number) => typeof v === 'number' ? `${(v * 100).toFixed(2)}%` : '—'}
              />
              <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 11, paddingTop: 8 }} />
              {MODEL_KEYS.map(mk => (
                <Line
                  key={mk}
                  type="monotone"
                  dataKey={MODEL_NAMES[mk]}
                  stroke={MODEL_COLORS[mk]}
                  dot={false}
                  strokeWidth={2}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
          <div className="flex items-start gap-2 mt-3 p-2.5 rounded bg-bg-secondary/40 border border-bg-border text-xs text-text-muted">
            <Info size={14} className="text-cyan-400 flex-shrink-0 mt-0.5" />
            <span>
              <strong>Validation Takeaway:</strong> Decision Tree exhibits flat plateaus because individual leaves produce discrete probability outputs. Ensembles (Random Forest, XGBoost) and Logistic Regression provide smooth continuous probabilities, making fine-grained threshold optimization effective.
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
