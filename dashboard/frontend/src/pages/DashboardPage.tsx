import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  ShieldAlert, ShieldCheck, Activity, Upload, BarChart3,
  Sliders, ArrowUpRight, Lock, CheckCircle2, AlertTriangle,
  Info, Cpu, FileSpreadsheet, Radio, ExternalLink
} from 'lucide-react'
import {
  PieChart, Pie, Cell, Tooltip, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, ResponsiveContainer, LineChart, Line
} from 'recharts'
import { useAnalysis } from '../services/AnalysisContext'
import { getOfficialMetrics, getThresholdAnalysis } from '../services/api'

const COLORS = { normal: '#10b981', attack: '#ef4444' }

const MODEL_COLORS: Record<string, string> = {
  logistic_regression: '#6366f1',
  decision_tree: '#f59e0b',
  random_forest: '#10b981',
  xgboost: '#06b6d4',
}

const MODEL_NAMES: Record<string, string> = {
  logistic_regression: 'Logistic Reg.',
  decision_tree: 'Decision Tree',
  random_forest: 'Random Forest',
  xgboost: 'XGBoost',
}

const BENCHMARK_DONUT = [
  { name: 'Normal Flows', value: 34206 },
  { name: 'Attack Flows', value: 21739 },
]

const PROB_BUCKETS = ['0–10%', '10–20%', '20–30%', '30–40%', '40–50%', '50–60%', '60–70%', '70–80%', '80–90%', '90–100%']

function probHistogram(rows: { probability: number }[]) {
  const buckets = Array(10).fill(0)
  rows.forEach(r => {
    const idx = Math.min(9, Math.floor(r.probability * 10))
    buckets[idx]++
  })
  return PROB_BUCKETS.map((name, i) => ({ name, count: buckets[i] }))
}

// Probability-based confidence heuristic for flagged flows
function getConfidenceSeverity(prob: number): { label: string; color: string; bg: string; border: string } {
  if (prob >= 0.90) {
    return { label: 'High', color: 'text-red-400', bg: 'bg-red-500/15', border: 'border-red-500/30' }
  } else if (prob >= 0.70) {
    return { label: 'Medium', color: 'text-amber-400', bg: 'bg-amber-500/15', border: 'border-amber-500/30' }
  }
  return { label: 'Low', color: 'text-blue-400', bg: 'bg-blue-500/15', border: 'border-blue-500/30' }
}

export default function DashboardPage() {
  const navigate = useNavigate()
  const { result } = useAnalysis()

  // Benchmark metrics & sweep data from API
  const [officialMetrics, setOfficialMetrics] = useState<Record<string, Record<string, Record<string, unknown>>> | null>(null)
  const [thresholdData, setThresholdData] = useState<Record<string, unknown>[]>([])
  const [sweepMetric, setSweepMetric] = useState<'attack_f1' | 'attack_recall' | 'accuracy'>('attack_f1')
  const [loadingMetrics, setLoadingMetrics] = useState(true)

  useEffect(() => {
    Promise.all([getOfficialMetrics(), getThresholdAnalysis()])
      .then(([m, t]) => {
        setOfficialMetrics(m.data.test_metrics as Record<string, Record<string, Record<string, unknown>>>)
        setThresholdData(t.data.threshold_analysis)
      })
      .catch(() => {})
      .finally(() => setLoadingMetrics(false))
  }, [])

  // Model comparison bar data for standard threshold 0.50
  const modelCompData = [
    { metric: 'Accuracy', lr: 0.7445, dt: 0.8221, rf: 0.8273, xgb: 0.8353 },
    { metric: 'Attack Recall', lr: 0.9581, dt: 0.9272, rf: 0.9751, xgb: 0.9733 },
    { metric: 'Attack F1', lr: 0.7445, dt: 0.8020, rf: 0.8144, xgb: 0.8212 },
    { metric: 'ROC-AUC', lr: 0.9193, dt: 0.8432, rf: 0.9655, xgb: 0.9719 },
  ]

  // If officialMetrics loaded from API, update dynamically:
  const dynamicModelCompData = officialMetrics ? [
    {
      metric: 'Accuracy',
      lr: officialMetrics.logistic_regression?.default_0_5?.accuracy ?? 0.7445,
      dt: officialMetrics.decision_tree?.default_0_5?.accuracy ?? 0.8221,
      rf: officialMetrics.random_forest?.default_0_5?.accuracy ?? 0.8273,
      xgb: officialMetrics.xgboost?.default_0_5?.accuracy ?? 0.8353,
    },
    {
      metric: 'Attack Recall',
      lr: officialMetrics.logistic_regression?.default_0_5?.attack_recall ?? 0.9581,
      dt: officialMetrics.decision_tree?.default_0_5?.attack_recall ?? 0.9272,
      rf: officialMetrics.random_forest?.default_0_5?.attack_recall ?? 0.9751,
      xgb: officialMetrics.xgboost?.default_0_5?.attack_recall ?? 0.9733,
    },
    {
      metric: 'Attack F1',
      lr: officialMetrics.logistic_regression?.default_0_5?.attack_f1 ?? 0.7445,
      dt: officialMetrics.decision_tree?.default_0_5?.attack_f1 ?? 0.8020,
      rf: officialMetrics.random_forest?.default_0_5?.attack_f1 ?? 0.8144,
      xgb: officialMetrics.xgboost?.default_0_5?.attack_f1 ?? 0.8212,
    },
    {
      metric: 'ROC-AUC',
      lr: officialMetrics.logistic_regression?.default_0_5?.roc_auc ?? 0.9193,
      dt: officialMetrics.decision_tree?.default_0_5?.roc_auc ?? 0.8432,
      rf: officialMetrics.random_forest?.default_0_5?.roc_auc ?? 0.9655,
      xgb: officialMetrics.xgboost?.default_0_5?.roc_auc ?? 0.9719,
    },
  ] : modelCompData

  // Threshold sweep line chart data
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
      Object.keys(MODEL_NAMES).forEach(mk => {
        const found = byModel[mk]?.find(v => Math.abs(v.threshold - th) < 0.0001)
        if (found) pt[MODEL_NAMES[mk]] = found.value
      })
      return pt
    })
  })()

  // Uploaded dataset variables
  const isUploaded = Boolean(result)
  const donutData = isUploaded && result
    ? [
        { name: 'Normal Flows', value: result.n_normal },
        { name: 'Attack Flows', value: result.n_attacks },
      ]
    : BENCHMARK_DONUT

  const totalDonutFlows = isUploaded && result
    ? result.n_rows
    : 55945

  const normalDonutFlows = isUploaded && result
    ? result.n_normal
    : 34206

  const attackDonutFlows = isUploaded && result
    ? result.n_attacks
    : 21739

  const normalPct = ((normalDonutFlows / totalDonutFlows) * 100).toFixed(1)
  const attackPct = ((attackDonutFlows / totalDonutFlows) * 100).toFixed(1)

  const histData = isUploaded && result ? probHistogram(result.rows) : []
  const topSuspicious = isUploaded && result
    ? [...result.rows].sort((a, b) => b.probability - a.probability).filter(r => r.prediction === 1).slice(0, 5)
    : []

  const ev = result?.evaluation

  return (
    <div className="space-y-4 max-w-[1600px] mx-auto">
      {/* ── TOP HEADER BAR (Subtle Readiness Indicator & Offline Mode) ─ */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-bg-border/60">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="w-2 h-2 rounded-full bg-emerald-400" />
            <h1 className="text-sm font-semibold text-text-primary tracking-tight uppercase">
              Network Intrusion Detection Dashboard
            </h1>
            <span className="text-[11px] text-text-muted">
              · Offline Analysis Mode
            </span>
          </div>
          <p className="text-xs text-text-muted mt-0.5">
            ML-Based Network Attack Classification on UNSW-NB15 (Phase 7.1 Final Models)
          </p>
        </div>

        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-bg-secondary/80 border border-bg-border/60 text-[11px] font-mono text-emerald-400">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            <span>Engine Ready</span>
          </div>
          <button
            onClick={() => navigate('/analyze')}
            className="btn-primary text-xs px-3 py-1.5 flex items-center gap-1.5 shadow-sm"
          >
            <Upload size={13} />
            {isUploaded ? 'Analyze Another CSV' : 'Analyze Traffic CSV'}
          </button>
        </div>
      </div>

      {/* ── KPI STRIP (4 Compact High-Density Cards) ──────────────────── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {isUploaded && result ? (
          <>
            <div className="card p-3 flex items-center justify-between bg-bg-card border-bg-border/60">
              <div>
                <span className="text-[10px] font-semibold uppercase text-text-muted tracking-wider block">
                  Total Flows Analyzed
                </span>
                <span className="text-xl font-bold font-mono text-text-primary mt-0.5 block">
                  {result.n_rows.toLocaleString()}
                </span>
                <span className="text-[10px] text-cyan-400 font-mono">User Uploaded Batch</span>
              </div>
              <div className="w-8 h-8 rounded-lg bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400">
                <FileSpreadsheet size={16} />
              </div>
            </div>

            <div className="card p-3 flex items-center justify-between bg-bg-card border-bg-border/60">
              <div>
                <span className="text-[10px] font-semibold uppercase text-text-muted tracking-wider block">
                  Predicted Attacks
                </span>
                <span className="text-xl font-bold font-mono text-red-400 mt-0.5 block">
                  {result.n_attacks.toLocaleString()}
                </span>
                <span className="text-[10px] text-red-400/80 font-mono">{result.attack_pct}% of total flows</span>
              </div>
              <div className="w-8 h-8 rounded-lg bg-red-500/10 border border-red-500/20 flex items-center justify-center text-red-400">
                <ShieldAlert size={16} />
              </div>
            </div>

            <div className="card p-3 flex items-center justify-between bg-bg-card border-bg-border/60">
              <div>
                <span className="text-[10px] font-semibold uppercase text-text-muted tracking-wider block">
                  Predicted Normal
                </span>
                <span className="text-xl font-bold font-mono text-emerald-400 mt-0.5 block">
                  {result.n_normal.toLocaleString()}
                </span>
                <span className="text-[10px] text-emerald-400/80 font-mono">{(100 - result.attack_pct).toFixed(1)}% of total flows</span>
              </div>
              <div className="w-8 h-8 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
                <ShieldCheck size={16} />
              </div>
            </div>

            <div className="card p-3 flex items-center justify-between bg-bg-card border-bg-border/60">
              <div>
                <span className="text-[10px] font-semibold uppercase text-text-muted tracking-wider block">
                  Active Model & Cutoff
                </span>
                <span className="text-base font-bold font-mono text-purple-400 mt-0.5 block truncate">
                  {result.model_key.replace('_', ' ').toUpperCase()}
                </span>
                <span className="text-[10px] text-text-muted font-mono">Threshold: {result.threshold.toFixed(2)}</span>
              </div>
              <div className="w-8 h-8 rounded-lg bg-purple-500/10 border border-purple-500/20 flex items-center justify-center text-purple-400">
                <Sliders size={16} />
              </div>
            </div>
          </>
        ) : (
          <>
            <div className="card p-3 flex items-center justify-between bg-bg-card border-bg-border/60">
              <div>
                <span className="text-[10px] font-semibold uppercase text-text-muted tracking-wider block">
                  Official Test Samples
                </span>
                <span className="text-xl font-bold font-mono text-text-primary mt-0.5 block">
                  55,945
                </span>
                <span className="text-[10px] text-text-muted font-mono">Official Benchmark Test Set</span>
              </div>
              <div className="w-8 h-8 rounded-lg bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400">
                <FileSpreadsheet size={16} />
              </div>
            </div>

            <div className="card p-3 flex items-center justify-between bg-bg-card border-bg-border/60">
              <div>
                <span className="text-[10px] font-semibold uppercase text-text-muted tracking-wider block">
                  Actual Attack Samples
                </span>
                <span className="text-xl font-bold font-mono text-red-400 mt-0.5 block">
                  21,739
                </span>
                <span className="text-[10px] text-red-400/80 font-mono">38.86% Attack Prevalence</span>
              </div>
              <div className="w-8 h-8 rounded-lg bg-red-500/10 border border-red-500/20 flex items-center justify-center text-red-400">
                <ShieldAlert size={16} />
              </div>
            </div>

            <div className="card p-3 flex items-center justify-between bg-bg-card border-bg-border/60">
              <div>
                <span className="text-[10px] font-semibold uppercase text-text-muted tracking-wider block">
                  Actual Normal Samples
                </span>
                <span className="text-xl font-bold font-mono text-emerald-400 mt-0.5 block">
                  34,206
                </span>
                <span className="text-[10px] text-emerald-400/80 font-mono">61.14% Normal Flows</span>
              </div>
              <div className="w-8 h-8 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
                <ShieldCheck size={16} />
              </div>
            </div>

            <div className="card p-3 flex items-center justify-between bg-bg-card border-bg-border/60">
              <div>
                <span className="text-[10px] font-semibold uppercase text-text-muted tracking-wider block">
                  XGBoost Attack Recall
                </span>
                <span className="text-xl font-bold font-mono text-cyan-400 mt-0.5 block">
                  97.33%
                </span>
                <span className="text-[10px] text-cyan-400/80 font-mono">Standard Threshold 0.50</span>
              </div>
              <div className="w-8 h-8 rounded-lg bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400">
                <Cpu size={16} />
              </div>
            </div>
          </>
        )}
      </div>

      {/* ── ROW 1 OF PANELS: Donut (A) | Probability Histogram (B) ─────── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-3">
        {/* Panel A — Traffic Classification Distribution */}
        <div className="card p-3.5 lg:col-span-5 flex flex-col justify-between border-bg-border/60">
          <div>
            <div className="flex items-center justify-between mb-1">
              <h2 className="text-xs font-semibold text-text-primary uppercase tracking-wider flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                Panel A — Flow Classification Distribution
              </h2>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-bg-secondary border border-bg-border/60 text-cyan-400">
                {isUploaded ? 'Uploaded Batch' : 'Official Benchmark Test'}
              </span>
            </div>
            <p className="text-[11px] text-text-muted mb-2.5">
              {isUploaded
                ? `Binary predictions across ${totalDonutFlows.toLocaleString()} user flows.`
                : 'Ground-truth class balance across the 55,945 official UNSW-NB15 test flows.'}
            </p>

            <div className="flex items-center justify-around gap-4 py-1">
              <div className="relative flex items-center justify-center">
                <ResponsiveContainer width={140} height={140}>
                  <PieChart>
                    <Pie
                      data={donutData}
                      cx="50%"
                      cy="50%"
                      innerRadius={44}
                      outerRadius={62}
                      paddingAngle={3}
                      dataKey="value"
                    >
                      <Cell fill={COLORS.normal} />
                      <Cell fill={COLORS.attack} />
                    </Pie>
                    <Tooltip
                      contentStyle={{ background: '#111827', border: '1px solid #1e2d3d', borderRadius: 8, fontSize: 11 }}
                      itemStyle={{ color: '#e2e8f0' }}
                    />
                  </PieChart>
                </ResponsiveContainer>
                <div className="absolute flex flex-col items-center justify-center pointer-events-none">
                  <span className="text-xs font-bold font-mono text-text-primary">{totalDonutFlows.toLocaleString()}</span>
                  <span className="text-[9px] text-text-muted uppercase">Flows</span>
                </div>
              </div>

              <div className="space-y-2 text-xs">
                <div className="p-2 rounded-lg bg-bg-secondary/70 border border-bg-border/60">
                  <div className="flex items-center gap-1.5 mb-0.5">
                    <span className="w-2 h-2 rounded-full bg-emerald-500" />
                    <span className="text-[10px] text-text-muted uppercase">Normal Flows</span>
                  </div>
                  <div className="font-mono font-bold text-xs text-emerald-400">
                    {normalDonutFlows.toLocaleString()} <span className="text-[10px] text-text-muted font-normal">({normalPct}%)</span>
                  </div>
                </div>

                <div className="p-2 rounded-lg bg-bg-secondary/70 border border-bg-border/60">
                  <div className="flex items-center gap-1.5 mb-0.5">
                    <span className="w-2 h-2 rounded-full bg-red-500" />
                    <span className="text-[10px] text-text-muted uppercase">Attack Flows</span>
                  </div>
                  <div className="font-mono font-bold text-xs text-red-400">
                    {attackDonutFlows.toLocaleString()} <span className="text-[10px] text-text-muted font-normal">({attackPct}%)</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="text-[10px] text-text-muted pt-2 border-t border-bg-border/50 flex items-center justify-between">
            <span>{isUploaded ? 'Filtered by active model threshold' : 'Reference ground truth partition'}</span>
            <span className="font-mono text-cyan-400">{isUploaded ? `Th: ${result?.threshold.toFixed(2)}` : 'Standard 0.50'}</span>
          </div>
        </div>

        {/* Panel B — Prediction Probability Distribution */}
        <div className="card p-3.5 lg:col-span-7 flex flex-col justify-between border-bg-border/60">
          <div>
            <div className="flex items-center justify-between mb-1">
              <h2 className="text-xs font-semibold text-text-primary uppercase tracking-wider flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                Panel B — Prediction Probability Distribution
              </h2>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-bg-secondary border border-bg-border/60 text-cyan-400">
                {isUploaded ? 'Model Output Density' : 'Awaiting Ingestion'}
              </span>
            </div>
            <p className="text-[11px] text-text-muted mb-2">
              Histogram of predicted attack probabilities across 10 decile buckets [0.0–1.0].
            </p>

            {isUploaded && histData.length > 0 ? (
              <ResponsiveContainer width="100%" height={145}>
                <BarChart data={histData} barSize={16}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e2d3d" vertical={false} />
                  <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 10 }} />
                  <YAxis tick={{ fill: '#64748b', fontSize: 10 }} />
                  <Tooltip
                    contentStyle={{ background: '#111827', border: '1px solid #1e2d3d', borderRadius: 8, fontSize: 11 }}
                    itemStyle={{ color: '#e2e8f0' }}
                  />
                  <Bar dataKey="count" fill="#06b6d4" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-[145px] flex flex-col items-center justify-center rounded-lg bg-bg-secondary/30 border border-bg-border/40 text-center p-3">
                <p className="text-xs text-text-secondary font-medium">Probability Distribution Awaiting Traffic Ingestion</p>
                <p className="text-[11px] text-text-muted max-w-sm mt-0.5">
                  Upload network traffic in the{' '}
                  <button
                    onClick={() => navigate('/analyze')}
                    className="text-cyan-400 hover:text-cyan-300 font-medium underline inline transition-colors"
                  >
                    Traffic Analyzer
                  </button>{' '}
                  to view posterior confidence scores.
                </p>
              </div>
            )}
          </div>

          <div className="text-[10px] text-text-muted pt-2 border-t border-bg-border/50 flex items-center justify-between">
            <span>Posterior confidence P(Attack | X)</span>
            <span className="font-mono text-cyan-400">10 Bins (Deciles)</span>
          </div>
        </div>
      </div>

      {/* ── ROW 2 OF PANELS: Model Comparison (C) | Threshold Sweep (D) ── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-3">
        {/* Panel C — Model Performance Comparison */}
        <div className="card p-3.5 lg:col-span-7 flex flex-col justify-between border-bg-border/60">
          <div>
            <div className="flex items-center justify-between mb-1">
              <h2 className="text-xs font-semibold text-text-primary uppercase tracking-wider flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                Panel C — Model Performance Comparison (Official Test Set: 55,945)
              </h2>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-bg-secondary border border-bg-border/60 text-cyan-400">
                Threshold 0.50
              </span>
            </div>
            <p className="text-[11px] text-text-muted mb-2">
              Standardized comparison of all 4 retrained models evaluated on the official test set.
            </p>

            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={dynamicModelCompData} barCategoryGap="20%">
                <CartesianGrid strokeDasharray="3 3" stroke="#1e2d3d" vertical={false} />
                <XAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                <YAxis domain={[0, 1.0]} tick={{ fill: '#64748b', fontSize: 10 }} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                <Tooltip
                  contentStyle={{ background: '#111827', border: '1px solid #1e2d3d', borderRadius: 8, fontSize: 11 }}
                  itemStyle={{ color: '#e2e8f0' }}
                  formatter={(v: number, name: string) => [`${(v * 100).toFixed(2)}%`, name]}
                />
                <Bar dataKey="lr" name="Logistic Reg." fill="#6366f1" radius={[3, 3, 0, 0]} />
                <Bar dataKey="dt" name="Decision Tree" fill="#f59e0b" radius={[3, 3, 0, 0]} />
                <Bar dataKey="rf" name="Random Forest" fill="#10b981" radius={[3, 3, 0, 0]} />
                <Bar dataKey="xgb" name="XGBoost" fill="#06b6d4" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="flex items-center justify-between pt-2 border-t border-bg-border/50 text-[10px] text-text-muted flex-wrap gap-2">
            <div className="flex items-center gap-3">
              <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-[#6366f1]" /> LR</span>
              <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-[#f59e0b]" /> DT</span>
              <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-[#10b981]" /> RF</span>
              <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-[#06b6d4]" /> XGBoost (0.9719 AUC)</span>
            </div>
            <button
              onClick={() => navigate('/performance')}
              className="text-cyan-400 hover:text-cyan-300 font-medium inline-flex items-center gap-1"
            >
              Full Test Matrix <ArrowUpRight size={11} />
            </button>
          </div>
        </div>

        {/* Panel D — Threshold Analysis */}
        <div className="card p-3.5 lg:col-span-5 flex flex-col justify-between border-bg-border/60">
          <div>
            <div className="flex items-center justify-between mb-1 flex-wrap gap-1">
              <h2 className="text-xs font-semibold text-text-primary uppercase tracking-wider flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                Panel D — Validation Threshold Sweep
              </h2>
              <div className="flex items-center gap-1 bg-bg-secondary p-0.5 rounded border border-bg-border/60 text-[10px]">
                {(['attack_f1', 'attack_recall', 'accuracy'] as const).map(k => (
                  <button
                    key={k}
                    onClick={() => setSweepMetric(k)}
                    className={`px-1.5 py-0.5 rounded transition-colors ${
                      sweepMetric === k ? 'bg-cyan-500/20 text-cyan-300 font-bold' : 'text-text-muted hover:text-text-primary'
                    }`}
                  >
                    {k === 'attack_f1' ? 'F1' : k === 'attack_recall' ? 'Recall' : 'Acc'}
                  </button>
                ))}
              </div>
            </div>
            <p className="text-[11px] text-text-muted mb-2">
              Trade-offs across decision thresholds [0.05–0.95] on 21,548 validation flows.
            </p>

            {sweepLines.length > 0 ? (
              <ResponsiveContainer width="100%" height={160}>
                <LineChart data={sweepLines}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e2d3d" vertical={false} />
                  <XAxis dataKey="threshold" tick={{ fill: '#64748b', fontSize: 10 }} tickFormatter={v => v.toFixed(2)} />
                  <YAxis domain={[0.6, 1.0]} tick={{ fill: '#64748b', fontSize: 10 }} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                  <Tooltip
                    contentStyle={{ background: '#111827', border: '1px solid #1e2d3d', borderRadius: 8, fontSize: 11 }}
                    itemStyle={{ color: '#e2e8f0' }}
                    formatter={(v: number) => `${(v * 100).toFixed(2)}%`}
                  />
                  {Object.keys(MODEL_NAMES).map(mk => (
                    <Line
                      key={mk}
                      type="monotone"
                      dataKey={MODEL_NAMES[mk]}
                      stroke={MODEL_COLORS[mk]}
                      dot={false}
                      strokeWidth={1.5}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-[160px] flex items-center justify-center text-xs text-text-muted">
                Loading validation curve…
              </div>
            )}
          </div>

          <div className="text-[10px] text-text-muted pt-2 border-t border-bg-border/50 flex items-center justify-between">
            <span>Leakage-free internal validation</span>
            <span className="font-mono text-cyan-400">21,548 Samples</span>
          </div>
        </div>
      </div>

      {/* ── ROW 3 OF PANELS: Suspicious Flows (E) | Analysis Summary (F) ── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-3">
        {/* Panel E — Recent Flagged Suspicious Flows with Severity Heuristic */}
        <div className="card p-3.5 lg:col-span-8 flex flex-col justify-between border-bg-border/60">
          <div>
            <div className="flex items-center justify-between mb-1">
              <h2 className="text-xs font-semibold text-text-primary uppercase tracking-wider flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                Panel E — Flagged Suspicious Flows
              </h2>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-bg-secondary border border-bg-border/60 text-cyan-400">
                {isUploaded ? `${result?.n_attacks.toLocaleString()} Flagged` : 'Awaiting Ingestion'}
              </span>
            </div>
            <p className="text-[11px] text-text-muted mb-2">
              Model-flagged network flows ranked by attack probability. Severity is a probability-based confidence heuristic (High: ≥90%, Med: ≥70%, Low: &lt;70%).
            </p>

            {isUploaded && topSuspicious.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-bg-border/60 text-text-muted uppercase font-medium">
                      <th className="py-2 px-2.5">Flow ID</th>
                      <th className="py-2 px-2.5">Attack Probability</th>
                      <th className="py-2 px-2.5">Prediction</th>
                      <th className="py-2 px-2.5">
                        <span className="flex items-center gap-1">
                          Severity <span className="text-[9px] font-normal text-text-muted">(Heuristic)</span>
                        </span>
                      </th>
                      {result?.has_ground_truth && <th className="py-2 px-2.5">True Label</th>}
                      {result?.rows[0]?.attack_cat !== undefined && <th className="py-2 px-2.5">Category</th>}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-bg-border/40 font-mono">
                    {topSuspicious.map(row => {
                      const sev = getConfidenceSeverity(row.probability)
                      return (
                        <tr key={row.row_index} className="hover:bg-bg-secondary/40 transition-colors">
                          <td className="py-2 px-2.5 text-text-muted">#{row.row_index}</td>
                          <td className="py-2 px-2.5">
                            <div className="flex items-center gap-2">
                              <div className="w-16 h-1.5 rounded-full bg-bg-border overflow-hidden">
                                <div
                                  className="h-full bg-red-500 rounded-full"
                                  style={{ width: `${row.probability * 100}%` }}
                                />
                              </div>
                              <span className="text-red-400 font-semibold">{(row.probability * 100).toFixed(1)}%</span>
                            </div>
                          </td>
                          <td className="py-2 px-2.5"><span className="badge-attack">Attack</span></td>
                          <td className="py-2 px-2.5">
                            <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium border ${sev.bg} ${sev.color} ${sev.border}`}>
                              {sev.label}
                            </span>
                          </td>
                          {result?.has_ground_truth && (
                            <td className="py-2 px-2.5">
                              {row.true_label === 1 ? <span className="badge-attack">Attack</span>
                                : row.true_label === 0 ? <span className="badge-normal">Normal</span>
                                : <span className="text-text-muted">—</span>}
                            </td>
                          )}
                          {row.attack_cat !== undefined && (
                            <td className="py-2 px-2.5 text-text-secondary">{row.attack_cat}</td>
                          )}
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="h-[110px] flex flex-col items-center justify-center rounded-lg bg-bg-secondary/20 border border-bg-border/40 text-center p-3">
                <p className="text-xs text-text-secondary font-medium">No Ingested Traffic</p>
                <p className="text-[11px] text-text-muted mt-0.5">
                  Analyze a CSV file in the{' '}
                  <button
                    onClick={() => navigate('/analyze')}
                    className="text-cyan-400 hover:text-cyan-300 underline font-medium inline transition-colors"
                  >
                    Traffic Analyzer
                  </button>{' '}
                  to inspect flagged flows.
                </p>
              </div>
            )}
          </div>

          <div className="text-[10px] text-text-muted pt-2 border-t border-bg-border/50 flex items-center justify-between">
            <span>Requires human SOC analyst triage</span>
            <button
              onClick={() => navigate('/analyze')}
              className="text-cyan-400 hover:text-cyan-300 font-medium inline-flex items-center gap-1"
            >
              Analyze / Export Full Table <ArrowUpRight size={11} />
            </button>
          </div>
        </div>

        {/* Panel F — Analysis Summary */}
        <div className="card p-3.5 lg:col-span-4 flex flex-col justify-between border-bg-border/60">
          <div>
            <div className="flex items-center justify-between mb-1">
              <h2 className="text-xs font-semibold text-text-primary uppercase tracking-wider flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                Panel F — Session Execution Summary
              </h2>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-bg-secondary border border-bg-border/60 text-cyan-400">
                {isUploaded ? 'Active Session' : 'Ready'}
              </span>
            </div>
            <p className="text-[11px] text-text-muted mb-2">
              Parameters & evaluation metrics for current inference session.
            </p>

            <div className="space-y-1.5 text-xs">
              <div className="flex items-center justify-between p-2 rounded bg-bg-secondary/70 border border-bg-border/60">
                <span className="text-text-muted">Selected Detection Model:</span>
                <span className="font-mono font-bold text-cyan-400">
                  {isUploaded ? result?.model_key.replace('_', ' ').toUpperCase() : 'XGBOOST (DEFAULT)'}
                </span>
              </div>

              <div className="flex items-center justify-between p-2 rounded bg-bg-secondary/70 border border-bg-border/60">
                <span className="text-text-muted">Decision Threshold:</span>
                <span className="font-mono font-bold text-text-primary">
                  {isUploaded ? result?.threshold.toFixed(2) : '0.49 (Strategy A)'}
                </span>
              </div>

              <div className="flex items-center justify-between p-2 rounded bg-bg-secondary/70 border border-bg-border/60">
                <span className="text-text-muted">Flows Ingested:</span>
                <span className="font-mono font-bold text-text-primary">
                  {isUploaded ? result?.n_rows.toLocaleString() : '0 (Awaiting Upload)'}
                </span>
              </div>

              <div className="flex items-center justify-between p-2 rounded bg-bg-secondary/70 border border-bg-border/60">
                <span className="text-text-muted">Ground Truth Labels:</span>
                <span className="font-mono font-bold text-emerald-400">
                  {isUploaded ? (result?.has_ground_truth ? 'Detected (Evaluated)' : 'None (Unsupervised)') : 'N/A'}
                </span>
              </div>

              {ev && (
                <div className="p-2 rounded bg-emerald-500/10 border border-emerald-500/20 space-y-1">
                  <div className="text-[10px] font-bold text-emerald-400 uppercase tracking-wide">
                    Ground-Truth Evaluation ({ev.n_labeled} flows):
                  </div>
                  <div className="grid grid-cols-2 gap-1 text-[11px] font-mono">
                    <div>Acc: <span className="font-bold text-text-primary">{(ev.accuracy * 100).toFixed(1)}%</span></div>
                    <div>Recall: <span className="font-bold text-emerald-400">{(ev.attack_recall * 100).toFixed(1)}%</span></div>
                    <div>Prec: <span className="font-bold text-cyan-400">{(ev.attack_precision * 100).toFixed(1)}%</span></div>
                    <div>F1: <span className="font-bold text-text-primary">{ev.attack_f1.toFixed(3)}</span></div>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="text-[10px] text-text-muted pt-2 border-t border-bg-border/50 flex items-center justify-between">
            <span>Offline ML Decision Support</span>
            <span className="font-mono text-cyan-400">UNSW-NB15</span>
          </div>
        </div>
      </div>

      {/* ── ROW 4: Compact Official Phase 7.1 Benchmark Comparison Table ── */}
      <div className="card p-3.5 border-bg-border/60">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
            <h2 className="text-xs font-semibold text-text-primary uppercase tracking-wider">
              Official Benchmark Results (Phase 7.1 Final Test Evaluation · 55,945 Samples)
            </h2>
          </div>
          <span className="text-[10px] font-mono text-text-muted">
            Quarantined Test Set · Standard Threshold 0.50
          </span>
        </div>
        <p className="text-[11px] text-text-muted mb-2.5">
          Verified baseline metrics evaluated on the untouched official test partition. Sourced directly from Phase 7.1 test evaluation artifacts.
        </p>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-bg-border/60 text-text-muted font-medium uppercase tracking-wider">
                <th className="py-2 px-3">Model</th>
                <th className="py-2 px-3">Architecture</th>
                <th className="py-2 px-3">Accuracy</th>
                <th className="py-2 px-3">Attack Recall</th>
                <th className="py-2 px-3">Attack F1</th>
                <th className="py-2 px-3">ROC-AUC</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-bg-border/40 font-mono text-xs">
              {[
                { name: 'XGBoost', key: 'xgboost', desc: 'Gradient Boosted Trees (Recommended)', acc: '83.53%', rec: '97.33%', f1: '0.8212', auc: '0.9719' },
                { name: 'Random Forest', key: 'random_forest', desc: '300-Tree Bagging Ensemble', acc: '82.73%', rec: '97.51%', f1: '0.8144', auc: '0.9655' },
                { name: 'Decision Tree', key: 'decision_tree', desc: 'Single CART Tree (Max Depth 15)', acc: '82.21%', rec: '92.72%', f1: '0.8020', auc: '0.8432' },
                { name: 'Logistic Regression', key: 'logistic_regression', desc: 'L2-Regularized Linear Baseline', acc: '74.45%', rec: '95.81%', f1: '0.7445', auc: '0.9193' },
              ].map(row => {
                const color = MODEL_COLORS[row.key] || '#94a3b8'
                return (
                  <tr key={row.key} className="hover:bg-bg-secondary/40 transition-colors">
                    <td className="py-2 px-3 font-semibold flex items-center gap-1.5" style={{ color }}>
                      <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: color }} />
                      {row.name}
                    </td>
                    <td className="py-2 px-3 text-text-muted font-sans text-[11px]">{row.desc}</td>
                    <td className="py-2 px-3 text-text-primary">{row.acc}</td>
                    <td className="py-2 px-3 text-emerald-400 font-semibold">{row.rec}</td>
                    <td className="py-2 px-3 text-text-primary">{row.f1}</td>
                    <td className="py-2 px-3 text-cyan-400">{row.auc}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
