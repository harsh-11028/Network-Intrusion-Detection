import { BrowserRouter, Routes, Route, NavLink, useLocation } from 'react-router-dom'
import {
  LayoutDashboard, Upload, BarChart3, Lightbulb, BookOpen,
  Shield, ChevronLeft, ChevronRight, Wifi, WifiOff
} from 'lucide-react'
import { useState } from 'react'
import { AnalysisProvider } from './services/AnalysisContext'

import DashboardPage from './pages/DashboardPage'
import TrafficAnalyzerPage from './pages/TrafficAnalyzerPage'
import ModelPerformancePage from './pages/ModelPerformancePage'
import ModelInsightsPage from './pages/ModelInsightsPage'
import MethodologyPage from './pages/MethodologyPage'

const NAV_ITEMS = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard', end: true },
  { to: '/analyze', icon: Upload, label: 'Traffic Analyzer' },
  { to: '/performance', icon: BarChart3, label: 'Model Performance' },
  { to: '/insights', icon: Lightbulb, label: 'Model Insights' },
  { to: '/methodology', icon: BookOpen, label: 'Methodology' },
]

function Sidebar({ collapsed, setCollapsed }: { collapsed: boolean; setCollapsed: (v: boolean) => void }) {
  return (
    <aside className={`flex flex-col bg-bg-secondary border-r border-bg-border transition-all duration-300 ${collapsed ? 'w-16' : 'w-60'} flex-shrink-0 h-screen sticky top-0`}>
      {/* Logo */}
      <div className="flex items-center gap-3 px-4 py-5 border-b border-bg-border">
        <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center flex-shrink-0">
          <Shield size={18} className="text-white" />
        </div>
        {!collapsed && (
          <div className="overflow-hidden">
            <div className="text-sm font-bold text-text-primary leading-tight">NIDS</div>
            <div className="text-[10px] text-text-muted leading-tight">Intrusion Detection</div>
          </div>
        )}
      </div>

      {/* Nav */}
      <nav className="flex-1 py-4 space-y-1 px-2">
        {NAV_ITEMS.map(({ to, icon: Icon, label, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
                isActive
                  ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30'
                  : 'text-text-secondary hover:text-text-primary hover:bg-bg-card'
              }`
            }
          >
            <Icon size={18} className="flex-shrink-0" />
            {!collapsed && <span className="truncate">{label}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Bottom info */}
      <div className="px-3 py-4 border-t border-bg-border space-y-3">
        {!collapsed && (
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
            <WifiOff size={12} className="text-emerald-400 flex-shrink-0" />
            <span className="text-[10px] text-emerald-400 font-medium">Offline CSV Analysis</span>
          </div>
        )}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="w-full flex items-center justify-center p-2 rounded-lg text-text-muted hover:text-text-primary hover:bg-bg-card transition-colors"
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </button>
      </div>
    </aside>
  )
}

function PageHeader() {
  const location = useLocation()
  const item = NAV_ITEMS.find(n => n.end ? location.pathname === n.to : location.pathname.startsWith(n.to))
  const titles: Record<string, { title: string; desc: string }> = {
    '/': { title: 'Dashboard', desc: 'Overview of the latest traffic analysis' },
    '/analyze': { title: 'Traffic Analyzer', desc: 'Upload and analyze CSV network traffic data' },
    '/performance': { title: 'Model Performance', desc: 'Official Phase 7.1 test-set evaluation metrics' },
    '/insights': { title: 'Model Insights', desc: 'Feature importances and model interpretability' },
    '/methodology': { title: 'Methodology', desc: 'Project phases, design decisions, and limitations' },
  }
  const info = titles[item?.to ?? '/'] ?? titles['/']
  return (
    <header className="sticky top-0 z-10 bg-bg-primary/90 backdrop-blur border-b border-bg-border px-6 py-4 flex items-center justify-between">
      <div>
        <h1 className="text-lg font-semibold text-text-primary">{info.title}</h1>
        <p className="text-xs text-text-muted">{info.desc}</p>
      </div>
      <div className="flex items-center gap-2 text-xs text-text-muted">
        <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
        <span>UNSW-NB15 · Phase 8</span>
      </div>
    </header>
  )
}

export default function App() {
  const [collapsed, setCollapsed] = useState(false)
  return (
    <BrowserRouter>
      <AnalysisProvider>
        <div className="flex h-screen overflow-hidden bg-bg-primary">
          <Sidebar collapsed={collapsed} setCollapsed={setCollapsed} />
          <div className="flex-1 flex flex-col overflow-hidden">
            <PageHeader />
            <main className="flex-1 overflow-y-auto p-6">
              <Routes>
                <Route path="/" element={<DashboardPage />} />
                <Route path="/analyze" element={<TrafficAnalyzerPage />} />
                <Route path="/performance" element={<ModelPerformancePage />} />
                <Route path="/insights" element={<ModelInsightsPage />} />
                <Route path="/methodology" element={<MethodologyPage />} />
              </Routes>
            </main>
          </div>
        </div>
      </AnalysisProvider>
    </BrowserRouter>
  )
}
