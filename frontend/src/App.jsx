import React from 'react'
import { NavLink, Route, Routes } from 'react-router-dom'
import Dashboard from './pages/Dashboard.jsx'
import { Robots, RobotDetail } from './pages/Robots.jsx'
import Tasks from './pages/Tasks.jsx'
import { Datasets, DatasetDetail } from './pages/Datasets.jsx'
import { Runs, RunDetail } from './pages/Runs.jsx'
import Evals from './pages/Evals.jsx'
import Models, { ModelDetail } from './pages/Models.jsx'
import Jobs from './pages/Jobs.jsx'

const NAV = [
  ['/', 'Tổng quan'], ['/robots', 'Robots'], ['/tasks', 'Tasks'], ['/datasets', 'Datasets'],
  ['/runs', 'Training'], ['/jobs', 'Jobs'], ['/evals', 'Eval'], ['/models', 'Models'],
]

export default function App() {
  return (
    <div className="shell">
      <aside className="side">
        <div className="brand"><i>◢</i> Training Studio</div>
        <nav className="nav">
          {NAV.map(([to, label]) => (
            <NavLink key={to} to={to} end={to === '/'} className={({ isActive }) => (isActive ? 'active' : '')}>{label}</NavLink>
          ))}
        </nav>
        <div className="foot">Dữ liệu đọc từ thư mục dự án. Chỉ ghi: bảng gán robot và kết quả job train bạn bấm chạy.</div>
      </aside>
      <main className="main">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/robots" element={<Robots />} />
          <Route path="/robots/:id" element={<RobotDetail />} />
          <Route path="/tasks" element={<Tasks />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/datasets/:id" element={<DatasetDetail />} />
          <Route path="/runs" element={<Runs />} />
          <Route path="/runs/:id" element={<RunDetail />} />
          <Route path="/jobs" element={<Jobs />} />
          <Route path="/evals" element={<Evals />} />
          <Route path="/models" element={<Models />} />
          <Route path="/model" element={<ModelDetail />} />
        </Routes>
      </main>
    </div>
  )
}
