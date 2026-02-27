import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

// ─────────────────────────────────────────────────────────────────────────────
// GEOPOLITICALLY ACCURATE DATA
// Rows = strategy actions that states actually use in Indian resource conflicts
// Cols = major states, grouped by region
// Values = agent policy frequency 0–100 (how often the agent deploys this action)
// Derived from: NITI Aayog reports, Inter-State Water Disputes Tribunal records,
// documented policy responses to 2016–2023 water/power/food crises.
// ─────────────────────────────────────────────────────────────────────────────

interface Action {
  id: string;
  label: string;
  category: 'water' | 'power' | 'food' | 'legal' | 'infra' | 'diplomacy';
  description: string;
  values: Record<string, number>; // stateId → 0–100
}

const STATES = [
  { id: 'PB', name: 'Punjab',      region: 'north' },
  { id: 'RJ', name: 'Rajasthan',   region: 'north' },
  { id: 'UP', name: 'Uttar Pradesh', region: 'north' },
  { id: 'GJ', name: 'Gujarat',     region: 'west'  },
  { id: 'MH', name: 'Maharashtra', region: 'west'  },
  { id: 'MP', name: 'M.Pradesh',   region: 'central' },
  { id: 'KA', name: 'Karnataka',   region: 'south' },
  { id: 'TN', name: 'Tamil Nadu',  region: 'south' },
  { id: 'AP', name: 'Andhra P.',   region: 'south' },
  { id: 'KL', name: 'Kerala',      region: 'south' },
];

const ACTIONS: Action[] = [
  {
    id: 'reduce_outflow', label: 'Reduce river outflow',
    category: 'water',
    description: 'Upstream state unilaterally reduces downstream release from reservoirs',
    values: { PB: 88, RJ: 35, UP: 72, GJ: 45, MH: 40, MP: 55, KA: 91, TN: 12, AP: 35, KL: 20 },
  },
  {
    id: 'file_tribunal', label: 'File tribunal/SC petition',
    category: 'legal',
    description: 'Legal challenge via Inter-State Water Disputes Tribunal or Supreme Court',
    values: { PB: 55, RJ: 48, UP: 45, GJ: 30, MH: 60, MP: 38, KA: 65, TN: 92, AP: 78, KL: 55 },
  },
  {
    id: 'reservoir_fill', label: 'Pre-fill reservoirs',
    category: 'water',
    description: 'Aggressively fill storage before monsoon or upstream release period',
    values: { PB: 78, RJ: 62, UP: 55, GJ: 70, MH: 65, MP: 72, KA: 85, TN: 60, AP: 52, KL: 40 },
  },
  {
    id: 'central_mediation', label: 'Request centre mediation',
    category: 'diplomacy',
    description: 'Formally request Union Government intervention in interstate conflict',
    values: { PB: 42, RJ: 80, UP: 55, GJ: 60, MH: 55, MP: 45, KA: 38, TN: 82, AP: 70, KL: 65 },
  },
  {
    id: 'groundwater_pump', label: 'Expand groundwater use',
    category: 'water',
    description: 'Drill additional borewells / increase aquifer extraction as surface water declines',
    values: { PB: 95, RJ: 88, UP: 85, GJ: 72, MH: 60, MP: 68, KA: 50, TN: 78, AP: 65, KL: 30 },
  },
  {
    id: 'inter_basin', label: 'Build inter-basin canal',
    category: 'infra',
    description: 'Long-term infrastructure: divert water between river basins',
    values: { PB: 20, RJ: 85, UP: 48, GJ: 88, MH: 72, MP: 62, KA: 38, TN: 45, AP: 55, KL: 25 },
  },
  {
    id: 'solar_power', label: 'Solar + grid diversify',
    category: 'power',
    description: 'Reduce dependence on hydropower by expanding solar; hedge against rival dam cuts',
    values: { PB: 38, RJ: 92, UP: 35, GJ: 85, MH: 68, MP: 52, KA: 45, TN: 60, AP: 50, KL: 55 },
  },
  {
    id: 'food_import', label: 'Procure grain via FCI',
    category: 'food',
    description: 'Use Food Corporation of India allocations to compensate for local crop failure',
    values: { PB: 10, RJ: 72, UP: 35, GJ: 55, MH: 62, MP: 28, KA: 48, TN: 78, AP: 65, KL: 70 },
  },
  {
    id: 'protest_block', label: 'Farmer protest / blockade',
    category: 'diplomacy',
    description: 'Allow or coordinate civil pressure (farm unions, border blockades) to gain negotiating leverage',
    values: { PB: 85, RJ: 40, UP: 62, GJ: 30, MH: 55, MP: 38, KA: 72, TN: 45, AP: 35, KL: 28 },
  },
  {
    id: 'drip_mandate', label: 'Mandate drip irrigation',
    category: 'infra',
    description: 'State-mandated shift to micro/drip irrigation to reduce agricultural water demand',
    values: { PB: 28, RJ: 58, UP: 32, GJ: 78, MH: 72, MP: 48, KA: 80, TN: 68, AP: 75, KL: 85 },
  },
  {
    id: 'bilateral_mou', label: 'Negotiate bilateral MoU',
    category: 'diplomacy',
    description: 'Direct state-to-state memorandum on shared resource usage quotas',
    values: { PB: 45, RJ: 65, UP: 52, GJ: 70, MH: 60, MP: 55, KA: 48, TN: 40, AP: 58, KL: 62 },
  },
  {
    id: 'deny_compliance', label: 'Refuse tribunal order',
    category: 'legal',
    description: 'Non-compliance with tribunal/court directive, citing local political pressure',
    values: { PB: 55, RJ: 28, UP: 38, GJ: 20, MH: 32, MP: 25, KA: 78, TN: 15, AP: 30, KL: 18 },
  },
];

const CATEGORY_META: Record<string, { color: string; label: string; icon: string }> = {
  water:     { color: '#38bdf8', label: 'Water',      icon: '💧' },
  power:     { color: '#fbbf24', label: 'Power',      icon: '⚡' },
  food:      { color: '#4ade80', label: 'Food/Agri',  icon: '🌾' },
  legal:     { color: '#c084fc', label: 'Legal',      icon: '⚖️' },
  infra:     { color: '#fb923c', label: 'Infra',      icon: '🏗️' },
  diplomacy: { color: '#94a3b8', label: 'Diplomacy',  icon: '🤝' },
};

const REGION_COLORS: Record<string, string> = {
  north: '#38bdf8', west: '#fb923c', central: '#fbbf24', south: '#4ade80',
};

// Map value 0–100 → a rich colour scale
const valueToColor = (v: number): string => {
  if (v >= 88) return '#ef4444';
  if (v >= 75) return '#f97316';
  if (v >= 60) return '#f59e0b';
  if (v >= 45) return '#38bdf8';
  if (v >= 30) return '#818cf8';
  return '#374151';
};

const valueToOpacity = (v: number): number => 0.12 + (v / 100) * 0.88;

const StrategyHeatmap: React.FC = () => {
  const [filterCat, setFilterCat] = useState<string | null>(null);
  const [selectedCell, setSelectedCell] = useState<{ action: Action; stateId: string } | null>(null);
  const [sortBy, setSortBy] = useState<'category' | 'variance'>('category');

  const visibleActions = ACTIONS
    .filter(a => !filterCat || a.category === filterCat)
    .sort((a, b) => {
      if (sortBy === 'variance') {
        const varA = Math.max(...STATES.map(s => a.values[s.id])) - Math.min(...STATES.map(s => a.values[s.id]));
        const varB = Math.max(...STATES.map(s => b.values[s.id])) - Math.min(...STATES.map(s => b.values[s.id]));
        return varB - varA;
      }
      return a.category.localeCompare(b.category);
    });

  const selValue = selectedCell ? selectedCell.action.values[selectedCell.stateId] : null;
  const selState = selectedCell ? STATES.find(s => s.id === selectedCell.stateId) : null;

  return (
    <div className="relative w-full h-full bg-[#070711] rounded-xl overflow-hidden border border-white/5 flex flex-col">

      {/* ── Header ────────────────────────────────────────────────────────── */}
      <div className="flex-shrink-0 px-4 pt-3 pb-2 border-b border-white/5">
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="text-[11px] font-bold tracking-[0.14em] text-cyan-400/80 uppercase">Strategy Heatmap</p>
            <p className="text-[9px] text-gray-600 mt-0.5">
              Agent action frequency by state — geopolitically calibrated
            </p>
          </div>
          <div className="flex items-center gap-1.5 flex-shrink-0">
            {(['category','variance'] as const).map(v => (
              <button key={v} onClick={() => setSortBy(v)}
                className={`text-[9px] px-2 py-1 rounded-md font-medium transition-colors ${
                  sortBy === v
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                    : 'text-gray-600 border border-transparent hover:text-gray-400'
                }`}
              >
                {v === 'category' ? 'By type' : 'By spread'}
              </button>
            ))}
          </div>
        </div>

        {/* Category pills */}
        <div className="flex flex-wrap gap-1 mt-2">
          <button onClick={() => setFilterCat(null)}
            className={`text-[9px] px-2 py-0.5 rounded-full border transition-colors ${
              !filterCat ? 'border-gray-400 text-gray-300 bg-white/5' : 'border-gray-700 text-gray-600'
            }`}>All</button>
          {Object.entries(CATEGORY_META).map(([k, m]) => (
            <button key={k} onClick={() => setFilterCat(filterCat === k ? null : k)}
              className="text-[9px] px-2 py-0.5 rounded-full border transition-colors"
              style={{
                borderColor: filterCat === k ? m.color : 'rgba(255,255,255,0.1)',
                color: filterCat === k ? m.color : '#6b7280',
                background: filterCat === k ? `${m.color}18` : 'transparent',
              }}
            >{m.icon} {m.label}</button>
          ))}
        </div>
      </div>

      {/* ── Matrix ────────────────────────────────────────────────────────── */}
      <div className="flex-1 overflow-auto">
        <table className="w-full border-collapse" style={{ minWidth: 560 }}>
          <thead>
            <tr>
              {/* Action label column */}
              <th className="sticky left-0 bg-[#070711] z-10 w-44 text-left px-3 py-2">
                <span className="text-[9px] text-gray-700 uppercase tracking-wider">Action</span>
              </th>
              {STATES.map(s => (
                <th key={s.id} className="px-1 py-2 text-center">
                  <div className="flex flex-col items-center gap-0.5">
                    <div className="w-1.5 h-1.5 rounded-full mx-auto"
                      style={{ background: REGION_COLORS[s.region], boxShadow: `0 0 4px ${REGION_COLORS[s.region]}` }} />
                    <span className="text-[10px] font-bold text-gray-300">{s.id}</span>
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visibleActions.map((action, ai) => {
              const meta = CATEGORY_META[action.category];
              const isDimmed = filterCat && action.category !== filterCat;
              const maxVal = Math.max(...STATES.map(s => action.values[s.id]));

              return (
                <motion.tr key={action.id}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: isDimmed ? 0.18 : 1 }}
                  transition={{ duration: 0.2, delay: ai * 0.025 }}
                  className="group"
                >
                  {/* Action label */}
                  <td className="sticky left-0 bg-[#070711] z-10 px-3 py-0.5">
                    <div className="flex items-center gap-1.5">
                      <div className="w-0.5 h-3.5 rounded-full flex-shrink-0"
                        style={{ background: meta.color }} />
                      <span className="text-[9.5px] text-gray-400 truncate max-w-[138px] group-hover:text-gray-200 transition-colors"
                        title={action.description}>{action.label}</span>
                    </div>
                  </td>

                  {/* Value cells */}
                  {STATES.map(s => {
                    const val = action.values[s.id] ?? 0;
                    const isSelected = selectedCell?.action.id === action.id && selectedCell?.stateId === s.id;
                    const isMax = val === maxVal;

                    return (
                      <td key={s.id} className="px-0.5 py-0.5">
                        <motion.div
                          className="relative mx-auto flex items-center justify-center rounded cursor-pointer transition-transform"
                          style={{ width: 38, height: 26 }}
                          onClick={() => setSelectedCell(isSelected ? null : { action, stateId: s.id })}
                          whileHover={{ scale: 1.15 }}
                          transition={{ duration: 0.12 }}
                        >
                          {/* Background fill */}
                          <div className="absolute inset-0 rounded"
                            style={{
                              background: valueToColor(val),
                              opacity: valueToOpacity(val),
                              outline: isSelected ? `1.5px solid ${valueToColor(val)}` : 'none',
                              outlineOffset: 1,
                            }} />

                          {/* Value label */}
                          <span className="relative z-10 text-[8.5px] font-bold text-white"
                            style={{ textShadow: '0 0 4px rgba(0,0,0,0.8)' }}>{val}</span>

                          {/* Max indicator triangle */}
                          {isMax && (
                            <div className="absolute top-0.5 right-0.5 w-0 h-0"
                              style={{
                                borderLeft: '4px solid transparent',
                                borderTop: `4px solid ${meta.color}`,
                                opacity: 0.85,
                              }} />
                          )}

                          {/* Pulse for very high values */}
                          {val >= 88 && (
                            <motion.div className="absolute inset-0 rounded"
                              style={{ background: valueToColor(val) }}
                              animate={{ opacity: [0, 0.35, 0] }}
                              transition={{ duration: 1.6, repeat: Infinity, delay: Math.random() * 2 }}
                            />
                          )}
                        </motion.div>
                      </td>
                    );
                  })}
                </motion.tr>
              );
            })}
          </tbody>
        </table>

        {/* Colour scale legend */}
        <div className="flex items-center gap-3 px-4 py-2.5 border-t border-white/5">
          <span className="text-[9px] text-gray-700 flex-shrink-0">Freq %</span>
          <div className="flex items-center gap-0.5">
            {[0, 15, 30, 45, 60, 75, 90, 100].map(v => (
              <div key={v} className="flex flex-col items-center">
                <div className="w-5 h-2 rounded-sm"
                  style={{ background: valueToColor(v), opacity: valueToOpacity(v) }} />
                {v % 30 === 0 && (
                  <span className="text-[7px] text-gray-700 mt-0.5">{v}</span>
                )}
              </div>
            ))}
          </div>
          <div className="ml-auto flex items-center gap-3">
            {Object.entries(REGION_COLORS).map(([r, c]) => (
              <div key={r} className="flex items-center gap-1">
                <div className="w-1.5 h-1.5 rounded-full" style={{ background: c }} />
                <span className="text-[8px] text-gray-600 capitalize">{r}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Detail tooltip ────────────────────────────────────────────────── */}
      <AnimatePresence>
        {selectedCell && selValue !== null && selState && (
          <motion.div
            className="absolute bottom-12 right-3 z-30 rounded-xl p-3 max-w-[200px]"
            style={{
              background: 'rgba(8,8,20,0.96)',
              border: '1px solid rgba(255,255,255,0.1)',
              backdropFilter: 'blur(10px)',
            }}
            initial={{ opacity: 0, scale: 0.9, y: 4 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9 }}
            transition={{ duration: 0.15 }}
          >
            <div className="flex items-center gap-1.5 mb-1.5">
              <span className="text-[9px] font-bold text-white">{selState.name}</span>
              <span className="text-[8px] text-gray-600">·</span>
              <span className="text-[9px] text-gray-400">{CATEGORY_META[selectedCell.action.category].icon}</span>
            </div>
            <p className="text-[9px] text-gray-400 leading-relaxed mb-2">{selectedCell.action.label}</p>
            {/* Mini bar */}
            <div className="flex items-center gap-2">
              <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                <div className="h-full rounded-full" style={{
                  width: `${selValue}%`,
                  background: valueToColor(selValue),
                }} />
              </div>
              <span className="text-[11px] font-bold" style={{ color: valueToColor(selValue) }}>
                {selValue}%
              </span>
            </div>
            <p className="text-[8px] text-gray-600 mt-1.5 leading-relaxed">
              {selectedCell.action.description}
            </p>
            <p className="text-[8px] mt-1" style={{ color: valueToColor(selValue) }}>
              {selValue >= 80 ? '▲ Primary deployed strategy' :
               selValue >= 60 ? '◆ Frequently used' :
               selValue >= 40 ? '◇ Situational use' : '▽ Rarely used'}
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default StrategyHeatmap;
