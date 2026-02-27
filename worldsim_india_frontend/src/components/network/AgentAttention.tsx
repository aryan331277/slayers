import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { indiaStates, agentAttentions } from '@/data/indiaStates';

interface AgentAttentionProps {
  selectedAgent?: string;
  simulationCycle?: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Geopolitically accurate decision context for each agent.
// Each entry explains WHAT the agent is currently optimising for, and WHY
// they focus attention on specific other states.
// ─────────────────────────────────────────────────────────────────────────────
const AGENT_INTEL: Record<string, {
  objective: string;
  rationale: string;
  rivalIds: string[];
  allyIds: string[];
  keyMetric: string;
}> = {
  GJ: {
    objective: 'Offer Narmada surplus power-for-water deal to RJ',
    rationale: 'Gujarat controls Sardar Sarovar dam. Rajasthan is water-stressed and needs energy for pump irrigation. GJ proposes a kWh-per-cusec swap, reducing Rajasthan\'s crisis while expanding GJ influence.',
    rivalIds: ['MH'],
    allyIds: ['RJ', 'MP'],
    keyMetric: 'Narmada allocation utilisation',
  },
  KA: {
    objective: 'Minimise Cauvery release while staying SC-compliant',
    rationale: 'Karnataka monitors Kabini & KRS reservoir levels daily. It must balance SC-mandated releases to TN against Bengaluru drinking water needs. Farmer lobbies threaten non-compliance.',
    rivalIds: ['TN', 'AP'],
    allyIds: ['MH'],
    keyMetric: 'KRS reservoir % capacity',
  },
  TN: {
    objective: 'Maximise Cauvery inflow from Karnataka',
    rationale: 'Tamil Nadu is 100% downstream on Cauvery. It watches Karnataka\'s reservoir levels obsessively and files SC contempt petitions when releases fall below the Bachawat Tribunal award.',
    rivalIds: ['KA'],
    allyIds: ['KL', 'AP'],
    keyMetric: 'Mettur Dam storage level',
  },
  RJ: {
    objective: 'Secure IGNP canal quota + emergency groundwater rights',
    rationale: 'Rajasthan receives Indus water via IGNP canal from Punjab. With Punjab restricting flows in drought years, RJ seeks Centre arbitration and Narmada augmentation from GJ.',
    rivalIds: ['PB'],
    allyIds: ['GJ', 'MP'],
    keyMetric: 'IGNP canal flow (MAF)',
  },
  MH: {
    objective: 'Assert Krishna basin upper-riparian priority vs AP',
    rationale: 'Maharashtra controls Koyna & Ujjani dams on the Krishna tributary. It faces a two-front dispute: AP demands downstream release, KA contests Bhima sub-basin allocations.',
    rivalIds: ['AP', 'KA'],
    allyIds: ['MP', 'GJ'],
    keyMetric: 'Krishna Tribunal Award compliance',
  },
  PB: {
    objective: 'Block SYL canal completion to preserve Ravi-Beas water',
    rationale: 'Punjab controls Ravi and Beas rivers but claims they are now insufficient for domestic use. Refuses SYL canal completion despite SC orders, using farmer protests as political cover.',
    rivalIds: ['RJ', 'HR'],
    allyIds: [],
    keyMetric: 'Ravi-Beas annual runoff (MAF)',
  },
  UP: {
    objective: 'Protect Ganga allotment vs upstream Bihar dam proposals',
    rationale: 'UP is the largest consumer of Ganga water. Monitors Bihar\'s dam proposals and inter-state barrages closely. Also tracks MP\'s Ken-Betwa link which diverts Bundelkhand tributaries.',
    rivalIds: ['BR', 'MP'],
    allyIds: ['HR', 'UT'],
    keyMetric: 'Ganga flow at Haridwar (m³/s)',
  },
  BR: {
    objective: 'Manage flood risk from UP upstream surplus release',
    rationale: 'Bihar is downstream of UP on Ganga and Gandak. Excess releases from UP dams flood Bihar agriculture. Bihar advocates for coordinated release schedules via Ganga River Basin Authority.',
    rivalIds: ['UP'],
    allyIds: ['WB', 'JH'],
    keyMetric: 'Gandak barrage discharge (m³/s)',
  },
  MP: {
    objective: 'Accelerate Ken-Betwa interlinking to move water north',
    rationale: 'Madhya Pradesh drives the Ken-Betwa river interlinking project, aiming to divert surplus from Ken (flows south into UP) to drought-hit Betwa basin. Benefits MP but UP objects to reduced inflow.',
    rivalIds: ['UP'],
    allyIds: ['RJ', 'GJ'],
    keyMetric: 'Ken-Betwa transfer progress (TCM)',
  },
  AP: {
    objective: 'Maximise Godavari surplus allocation before Telangana',
    rationale: 'Post-bifurcation, AP and TG share Krishna & Godavari. AP front-runs reservoir filling to establish prior appropriation precedent. Files GWDT-II petitions to lock in allocations.',
    rivalIds: ['TL', 'KA'],
    allyIds: ['TN'],
    keyMetric: 'Srisailam reservoir level',
  },
};

const STATUS_COLOUR: Record<string, string> = {
  green: '#22c55e', amber: '#f59e0b', red: '#ef4444', black: '#6b7280',
};

const cx = 200, cy = 210, ORBIT = 140;

const getOrbitPos = (n: number, total: number): { x: number; y: number } => {
  const angle = (n / total) * 2 * Math.PI - Math.PI / 2;
  return { x: cx + Math.cos(angle) * ORBIT, y: cy + Math.sin(angle) * ORBIT };
};

const AgentAttention: React.FC<AgentAttentionProps> = ({ selectedAgent = 'GJ', simulationCycle = 0 }) => {
  const [agent, setAgent] = useState(selectedAgent);
  const [tick, setTick] = useState(0);
  const [hovNode, setHovNode] = useState<string | null>(null);

  useEffect(() => { setAgent(selectedAgent); }, [selectedAgent]);

  // Continuous particle tick
  useEffect(() => {
    const id = setInterval(() => setTick(t => (t + 1) % 200), 35);
    return () => clearInterval(id);
  }, []);

  const attnData = agentAttentions.find(a => a.stateId === agent) ?? agentAttentions[0];
  const intel = AGENT_INTEL[agent];
  const agentState = indiaStates.find(s => s.id === agent);

  // Sort targets by attention weight desc
  const targets = [...attnData.attentionWeights].sort((a, b) => b.weight - a.weight);

  const hovTarget = targets.find(t => t.targetState === hovNode);
  const hovState  = hovNode ? indiaStates.find(s => s.id === hovNode) : null;

  // Beam colour based on weight
  const beamColor = (w: number) =>
    w >= 0.7 ? '#ef4444' : w >= 0.5 ? '#f97316' : w >= 0.3 ? '#38bdf8' : '#475569';

  return (
    <div className="relative w-full h-full bg-[#070711] rounded-xl overflow-hidden border border-white/5 flex flex-col">

      {/* ── Header ────────────────────────────────────────────────────────── */}
      <div className="flex-shrink-0 px-4 pt-3 pb-2 border-b border-white/5">
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1 min-w-0">
            <p className="text-[11px] font-bold tracking-[0.14em] text-cyan-400/80 uppercase">Agent Attention</p>
            {intel && (
              <p className="text-[9px] text-gray-500 mt-0.5 leading-relaxed truncate"
                title={intel.objective}>{agent}: {intel.objective}</p>
            )}
          </div>
          <select value={agent} onChange={e => setAgent(e.target.value)}
            className="text-[10px] bg-[#111120] border border-white/10 text-gray-300
              rounded-lg px-2 py-1.5 outline-none focus:border-cyan-500/60 flex-shrink-0 cursor-pointer">
            {agentAttentions.map(a => {
              const n = indiaStates.find(s => s.id === a.stateId)?.name ?? a.stateId;
              return (
                <option key={a.stateId} value={a.stateId} className="bg-[#111120]">
                  {a.stateId} — {n}
                </option>
              );
            })}
          </select>
        </div>
      </div>

      {/* ── Main view: SVG + sidebar ──────────────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden">

        {/* SVG Attention Graph */}
        <div className="flex-1 relative overflow-hidden">
          <svg viewBox="40 50 330 340" className="w-full h-full" preserveAspectRatio="xMidYMid meet">
            <defs>
              <filter id="aa-glow"><feGaussianBlur stdDeviation="3" result="b"/>
                <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
              <filter id="aa-glow-sm"><feGaussianBlur stdDeviation="1.5" result="b"/>
                <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
              <radialGradient id="aa-center-grad" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stopColor="#38bdf8" stopOpacity="1"/>
                <stop offset="100%" stopColor="#0284c7" stopOpacity="0.7"/>
              </radialGradient>
            </defs>

            {/* Orbit rings */}
            {[70, ORBIT, ORBIT + 22].map((r, i) => (
              <circle key={r} cx={cx} cy={cy} r={r}
                fill="none" stroke="rgba(56,189,248,0.05)"
                strokeWidth={i === 1 ? 0.8 : 0.4}
                strokeDasharray={i === 2 ? '2 5' : undefined}
              />
            ))}

            {/* Attention beams + particles */}
            {targets.map((t, i) => {
              const pos = getOrbitPos(i, targets.length);
              const w = t.weight;
              const col = beamColor(w);
              const sw  = 0.8 + w * 7;
              const isH = hovNode === t.targetState;

              // Particle position along beam (staggered by index)
              const progress = ((tick / 200) + i * (1 / targets.length)) % 1;
              const px = cx + (pos.x - cx) * progress;
              const py = cy + (pos.y - cy) * progress;

              return (
                <g key={t.targetState}>
                  {/* Glow halo on beam */}
                  <line x1={cx} y1={cy} x2={pos.x} y2={pos.y}
                    stroke={col} strokeWidth={sw + 5} strokeOpacity={isH ? 0.2 : 0.06}
                    strokeLinecap="round" />
                  {/* Core beam */}
                  <motion.line x1={cx} y1={cy} x2={pos.x} y2={pos.y}
                    stroke={col} strokeWidth={sw}
                    strokeLinecap="round"
                    animate={{ strokeOpacity: isH ? 0.95 : [0.35 + w * 0.3, 0.65 + w * 0.3, 0.35 + w * 0.3] }}
                    transition={{ duration: 2 + i * 0.3, repeat: Infinity }}
                  />
                  {/* Travelling particle */}
                  <circle cx={px} cy={py} r={2.5} fill={col} opacity={0.9}
                    filter="url(#aa-glow-sm)" />
                  {/* Mid-beam weight label */}
                  <text
                    x={(cx + pos.x) / 2 + (pos.y - cy) * 0.14}
                    y={(cy + pos.y) / 2 - (pos.x - cx) * 0.14}
                    textAnchor="middle" dominantBaseline="middle"
                    fill={col} fontSize="7" fontWeight="700" opacity={isH ? 1 : 0.75}
                    style={{ pointerEvents: 'none', userSelect: 'none' }}
                  >{Math.round(w * 100)}%</text>
                </g>
              );
            })}

            {/* Centre node (current agent) */}
            <motion.circle cx={cx} cy={cy} r={32}
              fill="rgba(56,189,248,0.07)" stroke="rgba(56,189,248,0.25)" strokeWidth={1}
              animate={{ r: [30, 35, 30] }} transition={{ duration: 3, repeat: Infinity }}
            />
            <circle cx={cx} cy={cy} r={22} fill="url(#aa-center-grad)" filter="url(#aa-glow)" />
            <text x={cx} y={cy} textAnchor="middle" dominantBaseline="middle"
              fill="#000" fontSize="12" fontWeight="900"
              style={{ userSelect: 'none', pointerEvents: 'none' }}>{agent}</text>
            <text x={cx} y={cy + 36} textAnchor="middle"
              fill="rgba(255,255,255,0.5)" fontSize="8"
              style={{ userSelect: 'none', pointerEvents: 'none' }}
            >{agentState?.name ?? agent}</text>

            {/* Target nodes */}
            {targets.map((t, i) => {
              const pos   = getOrbitPos(i, targets.length);
              const tState = indiaStates.find(s => s.id === t.targetState);
              const r     = 8 + t.weight * 12;
              const isH   = hovNode === t.targetState;
              const sc    = STATUS_COLOUR[tState?.status ?? 'green'];
              const isRival = intel?.rivalIds.includes(t.targetState);
              const isAlly  = intel?.allyIds.includes(t.targetState);

              return (
                <g key={t.targetState} style={{ cursor: 'pointer' }}
                  onMouseEnter={() => setHovNode(t.targetState)}
                  onMouseLeave={() => setHovNode(null)}
                >
                  {/* Rival/Ally outer ring */}
                  {(isRival || isAlly) && (
                    <circle cx={pos.x} cy={pos.y} r={r + 6}
                      fill="none"
                      stroke={isRival ? '#ef4444' : '#22c55e'}
                      strokeWidth={1} strokeDasharray={isRival ? '2 2' : '3 3'}
                      strokeOpacity={0.7}
                    />
                  )}
                  <motion.circle cx={pos.x} cy={pos.y} r={r}
                    fill={sc} fillOpacity={isH ? 0.92 : 0.55}
                    stroke={isH ? '#fff' : sc} strokeWidth={isH ? 1.5 : 0.8}
                    animate={{ r: isH ? r + 2 : r }}
                    transition={{ duration: 0.2 }}
                    filter={isH ? 'url(#aa-glow)' : undefined}
                  />
                  <text x={pos.x} y={pos.y} textAnchor="middle" dominantBaseline="middle"
                    fill="#fff" fontSize={isH ? 8.5 : 7.5} fontWeight="700"
                    style={{ pointerEvents: 'none', userSelect: 'none' }}>{t.targetState}</text>
                  <text x={pos.x} y={pos.y + r + 9} textAnchor="middle"
                    fill="rgba(255,255,255,0.45)" fontSize="6.5"
                    style={{ pointerEvents: 'none', userSelect: 'none' }}
                  >{(tState?.name ?? t.targetState).split(' ')[0]}</text>
                  {/* Rival/Ally icon */}
                  {isRival && <text x={pos.x + r + 3} y={pos.y - r}
                    fill="#ef4444" fontSize="8" style={{ userSelect: 'none' }}>⚔</text>}
                  {isAlly && !isRival && <text x={pos.x + r + 3} y={pos.y - r}
                    fill="#22c55e" fontSize="8" style={{ userSelect: 'none' }}>🤝</text>}
                </g>
              );
            })}
          </svg>
        </div>

        {/* ── Right sidebar ─────────────────────────────────────────────── */}
        <div className="w-36 flex-shrink-0 border-l border-white/5 flex flex-col overflow-hidden">

          {/* Ranked list */}
          <div className="flex-1 overflow-y-auto px-2.5 pt-2.5">
            <p className="text-[8.5px] uppercase tracking-wider text-gray-700 font-semibold mb-2">
              Attention Rank
            </p>
            {targets.map((t, i) => {
              const pct = Math.round(t.weight * 100);
              const col = beamColor(t.weight);
              const isH = hovNode === t.targetState;
              return (
                <div key={t.targetState}
                  className="mb-2 cursor-pointer"
                  onMouseEnter={() => setHovNode(t.targetState)}
                  onMouseLeave={() => setHovNode(null)}
                >
                  <div className="flex justify-between mb-0.5">
                    <span className="text-[9px] font-bold"
                      style={{ color: isH ? '#fff' : col }}>
                      {i + 1}. {t.targetState}
                    </span>
                    <span className="text-[9px] font-mono" style={{ color: col }}>{pct}%</span>
                  </div>
                  <div className="h-[3px] bg-gray-800 rounded-full overflow-hidden">
                    <motion.div className="h-full rounded-full" style={{ background: col }}
                      initial={{ width: 0 }}
                      animate={{ width: `${pct}%` }}
                      transition={{ duration: 0.5, delay: i * 0.06 }}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          {/* Context blurb */}
          {intel && (
            <div className="flex-shrink-0 border-t border-white/5 p-2.5">
              <p className="text-[7.5px] text-cyan-500/60 font-bold uppercase tracking-wider mb-1">
                Agent rationale
              </p>
              <p className="text-[7.5px] text-gray-600 leading-relaxed">{intel.rationale}</p>
              <div className="mt-1.5">
                <span className="text-[7px] text-gray-700">Tracking: </span>
                <span className="text-[7px] text-cyan-500/70">{intel.keyMetric}</span>
              </div>
            </div>
          )}

          <div className="flex-shrink-0 px-2.5 pb-2 text-center">
            <span className="text-[8px] text-gray-800">Cycle {simulationCycle}</span>
          </div>
        </div>
      </div>

      {/* ── Hover node detail card ────────────────────────────────────────── */}
      <AnimatePresence>
        {hovNode && hovTarget && hovState && (
          <motion.div className="absolute bottom-3 left-3 z-20 rounded-xl p-3 max-w-[195px]"
            style={{
              background: 'rgba(7,7,17,0.95)',
              border: '1px solid rgba(255,255,255,0.09)',
              backdropFilter: 'blur(10px)',
            }}
            initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
          >
            <p className="text-[10px] font-bold text-white">{hovState.name}</p>
            <p className="text-[8.5px] text-gray-500 mt-0.5">
              Attention: <span style={{ color: beamColor(hovTarget.weight) }} className="font-mono font-bold">
                {Math.round(hovTarget.weight * 100)}%
              </span>
            </p>
            <p className="text-[8px] text-gray-600 mt-1.5 leading-relaxed">
              {intel?.rivalIds.includes(hovNode)
                ? '⚔️ Key rival — contested resource access. Agent prioritises countermoves.'
                : intel?.allyIds.includes(hovNode)
                ? '🤝 Strategic ally — active cooperation agreements in play.'
                : 'Secondary influence tracked for downstream cascade effects.'}
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default AgentAttention;
