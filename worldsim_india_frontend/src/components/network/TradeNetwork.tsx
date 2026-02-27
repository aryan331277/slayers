import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { indiaStates, tradeConnections } from '@/data/indiaStates';

interface TradeNetworkProps {
  simulationCycle?: number;
}

// Hardcoded node positions for a clean network layout - NOT geography-based
const NODE_POSITIONS: Record<string, { x: number; y: number }> = {
  'PB': { x: 120, y: 80 },
  'HR': { x: 190, y: 80 },
  'RJ': { x: 100, y: 160 },
  'GJ': { x: 100, y: 240 },
  'MP': { x: 200, y: 200 },
  'UP': { x: 270, y: 110 },
  'DL': { x: 210, y: 130 },
  'HP': { x: 175, y: 50 },
  'UT': { x: 250, y: 55 },
  'BR': { x: 350, y: 120 },
  'WB': { x: 380, y: 170 },
  'JH': { x: 340, y: 185 },
  'OD': { x: 350, y: 240 },
  'CG': { x: 285, y: 245 },
  'MH': { x: 195, y: 290 },
  'TL': { x: 280, y: 310 },
  'AP': { x: 310, y: 350 },
  'KA': { x: 210, y: 360 },
  'TN': { x: 270, y: 400 },
  'KL': { x: 195, y: 400 },
  'GA': { x: 160, y: 330 },
};

const ACTIVE_NODES = ['PB', 'HR', 'RJ', 'GJ', 'MP', 'UP', 'DL', 'BR', 'WB', 'MH', 'KA', 'TN', 'KL', 'AP', 'CG', 'OD'];

const TradeNetwork: React.FC<TradeNetworkProps> = ({ simulationCycle = 0 }) => {
  const [brokenConnections, setBrokenConnections] = useState<string[]>([]);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    if (simulationCycle === 45) setBrokenConnections(['PB-TN']);
    else if (simulationCycle === 55) setBrokenConnections(['PB-TN', 'KA-TN']);
    else if (simulationCycle < 45) setBrokenConnections([]);
  }, [simulationCycle]);

  // Particle animation tick
  useEffect(() => {
    const interval = setInterval(() => setTick(t => (t + 1) % 100), 50);
    return () => clearInterval(interval);
  }, []);

  const typeColor: Record<string, string> = {
    water: '#00d4ff',
    food: '#00ff88',
    power: '#f7c600',
  };

  const activeLinks = tradeConnections.filter(c =>
    ACTIVE_NODES.includes(c.from) && ACTIVE_NODES.includes(c.to)
  );

  return (
    <div className="relative w-full h-full bg-[#0a0a0f] rounded-xl overflow-hidden">
      <div className="absolute inset-0 grid-bg opacity-50" />

      <div className="absolute top-4 left-4 z-10">
        <h3 className="text-lg font-bold text-white neon-text">Trade & Alliance Network</h3>
        <p className="text-xs text-gray-400 mt-0.5">Live interstate dependencies</p>
      </div>

      {/* Legend */}
      <div className="absolute top-4 right-4 z-10 glass rounded-lg p-2.5">
        <div className="text-xs font-semibold text-gray-300 mb-1.5">Connections</div>
        <div className="space-y-1">
          {Object.entries(typeColor).map(([type, color]) => (
            <div key={type} className="flex items-center gap-1.5">
              <div className="w-6 h-0.5" style={{ background: color }} />
              <span className="text-xs text-gray-300 capitalize">{type}</span>
            </div>
          ))}
          <div className="flex items-center gap-1.5">
            <div className="w-6 h-0.5 border-dashed border-t border-red-500" style={{ background: 'none' }} />
            <span className="text-xs text-gray-300">Broken</span>
          </div>
        </div>
      </div>

      <svg
        viewBox="75 30 340 390"
        className="w-full h-full"
        preserveAspectRatio="xMidYMid meet"
      >
        <defs>
          <filter id="net-glow">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>

        {/* Draw links */}
        {activeLinks.map((link, i) => {
          const src = NODE_POSITIONS[link.from];
          const tgt = NODE_POSITIONS[link.to];
          if (!src || !tgt) return null;

          const isBroken = brokenConnections.includes(`${link.from}-${link.to}`);
          const color = typeColor[link.type] || '#888';
          const strokeW = Math.max(0.8, link.volume / 35);

          // Particle position along line
          const progress = (tick / 100 + i * 0.15) % 1;
          const px = src.x + (tgt.x - src.x) * progress;
          const py = src.y + (tgt.y - src.y) * progress;

          return (
            <g key={`${link.from}-${link.to}`}>
              <line
                x1={src.x} y1={src.y}
                x2={tgt.x} y2={tgt.y}
                stroke={isBroken ? '#ff3333' : color}
                strokeWidth={isBroken ? 0.8 : strokeW}
                strokeOpacity={isBroken ? 0.3 : 0.45}
                strokeDasharray={isBroken ? '4,3' : undefined}
              />
              {!isBroken && link.active && (
                <circle
                  cx={px} cy={py}
                  r={2}
                  fill={color}
                  opacity={0.9}
                  filter="url(#net-glow)"
                />
              )}
            </g>
          );
        })}

        {/* Draw nodes */}
        {ACTIVE_NODES.map((id) => {
          const pos = NODE_POSITIONS[id];
          const state = indiaStates.find(s => s.id === id);
          if (!pos || !state) return null;

          const nodeColor = state.status === 'green' ? '#00ff88' : state.status === 'amber' ? '#f7c600' : '#ff3333';

          return (
            <g key={id} transform={`translate(${pos.x},${pos.y})`}>
              <circle r={8} fill={nodeColor} fillOpacity={0.25} stroke={nodeColor} strokeWidth={1.5} />
              <circle r={4} fill={nodeColor} filter="url(#net-glow)" />
              <text
                x={0} y={16}
                textAnchor="middle"
                fill="rgba(255,255,255,0.7)"
                fontSize="7"
                fontWeight="600"
                style={{ userSelect: 'none' }}
              >
                {id}
              </text>
            </g>
          );
        })}
      </svg>

      {/* Break Alert */}
      {brokenConnections.length > 0 && (
        <motion.div
          className="absolute bottom-3 left-3 right-3 z-20 bg-red-900/80 border border-red-500 rounded-lg p-2.5"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse flex-shrink-0" />
            <span className="text-xs font-semibold text-red-200">Alliance Break Detected</span>
          </div>
          <p className="text-xs text-red-300 mt-0.5 leading-relaxed">
            {brokenConnections.includes('PB-TN') && 'Punjab–Tamil Nadu food supply dissolved'}
            {brokenConnections.includes('KA-TN') && ' · KA–TN water sharing suspended'}
          </p>
        </motion.div>
      )}
    </div>
  );
};

export default TradeNetwork;
