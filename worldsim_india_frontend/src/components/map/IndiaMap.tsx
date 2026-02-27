import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { indiaStates } from '@/data/indiaStates';
import type { StateData } from '@/types';

interface IndiaMapProps {
  onStateClick?: (state: StateData) => void;
  selectedState?: string | null;
  simulationCycle?: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// STRATEGY: Use the actual India map PNG as SVG background image.
// The viewBox is 0 0 800 900 (matching the image's ~portrait aspect ratio).
// All state marker coordinates below are calibrated to this viewBox.
//
// Map image origin: top-left corner aligns with J&K top-left.
// The image shows India with states outlined — we overlay interactive dots.
// ─────────────────────────────────────────────────────────────────────────────

// ── Set true locally to click-to-log SVG coords for calibration ──────────────
const DEV_CALIBRATE = false;

// NOTE: Replace INDIA_MAP_SRC with the actual path/URL to the uploaded PNG.
// In a Next.js / Vite project this would be imported or referenced as a public asset.
const INDIA_MAP_SRC = '/india-map.png'; // <-- point this to your uploaded image

// ─────────────────────────────────────────────────────────────────────────────
// STATE MARKERS — coordinates calibrated for the standard India outline image
// viewBox: 0 0 800 900  (portrait)
// Origin (0,0) = top-left of image; positive X→right, positive Y→down
// ─────────────────────────────────────────────────────────────────────────────
const STATE_MARKERS: Record<string, {
  x: number; y: number;
  name: string;
  capital: string;
  abbr: string;
}> = {
  // ─────────────────────────────────────────────────────────────────────────
  // PIXEL-ACCURATE coordinates — derived by:
  // 1. Using Python/Pillow to flood-fill state regions from the 736×736 source image
  // 2. Computing exact centroid of each filled region
  // 3. Converting: svg_x = 40 + (img_x/736)*680, svg_y = 70 + (img_y/736)*680
  //    (image renders at x=40,y=70,w=680,h=680 in viewBox 800×900)
  // ─────────────────────────────────────────────────────────────────────────

  // ── North ─────────────────────────────────────────────────────────────────
  JK:  { x: 238, y: 162, name: 'Jammu & Kashmir',   capital: 'Srinagar',             abbr: 'JK'  },
  LA:  { x: 274, y: 136, name: 'Ladakh',             capital: 'Leh',                  abbr: 'LA'  },
  HP:  { x: 281, y: 203, name: 'Himachal Pradesh',   capital: 'Shimla',               abbr: 'HP'  },
  PB:  { x: 247, y: 227, name: 'Punjab',             capital: 'Chandigarh',           abbr: 'PB'  },
  UT:  { x: 318, y: 244, name: 'Uttarakhand',        capital: 'Dehradun',             abbr: 'UT'  },
  HR:  { x: 262, y: 264, name: 'Haryana',            capital: 'Chandigarh',           abbr: 'HR'  },
  DL:  { x: 273, y: 271, name: 'Delhi',              capital: 'New Delhi',            abbr: 'DL'  },

  // ── West ──────────────────────────────────────────────────────────────────
  RJ:  { x: 213, y: 320, name: 'Rajasthan',          capital: 'Jaipur',               abbr: 'RJ'  },
  GJ:  { x: 165, y: 402, name: 'Gujarat',            capital: 'Gandhinagar',          abbr: 'GJ'  },

  // ── Central ───────────────────────────────────────────────────────────────
  UP:  { x: 346, y: 315, name: 'Uttar Pradesh',      capital: 'Lucknow',              abbr: 'UP'  },
  MP:  { x: 300, y: 390, name: 'Madhya Pradesh',     capital: 'Bhopal',               abbr: 'MP'  },
  CG:  { x: 375, y: 437, name: 'Chhattisgarh',       capital: 'Raipur',               abbr: 'CG'  },

  // ── East ──────────────────────────────────────────────────────────────────
  BR:  { x: 445, y: 341, name: 'Bihar',              capital: 'Patna',                abbr: 'BR'  },
  JH:  { x: 445, y: 385, name: 'Jharkhand',          capital: 'Ranchi',               abbr: 'JH'  },
  WB:  { x: 494, y: 381, name: 'West Bengal',        capital: 'Kolkata',              abbr: 'WB'  },
  OD:  { x: 424, y: 453, name: 'Odisha',             capital: 'Bhubaneswar',          abbr: 'OD'  },

  // ── Northeast ─────────────────────────────────────────────────────────────
  SK:  { x: 481, y: 282, name: 'Sikkim',             capital: 'Gangtok',              abbr: 'SK'  },
  AR:  { x: 619, y: 275, name: 'Arunachal Pradesh',  capital: 'Itanagar',             abbr: 'AR'  },
  AS:  { x: 588, y: 315, name: 'Assam',              capital: 'Dispur',               abbr: 'AS'  },
  NL:  { x: 619, y: 319, name: 'Nagaland',           capital: 'Kohima',               abbr: 'NL'  },
  ML:  { x: 557, y: 336, name: 'Meghalaya',          capital: 'Shillong',             abbr: 'ML'  },
  MN:  { x: 611, y: 348, name: 'Manipur',            capital: 'Imphal',               abbr: 'MN'  },
  TR:  { x: 571, y: 374, name: 'Tripura',            capital: 'Agartala',             abbr: 'TR'  },
  MZ:  { x: 593, y: 382, name: 'Mizoram',            capital: 'Aizawl',               abbr: 'MZ'  },

  // ── South ─────────────────────────────────────────────────────────────────
  MH:  { x: 252, y: 477, name: 'Maharashtra',        capital: 'Mumbai',               abbr: 'MH'  },
  TL:  { x: 313, y: 515, name: 'Telangana',          capital: 'Hyderabad',            abbr: 'TL'  },
  AP:  { x: 334, y: 560, name: 'Andhra Pradesh',     capital: 'Amaravati',            abbr: 'AP'  },
  GA:  { x: 220, y: 537, name: 'Goa',                capital: 'Panaji',               abbr: 'GA'  },
  KA:  { x: 251, y: 582, name: 'Karnataka',          capital: 'Bengaluru',            abbr: 'KA'  },
  TN:  { x: 299, y: 666, name: 'Tamil Nadu',         capital: 'Chennai',              abbr: 'TN'  },
  KL:  { x: 252, y: 677, name: 'Kerala',             capital: 'Thiruvananthapuram',   abbr: 'KL'  },

  // ── Union Territories ─────────────────────────────────────────────────────
  AN:  { x: 641, y: 523, name: 'Andaman & Nicobar',  capital: 'Port Blair',           abbr: 'AN'  },
};

// Water dispute connections — geopolitically accurate
const DISPUTE_LINKS: Array<{
  from: string; to: string;
  label: string;
  color: string;
  minCycle: number;
}> = [
  { from: 'KA', to: 'TN', label: 'Cauvery',        color: '#38bdf8', minCycle: 0  },
  { from: 'KA', to: 'AP', label: 'Krishna',         color: '#818cf8', minCycle: 20 },
  { from: 'MH', to: 'KA', label: 'Krishna (upper)', color: '#818cf8', minCycle: 20 },
  { from: 'PB', to: 'RJ', label: 'Ravi-Beas',       color: '#38bdf8', minCycle: 10 },
  { from: 'UP', to: 'BR', label: 'Ganga basin',     color: '#38bdf8', minCycle: 30 },
  { from: 'AP', to: 'TN', label: 'Krishna-lower',   color: '#818cf8', minCycle: 25 },
  { from: 'GJ', to: 'RJ', label: 'Narmada',         color: '#4ade80', minCycle: 15 },
];

const STATUS_CFG = {
  green: { dot: '#22c55e', ring: '#16a34a44', glow: '#22c55e' },
  amber: { dot: '#f59e0b', ring: '#d9770644', glow: '#f59e0b' },
  red:   { dot: '#ef4444', ring: '#dc262644', glow: '#ef4444' },
  black: { dot: '#6b7280', ring: '#ef444444', glow: '#ef4444' },
};

const IndiaMap: React.FC<IndiaMapProps> = ({ onStateClick, selectedState, simulationCycle = 0 }) => {
  const [states, setStates] = useState<StateData[]>(indiaStates);
  const [hovered, setHovered] = useState<string | null>(null);

  useEffect(() => {
    setStates(indiaStates.map(s => {
      let status = s.status;
      if (simulationCycle >= 40 && simulationCycle < 55) {
        if (s.id === 'RJ') status = 'red';
        if (simulationCycle >= 43 && s.id === 'UP') status = 'amber';
        if (simulationCycle >= 44 && s.id === 'MP') status = 'amber';
        if (simulationCycle >= 46 && s.id === 'HR') status = 'amber';
      }
      if (simulationCycle >= 52 && simulationCycle < 70) {
        if (s.id === 'KA') status = 'red';
        if (simulationCycle >= 54 && s.id === 'TN') status = 'red';
        if (simulationCycle >= 55 && s.id === 'AP') status = 'amber';
      }
      return { ...s, status };
    }));
  }, [simulationCycle]);

  const focusId = hovered ?? selectedState;
  const focusState = focusId ? states.find(s => s.id === focusId) : null;
  const focusMarker = focusId ? STATE_MARKERS[focusId] : null;

  const activeLinks = DISPUTE_LINKS.filter(l => simulationCycle >= l.minCycle);

  // Label offset nudge to avoid dot overlap in dense regions
  const LABEL_OFFSETS: Record<string, [number, number]> = {
    DL:  [14, -10],
    HR:  [-14, -8],
    SK:  [14, -6],
    NL:  [16, 0],
    ML:  [-16, 6],
    MN:  [16, 4],
    TR:  [-16, 6],
    MZ:  [14, 6],
    LA:  [14, -8],
    AN:  [0, 16],
    GA:  [-14, 6],
    PB:  [-14, -6],
    UT:  [14, -6],
  };

  return (
    <div className="relative w-full h-full bg-[#060a12] rounded-xl overflow-hidden border border-white/5">
      {/* Ambient glow */}
      <div className="pointer-events-none absolute inset-0"
        style={{ background: 'radial-gradient(ellipse 70% 60% at 40% 45%, rgba(56,189,248,0.04) 0%, transparent 70%)' }} />

      {/* Header */}
      <div className="absolute top-0 left-0 right-0 z-10 flex items-center justify-between px-4 pt-3">
        <div>
          <p className="text-[11px] font-bold tracking-[0.16em] text-cyan-400/80 uppercase">
            India · Resource Status
          </p>
          <p className="text-[9px] text-gray-600 mt-0.5">Cycle {simulationCycle} / 100</p>
        </div>
        <div className="flex gap-3 items-center">
          {([['#22c55e', 'Stable'], ['#f59e0b', 'Stressed'], ['#ef4444', 'Critical']] as const).map(([c, l]) => (
            <div key={l} className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full" style={{ background: c, boxShadow: `0 0 5px ${c}` }} />
              <span className="text-[9px] text-gray-500">{l}</span>
            </div>
          ))}
        </div>
      </div>

      {/* ── SVG ─────────────────────────────────────────────────────────────── */}
      {/*
        viewBox="0 0 800 900" — portrait aspect matching the India map image.

        DEV TIP: Set DEV_CALIBRATE=true to enable click-to-log mode.
        Click anywhere on the map → browser console logs exact SVG x,y.
        Use those to fix any dot that's still misplaced in STATE_MARKERS.
        Set back to false before committing.
      */}
      <svg
        viewBox="0 0 800 900"
        className="w-full h-full"
        preserveAspectRatio="xMidYMid meet"
        onClick={(e) => {
          if (!DEV_CALIBRATE) return;
          const svg = e.currentTarget as SVGSVGElement;
          const pt = svg.createSVGPoint();
          pt.x = e.clientX; pt.y = e.clientY;
          const p = pt.matrixTransform(svg.getScreenCTM()!.inverse());
          console.log(`SVG coords → x: ${Math.round(p.x)}, y: ${Math.round(p.y)}`);
        }}
      >
        <defs>
          <filter id="im-glow">
            <feGaussianBlur stdDeviation="3" result="b" />
            <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <filter id="im-glow-sm">
            <feGaussianBlur stdDeviation="1.5" result="b" />
            <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <filter id="im-glow-lg">
            <feGaussianBlur stdDeviation="6" result="b" />
            <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>

          {/* Ocean dot pattern */}
          <pattern id="ocean-dots" x="0" y="0" width="14" height="14" patternUnits="userSpaceOnUse">
            <circle cx="7" cy="7" r="0.7" fill="rgba(56,189,248,0.12)" />
          </pattern>

          {/* Tint overlay to darken the white map image */}
          <filter id="map-tint" colorInterpolationFilters="sRGB">
            <feColorMatrix type="matrix"
              values="0.05 0    0    0 0.04
                      0    0.12 0    0 0.15
                      0    0    0.22 0 0.22
                      0    0    0    1 0" />
          </filter>
        </defs>

        {/* Ocean background */}
        <rect x="0" y="0" width="800" height="900" fill="#040c18" />
        <rect x="0" y="0" width="800" height="900" fill="url(#ocean-dots)" />

        {/*
          ── INDIA MAP IMAGE ──────────────────────────────────────────────────
          The uploaded PNG is the authoritative India state map.
          We apply a color matrix filter to turn the white/black line-art into
          our dark-navy + cyan-border aesthetic so it blends with the UI.

          IMPORTANT: In your project, copy the uploaded PNG to your public folder
          and update INDIA_MAP_SRC to the correct path, e.g. '/india-map.png'.
          In Next.js: place in /public/india-map.png → src="/india-map.png"
          In Vite/CRA: place in /public/ → src="/india-map.png"
        */}
        {/*
          Image is 736×736 square. We render it as a perfect square 680×680
          centered in the viewport. y=70 because (900-680)/2 ≈ 110... but we
          want it near the top, so y=70 gives a small top margin.
          The calculated render position from Python: x=40, y=70, w=680, h=680
        */}
        <image
          href={INDIA_MAP_SRC}
          x="40"
          y="70"
          width="680"
          height="680"
          preserveAspectRatio="xMidYMid meet"
          filter="url(#map-tint)"
          opacity={0.9}
        />
        <image
          href={INDIA_MAP_SRC}
          x="40"
          y="70"
          width="680"
          height="680"
          preserveAspectRatio="xMidYMid meet"
          opacity={0.08}
          style={{ mixBlendMode: 'screen' }}
        />

        {/* ── DISPUTE CONNECTION LINES ── */}
        {activeLinks.map((link, i) => {
          const a = STATE_MARKERS[link.from];
          const b = STATE_MARKERS[link.to];
          if (!a || !b) return null;
          return (
            <motion.line key={`${link.from}-${link.to}`}
              x1={a.x} y1={a.y} x2={b.x} y2={b.y}
              stroke={link.color} strokeWidth={1} strokeDasharray="4 4"
              animate={{ strokeOpacity: [0.15, 0.65, 0.15] }}
              transition={{ duration: 2.5, repeat: Infinity, delay: i * 0.3 }}
            />
          );
        })}

        {/* ── STATE MARKERS ── */}
        {states.map(state => {
          const m = STATE_MARKERS[state.id];
          if (!m) return null;

          const cfg    = STATUS_CFG[state.status] ?? STATUS_CFG.green;
          const isH    = hovered === state.id;
          const isS    = selectedState === state.id;
          const isCrit = state.status === 'red' || state.status === 'black';
          const active = isH || isS;

          const [lox, loy] = LABEL_OFFSETS[state.id] ?? [0, 0];

          // Label position: when active, push label further out; default nudge
          const lx = active
            ? m.x + (lox !== 0 ? lox * 2.2 : 0)
            : m.x + (lox !== 0 ? lox : 10);
          const ly = active
            ? m.y + (loy !== 0 ? loy * 2.2 : -14)
            : m.y + (loy !== 0 ? loy : -10);

          return (
            <g key={state.id} style={{ cursor: 'pointer' }}
              onClick={() => onStateClick?.(state)}
              onMouseEnter={() => setHovered(state.id)}
              onMouseLeave={() => setHovered(null)}
            >
              {/* Leader line for nudged labels when active */}
              {active && (lox !== 0 || loy !== 0) && (
                <line
                  x1={m.x} y1={m.y}
                  x2={lx} y2={ly}
                  stroke={cfg.dot} strokeWidth={0.7} strokeOpacity={0.5}
                />
              )}

              {/* Pulse ring for critical states */}
              {isCrit && (
                <motion.circle cx={m.x} cy={m.y} r={10}
                  fill="none" stroke={cfg.glow} strokeWidth={1.2}
                  animate={{ r: [7, 16, 7], opacity: [0.7, 0, 0.7] }}
                  transition={{ duration: 1.8, repeat: Infinity }}
                />
              )}

              {/* Selection ring */}
              {isS && (
                <circle cx={m.x} cy={m.y} r={11}
                  fill="none" stroke="#38bdf8" strokeWidth={1.8}
                  filter="url(#im-glow-sm)"
                />
              )}

              {/* Hover ring */}
              {isH && !isS && (
                <circle cx={m.x} cy={m.y} r={10}
                  fill="none" stroke={cfg.dot} strokeWidth={1}
                  opacity={0.5}
                />
              )}

              {/* Main dot */}
              <motion.circle
                cx={m.x} cy={m.y}
                r={active ? 6 : isCrit ? 5.5 : 4.5}
                fill={cfg.dot}
                stroke={active ? '#fff' : 'rgba(0,0,0,0.5)'}
                strokeWidth={active ? 1.4 : 0.8}
                filter={active || isCrit ? 'url(#im-glow-sm)' : undefined}
                animate={isCrit && !active ? { r: [4, 6, 4] } : undefined}
                transition={{ duration: 1.5, repeat: Infinity }}
              />

              {/* State abbreviation label */}
              <text
                x={lx}
                y={ly}
                textAnchor={lox < 0 ? 'end' : lox > 0 ? 'start' : 'middle'}
                dominantBaseline="middle"
                fill={active ? '#fff' : 'rgba(255,255,255,0.75)'}
                fontSize={active ? 10 : 7.5}
                fontWeight={active ? '700' : '500'}
                style={{ pointerEvents: 'none', userSelect: 'none' }}
              >
                {state.id}
              </text>
            </g>
          );
        })}

        {/* Cascade arrows during drought scenario */}
        {simulationCycle >= 42 && simulationCycle < 52 && (
          <>
            <motion.path
              d={`M ${STATE_MARKERS.RJ.x},${STATE_MARKERS.RJ.y} C ${(STATE_MARKERS.RJ.x + STATE_MARKERS.MP.x) / 2},${STATE_MARKERS.RJ.y} ${(STATE_MARKERS.RJ.x + STATE_MARKERS.MP.x) / 2},${STATE_MARKERS.MP.y} ${STATE_MARKERS.MP.x},${STATE_MARKERS.MP.y}`}
              stroke="#ef4444" strokeWidth={1.4} fill="none" strokeDasharray="5 4"
              animate={{ opacity: [0, 0.9, 0] }}
              transition={{ duration: 2, repeat: Infinity }}
            />
            <motion.path
              d={`M ${STATE_MARKERS.MP.x},${STATE_MARKERS.MP.y} C ${(STATE_MARKERS.MP.x + STATE_MARKERS.UP.x) / 2},${(STATE_MARKERS.MP.y + STATE_MARKERS.UP.y) / 2} ${(STATE_MARKERS.MP.x + STATE_MARKERS.UP.x) / 2},${STATE_MARKERS.UP.y} ${STATE_MARKERS.UP.x},${STATE_MARKERS.UP.y}`}
              stroke="#f59e0b" strokeWidth={1.2} fill="none" strokeDasharray="5 4"
              animate={{ opacity: [0, 0.7, 0] }}
              transition={{ duration: 2, repeat: Infinity, delay: 0.5 }}
            />
          </>
        )}
      </svg>

      {/* ── Info Card ─────────────────────────────────────────────────────── */}
      <AnimatePresence>
        {focusState && focusMarker && (
          <motion.div
            className="absolute bottom-3 left-3 z-20 rounded-xl p-3"
            style={{
              minWidth: 190,
              background: 'rgba(6,10,18,0.97)',
              border: '1px solid rgba(255,255,255,0.08)',
              backdropFilter: 'blur(12px)',
            }}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 8 }}
          >
            <div className="flex items-center justify-between mb-2">
              <div>
                <p className="text-xs font-bold text-white leading-tight">{focusMarker.name}</p>
                <p className="text-[9px] text-gray-500 mt-0.5">🏛 {focusMarker.capital}</p>
              </div>
              <span
                className="text-[9px] px-1.5 py-0.5 rounded-full capitalize font-semibold ml-2"
                style={{
                  color: STATUS_CFG[focusState.status]?.dot,
                  background: `${STATUS_CFG[focusState.status]?.dot}20`,
                  border: `1px solid ${STATUS_CFG[focusState.status]?.dot}44`,
                }}
              >
                {focusState.status}
              </span>
            </div>
            {([
              ['Water',       focusState.resources.water,       '#38bdf8'],
              ['Power',       focusState.resources.power,       '#fbbf24'],
              ['Agriculture', focusState.resources.agriculture, '#4ade80'],
            ] as [string, number, string][]).map(([l, v, c]) => (
              <div key={l} className="mb-1.5">
                <div className="flex justify-between text-[9px] mb-0.5">
                  <span className="text-gray-500">{l}</span>
                  <span style={{ color: c }} className="font-mono font-bold">{v}%</span>
                </div>
                <div className="h-[3px] bg-gray-800/80 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full rounded-full"
                    style={{ background: c }}
                    initial={{ width: 0 }}
                    animate={{ width: `${v}%` }}
                    transition={{ duration: 0.45 }}
                  />
                </div>
              </div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Drought shock banner */}
      <AnimatePresence>
        {simulationCycle >= 42 && simulationCycle <= 47 && (
          <motion.div
            className="absolute top-12 left-1/2 -translate-x-1/2 z-20 whitespace-nowrap rounded-lg px-3 py-1.5"
            style={{
              background: 'rgba(120,15,15,0.92)',
              border: '1px solid rgba(239,68,68,0.6)',
            }}
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
          >
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-red-400 animate-ping inline-block" />
              <span className="text-[10px] font-semibold text-red-200">
                DROUGHT SHOCK — Monsoon deficit 42% · Cycle {simulationCycle}
              </span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default IndiaMap;
