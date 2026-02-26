import React, { useEffect, useRef, useState } from 'react';
import { gsap } from 'gsap';

interface StrategyHeatmapProps {
  agentId?: string;
}

const StrategyHeatmap: React.FC<StrategyHeatmapProps> = ({ agentId = 'saudi' }) => {
  const heatmapRef = useRef<HTMLDivElement>(null);
  const [selectedPhase, setSelectedPhase] = useState<'early' | 'late'>('late');

  // Generate heatmap data
  const generateHeatmapData = (phase: 'early' | 'late') => {
    const actions = ['Trade', 'Sanction', 'Raid', 'Cooperate', 'Defect', 'Invest', 'Stockpile', 'Negotiate'];
    const cycles = Array.from({ length: 20 }, (_, i) => i + 1);

    return cycles.map((cycle) =>
      actions.map((action) => {
        if (phase === 'early') {
          // Early training: high entropy (random)
          return Math.random() * 0.4 + 0.1;
        } else {
          // Late training: concentrated patterns
          // Saudi focuses on energy leverage
          if (agentId === 'saudi' && action === 'Trade') {
            return cycle > 10 ? 0.8 + Math.random() * 0.2 : 0.3 + Math.random() * 0.3;
          }
          if (agentId === 'saudi' && action === 'Invest') {
            return cycle > 15 ? 0.7 + Math.random() * 0.2 : 0.2 + Math.random() * 0.2;
          }
          return Math.random() * 0.3;
        }
      })
    );
  };

  const heatmapData = generateHeatmapData(selectedPhase);
  const actions = ['Trade', 'Sanction', 'Raid', 'Cooperate', 'Defect', 'Invest', 'Stockpile', 'Negotiate'];

  useEffect(() => {
    if (!heatmapRef.current) return;

    const cells = heatmapRef.current.querySelectorAll('.heatmap-cell');
    gsap.fromTo(
      cells,
      { opacity: 0, scale: 0.9 },
      {
        opacity: 1,
        scale: 1,
        duration: 0.3,
        stagger: {
          each: 0.01,
          from: 'random',
        },
        ease: 'power2.out',
      }
    );
  }, [selectedPhase]);

  const getCellColor = (value: number): string => {
    if (value >= 0.8) return 'rgba(0, 240, 255, 0.95)';
    if (value >= 0.6) return 'rgba(0, 240, 255, 0.7)';
    if (value >= 0.4) return 'rgba(0, 240, 255, 0.45)';
    if (value >= 0.2) return 'rgba(0, 240, 255, 0.25)';
    return 'rgba(0, 240, 255, 0.08)';
  };

  const getCellIntensity = (value: number): string => {
    if (value >= 0.8) return 'Very High';
    if (value >= 0.6) return 'High';
    if (value >= 0.4) return 'Medium';
    if (value >= 0.2) return 'Low';
    return 'Very Low';
  };

  return (
    <div className="w-full h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-[#F4F7FF]">Strategy Heatmap</h3>
          <p className="text-xs text-[#A7B1C8] mt-1">
            {agentId.charAt(0).toUpperCase() + agentId.slice(1)} Agent - Action frequency over time
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setSelectedPhase('early')}
            className={`
              px-3 py-1 text-xs rounded-full transition-all duration-300
              ${selectedPhase === 'early'
                ? 'bg-[#FF2D55]/20 text-[#FF2D55] border border-[#FF2D55]/50'
                : 'bg-[#0B1022] text-[#A7B1C8] border border-[#00F0FF]/20 hover:border-[#00F0FF]/40'
              }
            `}
          >
            Early Training
          </button>
          <button
            onClick={() => setSelectedPhase('late')}
            className={`
              px-3 py-1 text-xs rounded-full transition-all duration-300
              ${selectedPhase === 'late'
                ? 'bg-[#00E08A]/20 text-[#00E08A] border border-[#00E08A]/50'
                : 'bg-[#0B1022] text-[#A7B1C8] border border-[#00F0FF]/20 hover:border-[#00F0FF]/40'
              }
            `}
          >
            Late Training
          </button>
        </div>
      </div>

      {/* Heatmap */}
      <div ref={heatmapRef} className="flex-1 overflow-auto">
        <div className="grid gap-1" style={{ gridTemplateColumns: '80px repeat(20, 1fr)' }}>
          {/* Header row */}
          <div className="p-2" />
          {Array.from({ length: 20 }, (_, i) => (
            <div
              key={`cycle-${i}`}
              className="p-1 text-xs text-[#A7B1C8] text-center"
            >
              C{i + 1}
            </div>
          ))}

          {/* Action rows */}
          {actions.map((action, actionIndex) => (
            <React.Fragment key={`action-${action}`}>
              {/* Action label */}
              <div className="p-2 text-xs font-medium text-[#A7B1C8] flex items-center">
                {action}
              </div>
              {/* Cells */}
              {heatmapData.map((cycleData, cycleIndex) => {
                const value = cycleData[actionIndex];
                return (
                  <div
                    key={`cell-${actionIndex}-${cycleIndex}`}
                    className="heatmap-cell aspect-square rounded flex items-center justify-center cursor-pointer transition-all duration-200 hover:scale-110 hover:z-10"
                    style={{ background: getCellColor(value) }}
                    title={`${action} at Cycle ${cycleIndex + 1}: ${getCellIntensity(value)} (${(value * 100).toFixed(0)}%)`}
                  >
                    {value >= 0.6 && (
                      <span className="text-[8px] font-mono text-[#070A12] font-bold">
                        {(value * 100).toFixed(0)}
                      </span>
                    )}
                  </div>
                );
              })}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="mt-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <span className="text-xs text-[#A7B1C8]">Intensity:</span>
          <div className="flex items-center gap-2">
            {[
              { color: 'rgba(0, 240, 255, 0.08)', label: '0%' },
              { color: 'rgba(0, 240, 255, 0.25)', label: '25%' },
              { color: 'rgba(0, 240, 255, 0.45)', label: '45%' },
              { color: 'rgba(0, 240, 255, 0.7)', label: '70%' },
              { color: 'rgba(0, 240, 255, 0.95)', label: '95%' },
            ].map((item) => (
              <div key={item.label} className="flex items-center gap-1">
                <div
                  className="w-4 h-4 rounded"
                  style={{ background: item.color }}
                />
                <span className="text-xs text-[#A7B1C8]">{item.label}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Strategy insight */}
        <div className="panel px-3 py-2">
          <div className="text-xs">
            <span className="text-[#00F0FF] font-semibold">Emergent Strategy:</span>{' '}
            <span className="text-[#A7B1C8]">
              {selectedPhase === 'early'
                ? 'High entropy - random exploration'
                : `${agentId.charAt(0).toUpperCase() + agentId.slice(1)} discovered energy leverage as viable path`
              }
            </span>
          </div>
        </div>
      </div>

      {/* Entropy comparison */}
      <div className="mt-3 grid grid-cols-2 gap-4">
        <div className="panel p-3">
          <div className="text-xs text-[#A7B1C8] mb-1">Entropy Level</div>
          <div className="flex items-center gap-2">
            <div className="flex-1 h-2 bg-[#0B1022] rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-[#00E08A] to-[#FF2D55] transition-all duration-500"
                style={{ width: selectedPhase === 'early' ? '85%' : '25%' }}
              />
            </div>
            <span className="text-xs font-mono text-[#F4F7FF]">
              {selectedPhase === 'early' ? 'High' : 'Low'}
            </span>
          </div>
        </div>
        <div className="panel p-3">
          <div className="text-xs text-[#A7B1C8] mb-1">Strategy Concentration</div>
          <div className="flex items-center gap-2">
            <div className="flex-1 h-2 bg-[#0B1022] rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-[#FF2D55] to-[#00E08A] transition-all duration-500"
                style={{ width: selectedPhase === 'early' ? '20%' : '90%' }}
              />
            </div>
            <span className="text-xs font-mono text-[#F4F7FF]">
              {selectedPhase === 'early' ? 'Low' : 'High'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StrategyHeatmap;
