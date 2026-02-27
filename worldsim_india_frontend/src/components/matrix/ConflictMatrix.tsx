import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { conflictProbabilities } from '@/data/indiaStates';

interface ConflictMatrixProps {
  simulationCycle?: number;
}

const ConflictMatrix: React.FC<ConflictMatrixProps> = ({ simulationCycle = 0 }) => {
  const [probabilities, setProbabilities] = useState(conflictProbabilities);
  const [hoveredCell, setHoveredCell] = useState<{i: number, j: number} | null>(null);

  const stateIds = ['KA', 'TN', 'PB', 'HR', 'UP', 'BR', 'RJ', 'MH', 'GJ', 'MP'];
  const stateNames: Record<string, string> = {
    'KA': 'Karnataka', 'TN': 'Tamil Nadu', 'PB': 'Punjab', 'HR': 'Haryana',
    'UP': 'Uttar Pradesh', 'BR': 'Bihar', 'RJ': 'Rajasthan', 'MH': 'Maharashtra',
    'GJ': 'Gujarat', 'MP': 'Madhya Pradesh'
  };

  useEffect(() => {
    // Simulate probability changes based on cycle
    const updated = conflictProbabilities.map(cp => {
      let newProb = cp.probability;
      
      if (cp.state1 === 'KA' && cp.state2 === 'TN') {
        if (simulationCycle > 50) newProb = Math.min(0.95, cp.probability + 0.1);
        if (simulationCycle > 55) newProb = Math.min(0.98, cp.probability + 0.15);
      }
      
      if (cp.state1 === 'UP' && cp.state2 === 'BR') {
        if (simulationCycle > 48) newProb = Math.min(0.85, cp.probability + 0.1);
      }
      
      return { ...cp, probability: newProb };
    });
    
    setProbabilities(updated);
  }, [simulationCycle]);

  const getProbability = (state1: string, state2: string) => {
    const prob = probabilities.find(
      p => (p.state1 === state1 && p.state2 === state2) || (p.state1 === state2 && p.state2 === state1)
    );
    return prob?.probability || 0;
  };

  const getCellColor = (probability: number) => {
    if (probability < 0.3) return 'rgba(255, 255, 255, 0.1)';
    if (probability < 0.5) return 'rgba(247, 198, 0, 0.3)';
    if (probability < 0.7) return 'rgba(255, 107, 53, 0.5)';
    return 'rgba(255, 51, 51, 0.8)';
  };

  const getCellGlow = (probability: number) => {
    if (probability >= 0.8) return '0 0 15px rgba(255, 51, 51, 0.8)';
    if (probability >= 0.6) return '0 0 10px rgba(255, 107, 53, 0.5)';
    return 'none';
  };

  const isPulsing = (probability: number) => probability >= 0.8;

  return (
    <div className="relative w-full h-full bg-[#0a0a0f] rounded-xl overflow-hidden p-4">
      {/* Grid Background */}
      <div className="absolute inset-0 grid-bg opacity-50" />
      
      {/* Title */}
      <div className="relative z-10 mb-4">
        <h3 className="text-xl font-bold text-white neon-text">Conflict Probability Matrix</h3>
        <p className="text-sm text-gray-400 mt-1">Inter-state escalation likelihood</p>
      </div>

      {/* Matrix Grid */}
      <div className="relative z-10 overflow-auto">
        <div className="inline-block">
          {/* Header Row */}
          <div className="flex">
            <div className="w-24 h-10" /> {/* Corner cell */}
            {stateIds.map(id => (
              <div 
                key={id} 
                className="w-14 h-10 flex items-center justify-center text-xs font-semibold text-gray-400"
              >
                {id}
              </div>
            ))}
          </div>
          
          {/* Matrix Rows */}
          {stateIds.map((rowId, i) => (
            <div key={rowId} className="flex">
              {/* Row Header */}
              <div className="w-24 h-14 flex items-center justify-center text-xs font-semibold text-gray-400">
                {rowId}
              </div>
              
              {/* Cells */}
              {stateIds.map((colId, j) => {
                const probability = getProbability(rowId, colId);
                const isDiagonal = i === j;
                
                return (
                  <motion.div
                    key={`${rowId}-${colId}`}
                    className="w-14 h-14 flex items-center justify-center relative"
                    style={{
                      background: isDiagonal ? '#1a1a1a' : getCellColor(probability),
                      boxShadow: isDiagonal ? 'none' : getCellGlow(probability),
                      border: '1px solid rgba(42, 42, 58, 0.5)',
                    }}
                    animate={isPulsing(probability) && !isDiagonal ? {
                      opacity: [0.7, 1, 0.7],
                    } : {}}
                    transition={{
                      duration: 1.5,
                      repeat: Infinity,
                      ease: 'easeInOut',
                    }}
                    onMouseEnter={() => setHoveredCell({i, j})}
                    onMouseLeave={() => setHoveredCell(null)}
                  >
                    {!isDiagonal && (
                      <span className={`text-xs font-bold ${probability > 0.5 ? 'text-white' : 'text-gray-400'}`}>
                        {(probability * 100).toFixed(0)}%
                      </span>
                    )}
                    
                    {/* Diagonal indicator */}
                    {isDiagonal && (
                      <div className="w-2 h-2 bg-gray-600 rounded-full" />
                    )}
                  </motion.div>
                );
              })}
            </div>
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="relative z-10 mt-4 flex items-center gap-4">
        <span className="text-xs text-gray-400">Probability:</span>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded" style={{ background: 'rgba(255, 255, 255, 0.1)' }} />
          <span className="text-xs text-gray-500">&lt;30%</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded" style={{ background: 'rgba(247, 198, 0, 0.3)' }} />
          <span className="text-xs text-gray-500">30-50%</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded" style={{ background: 'rgba(255, 107, 53, 0.5)' }} />
          <span className="text-xs text-gray-500">50-70%</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded animate-pulse" style={{ background: 'rgba(255, 51, 51, 0.8)' }} />
          <span className="text-xs text-gray-500">&gt;70%</span>
        </div>
      </div>

      {/* Tooltip */}
      {hoveredCell && hoveredCell.i !== hoveredCell.j && (
        <div className="absolute bottom-4 right-4 z-20 glass rounded-lg p-3">
          <div className="text-sm font-bold text-white">
            {stateNames[stateIds[hoveredCell.i]]} ↔ {stateNames[stateIds[hoveredCell.j]]}
          </div>
          <div className="text-xs text-gray-400 mt-1">
            Conflict Probability: <span className="text-red-400 font-bold">
              {(getProbability(stateIds[hoveredCell.i], stateIds[hoveredCell.j]) * 100).toFixed(1)}%
            </span>
          </div>
          <div className="text-xs text-gray-500 mt-1">
            {getProbability(stateIds[hoveredCell.i], stateIds[hoveredCell.j]) > 0.7 
              ? '⚠️ High risk of escalation' 
              : getProbability(stateIds[hoveredCell.i], stateIds[hoveredCell.j]) > 0.5 
                ? '⚡ Elevated tensions' 
                : '✓ Relatively stable'}
          </div>
        </div>
      )}

      {/* Alert for high conflict */}
      {simulationCycle > 50 && (
        <motion.div 
          className="absolute top-16 right-4 z-20 bg-red-900/80 border border-red-500 rounded-lg p-2"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
        >
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-red-500 rounded-full animate-ping" />
            <span className="text-xs font-semibold text-red-200">KA-TN: 85%</span>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default ConflictMatrix;
