import { useEffect, useRef, useState } from 'react';
import { gsap } from 'gsap';
import { conflictProbabilities, regions } from '@/data/simulationData';

interface ConflictMatrixProps {
  highlightedPair?: string | null;
}

const ConflictMatrix = ({ highlightedPair }: ConflictMatrixProps) => {
  const matrixRef = useRef<HTMLDivElement>(null);
  const [hoveredCell, setHoveredCell] = useState<string | null>(null);

  const regionIds = regions.slice(0, 8).map((r) => r.id);
  const regionNames = regions.slice(0, 8).map((r) => r.name);

  useEffect(() => {
    if (!matrixRef.current) return;

    const cells = matrixRef.current.querySelectorAll('.matrix-cell');
    gsap.fromTo(
      cells,
      { opacity: 0, scale: 0.8 },
      {
        opacity: 1,
        scale: 1,
        duration: 0.4,
        stagger: {
          each: 0.02,
          from: 'random',
        },
        ease: 'power2.out',
      }
    );
  }, []);

  const getProbability = (id1: string, id2: string): number => {
    const prob = conflictProbabilities.find(
      (p) =>
        (p.region1 === id1 && p.region2 === id2) ||
        (p.region1 === id2 && p.region2 === id1)
    );
    return prob?.probability || Math.random() * 30;
  };

  const getCellColor = (probability: number): string => {
    if (probability >= 80) return 'rgba(255, 45, 85, 0.9)';
    if (probability >= 60) return 'rgba(255, 45, 85, 0.6)';
    if (probability >= 40) return 'rgba(255, 184, 0, 0.7)';
    if (probability >= 20) return 'rgba(255, 184, 0, 0.4)';
    return 'rgba(0, 224, 138, 0.3)';
  };

  const getCellTextColor = (probability: number): string => {
    if (probability >= 60) return '#fff';
    if (probability >= 40) return '#fff';
    return '#A7B1C8';
  };

  return (
    <div className="w-full h-full flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-[#F4F7FF]">Conflict Probability Matrix</h3>
        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-[#00E08A] opacity-30" />
            <span className="text-[#A7B1C8]">Low</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-[#FFB800] opacity-50" />
            <span className="text-[#A7B1C8]">Medium</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-[#FF2D55] opacity-70" />
            <span className="text-[#A7B1C8]">High</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-[#FF2D55] opacity-90" />
            <span className="text-[#A7B1C8]">Critical</span>
          </div>
        </div>
      </div>

      <div ref={matrixRef} className="flex-1 overflow-auto">
        <div className="grid" style={{ gridTemplateColumns: `100px repeat(${regionIds.length}, 1fr)` }}>
          {/* Header row */}
          <div className="p-2" />
          {regionNames.map((name) => (
            <div
              key={`header-${name}`}
              className="p-2 text-xs font-medium text-[#A7B1C8] text-center truncate"
              title={name}
            >
              {name.slice(0, 3).toUpperCase()}
            </div>
          ))}

          {/* Matrix rows */}
          {regionIds.map((rowId, rowIndex) => (
            <div key={`row-${rowId}`} className="contents">
              {/* Row label */}
              <div className="p-2 text-xs font-medium text-[#A7B1C8] flex items-center">
                {regionNames[rowIndex]}
              </div>
              {/* Cells */}
              {regionIds.map((colId, colIndex) => {
                const probability = getProbability(rowId, colId);
                const cellId = `${rowId}-${colId}`;
                const isHighlighted = highlightedPair === cellId || highlightedPair === `${colId}-${rowId}`;
                const isHovered = hoveredCell === cellId;
                const isDiagonal = rowIndex === colIndex;

                return (
                  <div
                    key={cellId}
                    className={`matrix-cell p-1 ${isHighlighted ? 'animate-pulse-critical' : ''}`}
                    onMouseEnter={() => setHoveredCell(cellId)}
                    onMouseLeave={() => setHoveredCell(null)}
                  >
                    <div
                      className={`
                        w-full aspect-square rounded flex items-center justify-center
                        transition-all duration-300 cursor-pointer
                        ${isHovered ? 'scale-110 z-10 relative' : ''}
                      `}
                      style={{
                        background: isDiagonal
                          ? 'rgba(0, 0, 0, 0.5)'
                          : getCellColor(probability),
                        border: isHighlighted
                          ? '2px solid #FF2D55'
                          : isHovered
                          ? '1px solid #00F0FF'
                          : '1px solid transparent',
                      }}
                      title={`${regionNames[rowIndex]} ↔ ${regionNames[colIndex]}: ${probability.toFixed(1)}%`}
                    >
                      {!isDiagonal && (
                        <span
                          className="text-xs font-mono font-semibold"
                          style={{ color: getCellTextColor(probability) }}
                        >
                          {probability.toFixed(0)}%
                        </span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>

      {/* Hot pairs indicator */}
      <div className="mt-4 p-3 bg-[#0B1022] rounded-lg border border-[#FF2D55]/30">
        <div className="flex items-center gap-2 mb-2">
          <div className="w-2 h-2 rounded-full bg-[#FF2D55] animate-pulse" />
          <span className="text-sm font-medium text-[#FF2D55]">Critical Tensions</span>
        </div>
        <div className="space-y-1">
          {conflictProbabilities
            .filter((p) => p.probability >= 70)
            .slice(0, 3)
            .map((pair) => (
              <div key={`${pair.region1}-${pair.region2}`} className="flex justify-between text-xs">
                <span className="text-[#A7B1C8]">
                  {pair.region1.charAt(0).toUpperCase() + pair.region1.slice(1)} ↔{' '}
                  {pair.region2.charAt(0).toUpperCase() + pair.region2.slice(1)}
                </span>
                <span className="text-[#FF2D55] font-mono font-semibold">{pair.probability}%</span>
              </div>
            ))}
        </div>
      </div>
    </div>
  );
};

export default ConflictMatrix;
