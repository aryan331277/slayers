import { useEffect, useRef, useState } from 'react';
import { gsap } from 'gsap';
import { caseStudies } from '@/data/simulationData';
import type { CaseStudy } from '@/types';
import { AlertCircle, BookOpen, ExternalLink, TrendingUp, History } from 'lucide-react';

interface CaseStudyPanelProps {
  detectedPattern?: string;
}

const CaseStudyPanel = ({ detectedPattern }: CaseStudyPanelProps) => {
  const panelRef = useRef<HTMLDivElement>(null);
  const [selectedCase, setSelectedCase] = useState<CaseStudy>(caseStudies[0]);
  const [isDetected, setIsDetected] = useState(false);

  useEffect(() => {
    if (!panelRef.current) return;

    gsap.fromTo(
      panelRef.current,
      { scale: 0.95, opacity: 0 },
      { scale: 1, opacity: 1, duration: 0.6, ease: 'power2.out' }
    );
  }, []);

  // Simulate pattern detection
  useEffect(() => {
    if (detectedPattern) {
      const matchedCase = caseStudies.find((c) =>
        c.pattern.toLowerCase().includes(detectedPattern.toLowerCase())
      );
      if (matchedCase) {
        setSelectedCase(matchedCase);
        setIsDetected(true);

        // Flash animation
        if (panelRef.current) {
          gsap.fromTo(
            panelRef.current,
            { boxShadow: '0 0 0 rgba(255, 45, 85, 0)' },
            {
              boxShadow: '0 0 30px rgba(255, 45, 85, 0.5)',
              duration: 0.5,
              yoyo: true,
              repeat: 3,
            }
          );
        }
      }
    }
  }, [detectedPattern]);

  return (
    <div ref={panelRef} className="w-full h-full flex flex-col">
      {/* Detection alert */}
      {isDetected && (
        <div className="mb-4 p-3 bg-[#FF2D55]/10 border border-[#FF2D55]/30 rounded-lg flex items-center gap-3 animate-pulse-critical">
          <AlertCircle className="w-5 h-5 text-[#FF2D55]" />
          <div>
            <div className="text-sm font-semibold text-[#FF2D55]">Pattern Detected</div>
            <div className="text-xs text-[#A7B1C8]">Historical parallel identified</div>
          </div>
        </div>
      )}

      {/* Case selector */}
      <div className="flex items-center gap-2 mb-4 overflow-x-auto pb-2">
        {caseStudies.map((caseStudy) => (
          <button
            key={caseStudy.pattern}
            onClick={() => {
              setSelectedCase(caseStudy);
              setIsDetected(false);
            }}
            className={`
              px-3 py-2 text-xs rounded-lg whitespace-nowrap transition-all duration-300
              ${selectedCase.pattern === caseStudy.pattern
                ? 'bg-[#00F0FF]/20 text-[#00F0FF] border border-[#00F0FF]/50'
                : 'bg-[#0B1022] text-[#A7B1C8] border border-[#00F0FF]/20 hover:border-[#00F0FF]/40'
              }
            `}
          >
            {caseStudy.pattern}
          </button>
        ))}
      </div>

      {/* Case study card */}
      <div className="flex-1 panel p-5">
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <History className="w-4 h-4 text-[#00F0FF]" />
              <span className="text-xs text-[#00F0FF] font-medium">HISTORICAL PARALLEL</span>
            </div>
            <h3 className="text-xl font-bold text-[#F4F7FF]">{selectedCase.pattern}</h3>
          </div>
          <div className="text-right">
            <div className="text-2xl font-bold font-mono text-[#FFB800]">{selectedCase.year}</div>
            <div className="text-xs text-[#A7B1C8]">Year</div>
          </div>
        </div>

        {/* Description */}
        <div className="mb-4">
          <p className="text-sm text-[#A7B1C8] leading-relaxed">{selectedCase.description}</p>
        </div>

        {/* Historical parallel */}
        <div className="p-4 bg-[#070A12] rounded-lg border border-[#00F0FF]/20 mb-4">
          <div className="flex items-center gap-2 mb-2">
            <BookOpen className="w-4 h-4 text-[#00E08A]" />
            <span className="text-xs text-[#00E08A] font-medium">REAL-WORLD EVENT</span>
          </div>
          <p className="text-sm text-[#F4F7FF] leading-relaxed">{selectedCase.historicalParallel}</p>
        </div>

        {/* Comparison chart */}
        <div className="mb-4">
          <div className="flex items-center gap-2 mb-3">
            <TrendingUp className="w-4 h-4 text-[#FFB800]" />
            <span className="text-xs text-[#FFB800] font-medium">SIMULATION VS HISTORY</span>
          </div>
          <div className="h-24 bg-[#070A12] rounded-lg p-3 relative overflow-hidden">
            {/* Simulated comparison curves */}
            <svg viewBox="0 0 300 80" className="w-full h-full">
              {/* Grid */}
              <g opacity="0.2">
                {[0, 20, 40, 60, 80].map((y) => (
                  <line key={y} x1="0" y1={y} x2="300" y2={y} stroke="#00F0FF" strokeWidth="0.5" />
                ))}
              </g>
              {/* Historical curve */}
              <path
                d="M0,60 Q50,55 100,45 T200,35 T300,25"
                fill="none"
                stroke="#A7B1C8"
                strokeWidth="2"
                strokeDasharray="4,4"
                opacity="0.6"
              />
              {/* Simulation curve */}
              <path
                d="M0,65 Q50,58 100,48 T200,38 T300,28"
                fill="none"
                stroke="#00F0FF"
                strokeWidth="2"
              />
              {/* Data points */}
              <circle cx="100" cy="48" r="3" fill="#00F0FF" />
              <circle cx="200" cy="38" r="3" fill="#00F0FF" />
              <circle cx="300" cy="28" r="3" fill="#00F0FF" />
            </svg>
            {/* Legend */}
            <div className="absolute bottom-2 right-2 flex items-center gap-3 text-[10px]">
              <div className="flex items-center gap-1">
                <div className="w-3 h-0.5 bg-[#00F0FF]" />
                <span className="text-[#A7B1C8]">Simulation</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-3 h-0.5 bg-[#A7B1C8]" style={{ background: 'repeating-linear-gradient(90deg, #A7B1C8, #A7B1C8 2px, transparent 2px, transparent 4px)' }} />
                <span className="text-[#A7B1C8]">Historical</span>
              </div>
            </div>
          </div>
        </div>

        {/* Citation */}
        <div className="flex items-center justify-between pt-3 border-t border-[#00F0FF]/10">
          <div className="flex items-center gap-2">
            <span className="text-xs text-[#A7B1C8]">Source:</span>
            <span className="text-xs text-[#00F0FF]">{selectedCase.citation}</span>
          </div>
          <button className="flex items-center gap-1 text-xs text-[#00F0FF] hover:text-[#00E08A] transition-colors">
            <span>View Details</span>
            <ExternalLink className="w-3 h-3" />
          </button>
        </div>
      </div>

      {/* Bottom note */}
      <div className="mt-3 text-center">
        <p className="text-xs text-[#A7B1C8] italic">
          &ldquo;Nobody programmed this parallel. It <span className="text-[#00F0FF]">emerged</span>.&rdquo;
        </p>
      </div>
    </div>
  );
};

export default CaseStudyPanel;
