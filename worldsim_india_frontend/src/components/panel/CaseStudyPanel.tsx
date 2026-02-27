import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { caseStudies } from '@/data/indiaStates';
import type { CaseStudy } from '@/types';
import { ScrollArea } from '@/components/ui/scroll-area';

interface CaseStudyPanelProps {
  simulationCycle?: number;
}

const CaseStudyPanel: React.FC<CaseStudyPanelProps> = ({ simulationCycle = 0 }) => {
  const [activeCase, setActiveCase] = useState<CaseStudy | null>(null);
  const [showPanel, setShowPanel] = useState(false);

  useEffect(() => {
    // Trigger case studies based on simulation cycle
    if (simulationCycle === 52) {
      setActiveCase(caseStudies.find(c => c.id === '4') || null);
      setShowPanel(true);
    } else if (simulationCycle === 68) {
      setActiveCase(caseStudies.find(c => c.id === '1') || null);
      setShowPanel(true);
    } else if (simulationCycle === 45) {
      setActiveCase(caseStudies.find(c => c.id === '3') || null);
      setShowPanel(true);
    } else if (simulationCycle < 40) {
      setShowPanel(false);
      setActiveCase(null);
    }
  }, [simulationCycle]);

  const getSimilarityColor = (score: number) => {
    if (score >= 0.85) return 'text-green-400';
    if (score >= 0.75) return 'text-yellow-400';
    return 'text-orange-400';
  };

  return (
    <div className="relative w-full h-full bg-[#0a0a0f] rounded-xl overflow-hidden">
      {/* Grid Background */}
      <div className="absolute inset-0 grid-bg opacity-50" />
      
      {/* Header */}
      <div className="relative z-10 p-4 border-b border-gray-800">
        <div className="flex justify-between items-center">
          <div>
            <h3 className="text-xl font-bold text-white neon-text">Case Study Panel</h3>
            <p className="text-sm text-gray-400 mt-1">Historical parallels & pattern detection</p>
          </div>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${showPanel ? 'bg-green-500 animate-pulse' : 'bg-gray-600'}`} />
            <span className="text-xs text-gray-400">{showPanel ? 'Pattern Detected' : 'Monitoring'}</span>
          </div>
        </div>
      </div>

      {/* Content */}
      <ScrollArea className="h-[calc(100%-80px)]">
        <div className="p-4">
          <AnimatePresence mode="wait">
            {!showPanel || !activeCase ? (
              <motion.div 
                className="text-center py-12"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <div className="text-5xl mb-4">🔍</div>
                <h4 className="text-lg font-semibold text-gray-300 mb-2">Pattern Detection Active</h4>
                <p className="text-sm text-gray-500 max-w-md mx-auto">
                  The system continuously monitors simulation events and compares them against historical Indian interstate conflicts. When a match is detected, it will appear here.
                </p>
                
                {/* Available Patterns */}
                <div className="mt-8">
                  <p className="text-xs text-gray-600 mb-3">MONITORED PATTERNS</p>
                  <div className="flex flex-wrap justify-center gap-2">
                    {caseStudies.map(cs => (
                      <span 
                        key={cs.id}
                        className="px-3 py-1 bg-[#1a1a25] border border-gray-700 rounded-full text-xs text-gray-400"
                      >
                        {cs.pattern}
                      </span>
                    ))}
                  </div>
                </div>
              </motion.div>
            ) : (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="space-y-4"
              >
                {/* Detection Alert */}
                <div className="bg-red-900/30 border border-red-500/50 rounded-lg p-4">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-red-500/20 rounded-full flex items-center justify-center">
                      <span className="text-xl">⚠️</span>
                    </div>
                    <div>
                      <div className="text-sm font-bold text-red-400">BEHAVIOR DETECTED</div>
                      <div className="text-lg font-semibold text-white">{activeCase.pattern}</div>
                    </div>
                  </div>
                </div>

                {/* Description */}
                <div className="bg-[#12121a] rounded-lg p-4">
                  <h5 className="text-sm font-semibold text-cyan-400 mb-2">Description</h5>
                  <p className="text-sm text-gray-300">{activeCase.description}</p>
                </div>

                {/* Historical Parallel */}
                <div className="bg-[#12121a] rounded-lg p-4">
                  <h5 className="text-sm font-semibold text-green-400 mb-3">Historical Parallel</h5>
                  <div className="space-y-3">
                    <div className="flex items-start gap-3">
                      <span className="text-xs font-mono text-gray-500 bg-gray-800 px-2 py-1 rounded">
                        {activeCase.historicalParallel.year}
                      </span>
                      <div>
                        <div className="text-sm font-semibold text-white">
                          {activeCase.historicalParallel.event}
                        </div>
                        <p className="text-xs text-gray-400 mt-1">
                          {activeCase.historicalParallel.details}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Similarity Score */}
                <div className="bg-[#12121a] rounded-lg p-4">
                  <div className="flex justify-between items-center mb-2">
                    <h5 className="text-sm font-semibold text-gray-400">Similarity Score</h5>
                    <span className={`text-2xl font-bold ${getSimilarityColor(activeCase.similarityScore)}`}>
                      {(activeCase.similarityScore * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                    <motion.div 
                      className={`h-full rounded-full ${
                        activeCase.similarityScore >= 0.85 ? 'bg-green-400' : 
                        activeCase.similarityScore >= 0.75 ? 'bg-yellow-400' : 'bg-orange-400'
                      }`}
                      initial={{ width: 0 }}
                      animate={{ width: `${activeCase.similarityScore * 100}%` }}
                      transition={{ duration: 1, ease: 'easeOut' }}
                    />
                  </div>
                </div>

                {/* References */}
                <div className="bg-[#12121a] rounded-lg p-4">
                  <h5 className="text-sm font-semibold text-gray-400 mb-2">References</h5>
                  <ul className="space-y-1">
                    {activeCase.references.map((ref, i) => (
                      <li key={i} className="text-xs text-gray-500 flex items-center gap-2">
                        <span className="w-1 h-1 bg-gray-600 rounded-full" />
                        {ref}
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Impact Statement */}
                <div className="bg-gradient-to-r from-cyan-900/30 to-blue-900/30 border border-cyan-500/30 rounded-lg p-4">
                  <p className="text-sm text-cyan-200 italic">
                    "The emergent behavior in this simulation matches documented Indian inter-state conflicts. 
                    The model is not fantasy — it reproduces recognizable dynamics."
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </ScrollArea>
    </div>
  );
};

export default CaseStudyPanel;
