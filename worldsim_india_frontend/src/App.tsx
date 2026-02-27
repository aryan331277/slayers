import { useState, useEffect, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Play, Pause, RotateCcw, FastForward } from 'lucide-react';

// Components
import IndiaMap from '@/components/map/IndiaMap';
import TradeNetwork from '@/components/network/TradeNetwork';
import ConflictMatrix from '@/components/matrix/ConflictMatrix';
import AgentAttention from '@/components/network/AgentAttention';
import ResourceHistory from '@/components/charts/ResourceHistory';
import StrategyHeatmap from '@/components/charts/StrategyHeatmap';
import EventFeed from '@/components/feed/EventFeed';
import CaseStudyPanel from '@/components/panel/CaseStudyPanel';
import ScenarioInput from '@/components/ScenarioInput';

import type { StateData } from '@/types';

function App() {
  const [simulationCycle, setSimulationCycle] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [selectedState, setSelectedState] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'simulation' | 'strategy'>('simulation');

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isPlaying) {
      interval = setInterval(() => {
        setSimulationCycle(prev => {
          if (prev >= 100) { setIsPlaying(false); return 100; }
          return prev + 1;
        });
      }, 1000 / speed);
    }
    return () => clearInterval(interval);
  }, [isPlaying, speed]);

  const handlePlayPause = () => setIsPlaying(!isPlaying);
  const handleReset = () => { setIsPlaying(false); setSimulationCycle(0); setSelectedState(null); };
  const handleFastForward = () => setSpeed(prev => Math.min(prev + 1, 5));

  const handleStateClick = useCallback((state: StateData) => {
    setSelectedState(state.id);
  }, []);

  const handleEventClick = useCallback((cycle: number) => {
    setSimulationCycle(cycle);
  }, []);

  const handleScenarioSubmit = useCallback((_scenario: string, affectedStates: string[]) => {
    setSimulationCycle(45);
    if (affectedStates.length > 0) setSelectedState(affectedStates[0]);
  }, []);

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white flex flex-col">
      {/* Header */}
      <header className="border-b border-gray-800 bg-[#0f0f15] flex-shrink-0">
        <div className="px-5 py-3">
          <div className="flex justify-between items-center gap-4">
            {/* Logo + Title */}
            <div className="flex items-center gap-3 flex-shrink-0">
              <div className="w-9 h-9 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-lg flex items-center justify-center">
                <span className="text-lg font-bold text-black">M</span>
              </div>
              <div>
                <h1 className="text-xl font-bold neon-text leading-tight">MAPPO</h1>
                <p className="text-[10px] text-gray-400 leading-tight">Multi-Agent Proximal Policy Optimization</p>
              </div>
            </div>

            {/* Tabs */}
            <div className="flex bg-[#1a1a25] rounded-lg p-1">
              {(['simulation', 'strategy'] as const).map(tab => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors capitalize ${
                    activeTab === tab ? 'bg-cyan-500 text-black' : 'text-gray-400 hover:text-white'
                  }`}
                >
                  {tab === 'simulation' ? 'Live Simulation' : 'Strategy Generator'}
                </button>
              ))}
            </div>

            {/* Controls */}
            {activeTab === 'simulation' && (
              <div className="flex items-center gap-2 flex-shrink-0">
                <div className="flex items-center gap-2 bg-[#1a1a25] rounded-lg px-3 py-1.5">
                  <span className="text-xs text-gray-400">Cycle</span>
                  <span className="text-base font-mono font-bold text-cyan-400 min-w-[2.5rem] text-right">{simulationCycle}</span>
                </div>
                <div className="flex items-center gap-1">
                  <Button onClick={handleReset} variant="outline" size="icon" className="border-gray-700 hover:bg-gray-800 h-8 w-8">
                    <RotateCcw className="w-3.5 h-3.5" />
                  </Button>
                  <Button
                    onClick={handlePlayPause}
                    size="sm"
                    className={`${isPlaying ? 'bg-red-500 hover:bg-red-600' : 'bg-cyan-500 hover:bg-cyan-600'} text-black px-3`}
                  >
                    {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                  </Button>
                  <Button onClick={handleFastForward} variant="outline" size="sm" className="border-gray-700 hover:bg-gray-800">
                    <FastForward className="w-3.5 h-3.5" />
                    <span className="ml-1 text-xs">{speed}x</span>
                  </Button>
                </div>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 p-4 overflow-auto">
        {activeTab === 'simulation' ? (
          <div className="grid grid-cols-12 gap-3 auto-rows-min">
            {/* Row 1: Map + Network */}
            <div className="col-span-12 lg:col-span-6" style={{ height: '440px' }}>
              <IndiaMap
                onStateClick={handleStateClick}
                selectedState={selectedState}
                simulationCycle={simulationCycle}
              />
            </div>
            <div className="col-span-12 lg:col-span-6" style={{ height: '440px' }}>
              <TradeNetwork simulationCycle={simulationCycle} />
            </div>

            {/* Row 2: Conflict Matrix + Agent Attention */}
            <div className="col-span-12 lg:col-span-6" style={{ height: '380px' }}>
              <ConflictMatrix simulationCycle={simulationCycle} />
            </div>
            <div className="col-span-12 lg:col-span-6" style={{ height: '380px' }}>
              <AgentAttention
                selectedAgent={selectedState || 'GJ'}
                simulationCycle={simulationCycle}
              />
            </div>

            {/* Row 3: Resource History + Strategy Heatmap */}
            <div className="col-span-12 lg:col-span-6" style={{ height: '460px' }}>
              <ResourceHistory
                selectedState={selectedState || 'RJ'}
                simulationCycle={simulationCycle}
              />
            </div>
            <div className="col-span-12 lg:col-span-6" style={{ height: '460px' }}>
              <StrategyHeatmap />
            </div>

            {/* Row 4: Event Feed + Case Study */}
            <div className="col-span-12 lg:col-span-6" style={{ height: '380px' }}>
              <EventFeed
                simulationCycle={simulationCycle}
                onEventClick={handleEventClick}
              />
            </div>
            <div className="col-span-12 lg:col-span-6" style={{ height: '380px' }}>
              <CaseStudyPanel simulationCycle={simulationCycle} />
            </div>
          </div>
        ) : (
          <div className="max-w-6xl mx-auto">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
              <div style={{ height: '680px' }}>
                <ScenarioInput onScenarioSubmit={handleScenarioSubmit} />
              </div>
              <div className="space-y-4">
                <div className="bg-[#12121a] rounded-xl p-5 border border-gray-800">
                  <h3 className="text-base font-semibold text-white mb-4">System Capabilities</h3>
                  <div className="grid grid-cols-2 gap-3">
                    {[
                      { value: '28', label: 'Indian States', color: 'text-cyan-400' },
                      { value: '150+', label: 'Trade Connections', color: 'text-green-400' },
                      { value: '45', label: 'Action Types', color: 'text-yellow-400' },
                      { value: '10K+', label: 'Training Episodes', color: 'text-red-400' },
                    ].map(({ value, label, color }) => (
                      <div key={label} className="bg-[#0a0a0f] rounded-lg p-3">
                        <div className={`text-2xl font-bold ${color}`}>{value}</div>
                        <div className="text-xs text-gray-400 mt-0.5">{label}</div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-[#12121a] rounded-xl p-5 border border-gray-800">
                  <h3 className="text-base font-semibold text-white mb-3">How It Works</h3>
                  <div className="space-y-2.5">
                    {[
                      'Describe a scenario or select from quick options',
                      'MAPPO agents analyze and simulate outcomes',
                      'Receive comprehensive resource allocation strategy',
                      'Export or print for implementation',
                    ].map((step, i) => (
                      <div key={i} className="flex items-start gap-3">
                        <div className="w-5 h-5 bg-cyan-500/20 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                          <span className="text-xs text-cyan-400">{i + 1}</span>
                        </div>
                        <p className="text-sm text-gray-400">{step}</p>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-gradient-to-r from-cyan-900/30 to-blue-900/30 rounded-xl p-5 border border-cyan-500/30">
                  <h3 className="text-base font-semibold text-cyan-400 mb-2">Example Queries</h3>
                  <ul className="space-y-1.5 text-sm text-gray-300">
                    <li>• "Maharashtra Region is facing droughts"</li>
                    <li>• "Karnataka-Tamil Nadu water dispute escalation"</li>
                    <li>• "Punjab agricultural crisis due to groundwater depletion"</li>
                    <li>• "North India monsoon failure affecting Rajasthan"</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-800 bg-[#0f0f15] flex-shrink-0">
        <div className="px-5 py-2.5">
          <div className="flex justify-between items-center text-xs text-gray-500">
            <div>MAPPO · Multi-Agent Reinforcement Learning for Interstate Resource Allocation</div>
            <div className="flex gap-4">
              <span>Cycle: {simulationCycle}/100</span>
              <span className={isPlaying ? 'text-green-400' : 'text-gray-500'}>
                {isPlaying ? '● Running' : '○ Paused'}
              </span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
