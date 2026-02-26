import { useState, useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import {
  Globe,
  Network,
  AlertTriangle,
  Eye,
  TrendingUp,
  Grid3X3,
  Newspaper,
  BookOpen,
  MessageSquare,
  Play,
  Pause,
  RotateCcw,
} from 'lucide-react';

// Components
import WorldMap from '@/components/map/WorldMap';
import TradeNetwork from '@/components/visualizations/TradeNetwork';
import ConflictMatrix from '@/components/visualizations/ConflictMatrix';
import AttentionVisualizer from '@/components/visualizations/AttentionVisualizer';
import ResourceHistoryChart from '@/components/charts/ResourceHistoryChart';
import StrategyHeatmap from '@/components/visualizations/StrategyHeatmap';
import EventFeed from '@/components/panels/EventFeed';
import CaseStudyPanel from '@/components/panels/CaseStudyPanel';
import ScenarioInput from '@/components/panels/ScenarioInput';
import StrategyOutput from '@/components/panels/StrategyOutput';

// Data
import { regions, getStatusColor } from '@/data/simulationData';
import type { Region, AllocationStrategy, SimulationEvent } from '@/types';

function App() {
  const [activeTab, setActiveTab] = useState<string>('map');
  const [isSimulationRunning, setIsSimulationRunning] = useState(false);
  const [currentCycle, setCurrentCycle] = useState(47);
  const [selectedRegion, setSelectedRegion] = useState<Region | null>(null);
  const [generatedStrategy, setGeneratedStrategy] = useState<AllocationStrategy | null>(null);
  const [showStrategyModal, setShowStrategyModal] = useState(false);
  const [detectedPattern, setDetectedPattern] = useState<string | null>(null);
  const [pulseRegions, setPulseRegions] = useState<string[]>([]);
  const [brokenConnections, setBrokenConnections] = useState<string[]>([]);
  
  const headerRef = useRef<HTMLDivElement>(null);

  // Header animation on mount
  useEffect(() => {
    if (headerRef.current) {
      gsap.fromTo(
        headerRef.current,
        { y: -50, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.8, ease: 'power2.out' }
      );
    }
  }, []);

  // Simulation loop
  useEffect(() => {
    if (!isSimulationRunning) return;

    const interval = setInterval(() => {
      setCurrentCycle((prev) => prev + 1);
      
      // Randomly pulse critical regions
      if (Math.random() > 0.7) {
        const criticalRegions = regions
          .filter((r) => r.status === 'critical' || r.status === 'stressed')
          .map((r) => r.id);
        setPulseRegions(criticalRegions.slice(0, 2));
      }

      // Randomly break connections
      if (Math.random() > 0.9) {
        setBrokenConnections(['india-china']);
        setTimeout(() => setBrokenConnections([]), 3000);
      }

      // Detect patterns
      if (Math.random() > 0.95) {
        setDetectedPattern('Upstream Infrastructure Cutoff');
        setTimeout(() => setDetectedPattern(null), 5000);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [isSimulationRunning]);

  const tabs = [
    { id: 'map', label: 'World Map', icon: Globe },
    { id: 'network', label: 'Trade Network', icon: Network },
    { id: 'conflict', label: 'Conflict Matrix', icon: AlertTriangle },
    { id: 'attention', label: 'Agent Attention', icon: Eye },
    { id: 'charts', label: 'Resource History', icon: TrendingUp },
    { id: 'heatmap', label: 'Strategy Heatmap', icon: Grid3X3 },
    { id: 'events', label: 'Event Feed', icon: Newspaper },
    { id: 'cases', label: 'Case Studies', icon: BookOpen },
  ];

  const handleStrategyGenerated = (strategy: AllocationStrategy) => {
    setGeneratedStrategy(strategy);
    setShowStrategyModal(true);
  };

  const handleEventClick = (event: SimulationEvent) => {
    // Highlight related regions
    setPulseRegions(event.regions);
    setTimeout(() => setPulseRegions([]), 3000);
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'map':
        return (
          <div className="h-full flex flex-col lg:flex-row gap-4">
            <div className="flex-1 panel p-4">
              <WorldMap
                onRegionClick={setSelectedRegion}
                highlightedRegions={selectedRegion ? [selectedRegion.id] : []}
                showTradeRoutes={true}
                pulseRegions={pulseRegions}
              />
            </div>
            <div className="lg:w-80 space-y-4">
              <div className="panel p-4">
                <h3 className="text-sm font-semibold text-[#F4F7FF] mb-3">Region Status</h3>
                <div className="space-y-2">
                  {regions.slice(0, 6).map((region) => (
                    <div
                      key={region.id}
                      className="flex items-center justify-between p-2 rounded-lg bg-[#070A12] cursor-pointer hover:bg-[#00F0FF]/10 transition-colors"
                      onClick={() => setSelectedRegion(region)}
                    >
                      <div className="flex items-center gap-2">
                        <div
                          className="w-3 h-3 rounded-full"
                          style={{ background: getStatusColor(region.status) }}
                        />
                        <span className="text-sm text-[#F4F7FF]">{region.name}</span>
                      </div>
                      <span
                        className="text-xs px-2 py-0.5 rounded-full capitalize"
                        style={{
                          background: `${getStatusColor(region.status)}20`,
                          color: getStatusColor(region.status),
                        }}
                      >
                        {region.status}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
              {selectedRegion && (
                <div className="panel p-4">
                  <h3 className="text-sm font-semibold text-[#F4F7FF] mb-3">
                    {selectedRegion.name} Details
                  </h3>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-[#A7B1C8]">Water:</span>
                      <span className="text-[#00F0FF] font-mono">{selectedRegion.resources.water}%</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-[#A7B1C8]">Food:</span>
                      <span className="text-[#00E08A] font-mono">{selectedRegion.resources.food}%</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-[#A7B1C8]">Energy:</span>
                      <span className="text-[#FFB800] font-mono">{selectedRegion.resources.energy}%</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-[#A7B1C8]">Population:</span>
                      <span className="text-[#F4F7FF] font-mono">{selectedRegion.population}M</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        );

      case 'network':
        return (
          <div className="h-full flex flex-col lg:flex-row gap-4">
            <div className="flex-1 panel p-4">
              <TradeNetwork brokenConnections={brokenConnections} />
            </div>
            <div className="lg:w-80 space-y-4">
              <div className="panel p-4">
                <h3 className="text-sm font-semibold text-[#F4F7FF] mb-3">Network Insights</h3>
                <div className="space-y-3">
                  <div className="p-3 bg-[#070A12] rounded-lg">
                    <div className="text-xs text-[#A7B1C8] mb-1">Egypt's Food Dependency</div>
                    <div className="text-lg font-bold text-[#FF2D55]">40%</div>
                    <div className="text-xs text-[#A7B1C8]">Imported via trade</div>
                  </div>
                  <div className="p-3 bg-[#070A12] rounded-lg">
                    <div className="text-xs text-[#A7B1C8] mb-1">Ethiopia Water Cascade</div>
                    <div className="text-lg font-bold text-[#FFB800]">2-3 cycles</div>
                    <div className="text-xs text-[#A7B1C8]">To affect downstream</div>
                  </div>
                  <div className="p-3 bg-[#070A12] rounded-lg">
                    <div className="text-xs text-[#A7B1C8] mb-1">Isolated Regions</div>
                    <div className="text-lg font-bold text-[#00F0FF]">3x faster</div>
                    <div className="text-xs text-[#A7B1C8]">Collapse rate</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );

      case 'conflict':
        return (
          <div className="h-full flex flex-col lg:flex-row gap-4">
            <div className="flex-1 panel p-4">
              <ConflictMatrix />
            </div>
            <div className="lg:w-80">
              <div className="panel p-4">
                <h3 className="text-sm font-semibold text-[#F4F7FF] mb-3">Critical Tensions</h3>
                <div className="space-y-3">
                  <div className="p-3 bg-[#FF2D55]/10 border border-[#FF2D55]/30 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <AlertTriangle className="w-4 h-4 text-[#FF2D55]" />
                      <span className="text-sm font-semibold text-[#FF2D55]">Egypt ↔ Ethiopia</span>
                    </div>
                    <div className="text-2xl font-bold text-[#FF2D55]">78%</div>
                    <div className="text-xs text-[#A7B1C8]">Conflict probability</div>
                  </div>
                  <div className="p-3 bg-[#070A12] rounded-lg">
                    <div className="text-xs text-[#A7B1C8] mb-2">Judge Quote</div>
                    <p className="text-sm text-[#F4F7FF] italic">
                      "Anyone who knows the Grand Ethiopian Renaissance Dam crisis will go quiet."
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );

      case 'attention':
        return (
          <div className="h-full flex flex-col lg:flex-row gap-4">
            <div className="flex-1 panel p-4">
              <AttentionVisualizer
                agentId="india"
                attentionData={[
                  { target: 'china', intensity: 0.85, resource: 'water' },
                  { target: 'saudi', intensity: 0.72, resource: 'food' },
                  { target: 'germany', intensity: 0.25 },
                  { target: 'brazil', intensity: 0.15 },
                ]}
              />
            </div>
            <div className="lg:w-80">
              <div className="panel p-4">
                <h3 className="text-sm font-semibold text-[#F4F7FF] mb-3">Decision Context</h3>
                <div className="p-3 bg-[#00F0FF]/10 border border-[#00F0FF]/30 rounded-lg mb-3">
                  <div className="text-xs text-[#00F0FF] mb-1">Current Decision</div>
                  <div className="text-sm text-[#F4F7FF]">India → Trade offer to China</div>
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between p-2 bg-[#070A12] rounded-lg">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-[#00F0FF]" />
                      <span className="text-sm text-[#A7B1C8]">China Water Stress</span>
                    </div>
                    <span className="text-sm font-mono text-[#00F0FF]">85%</span>
                  </div>
                  <div className="flex items-center justify-between p-2 bg-[#070A12] rounded-lg">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-[#00E08A]" />
                      <span className="text-sm text-[#A7B1C8]">Saudi Food Surplus</span>
                    </div>
                    <span className="text-sm font-mono text-[#00E08A]">72%</span>
                  </div>
                  <div className="flex items-center justify-between p-2 bg-[#070A12] rounded-lg">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-[#A7B1C8]" />
                      <span className="text-sm text-[#A7B1C8]">Germany Demand</span>
                    </div>
                    <span className="text-sm font-mono text-[#A7B1C8]">25%</span>
                  </div>
                </div>
                <div className="mt-4 p-3 bg-[#070A12] rounded-lg">
                  <p className="text-sm text-[#F4F7FF] text-center">
                    "It's <span className="text-[#00F0FF]">actually thinking</span>."
                  </p>
                </div>
              </div>
            </div>
          </div>
        );

      case 'charts':
        return (
          <div className="h-full flex flex-col gap-4">
            <div className="flex-1 panel p-4">
              <ResourceHistoryChart regionId="egypt" />
            </div>
          </div>
        );

      case 'heatmap':
        return (
          <div className="h-full flex flex-col lg:flex-row gap-4">
            <div className="flex-1 panel p-4">
              <StrategyHeatmap agentId="saudi" />
            </div>
            <div className="lg:w-80">
              <div className="panel p-4">
                <h3 className="text-sm font-semibold text-[#F4F7FF] mb-3">Learning Progress</h3>
                <div className="space-y-3">
                  <div className="p-3 bg-[#070A12] rounded-lg">
                    <div className="text-xs text-[#A7B1C8] mb-1">Training Phase</div>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-2 bg-[#0B1022] rounded-full overflow-hidden">
                        <div className="h-full bg-gradient-to-r from-[#FF2D55] to-[#00E08A]" style={{ width: '75%' }} />
                      </div>
                      <span className="text-sm font-mono text-[#00E08A]">75%</span>
                    </div>
                  </div>
                  <div className="p-3 bg-[#00E08A]/10 border border-[#00E08A]/30 rounded-lg">
                    <div className="text-xs text-[#00E08A] mb-1">Emergent Strategy</div>
                    <div className="text-sm text-[#F4F7FF]">Energy Leverage</div>
                    <div className="text-xs text-[#A7B1C8]">Discovered at cycle 847</div>
                  </div>
                  <div className="p-3 bg-[#070A12] rounded-lg">
                    <div className="text-xs text-[#A7B1C8] mb-1">Quote</div>
                    <p className="text-sm text-[#F4F7FF] italic">
                      "Nobody told it that. It discovered it through survival pressure."
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );

      case 'events':
        return (
          <div className="h-full flex flex-col lg:flex-row gap-4">
            <div className="flex-1 panel p-4">
              <EventFeed onEventClick={handleEventClick} />
            </div>
            <div className="lg:w-80">
              <div className="panel p-4">
                <h3 className="text-sm font-semibold text-[#F4F7FF] mb-3">Feed Statistics</h3>
                <div className="grid grid-cols-2 gap-2">
                  <div className="p-3 bg-[#070A12] rounded-lg text-center">
                    <div className="text-2xl font-bold text-[#00F0FF]">47</div>
                    <div className="text-xs text-[#A7B1C8]">Total Events</div>
                  </div>
                  <div className="p-3 bg-[#070A12] rounded-lg text-center">
                    <div className="text-2xl font-bold text-[#FF2D55]">8</div>
                    <div className="text-xs text-[#A7B1C8]">Critical</div>
                  </div>
                  <div className="p-3 bg-[#070A12] rounded-lg text-center">
                    <div className="text-2xl font-bold text-[#00E08A]">12</div>
                    <div className="text-xs text-[#A7B1C8]">Deals</div>
                  </div>
                  <div className="p-3 bg-[#070A12] rounded-lg text-center">
                    <div className="text-2xl font-bold text-[#FFB800]">6</div>
                    <div className="text-xs text-[#A7B1C8]">Conflicts</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );

      case 'cases':
        return (
          <div className="h-full flex flex-col gap-4">
            <div className="flex-1 panel p-4">
              <CaseStudyPanel detectedPattern={detectedPattern || undefined} />
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-[#070A12] text-[#F4F7FF]">
      {/* Noise overlay */}
      <div className="noise-overlay" />

      {/* Header */}
      <header ref={headerRef} className="fixed top-0 left-0 right-0 z-50 panel mx-4 mt-4">
        <div className="flex items-center justify-between px-6 py-4">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-lg bg-[#00F0FF]/20 flex items-center justify-center">
              <Globe className="w-6 h-6 text-[#00F0FF]" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-[#F4F7FF]">MAPPO Dashboard</h1>
              <p className="text-xs text-[#A7B1C8]">Multi-Agent Resource Allocation Simulator</p>
            </div>
          </div>

          <div className="flex items-center gap-6">
            {/* Cycle counter */}
            <div className="text-center">
              <div className="text-xs text-[#A7B1C8]">Current Cycle</div>
              <div className="text-2xl font-mono font-bold text-[#00F0FF]">{currentCycle}</div>
            </div>

            {/* Simulation controls */}
            <div className="flex items-center gap-2">
              <button
                onClick={() => setIsSimulationRunning(!isSimulationRunning)}
                className={`
                  p-2 rounded-lg transition-all duration-300
                  ${isSimulationRunning
                    ? 'bg-[#FF2D55]/20 text-[#FF2D55]'
                    : 'bg-[#00E08A]/20 text-[#00E08A]'
                  }
                `}
              >
                {isSimulationRunning ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
              </button>
              <button
                onClick={() => setCurrentCycle(47)}
                className="p-2 rounded-lg bg-[#0B1022] text-[#A7B1C8] hover:text-[#F4F7FF] transition-colors"
              >
                <RotateCcw className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>

        {/* Navigation tabs */}
        <div className="flex items-center gap-1 px-2 pb-2 overflow-x-auto">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`
                flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-300 whitespace-nowrap
                ${activeTab === tab.id
                  ? 'bg-[#00F0FF]/20 text-[#00F0FF]'
                  : 'text-[#A7B1C8] hover:text-[#F4F7FF] hover:bg-[#0B1022]'
                }
              `}
            >
              <tab.icon className="w-4 h-4" />
              <span>{tab.label}</span>
            </button>
          ))}
        </div>
      </header>

      {/* Main content */}
      <main className="pt-36 pb-8 px-4">
        <div className="max-w-[1800px] mx-auto h-[calc(100vh-180px)]">
          {renderContent()}
        </div>
      </main>

      {/* Scenario Input Section */}
      <section className="px-4 pb-8">
        <div className="max-w-[1800px] mx-auto">
          <div className="panel p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-lg bg-[#00F0FF]/20 flex items-center justify-center">
                <MessageSquare className="w-5 h-5 text-[#00F0FF]" />
              </div>
              <div>
                <h2 className="text-lg font-bold text-[#F4F7FF]">Custom Scenario Analysis</h2>
                <p className="text-sm text-[#A7B1C8]">Describe a resource scenario to generate an allocation strategy</p>
              </div>
            </div>
            <ScenarioInput onStrategyGenerated={handleStrategyGenerated} />
          </div>
        </div>
      </section>

      {/* Strategy Modal */}
      {showStrategyModal && generatedStrategy && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
          <div className="w-full max-w-4xl max-h-[90vh] overflow-auto">
            <StrategyOutput
              strategy={generatedStrategy}
              onClose={() => setShowStrategyModal(false)}
            />
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="px-4 py-6 border-t border-[#00F0FF]/10">
        <div className="max-w-[1800px] mx-auto text-center">
          <p className="text-sm text-[#A7B1C8]">
            MAPPO Multi-Agent Proximal Policy Optimization Resource Allocation System
          </p>
          <p className="text-xs text-[#A7B1C8]/60 mt-1">
            See the world's resource future before it happens.
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
