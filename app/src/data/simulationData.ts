import type { Region, TradeConnection, ConflictProbability, SimulationEvent, CaseStudy, ResourceHistory } from '@/types';

export const regions: Region[] = [
  { id: 'egypt', name: 'Egypt', x: 520, y: 280, status: 'stressed', resources: { water: 45, food: 55, energy: 60 }, population: 104 },
  { id: 'ethiopia', name: 'Ethiopia', x: 540, y: 320, status: 'stressed', resources: { water: 65, food: 40, energy: 35 }, population: 120 },
  { id: 'india', name: 'India', x: 680, y: 270, status: 'abundant', resources: { water: 75, food: 70, energy: 65 }, population: 1380 },
  { id: 'china', name: 'China', x: 720, y: 220, status: 'abundant', resources: { water: 70, food: 80, energy: 85 }, population: 1410 },
  { id: 'brazil', name: 'Brazil', x: 320, y: 380, status: 'abundant', resources: { water: 90, food: 85, energy: 75 }, population: 215 },
  { id: 'usa', name: 'USA', x: 180, y: 200, status: 'abundant', resources: { water: 80, food: 90, energy: 95 }, population: 331 },
  { id: 'saudi', name: 'Saudi Arabia', x: 560, y: 260, status: 'critical', resources: { water: 25, food: 35, energy: 95 }, population: 35 },
  { id: 'germany', name: 'Germany', x: 480, y: 160, status: 'abundant', resources: { water: 75, food: 80, energy: 80 }, population: 83 },
  { id: 'turkey', name: 'Turkey', x: 520, y: 200, status: 'stressed', resources: { water: 60, food: 65, energy: 70 }, population: 85 },
  { id: 'iraq', name: 'Iraq', x: 560, y: 240, status: 'critical', resources: { water: 35, food: 45, energy: 55 }, population: 40 },
  { id: 'australia', name: 'Australia', x: 780, y: 420, status: 'stressed', resources: { water: 40, food: 70, energy: 75 }, population: 26 },
  { id: 'nigeria', name: 'Nigeria', x: 460, y: 300, status: 'stressed', resources: { water: 50, food: 45, energy: 40 }, population: 206 },
];

export const tradeConnections: TradeConnection[] = [
  { from: 'saudi', to: 'egypt', type: 'energy', volume: 80, active: true },
  { from: 'egypt', to: 'saudi', type: 'food', volume: 40, active: true },
  { from: 'ethiopia', to: 'egypt', type: 'water', volume: 60, active: true },
  { from: 'india', to: 'china', type: 'water', volume: 50, active: true },
  { from: 'china', to: 'india', type: 'energy', volume: 70, active: true },
  { from: 'brazil', to: 'usa', type: 'food', volume: 85, active: true },
  { from: 'usa', to: 'brazil', type: 'energy', volume: 60, active: true },
  { from: 'germany', to: 'turkey', type: 'energy', volume: 55, active: true },
  { from: 'turkey', to: 'iraq', type: 'water', volume: 45, active: true },
  { from: 'australia', to: 'china', type: 'food', volume: 75, active: true },
  { from: 'nigeria', to: 'germany', type: 'energy', volume: 50, active: true },
  { from: 'saudi', to: 'india', type: 'energy', volume: 65, active: true },
  { from: 'china', to: 'saudi', type: 'food', volume: 55, active: true },
  { from: 'usa', to: 'germany', type: 'food', volume: 70, active: true },
  { from: 'ethiopia', to: 'nigeria', type: 'energy', volume: 35, active: true },
];

export const conflictProbabilities: ConflictProbability[] = [
  { region1: 'egypt', region2: 'ethiopia', probability: 78 },
  { region1: 'india', region2: 'china', probability: 65 },
  { region1: 'turkey', region2: 'iraq', probability: 72 },
  { region1: 'saudi', region2: 'iran', probability: 58 },
  { region1: 'india', region2: 'pakistan', probability: 82 },
  { region1: 'usa', region2: 'china', probability: 45 },
  { region1: 'israel', region2: 'palestine', probability: 88 },
  { region1: 'russia', region2: 'ukraine', probability: 91 },
  { region1: 'northkorea', region2: 'southkorea', probability: 75 },
  { region1: 'china', region2: 'taiwan', probability: 68 },
];

export const simulationEvents: SimulationEvent[] = [
  { cycle: 47, type: 'deal', description: 'Saudi Arabia offered Egypt a water deal. Egypt accepted. Scarcity: critical.', regions: ['saudi', 'egypt'], severity: 'high' },
  { cycle: 48, type: 'conflict', description: 'China raided India\'s water. India sanctioned China.', regions: ['china', 'india'], severity: 'critical' },
  { cycle: 49, type: 'collapse', description: 'China\'s food collapsed due to trade sanctions.', regions: ['china'], severity: 'critical' },
  { cycle: 46, type: 'climate', description: 'Drought detected in Ethiopia region. Water reserves declining.', regions: ['ethiopia'], severity: 'high' },
  { cycle: 45, type: 'trade', description: 'Brazil increased food exports to USA by 15%.', regions: ['brazil', 'usa'], severity: 'low' },
  { cycle: 44, type: 'deal', description: 'Germany and Turkey signed energy cooperation agreement.', regions: ['germany', 'turkey'], severity: 'medium' },
  { cycle: 43, type: 'conflict', description: 'Turkey reduced Euphrates flow to Iraq by 40%.', regions: ['turkey', 'iraq'], severity: 'high' },
  { cycle: 42, type: 'climate', description: 'Heatwave affecting Australia\'s agricultural output.', regions: ['australia'], severity: 'medium' },
  { cycle: 41, type: 'trade', description: 'Nigeria increased oil exports to European markets.', regions: ['nigeria', 'germany'], severity: 'low' },
  { cycle: 40, type: 'deal', description: 'India and Saudi Arabia signed renewable energy pact.', regions: ['india', 'saudi'], severity: 'medium' },
  { cycle: 39, type: 'conflict', description: 'Border tensions rose between India and China over water rights.', regions: ['india', 'china'], severity: 'high' },
  { cycle: 38, type: 'collapse', description: 'Somalia\'s food system collapsed due to prolonged drought.', regions: ['somalia'], severity: 'critical' },
  { cycle: 37, type: 'climate', description: 'Monsoon delays affecting Indian agricultural regions.', regions: ['india'], severity: 'medium' },
  { cycle: 36, type: 'trade', description: 'USA increased LNG exports to European allies.', regions: ['usa', 'germany'], severity: 'low' },
];

export const caseStudies: CaseStudy[] = [
  {
    pattern: 'Upstream Infrastructure Cutoff',
    description: 'Upstream nation restricts water flow to downstream neighbors using dam infrastructure.',
    historicalParallel: 'Turkey reduced Euphrates flow to Iraq by 40% in 2021. Conflict followed within 18 months.',
    citation: 'UN record #14023',
    year: 2021,
  },
  {
    pattern: 'Hegemon Isolation',
    description: 'Major power faces coordinated sanctions leading to resource collapse.',
    historicalParallel: 'Iran faced international sanctions (2018-2021) leading to 80% currency devaluation and food shortages.',
    citation: 'World Bank Report 2022',
    year: 2018,
  },
  {
    pattern: 'Commons Depletion',
    description: 'Shared water resource over-extraction leads to regional crisis.',
    historicalParallel: 'Aral Sea depletion (1960-2000) caused collapse of fishing industry and health crisis.',
    citation: 'UNEP Assessment Report',
    year: 2000,
  },
  {
    pattern: 'Energy Leverage',
    description: 'Energy-rich nation uses resource as geopolitical leverage.',
    historicalParallel: 'Russia\'s 2006 and 2009 gas disputes with Ukraine affected European supply.',
    citation: 'IEA Energy Security Report',
    year: 2009,
  },
];

export const resourceHistory: ResourceHistory[] = Array.from({ length: 50 }, (_, i) => ({
  cycle: i + 1,
  regions: {
    egypt: { water: 50 - i * 0.3 + Math.random() * 10, food: 60 - i * 0.2 + Math.random() * 8, energy: 65 + Math.random() * 5 },
    ethiopia: { water: 70 - i * 0.5 + Math.random() * 12, food: 45 - i * 0.3 + Math.random() * 10, energy: 40 + i * 0.1 },
    india: { water: 80 - i * 0.2 + Math.random() * 8, food: 75 - i * 0.15 + Math.random() * 6, energy: 70 + i * 0.05 },
    china: { water: 75 - i * 0.25 + Math.random() * 9, food: 85 - i * 0.4 + Math.random() * 7, energy: 90 - i * 0.1 },
    brazil: { water: 95 - i * 0.1 + Math.random() * 5, food: 90 - i * 0.15 + Math.random() * 6, energy: 80 + i * 0.02 },
    usa: { water: 85 - i * 0.05 + Math.random() * 4, food: 95 - i * 0.08 + Math.random() * 3, energy: 98 - i * 0.03 },
    saudi: { water: 30 - i * 0.15 + Math.random() * 8, food: 40 - i * 0.1 + Math.random() * 6, energy: 98 - i * 0.02 },
  },
}));

export const strategyHeatmapData = Array.from({ length: 12 }, () =>
  Array.from({ length: 20 }, () => Math.random())
);

export const getRegionStatus = (resources: { water: number; food: number; energy: number }): 'abundant' | 'stressed' | 'critical' | 'collapsed' => {
  const avg = (resources.water + resources.food + resources.energy) / 3;
  if (avg >= 70) return 'abundant';
  if (avg >= 45) return 'stressed';
  if (avg >= 25) return 'critical';
  return 'collapsed';
};

export const getStatusColor = (status: 'abundant' | 'stressed' | 'critical' | 'collapsed'): string => {
  switch (status) {
    case 'abundant': return '#00E08A';
    case 'stressed': return '#FFB800';
    case 'critical': return '#FF2D55';
    case 'collapsed': return '#8B5CF6';
    default: return '#00E08A';
  }
};

export const getConflictColor = (probability: number): string => {
  if (probability >= 80) return '#FF2D55';
  if (probability >= 60) return '#FFB800';
  if (probability >= 40) return '#F59E0B';
  return '#10B981';
};
