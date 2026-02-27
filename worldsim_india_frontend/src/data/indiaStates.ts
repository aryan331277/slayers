import type { StateData, TradeConnection, ConflictProbability, Event, CaseStudy, StateStrategy, AgentAttention, ResourceHistory } from '@/types';

export const indiaStates: StateData[] = [
  { id: 'JK', name: 'Jammu & Kashmir', status: 'amber', resources: { water: 65, power: 45, agriculture: 55 }, position: { x: 180, y: 80 } },
  { id: 'PB', name: 'Punjab', status: 'green', resources: { water: 85, power: 75, agriculture: 95 }, position: { x: 200, y: 140 } },
  { id: 'HP', name: 'Himachal Pradesh', status: 'green', resources: { water: 90, power: 80, agriculture: 70 }, position: { x: 230, y: 110 } },
  { id: 'HR', name: 'Haryana', status: 'green', resources: { water: 80, power: 70, agriculture: 90 }, position: { x: 210, y: 170 } },
  { id: 'DL', name: 'Delhi', status: 'amber', resources: { water: 40, power: 60, agriculture: 20 }, position: { x: 230, y: 185 } },
  { id: 'RJ', name: 'Rajasthan', status: 'red', resources: { water: 25, power: 55, agriculture: 45 }, position: { x: 150, y: 200 } },
  { id: 'UP', name: 'Uttar Pradesh', status: 'amber', resources: { water: 60, power: 50, agriculture: 80 }, position: { x: 280, y: 200 } },
  { id: 'BR', name: 'Bihar', status: 'red', resources: { water: 45, power: 35, agriculture: 65 }, position: { x: 350, y: 220 } },
  { id: 'UT', name: 'Uttarakhand', status: 'green', resources: { water: 85, power: 75, agriculture: 75 }, position: { x: 250, y: 140 } },
  { id: 'SK', name: 'Sikkim', status: 'green', resources: { water: 80, power: 60, agriculture: 65 }, position: { x: 420, y: 180 } },
  { id: 'AR', name: 'Arunachal Pradesh', status: 'green', resources: { water: 90, power: 50, agriculture: 70 }, position: { x: 480, y: 160 } },
  { id: 'NL', name: 'Nagaland', status: 'amber', resources: { water: 70, power: 40, agriculture: 60 }, position: { x: 470, y: 200 } },
  { id: 'MN', name: 'Manipur', status: 'amber', resources: { water: 65, power: 35, agriculture: 55 }, position: { x: 460, y: 230 } },
  { id: 'MZ', name: 'Mizoram', status: 'green', resources: { water: 75, power: 45, agriculture: 65 }, position: { x: 440, y: 250 } },
  { id: 'TR', name: 'Tripura', status: 'amber', resources: { water: 60, power: 40, agriculture: 60 }, position: { x: 420, y: 260 } },
  { id: 'ML', name: 'Meghalaya', status: 'green', resources: { water: 85, power: 45, agriculture: 70 }, position: { x: 430, y: 230 } },
  { id: 'AS', name: 'Assam', status: 'amber', resources: { water: 70, power: 40, agriculture: 75 }, position: { x: 420, y: 210 } },
  { id: 'WB', name: 'West Bengal', status: 'amber', resources: { water: 65, power: 45, agriculture: 80 }, position: { x: 380, y: 260 } },
  { id: 'JH', name: 'Jharkhand', status: 'red', resources: { water: 50, power: 55, agriculture: 60 }, position: { x: 340, y: 260 } },
  { id: 'OD', name: 'Odisha', status: 'amber', resources: { water: 60, power: 50, agriculture: 70 }, position: { x: 320, y: 300 } },
  { id: 'CG', name: 'Chhattisgarh', status: 'amber', resources: { water: 65, power: 60, agriculture: 75 }, position: { x: 280, y: 280 } },
  { id: 'MP', name: 'Madhya Pradesh', status: 'amber', resources: { water: 55, power: 65, agriculture: 80 }, position: { x: 230, y: 260 } },
  { id: 'GJ', name: 'Gujarat', status: 'green', resources: { water: 70, power: 85, agriculture: 85 }, position: { x: 120, y: 260 } },
  { id: 'MH', name: 'Maharashtra', status: 'red', resources: { water: 35, power: 70, agriculture: 65 }, position: { x: 180, y: 320 } },
  { id: 'GA', name: 'Goa', status: 'green', resources: { water: 75, power: 60, agriculture: 70 }, position: { x: 140, y: 360 } },
  { id: 'KA', name: 'Karnataka', status: 'red', resources: { water: 30, power: 65, agriculture: 70 }, position: { x: 180, y: 380 } },
  { id: 'TL', name: 'Telangana', status: 'amber', resources: { water: 45, power: 60, agriculture: 75 }, position: { x: 220, y: 350 } },
  { id: 'AP', name: 'Andhra Pradesh', status: 'amber', resources: { water: 50, power: 65, agriculture: 80 }, position: { x: 240, y: 380 } },
  { id: 'TN', name: 'Tamil Nadu', status: 'red', resources: { water: 25, power: 60, agriculture: 75 }, position: { x: 220, y: 420 } },
  { id: 'KL', name: 'Kerala', status: 'green', resources: { water: 80, power: 55, agriculture: 85 }, position: { x: 180, y: 440 } },
];

export const tradeConnections: TradeConnection[] = [
  { from: 'PB', to: 'HR', type: 'food', volume: 85, active: true },
  { from: 'PB', to: 'DL', type: 'food', volume: 70, active: true },
  { from: 'PB', to: 'TN', type: 'food', volume: 60, active: true },
  { from: 'HR', to: 'DL', type: 'water', volume: 75, active: true },
  { from: 'UP', to: 'DL', type: 'power', volume: 65, active: true },
  { from: 'RJ', to: 'GJ', type: 'power', volume: 55, active: true },
  { from: 'GJ', to: 'MH', type: 'power', volume: 80, active: true },
  { from: 'MP', to: 'GJ', type: 'power', volume: 70, active: true },
  { from: 'KA', to: 'TN', type: 'water', volume: 45, active: true },
  { from: 'AP', to: 'TN', type: 'power', volume: 60, active: true },
  { from: 'KL', to: 'TN', type: 'water', volume: 50, active: true },
  { from: 'WB', to: 'JH', type: 'food', volume: 55, active: true },
  { from: 'BR', to: 'WB', type: 'water', volume: 65, active: true },
  { from: 'OD', to: 'WB', type: 'power', volume: 50, active: true },
  { from: 'HP', to: 'PB', type: 'water', volume: 80, active: true },
  { from: 'UT', to: 'UP', type: 'water', volume: 70, active: true },
  { from: 'MH', to: 'KA', type: 'food', volume: 65, active: true },
  { from: 'CG', to: 'MH', type: 'power', volume: 60, active: true },
];

export const conflictProbabilities: ConflictProbability[] = [
  { state1: 'KA', state2: 'TN', probability: 0.85, trend: 'rising' },
  { state1: 'PB', state2: 'HR', probability: 0.45, trend: 'stable' },
  { state1: 'UP', state2: 'BR', probability: 0.65, trend: 'rising' },
  { state1: 'RJ', state2: 'PB', probability: 0.55, trend: 'rising' },
  { state1: 'MH', state2: 'KA', probability: 0.40, trend: 'stable' },
  { state1: 'AP', state2: 'TN', probability: 0.35, trend: 'falling' },
  { state1: 'WB', state2: 'BR', probability: 0.50, trend: 'stable' },
  { state1: 'GJ', state2: 'RJ', probability: 0.30, trend: 'falling' },
  { state1: 'MP', state2: 'UP', probability: 0.45, trend: 'stable' },
  { state1: 'KL', state2: 'TN', probability: 0.25, trend: 'stable' },
  { state1: 'OD', state2: 'WB', probability: 0.40, trend: 'rising' },
  { state1: 'JH', state2: 'BR', probability: 0.55, trend: 'rising' },
];

export const events: Event[] = [
  { id: '1', cycle: 38, timestamp: new Date(), description: 'Punjab reduced wheat supply commitment to Tamil Nadu. Tamil Nadu issued public protest.', type: 'trade', states: ['PB', 'TN'] },
  { id: '2', cycle: 41, timestamp: new Date(), description: 'Karnataka released 4.2 TMC extra from Kabini to Tamil Nadu. Gratitude expressed in Chennai.', type: 'alliance', states: ['KA', 'TN'] },
  { id: '3', cycle: 47, timestamp: new Date(), description: 'Gujarat offered 1200 MW round-the-clock power to Rajasthan. Agreement signed.', type: 'trade', states: ['GJ', 'RJ'] },
  { id: '4', cycle: 52, timestamp: new Date(), description: 'Uttar Pradesh filed fresh Supreme Court case against Bihar regarding Ganga water sharing.', type: 'conflict', states: ['UP', 'BR'] },
  { id: '5', cycle: 55, timestamp: new Date(), description: 'Maharashtra declared drought emergency in Marathwada region. Requested central assistance.', type: 'crisis', states: ['MH'] },
  { id: '6', cycle: 58, timestamp: new Date(), description: 'Rajasthan begged for water guarantee from Gujarat. Gujarat asked for power purchase agreement.', type: 'trade', states: ['RJ', 'GJ'] },
  { id: '7', cycle: 62, timestamp: new Date(), description: 'Punjab threatened to stop wheat supply to Tamil Nadu due to payment delays.', type: 'conflict', states: ['PB', 'TN'] },
  { id: '8', cycle: 65, timestamp: new Date(), description: 'Haryana agreed to increase water supply to Delhi by 15% during summer peak.', type: 'alliance', states: ['HR', 'DL'] },
  { id: '9', cycle: 68, timestamp: new Date(), description: 'Karnataka reduced Kabini outflow to Tamil Nadu by 35%. Protests erupted in Mandya.', type: 'crisis', states: ['KA', 'TN'] },
  { id: '10', cycle: 72, timestamp: new Date(), description: 'West Bengal agreed to supply 500 MW power to Jharkhand. Interstate agreement signed.', type: 'trade', states: ['WB', 'JH'] },
];

export const caseStudies: CaseStudy[] = [
  {
    id: '1',
    pattern: 'Upstream reservoir reduction → downstream protest',
    description: 'Upstream state reduces water flow during acute downstream distress',
    historicalParallel: {
      event: 'Karnataka reduced Kabini & Krishnaraja Sagar outflow to Tamil Nadu',
      year: '2016-17',
      details: '35-40% reduction during peak crisis → violence in Mandya & Bengaluru → Supreme Court monitored release order'
    },
    similarityScore: 0.87,
    references: ['NITI Aayog Report 2017', 'CAG Audit 2018', 'Supreme Court Order 2018']
  },
  {
    id: '2',
    pattern: 'Chronic power over-drawal without payment',
    description: 'State consistently draws more power than allocated without timely payment',
    historicalParallel: {
      event: 'Punjab over-drew from Northern Grid',
      year: '2012-14',
      details: 'Chronic over-drawal led to grid collapse in July 2012 affecting 600 million people'
    },
    similarityScore: 0.82,
    references: ['CERC Report 2013', 'POSOCO Analysis 2014']
  },
  {
    id: '3',
    pattern: 'Foodgrain supply defection during scarcity',
    description: 'State fails to honor food supply commitments during scarcity period',
    historicalParallel: {
      event: 'Punjab wheat supply disruption',
      year: '2006-07',
      details: 'Reduced wheat allocation to southern states during low production year'
    },
    similarityScore: 0.78,
    references: ['FCI Records 2007', 'Ministry of Agriculture Report']
  },
  {
    id: '4',
    pattern: 'Repeated tribunal/Supreme Court violation',
    description: 'State repeatedly violates water tribunal or court orders',
    historicalParallel: {
      event: 'Karnataka-Tamil Nadu Cauvery disputes',
      year: '1990-2024',
      details: 'Multiple instances of non-compliance with Cauvery Tribunal awards and SC orders'
    },
    similarityScore: 0.91,
    references: ['Cauvery Tribunal Award', 'Supreme Court Orders 2018', 'CWRC Reports']
  },
  {
    id: '5',
    pattern: 'Hegemonic state gradually isolated',
    description: 'Previously dominant state loses alliances due to aggressive policies',
    historicalParallel: {
      event: 'Punjab gradual isolation in northern grid',
      year: '2015-2020',
      details: 'Aggressive stance on water sharing led to reduced cooperation from neighboring states'
    },
    similarityScore: 0.75,
    references: ['NRSC Reports', 'Inter-State Council Minutes']
  },
];

export const stateStrategies: StateStrategy[] = [
  {
    stateId: 'RJ',
    actions: [
      { name: 'Offer long-term water sharing deal', frequency: 73 },
      { name: 'Build solar & beg for grid connectivity', frequency: 68 },
      { name: 'Request central drought assistance', frequency: 45 },
      { name: 'Import food from neighboring states', frequency: 38 },
    ]
  },
  {
    stateId: 'KA',
    actions: [
      { name: 'Negotiate water release agreements', frequency: 65 },
      { name: 'Build micro-irrigation infrastructure', frequency: 58 },
      { name: 'Request tribunal intervention', frequency: 42 },
      { name: 'Develop drought-resistant crops', frequency: 35 },
    ]
  },
  {
    stateId: 'TN',
    actions: [
      { name: 'File Supreme Court petitions', frequency: 62 },
      { name: 'Build seawater desalination plants', frequency: 55 },
      { name: 'Negotiate with upstream states', frequency: 48 },
      { name: 'Implement water conservation', frequency: 40 },
    ]
  },
  {
    stateId: 'MH',
    actions: [
      { name: 'Declare drought emergency', frequency: 58 },
      { name: 'Request inter-basin water transfer', frequency: 52 },
      { name: 'Build more reservoirs', frequency: 45 },
      { name: 'Promote drip irrigation', frequency: 38 },
    ]
  },
  {
    stateId: 'PB',
    actions: [
      { name: 'Maintain foodgrain supply commitments', frequency: 70 },
      { name: 'Negotiate water sharing agreements', frequency: 55 },
      { name: 'Invest in crop diversification', frequency: 48 },
      { name: 'Build water conservation structures', frequency: 42 },
    ]
  },
];

export const agentAttentions: AgentAttention[] = [
  {
    stateId: 'GJ',
    attentionWeights: [
      { targetState: 'RJ', weight: 0.85 },
      { targetState: 'MP', weight: 0.72 },
      { targetState: 'MH', weight: 0.55 },
      { targetState: 'KL', weight: 0.15 },
    ]
  },
  {
    stateId: 'KA',
    attentionWeights: [
      { targetState: 'TN', weight: 0.90 },
      { targetState: 'MH', weight: 0.60 },
      { targetState: 'AP', weight: 0.45 },
      { targetState: 'KL', weight: 0.40 },
    ]
  },
  {
    stateId: 'UP',
    attentionWeights: [
      { targetState: 'BR', weight: 0.75 },
      { targetState: 'MP', weight: 0.65 },
      { targetState: 'RJ', weight: 0.50 },
      { targetState: 'HR', weight: 0.45 },
    ]
  },
];

export const generateResourceHistory = (stateId: string): ResourceHistory[] => {
  const history: ResourceHistory[] = [];
  const baseValues: Record<string, { groundwater: number; reservoir: number; yield: number; powerDeficit: number; debtRatio: number }> = {
    'RJ': { groundwater: 45, reservoir: 35, yield: 55, powerDeficit: 25, debtRatio: 32 },
    'KA': { groundwater: 40, reservoir: 30, yield: 60, powerDeficit: 30, debtRatio: 28 },
    'TN': { groundwater: 35, reservoir: 25, yield: 65, powerDeficit: 35, debtRatio: 35 },
    'MH': { groundwater: 50, reservoir: 40, yield: 70, powerDeficit: 28, debtRatio: 38 },
    'PB': { groundwater: 75, reservoir: 80, yield: 90, powerDeficit: 15, debtRatio: 42 },
  };
  
  const base = baseValues[stateId] || { groundwater: 55, reservoir: 50, yield: 65, powerDeficit: 25, debtRatio: 30 };
  
  for (let i = 0; i < 60; i++) {
    const cycle = i + 1;
    const shock = i === 42 ? -20 : i === 35 ? -10 : i === 50 ? -15 : 0;
    const randomVariation = () => (Math.random() - 0.5) * 10;
    
    history.push({
      cycle,
      groundwater: Math.max(0, Math.min(100, base.groundwater + randomVariation() + (i > 42 ? -15 : 0))),
      reservoir: Math.max(0, Math.min(100, base.reservoir + randomVariation() + shock + (i > 42 ? -20 : 0))),
      yield: Math.max(0, Math.min(100, base.yield + randomVariation() + (i > 44 ? -25 : 0))),
      powerDeficit: Math.max(0, Math.min(100, base.powerDeficit + randomVariation() + (i > 45 ? 20 : 0))),
      debtRatio: Math.max(0, Math.min(100, base.debtRatio + randomVariation() * 0.5 + (i > 48 ? 10 : 0))),
    });
  }
  
  return history;
};

export const scenarios = [
  {
    id: '1',
    name: 'Maharashtra Drought Crisis',
    description: 'Maharashtra Region is facing severe droughts affecting 15 districts',
    affectedStates: ['MH', 'KA', 'TG'],
    resourceImpact: { water: -40, agriculture: -35, power: -20 }
  },
  {
    id: '2',
    name: 'North India Monsoon Failure',
    description: '2026-like monsoon failure hits Rajasthan and surrounding states',
    affectedStates: ['RJ', 'UP', 'MP', 'HR'],
    resourceImpact: { water: -50, agriculture: -45, power: -15 }
  },
  {
    id: '3',
    name: 'Cauvery Water Dispute Escalation',
    description: 'Karnataka-Tamil Nadu water sharing conflict intensifies',
    affectedStates: ['KA', 'TN', 'KL'],
    resourceImpact: { water: -30, agriculture: -25, power: -10 }
  },
  {
    id: '4',
    name: 'Punjab Agricultural Crisis',
    description: 'Punjab faces groundwater depletion and crop failure',
    affectedStates: ['PB', 'HR', 'UP'],
    resourceImpact: { water: -45, agriculture: -40, power: -10 }
  },
];
