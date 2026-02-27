export interface StateData {
  id: string;
  name: string;
  status: 'green' | 'amber' | 'red' | 'black';
  resources: {
    water: number;
    power: number;
    agriculture: number;
  };
  position: { x: number; y: number };
}

export interface TradeConnection {
  from: string;
  to: string;
  type: 'water' | 'food' | 'power';
  volume: number;
  active: boolean;
}

export interface ConflictProbability {
  state1: string;
  state2: string;
  probability: number;
  trend: 'rising' | 'stable' | 'falling';
}

export interface ResourceHistory {
  cycle: number;
  groundwater: number;
  reservoir: number;
  yield: number;
  powerDeficit: number;
  debtRatio: number;
}

export interface StateStrategy {
  stateId: string;
  actions: {
    name: string;
    frequency: number;
  }[];
}

export interface Event {
  id: string;
  cycle: number;
  timestamp: Date;
  description: string;
  type: 'trade' | 'conflict' | 'alliance' | 'crisis';
  states: string[];
}

export interface CaseStudy {
  id: string;
  pattern: string;
  description: string;
  historicalParallel: {
    event: string;
    year: string;
    details: string;
  };
  similarityScore: number;
  references: string[];
}

export interface AgentAttention {
  stateId: string;
  attentionWeights: {
    targetState: string;
    weight: number;
  }[];
}

export interface SimulationScenario {
  id: string;
  name: string;
  description: string;
  affectedStates: string[];
  resourceImpact: {
    water?: number;
    power?: number;
    agriculture?: number;
  };
}
