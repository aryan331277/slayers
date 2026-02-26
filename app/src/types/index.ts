export type RegionStatus = 'abundant' | 'stressed' | 'critical' | 'collapsed';
export type ResourceType = 'water' | 'food' | 'energy';
export type TradeType = 'water' | 'food' | 'energy' | 'alliance';

export interface Region {
  id: string;
  name: string;
  x: number;
  y: number;
  status: RegionStatus;
  resources: {
    water: number;
    food: number;
    energy: number;
  };
  population: number;
}

export interface TradeConnection {
  from: string;
  to: string;
  type: TradeType;
  volume: number;
  active: boolean;
}

export interface ConflictProbability {
  region1: string;
  region2: string;
  probability: number;
}

export interface AttentionData {
  target: string;
  intensity: number;
  resource?: ResourceType;
}

export interface AgentDecision {
  agent: string;
  action: string;
  attention: AttentionData[];
}

export interface ResourceHistory {
  cycle: number;
  regions: Record<string, {
    water: number;
    food: number;
    energy: number;
  }>;
  events?: SimulationEvent[];
}

export interface SimulationEvent {
  cycle: number;
  type: 'trade' | 'conflict' | 'climate' | 'collapse' | 'sanction' | 'deal';
  description: string;
  regions: string[];
  severity?: 'low' | 'medium' | 'high' | 'critical';
}

export interface StrategyAction {
  action: string;
  frequency: number;
}

export interface CaseStudy {
  pattern: string;
  description: string;
  historicalParallel: string;
  citation: string;
  year: number;
}

export interface AllocationStrategy {
  region: string;
  scenario: string;
  recommendations: {
    resource: ResourceType;
    action: string;
    priority: 'immediate' | 'short-term' | 'long-term';
    details: string;
  }[];
  tradeProposals: {
    partner: string;
    resource: ResourceType;
    amount: number;
    terms: string;
  }[];
  riskMitigation: string[];
}
