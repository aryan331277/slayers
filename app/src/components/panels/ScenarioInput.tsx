import { useState } from 'react';
import { Send, Droplets, Wheat, Zap, AlertTriangle } from 'lucide-react';
import type { AllocationStrategy } from '@/types';

interface ScenarioInputProps {
  onStrategyGenerated?: (strategy: AllocationStrategy) => void;
}

const ScenarioInput = ({ onStrategyGenerated }: ScenarioInputProps) => {
  const [input, setInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [suggestions] = useState([
    'Maharashtra region is facing severe droughts',
    'Nile water dispute between Egypt and Ethiopia escalates',
    'India-China border tensions over water resources',
    'Saudi Arabia oil production cuts affect global energy',
    'Brazil Amazon deforestation impacts food production',
  ]);

  const generateStrategy = async (scenario: string): Promise<AllocationStrategy> => {
    await new Promise((resolve) => setTimeout(resolve, 2000));

    const lowerScenario = scenario.toLowerCase();
    const isDrought = lowerScenario.includes('drought');
    const isWater = lowerScenario.includes('water') || isDrought;
    const isFood = lowerScenario.includes('food') || lowerScenario.includes('agriculture');
    const region = lowerScenario.includes('maharashtra')
      ? 'India'
      : lowerScenario.includes('egypt')
      ? 'Egypt'
      : lowerScenario.includes('ethiopia')
      ? 'Ethiopia'
      : lowerScenario.includes('saudi')
      ? 'Saudi Arabia'
      : lowerScenario.includes('brazil')
      ? 'Brazil'
      : 'Global';

    const strategy: AllocationStrategy = {
      region,
      scenario,
      recommendations: [
        {
          resource: isWater || isDrought ? 'water' : isFood ? 'food' : 'energy',
          action: isDrought
            ? 'Implement emergency water rationing and import from neighboring regions'
            : isWater
            ? 'Negotiate water-sharing agreements and invest in desalination'
            : isFood
            ? 'Diversify food sources and establish strategic reserves'
            : 'Diversify energy mix and secure alternative suppliers',
          priority: 'immediate',
          details: isDrought
            ? 'Reduce agricultural water usage by 40%, prioritize drinking water supply, activate emergency pipelines'
            : 'Establish bilateral agreements, invest 15% GDP in infrastructure',
        },
        {
          resource: isDrought ? 'food' : 'water',
          action: 'Secure alternative supply chains',
          priority: 'short-term',
          details: 'Identify 3+ alternative suppliers, negotiate emergency contracts',
        },
        {
          resource: 'energy',
          action: 'Optimize energy allocation for critical infrastructure',
          priority: 'immediate',
          details: 'Prioritize hospitals, water treatment, and food storage facilities',
        },
      ],
      tradeProposals: [
        {
          partner: region === 'India' ? 'China' : region === 'Egypt' ? 'Ethiopia' : 'USA',
          resource: isDrought ? 'water' : isFood ? 'food' : 'energy',
          amount: 500000,
          terms: 'Emergency supply agreement with 20% premium pricing',
        },
        {
          partner: region === 'India' ? 'Saudi Arabia' : 'Germany',
          resource: 'energy',
          amount: 1000000,
          terms: 'Long-term contract with technology transfer clause',
        },
      ],
      riskMitigation: [
        'Establish 90-day strategic resource reserves',
        'Diversify supplier base to reduce single-point-of-failure',
        'Invest in domestic production capacity',
        'Develop early warning systems for resource shortages',
        'Create regional cooperation frameworks',
      ],
    };

    return strategy;
  };

  const handleSubmit = async () => {
    if (!input.trim()) return;

    setIsProcessing(true);
    const strategy = await generateStrategy(input);
    setIsProcessing(false);
    onStrategyGenerated?.(strategy);
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInput(suggestion);
  };

  return (
    <div className="w-full">
      <div className="relative">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Describe a scenario (e.g., 'Maharashtra region is facing severe droughts')..."
          className="input-field min-h-[100px] resize-none pr-14"
          disabled={isProcessing}
        />
        <button
          onClick={handleSubmit}
          disabled={!input.trim() || isProcessing}
          className={`
            absolute bottom-3 right-3 p-2 rounded-lg transition-all duration-300
            ${input.trim() && !isProcessing
              ? 'bg-[#00F0FF] text-[#070A12] hover:bg-[#00E08A]'
              : 'bg-[#0B1022] text-[#A7B1C8] cursor-not-allowed'
            }
          `}
        >
          {isProcessing ? (
            <div className="w-5 h-5 border-2 border-[#A7B1C8] border-t-transparent rounded-full animate-spin" />
          ) : (
            <Send className="w-5 h-5" />
          )}
        </button>
      </div>

      <div className="mt-3">
        <div className="text-xs text-[#A7B1C8] mb-2">Try these scenarios:</div>
        <div className="flex flex-wrap gap-2">
          {suggestions.map((suggestion) => (
            <button
              key={suggestion}
              onClick={() => handleSuggestionClick(suggestion)}
              className="text-xs px-3 py-1.5 rounded-full bg-[#0B1022] text-[#A7B1C8] border border-[#00F0FF]/20 hover:border-[#00F0FF]/50 hover:text-[#00F0FF] transition-all duration-300"
            >
              {suggestion}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-4 grid grid-cols-4 gap-2">
        {[
          { icon: Droplets, label: 'Water Crisis', color: '#00F0FF' },
          { icon: Wheat, label: 'Food Shortage', color: '#00E08A' },
          { icon: Zap, label: 'Energy Cut', color: '#FFB800' },
          { icon: AlertTriangle, label: 'Conflict', color: '#FF2D55' },
        ].map((action) => (
          <button
            key={action.label}
            onClick={() => setInput(`${action.label} in `)}
            className="flex flex-col items-center gap-1 p-2 rounded-lg bg-[#0B1022] border border-[#00F0FF]/10 hover:border-[#00F0FF]/30 transition-all duration-300"
          >
            <action.icon className="w-5 h-5" style={{ color: action.color }} />
            <span className="text-[10px] text-[#A7B1C8]">{action.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
};

export default ScenarioInput;
