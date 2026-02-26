import React, { useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  ReferenceArea,
} from 'recharts';
import { resourceHistory } from '@/data/simulationData';

interface ResourceHistoryChartProps {
  regionId?: string;
  showAllRegions?: boolean;
}

const ResourceHistoryChart: React.FC<ResourceHistoryChartProps> = ({
  regionId = 'egypt',
  showAllRegions = false,
}) => {
  const [selectedResource, setSelectedResource] = useState<'all' | 'water' | 'food' | 'energy'>('all');

  // Transform data for chart
  const chartData = resourceHistory.map((cycle) => ({
    cycle: cycle.cycle,
    ...(showAllRegions
      ? Object.entries(cycle.regions).reduce(
          (acc, [id, resources]) => ({
            ...acc,
            [`${id}_water`]: resources.water,
            [`${id}_food`]: resources.food,
            [`${id}_energy`]: resources.energy,
          }),
          {}
        )
      : cycle.regions[regionId as keyof typeof cycle.regions] || {}),
  }));

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="chart-tooltip">
          <p className="text-[#F4F7FF] font-semibold mb-2">Cycle {label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} className="text-xs" style={{ color: entry.color }}>
              {entry.name}: {entry.value?.toFixed(1)}%
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  const getLineColor = (resource: string): string => {
    switch (resource) {
      case 'water': return '#00F0FF';
      case 'food': return '#00E08A';
      case 'energy': return '#FFB800';
      default: return '#00F0FF';
    }
  };

  const climateShocks = [
    { cycle: 15, label: 'Drought', severity: 'high' },
    { cycle: 28, label: 'Heatwave', severity: 'medium' },
    { cycle: 38, label: 'Flood', severity: 'high' },
    { cycle: 46, label: 'Drought', severity: 'critical' },
  ];

  return (
    <div className="w-full h-full flex flex-col">
      {/* Controls */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-[#F4F7FF]">
          Resource History {showAllRegions ? '(All Regions)' : `- ${regionId.charAt(0).toUpperCase() + regionId.slice(1)}`}
        </h3>
        <div className="flex items-center gap-2">
          {(['all', 'water', 'food', 'energy'] as const).map((resource) => (
            <button
              key={resource}
              onClick={() => setSelectedResource(resource)}
              className={`
                px-3 py-1 text-xs rounded-full transition-all duration-300
                ${selectedResource === resource
                  ? 'bg-[#00F0FF]/20 text-[#00F0FF] border border-[#00F0FF]/50'
                  : 'bg-[#0B1022] text-[#A7B1C8] border border-[#00F0FF]/20 hover:border-[#00F0FF]/40'
                }
              `}
            >
              {resource.charAt(0).toUpperCase() + resource.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Chart */}
      <div className="flex-1 min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3,3" stroke="rgba(0, 240, 255, 0.1)" />
            <XAxis
              dataKey="cycle"
              stroke="#A7B1C8"
              tick={{ fill: '#A7B1C8', fontSize: 11 }}
              tickLine={{ stroke: '#A7B1C8' }}
              label={{ value: 'Simulation Cycle', position: 'bottom', fill: '#A7B1C8', fontSize: 12 }}
            />
            <YAxis
              stroke="#A7B1C8"
              tick={{ fill: '#A7B1C8', fontSize: 11 }}
              tickLine={{ stroke: '#A7B1C8' }}
              domain={[0, 100]}
              label={{ value: 'Resource Level (%)', angle: -90, position: 'insideLeft', fill: '#A7B1C8', fontSize: 12 }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ paddingTop: '10px' }}
              formatter={(value) => <span className="text-[#A7B1C8] text-xs">{value}</span>}
            />

            {/* Critical threshold band */}
            <ReferenceArea y1={0} y2={30} fill="#FF2D55" fillOpacity={0.1} />
            <ReferenceLine y={30} stroke="#FF2D55" strokeDasharray="5,5" strokeOpacity={0.5}>
              <text x="10" y="25" fill="#FF2D55" fontSize="10">Critical Threshold</text>
            </ReferenceLine>

            {/* Climate shock markers */}
            {climateShocks.map((shock) => (
              <ReferenceLine
                key={shock.cycle}
                x={shock.cycle}
                stroke={shock.severity === 'critical' ? '#FF2D55' : shock.severity === 'high' ? '#FFB800' : '#00F0FF'}
                strokeDasharray="3,3"
                strokeOpacity={0.6}
              />
            ))}

            {/* Resource lines */}
            {(selectedResource === 'all' || selectedResource === 'water') && (
              <Line
                type="monotone"
                dataKey={showAllRegions ? `${regionId}_water` : 'water'}
                stroke={getLineColor('water')}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 6, fill: getLineColor('water'), stroke: '#fff', strokeWidth: 2 }}
                name="Water"
              />
            )}
            {(selectedResource === 'all' || selectedResource === 'food') && (
              <Line
                type="monotone"
                dataKey={showAllRegions ? `${regionId}_food` : 'food'}
                stroke={getLineColor('food')}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 6, fill: getLineColor('food'), stroke: '#fff', strokeWidth: 2 }}
                name="Food"
              />
            )}
            {(selectedResource === 'all' || selectedResource === 'energy') && (
              <Line
                type="monotone"
                dataKey={showAllRegions ? `${regionId}_energy` : 'energy'}
                stroke={getLineColor('energy')}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 6, fill: getLineColor('energy'), stroke: '#fff', strokeWidth: 2 }}
                name="Energy"
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Climate shock legend */}
      <div className="mt-3 flex items-center gap-4 text-xs">
        <span className="text-[#A7B1C8]">Climate Events:</span>
        {climateShocks.map((shock) => (
          <div key={shock.cycle} className="flex items-center gap-1">
            <div
              className="w-2 h-2 rounded-full"
              style={{
                background: shock.severity === 'critical' ? '#FF2D55' : shock.severity === 'high' ? '#FFB800' : '#00F0FF',
              }}
            />
            <span className="text-[#A7B1C8]">
              Cycle {shock.cycle}: {shock.label}
            </span>
          </div>
        ))}
      </div>

      {/* Water-Food-Energy Nexus explanation */}
      <div className="mt-3 p-3 bg-[#0B1022] rounded-lg border border-[#00F0FF]/20">
        <div className="text-xs text-[#A7B1C8]">
          <span className="text-[#00F0FF] font-semibold">Water-Food-Energy Nexus:</span>{' '}
          Drought hits → Water drops → Food collapses 2 cycles later. Physical dependency documented in agricultural studies.
        </div>
      </div>
    </div>
  );
};

export default ResourceHistoryChart;
