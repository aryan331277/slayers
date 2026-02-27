import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ReferenceArea,
} from 'recharts';
import { generateResourceHistory, indiaStates } from '@/data/indiaStates';

interface ResourceHistoryProps {
  selectedState?: string;
  simulationCycle?: number;
}

const ResourceHistory: React.FC<ResourceHistoryProps> = ({ selectedState = 'RJ', simulationCycle = 0 }) => {
  const [data, setData] = useState<any[]>([]);
  const [currentState, setCurrentState] = useState(selectedState);

  useEffect(() => {
    setCurrentState(selectedState);
  }, [selectedState]);

  useEffect(() => {
    const history = generateResourceHistory(currentState);
    setData(history);
  }, [currentState]);

  const stateInfo = indiaStates.find(s => s.id === currentState);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-[#1a1a25] border border-gray-700 rounded-lg p-3">
          <p className="text-sm font-semibold text-white">Cycle {label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} className="text-xs mt-1" style={{ color: entry.color }}>
              {entry.name}: {entry.value.toFixed(1)}%
            </p>
          ))}
          {label === 42 && (
            <p className="text-xs text-red-400 mt-2 font-semibold">⚠️ Monsoon Failure</p>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="relative w-full h-full bg-[#0a0a0f] rounded-xl overflow-hidden p-4">
      {/* Grid Background */}
      <div className="absolute inset-0 grid-bg opacity-50" />
      
      {/* Title */}
      <div className="relative z-10 flex justify-between items-start mb-4">
        <div>
          <h3 className="text-xl font-bold text-white neon-text">Resource History</h3>
          <p className="text-sm text-gray-400 mt-1">{stateInfo?.name} - Time Series Analysis</p>
        </div>
        
        {/* State Selector */}
        <select 
          value={currentState}
          onChange={(e) => setCurrentState(e.target.value)}
          className="bg-[#1a1a25] text-white text-sm border border-gray-600 rounded px-3 py-1.5 focus:outline-none focus:border-cyan-400"
        >
          {indiaStates.map(s => (
            <option key={s.id} value={s.id}>
              {s.id} - {s.name}
            </option>
          ))}
        </select>
      </div>

      {/* Charts Grid */}
      <div className="relative z-10 grid grid-cols-2 gap-4 h-[calc(100%-80px)]">
        {/* Groundwater Chart */}
        <div className="bg-[#12121a] rounded-lg p-3">
          <div className="text-xs font-semibold text-gray-400 mb-2">Groundwater Depth</div>
          <ResponsiveContainer width="100%" height="85%">
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
              <XAxis dataKey="cycle" stroke="#666" fontSize={10} />
              <YAxis stroke="#666" fontSize={10} domain={[0, 100]} />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={30} stroke="#ff3333" strokeDasharray="3 3" label={{ value: 'Critical', fill: '#ff3333', fontSize: 10 }} />
              <ReferenceArea x1={42} x2={45} fill="#ff3333" fillOpacity={0.1} />
              <Line 
                type="monotone" 
                dataKey="groundwater" 
                stroke="#00d4ff" 
                strokeWidth={2}
                dot={false}
                name="Groundwater"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Reservoir Storage Chart */}
        <div className="bg-[#12121a] rounded-lg p-3">
          <div className="text-xs font-semibold text-gray-400 mb-2">Reservoir Storage %</div>
          <ResponsiveContainer width="100%" height="85%">
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
              <XAxis dataKey="cycle" stroke="#666" fontSize={10} />
              <YAxis stroke="#666" fontSize={10} domain={[0, 100]} />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={20} stroke="#ff3333" strokeDasharray="3 3" label={{ value: 'Critical', fill: '#ff3333', fontSize: 10 }} />
              <ReferenceArea x1={42} x2={45} fill="#ff3333" fillOpacity={0.1} />
              <Line 
                type="monotone" 
                dataKey="reservoir" 
                stroke="#00ff88" 
                strokeWidth={2}
                dot={false}
                name="Reservoir"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Agricultural Yield Chart */}
        <div className="bg-[#12121a] rounded-lg p-3">
          <div className="text-xs font-semibold text-gray-400 mb-2">Agricultural Yield</div>
          <ResponsiveContainer width="100%" height="85%">
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
              <XAxis dataKey="cycle" stroke="#666" fontSize={10} />
              <YAxis stroke="#666" fontSize={10} domain={[0, 100]} />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={40} stroke="#ff3333" strokeDasharray="3 3" label={{ value: 'Migration Risk', fill: '#ff3333', fontSize: 10 }} />
              <ReferenceArea x1={42} x2={45} fill="#ff3333" fillOpacity={0.1} />
              <Line 
                type="monotone" 
                dataKey="yield" 
                stroke="#f7c600" 
                strokeWidth={2}
                dot={false}
                name="Yield"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Power Deficit Chart */}
        <div className="bg-[#12121a] rounded-lg p-3">
          <div className="text-xs font-semibold text-gray-400 mb-2">Power Deficit %</div>
          <ResponsiveContainer width="100%" height="85%">
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
              <XAxis dataKey="cycle" stroke="#666" fontSize={10} />
              <YAxis stroke="#666" fontSize={10} domain={[0, 100]} />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={50} stroke="#ff3333" strokeDasharray="3 3" label={{ value: 'Critical', fill: '#ff3333', fontSize: 10 }} />
              <ReferenceArea x1={42} x2={45} fill="#ff3333" fillOpacity={0.1} />
              <Line 
                type="monotone" 
                dataKey="powerDeficit" 
                stroke="#ff6b35" 
                strokeWidth={2}
                dot={false}
                name="Power Deficit"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Shock Marker */}
      {simulationCycle >= 42 && simulationCycle <= 45 && (
        <motion.div 
          className="absolute top-16 left-1/2 transform -translate-x-1/2 z-20 bg-red-900/90 border border-red-500 rounded-lg px-4 py-2"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-red-500 rounded-full animate-ping" />
            <span className="text-sm font-semibold text-red-200">Monsoon Failure - Cycle 42</span>
          </div>
        </motion.div>
      )}

      {/* Legend */}
      <div className="relative z-10 mt-2 flex items-center gap-4 text-xs">
        <div className="flex items-center gap-2">
          <div className="w-3 h-0.5 bg-red-500 border-dashed" style={{ borderTop: '1px dashed #ff3333' }} />
          <span className="text-gray-500">Critical Threshold</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-red-500/10 rounded" />
          <span className="text-gray-500">Shock Period</span>
        </div>
      </div>
    </div>
  );
};

export default ResourceHistory;
