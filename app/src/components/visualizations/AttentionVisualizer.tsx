import React, { useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import { regions } from '@/data/simulationData';

interface AttentionData {
  target: string;
  intensity: number;
  resource?: string;
}

interface AttentionVisualizerProps {
  agentId: string;
  attentionData: AttentionData[];
}

const AttentionVisualizer: React.FC<AttentionVisualizerProps> = ({
  agentId,
  attentionData,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  const agent = regions.find((r) => r.id === agentId);
  if (!agent) return null;

  useEffect(() => {
    if (!svgRef.current) return;

    // Animate attention pulses
    const pulses = svgRef.current.querySelectorAll('.attention-pulse');
    pulses.forEach((pulse, index) => {
      gsap.to(pulse, {
        opacity: 0.3,
        strokeWidth: 2,
        duration: 1 + index * 0.2,
        yoyo: true,
        repeat: -1,
        ease: 'power1.inOut',
      });
    });

    // Animate the central agent node
    const agentNode = svgRef.current.querySelector('.agent-node');
    gsap.fromTo(
      agentNode,
      { scale: 0.8, opacity: 0 },
      { scale: 1, opacity: 1, duration: 0.6, ease: 'back.out(1.7)' }
    );
  }, [attentionData]);

  const getAttentionColor = (intensity: number): string => {
    if (intensity >= 0.8) return '#00F0FF';
    if (intensity >= 0.5) return '#00E08A';
    return '#A7B1C8';
  };

  const getResourceIcon = (resource?: string): string => {
    switch (resource) {
      case 'water': return '💧';
      case 'food': return '🌾';
      case 'energy': return '⚡';
      default: return '📊';
    }
  };

  return (
    <div className="w-full h-full relative">
      <svg
        ref={svgRef}
        viewBox="0 0 900 500"
        className="w-full h-full"
      >
        <defs>
          <filter id="attentionGlow">
            <feGaussianBlur stdDeviation="4" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <radialGradient id="agentGradient" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#00F0FF" stopOpacity="0.8" />
            <stop offset="100%" stopColor="#00F0FF" stopOpacity="0" />
          </radialGradient>
        </defs>

        {/* Background */}
        <rect width="900" height="500" fill="rgba(7, 10, 18, 0.5)" rx="12" />

        {/* Attention connections */}
        {attentionData.map((attention, index) => {
          const targetRegion = regions.find((r) => r.id === attention.target);
          if (!targetRegion) return null;

          const color = getAttentionColor(attention.intensity);
          const isHighAttention = attention.intensity >= 0.7;

          return (
            <g key={`attention-${attention.target}`}>
              {/* Connection line */}
              <line
                className="attention-pulse"
                x1={agent.x}
                y1={agent.y}
                x2={targetRegion.x}
                y2={targetRegion.y}
                stroke={color}
                strokeWidth={attention.intensity * 6}
                strokeOpacity={0.6}
                strokeLinecap="round"
                filter={isHighAttention ? 'url(#attentionGlow)' : ''}
                style={{
                  animation: isHighAttention ? 'attention-flow 1.5s ease-in-out infinite' : 'none',
                }}
              />
              
              {/* Pulse animation for high attention */}
              {isHighAttention && (
                <>
                  <circle r="4" fill={color} opacity="0.9">
                    <animateMotion
                      dur={`${1.5 + index * 0.3}s`}
                      repeatCount="indefinite"
                      path={`M${agent.x},${agent.y} L${targetRegion.x},${targetRegion.y}`}
                    />
                  </circle>
                  <circle r="6" fill="none" stroke={color} strokeWidth="1" opacity="0.5">
                    <animateMotion
                      dur={`${1.5 + index * 0.3}s`}
                      repeatCount="indefinite"
                      path={`M${agent.x},${agent.y} L${targetRegion.x},${targetRegion.y}`}
                    />
                    <animate
                      attributeName="r"
                      values="4;12;4"
                      dur="1.5s"
                      repeatCount="indefinite"
                    />
                    <animate
                      attributeName="opacity"
                      values="0.8;0;0.8"
                      dur="1.5s"
                      repeatCount="indefinite"
                    />
                  </circle>
                </>
              )}
            </g>
          );
        })}

        {/* Target region nodes */}
        {attentionData.map((attention) => {
          const targetRegion = regions.find((r) => r.id === attention.target);
          if (!targetRegion) return null;

          const color = getAttentionColor(attention.intensity);
          const isHighAttention = attention.intensity >= 0.7;

          return (
            <g
              key={`target-${attention.target}`}
              transform={`translate(${targetRegion.x}, ${targetRegion.y})`}
            >
              {/* Glow ring for high attention */}
              {isHighAttention && (
                <circle
                  r="25"
                  fill="none"
                  stroke={color}
                  strokeWidth="1"
                  opacity="0.4"
                  filter="url(#attentionGlow)"
                >
                  <animate
                    attributeName="r"
                    values="20;30;20"
                    dur="2s"
                    repeatCount="indefinite"
                  />
                </circle>
              )}
              {/* Node */}
              <circle
                r={10 + attention.intensity * 5}
                fill="#0B1022"
                stroke={color}
                strokeWidth={isHighAttention ? 3 : 2}
              />
              {/* Inner fill */}
              <circle
                r={6}
                fill={color}
                opacity={0.8}
              />
              {/* Label */}
              <text
                y={28}
                textAnchor="middle"
                fill="#F4F7FF"
                fontSize="10"
                fontWeight="600"
              >
                {targetRegion.name}
              </text>
              {/* Resource indicator */}
              {attention.resource && (
                <text
                  y={-20}
                  textAnchor="middle"
                  fontSize="14"
                >
                  {getResourceIcon(attention.resource)}
                </text>
              )}
              {/* Intensity percentage */}
              <g transform="translate(18, -18)">
                <circle r="10" fill={color} opacity="0.9" />
                <text
                  y="3"
                  textAnchor="middle"
                  fill="#fff"
                  fontSize="7"
                  fontWeight="700"
                  className="mono"
                >
                  {Math.round(attention.intensity * 100)}%
                </text>
              </g>
            </g>
          );
        })}

        {/* Central agent node */}
        <g className="agent-node" transform={`translate(${agent.x}, ${agent.y})`}>
          {/* Outer glow */}
          <circle r="50" fill="url(#agentGradient)" opacity="0.5">
            <animate
              attributeName="r"
              values="40;55;40"
              dur="3s"
              repeatCount="indefinite"
            />
          </circle>
          {/* Main node */}
          <circle r="20" fill="#0B1022" stroke="#00F0FF" strokeWidth="3" filter="url(#attentionGlow)" />
          {/* Inner core */}
          <circle r="12" fill="#00F0FF" opacity="0.9">
            <animate
              attributeName="opacity"
              values="0.7;1;0.7"
              dur="1.5s"
              repeatCount="indefinite"
            />
          </circle>
          {/* Label */}
          <text y="45" textAnchor="middle" fill="#00F0FF" fontSize="12" fontWeight="700">
            {agent.name}
          </text>
          <text y="58" textAnchor="middle" fill="#A7B1C8" fontSize="9">
            DECIDING...
          </text>
        </g>
      </svg>

      {/* Attention legend */}
      <div className="absolute bottom-4 left-4 panel p-3">
        <div className="text-xs font-medium text-[#F4F7FF] mb-2">Attention Level</div>
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <div className="w-8 h-1 bg-[#00F0FF] rounded" />
            <span className="text-xs text-[#A7B1C8]">High (80-100%)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-8 h-1 bg-[#00E08A] rounded" />
            <span className="text-xs text-[#A7B1C8]">Medium (50-79%)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-8 h-1 bg-[#A7B1C8] rounded" />
            <span className="text-xs text-[#A7B1C8]">Low (0-49%)</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AttentionVisualizer;
