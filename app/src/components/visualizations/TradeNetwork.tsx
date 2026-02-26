import { useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import { regions, tradeConnections } from '@/data/simulationData';
import type { TradeConnection } from '@/types';

interface TradeNetworkProps {
  highlightedConnection?: string | null;
  brokenConnections?: string[];
}

const TradeNetwork = ({
  highlightedConnection,
  brokenConnections = [],
}: TradeNetworkProps) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current) return;

    // Animate nodes
    const nodes = svgRef.current.querySelectorAll('.network-node');
    gsap.fromTo(
      nodes,
      { scale: 0, opacity: 0 },
      {
        scale: 1,
        opacity: 1,
        duration: 0.6,
        stagger: 0.05,
        ease: 'back.out(1.7)',
      }
    );

    // Animate connections
    const connections = svgRef.current.querySelectorAll('.network-connection');
    gsap.fromTo(
      connections,
      { strokeDashoffset: 300, opacity: 0 },
      {
        strokeDashoffset: 0,
        opacity: 1,
        duration: 1.2,
        stagger: 0.08,
        ease: 'power2.out',
      }
    );
  }, []);

  useEffect(() => {
    if (!svgRef.current || !highlightedConnection) return;

    const connection = svgRef.current.querySelector(`[data-connection="${highlightedConnection}"]`);
    if (connection) {
      gsap.to(connection, {
        strokeWidth: 6,
        opacity: 1,
        duration: 0.3,
        yoyo: true,
        repeat: 3,
      });
    }
  }, [highlightedConnection]);

  const getConnectionColor = (type: string): string => {
    switch (type) {
      case 'water': return '#00F0FF';
      case 'food': return '#00E08A';
      case 'energy': return '#FFB800';
      default: return '#00F0FF';
    }
  };

  const getConnectionId = (conn: TradeConnection): string => {
    return `${conn.from}-${conn.to}`;
  };

  return (
    <div className="w-full h-full relative">
      <svg
        ref={svgRef}
        viewBox="0 0 900 500"
        className="w-full h-full"
      >
        <defs>
          <filter id="networkGlow">
            <feGaussianBlur stdDeviation="2" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <marker
            id="arrowhead"
            markerWidth="10"
            markerHeight="7"
            refX="9"
            refY="3.5"
            orient="auto"
          >
            <polygon points="0 0, 10 3.5, 0 7" fill="#00F0FF" opacity="0.6" />
          </marker>
        </defs>

        {/* Background */}
        <rect width="900" height="500" fill="rgba(7, 10, 18, 0.5)" rx="12" />

        {/* Connection lines */}
        {tradeConnections.map((connection, index) => {
          const fromRegion = regions.find((r) => r.id === connection.from);
          const toRegion = regions.find((r) => r.id === connection.to);
          if (!fromRegion || !toRegion) return null;

          const connId = getConnectionId(connection);
          const isBroken = brokenConnections.includes(connId);
          const isHighlighted = highlightedConnection === connId;

          return (
            <g key={`route-${index}`}>
              <line
                className="network-connection"
                data-connection={connId}
                x1={fromRegion.x}
                y1={fromRegion.y}
                x2={toRegion.x}
                y2={toRegion.y}
                stroke={getConnectionColor(connection.type)}
                strokeWidth={isHighlighted ? 5 : connection.volume / 12}
                strokeOpacity={isBroken ? 0.2 : isHighlighted ? 1 : 0.5}
                strokeDasharray={isBroken ? '4,8' : '6,3'}
                filter="url(#networkGlow)"
                style={{
                  transition: 'all 0.5s ease',
                }}
              />
              {isBroken && (
                <text
                  x={(fromRegion.x + toRegion.x) / 2}
                  y={(fromRegion.y + toRegion.y) / 2 - 10}
                  fill="#FF2D55"
                  fontSize="10"
                  textAnchor="middle"
                  className="animate-pulse-critical"
                >
                  DEFECTED
                </text>
              )}
              {/* Data flow animation */}
              {!isBroken && (
                <circle r="3" fill={getConnectionColor(connection.type)} opacity="0.8">
                  <animateMotion
                    dur={`${2 + index * 0.3}s`}
                    repeatCount="indefinite"
                    path={`M${fromRegion.x},${fromRegion.y} L${toRegion.x},${toRegion.y}`}
                  />
                </circle>
              )}
            </g>
          );
        })}

        {/* Region nodes */}
        {regions.map((region) => {
          const activeConnections = tradeConnections.filter(
            (c) => (c.from === region.id || c.to === region.id) && c.active
          );
          const totalVolume = activeConnections.reduce((sum, c) => sum + c.volume, 0);

          return (
            <g
              key={`node-${region.id}`}
              className="network-node"
              transform={`translate(${region.x}, ${region.y})`}
            >
              {/* Node glow */}
              <circle
                r={20 + totalVolume / 10}
                fill="none"
                stroke="#00F0FF"
                strokeWidth="1"
                opacity="0.2"
                filter="url(#networkGlow)"
              />
              {/* Main node */}
              <circle
                r={12}
                fill="#0B1022"
                stroke="#00F0FF"
                strokeWidth="2"
              />
              {/* Inner dot */}
              <circle
                r={6}
                fill="#00F0FF"
                opacity="0.8"
              />
              {/* Label */}
              <text
                y={30}
                textAnchor="middle"
                fill="#F4F7FF"
                fontSize="10"
                fontWeight="600"
              >
                {region.name}
              </text>
              {/* Connection count */}
              <g transform="translate(15, -15)">
                <circle r="8" fill="#FF2D55" opacity="0.9" />
                <text
                  y="3"
                  textAnchor="middle"
                  fill="#fff"
                  fontSize="7"
                  fontWeight="700"
                >
                  {activeConnections.length}
                </text>
              </g>
            </g>
          );
        })}

        {/* Legend */}
        <g transform="translate(20, 20)">
          <rect width="140" height="90" rx="8" fill="rgba(11, 16, 34, 0.95)" stroke="rgba(0, 240, 255, 0.2)" />
          <text x="10" y="18" fill="#F4F7FF" fontSize="11" fontWeight="600">Trade Types</text>
          {[
            { color: '#00F0FF', label: 'Water' },
            { color: '#00E08A', label: 'Food' },
            { color: '#FFB800', label: 'Energy' },
          ].map((item, i) => (
            <g key={item.label} transform={`translate(10, ${32 + i * 18})`}>
              <line x1="0" y1="0" x2="20" y2="0" stroke={item.color} strokeWidth="3" />
              <text x="28" y="4" fill="#A7B1C8" fontSize="10">{item.label}</text>
            </g>
          ))}
        </g>
      </svg>
    </div>
  );
};

export default TradeNetwork;
