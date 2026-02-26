import { useEffect, useRef, useState } from 'react';
import { gsap } from 'gsap';
import { regions, getStatusColor } from '@/data/simulationData';
import type { Region } from '@/types';

interface WorldMapProps {
  onRegionClick?: (region: Region) => void;
  highlightedRegions?: string[];
  showTradeRoutes?: boolean;
  pulseRegions?: string[];
}

const WorldMap = ({
  onRegionClick,
  highlightedRegions = [],
  showTradeRoutes = true,
  pulseRegions = [],
}: WorldMapProps) => {
  const mapRef = useRef<SVGSVGElement>(null);
  const [hoveredRegion, setHoveredRegion] = useState<string | null>(null);

  useEffect(() => {
    if (!mapRef.current) return;

    // Animate regions on mount
    const regionNodes = mapRef.current.querySelectorAll('.region-node');
    gsap.fromTo(
      regionNodes,
      { scale: 0.6, opacity: 0 },
      {
        scale: 1,
        opacity: 1,
        duration: 0.8,
        stagger: 0.08,
        ease: 'back.out(1.7)',
      }
    );

    // Animate trade routes
    if (showTradeRoutes) {
      const routes = mapRef.current.querySelectorAll('.trade-route');
      gsap.fromTo(
        routes,
        { strokeDashoffset: 200, opacity: 0 },
        {
          strokeDashoffset: 0,
          opacity: 1,
          duration: 1.5,
          stagger: 0.1,
          ease: 'power2.out',
        }
      );
    }
  }, [showTradeRoutes]);

  useEffect(() => {
    if (!mapRef.current) return;

    // Pulse animation for critical regions
    pulseRegions.forEach((regionId) => {
      const node = mapRef.current?.querySelector(`[data-region="${regionId}"]`);
      if (node) {
        gsap.to(node, {
          scale: 1.15,
          duration: 0.8,
          yoyo: true,
          repeat: -1,
          ease: 'power1.inOut',
        });
      }
    });
  }, [pulseRegions]);

  const getRegionColor = (region: Region): string => {
    if (highlightedRegions.includes(region.id)) {
      return '#00F0FF';
    }
    return getStatusColor(region.status);
  };

  const getTradeRouteColor = (type: string): string => {
    switch (type) {
      case 'water': return '#00F0FF';
      case 'food': return '#00E08A';
      case 'energy': return '#FFB800';
      default: return '#00F0FF';
    }
  };

  return (
    <svg
      ref={mapRef}
      viewBox="0 0 900 500"
      className="w-full h-full"
      style={{ filter: 'drop-shadow(0 0 20px rgba(0, 240, 255, 0.1))' }}
    >
      {/* World map background - simplified continents */}
      <defs>
        <linearGradient id="oceanGradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#070A12" />
          <stop offset="100%" stopColor="#0B1022" />
        </linearGradient>
        <filter id="glow">
          <feGaussianBlur stdDeviation="3" result="coloredBlur" />
          <feMerge>
            <feMergeNode in="coloredBlur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Ocean background */}
      <rect width="900" height="500" fill="url(#oceanGradient)" rx="12" />

      {/* Simplified continent shapes */}
      <g className="continents" opacity="0.3">
        {/* North America */}
        <path
          d="M80,120 Q120,80 180,100 L200,180 Q160,220 100,200 L60,150 Z"
          fill="#1a1f35"
          stroke="#00F0FF"
          strokeWidth="0.5"
        />
        {/* South America */}
        <path
          d="M200,280 Q240,260 260,300 L280,400 Q240,440 200,400 L180,320 Z"
          fill="#1a1f35"
          stroke="#00F0FF"
          strokeWidth="0.5"
        />
        {/* Europe */}
        <path
          d="M420,100 Q480,80 520,120 L500,180 Q460,190 420,160 Z"
          fill="#1a1f35"
          stroke="#00F0FF"
          strokeWidth="0.5"
        />
        {/* Africa */}
        <path
          d="M420,220 Q500,200 540,260 L520,400 Q460,420 420,380 L400,280 Z"
          fill="#1a1f35"
          stroke="#00F0FF"
          strokeWidth="0.5"
        />
        {/* Asia */}
        <path
          d="M540,80 Q720,60 780,140 L760,280 Q680,300 600,260 L540,180 Z"
          fill="#1a1f35"
          stroke="#00F0FF"
          strokeWidth="0.5"
        />
        {/* Australia */}
        <path
          d="M720,360 Q800,340 820,400 L780,460 Q720,450 700,400 Z"
          fill="#1a1f35"
          stroke="#00F0FF"
          strokeWidth="0.5"
        />
      </g>

      {/* Grid lines */}
      <g opacity="0.1">
        {Array.from({ length: 10 }, (_, i) => (
          <line key={`h-${i}`} x1="0" y1={i * 50} x2="900" y2={i * 50} stroke="#00F0FF" strokeWidth="0.5" />
        ))}
        {Array.from({ length: 18 }, (_, i) => (
          <line key={`v-${i}`} x1={i * 50} y1="0" x2={i * 50} y2="500" stroke="#00F0FF" strokeWidth="0.5" />
        ))}
      </g>

      {/* Trade routes */}
      {showTradeRoutes && regions.map((region, index) => {
        // Create some visual trade routes between nearby regions
        const nearbyRegions = regions.filter((r, i) => {
          if (i <= index) return false;
          const dx = r.x - region.x;
          const dy = r.y - region.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          return distance < 200 && Math.random() > 0.5;
        });

        return nearbyRegions.map((targetRegion, i) => {
          const type = ['water', 'food', 'energy'][Math.floor(Math.random() * 3)];
          return (
            <g key={`route-${region.id}-${targetRegion.id}`}>
              <line
                className="trade-route"
                x1={region.x}
                y1={region.y}
                x2={targetRegion.x}
                y2={targetRegion.y}
                stroke={getTradeRouteColor(type)}
                strokeWidth={2}
                strokeOpacity={0.4}
                strokeDasharray="8,4"
              />
              <circle r="2" fill={getTradeRouteColor(type)} opacity="0.8">
                <animateMotion
                  dur={`${3 + i * 0.5}s`}
                  repeatCount="indefinite"
                  path={`M${region.x},${region.y} L${targetRegion.x},${targetRegion.y}`}
                />
              </circle>
            </g>
          );
        });
      })}

      {/* Region nodes */}
      {regions.map((region) => (
        <g
          key={region.id}
          className="region-node"
          data-region={region.id}
          transform={`translate(${region.x}, ${region.y})`}
          onClick={() => onRegionClick?.(region)}
          onMouseEnter={() => setHoveredRegion(region.id)}
          onMouseLeave={() => setHoveredRegion(null)}
          style={{ cursor: onRegionClick ? 'pointer' : 'default' }}
        >
          {/* Outer glow ring */}
          <circle
            r={hoveredRegion === region.id ? 25 : 18}
            fill="none"
            stroke={getRegionColor(region)}
            strokeWidth="1"
            opacity={0.4}
            style={{
              filter: 'url(#glow)',
              transition: 'all 0.3s ease',
            }}
          />
          {/* Main node */}
          <circle
            r={hoveredRegion === region.id ? 14 : 10}
            fill={getRegionColor(region)}
            style={{
              filter: 'url(#glow)',
              transition: 'all 0.3s ease',
            }}
          />
          {/* Inner highlight */}
          <circle
            r={6}
            fill="#fff"
            opacity={0.3}
          />
          {/* Region label */}
          <text
            y={28}
            textAnchor="middle"
            fill="#F4F7FF"
            fontSize="10"
            fontWeight="600"
            style={{ pointerEvents: 'none' }}
          >
            {region.name}
          </text>
          {/* Resource indicator */}
          <g transform="translate(-15, -25)">
            <rect width="30" height="12" rx="3" fill="rgba(0,0,0,0.6)" />
            <text
              x="15"
              y="8"
              textAnchor="middle"
              fill="#fff"
              fontSize="7"
              className="mono"
            >
              {Math.round((region.resources.water + region.resources.food + region.resources.energy) / 3)}%
            </text>
          </g>
        </g>
      ))}

      {/* Legend */}
      <g transform="translate(20, 420)">
        <rect width="140" height="70" rx="8" fill="rgba(11, 16, 34, 0.9)" stroke="rgba(0, 240, 255, 0.2)" />
        <text x="10" y="18" fill="#F4F7FF" fontSize="11" fontWeight="600">Region Status</text>
        {[
          { color: '#00E08A', label: 'Abundant' },
          { color: '#FFB800', label: 'Stressed' },
          { color: '#FF2D55', label: 'Critical' },
          { color: '#8B5CF6', label: 'Collapsed' },
        ].map((item, i) => (
          <g key={item.label} transform={`translate(10, ${30 + i * 10})`}>
            <circle r="4" fill={item.color} />
            <text x="12" y="3" fill="#A7B1C8" fontSize="8">{item.label}</text>
          </g>
        ))}
      </g>
    </svg>
  );
};

export default WorldMap;
