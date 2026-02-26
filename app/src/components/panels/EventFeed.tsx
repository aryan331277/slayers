import { useEffect, useRef, useState } from 'react';
import { gsap } from 'gsap';
import { simulationEvents } from '@/data/simulationData';
import type { SimulationEvent } from '@/types';
import { AlertTriangle, Handshake, ShieldAlert, Skull } from 'lucide-react';

interface EventFeedProps {
  onEventClick?: (event: SimulationEvent) => void;
  maxEvents?: number;
}

const EventFeed = ({ onEventClick, maxEvents = 10 }: EventFeedProps) => {
  const feedRef = useRef<HTMLDivElement>(null);
  const [events, setEvents] = useState<SimulationEvent[]>(simulationEvents.slice(0, maxEvents));
  const [newEventId, setNewEventId] = useState<number | null>(null);

  useEffect(() => {
    if (!feedRef.current) return;

    const items = feedRef.current.querySelectorAll('.event-item');
    gsap.fromTo(
      items,
      { x: 30, opacity: 0 },
      {
        x: 0,
        opacity: 1,
        duration: 0.4,
        stagger: 0.08,
        ease: 'power2.out',
      }
    );
  }, []);

  // Simulate new events coming in
  useEffect(() => {
    const interval = setInterval(() => {
      const newEvent: SimulationEvent = {
        cycle: events[0]?.cycle + 1 || 50,
        type: ['trade', 'conflict', 'climate', 'deal'][Math.floor(Math.random() * 4)] as SimulationEvent['type'],
        description: generateRandomEvent(),
        regions: ['egypt', 'ethiopia'],
        severity: ['low', 'medium', 'high', 'critical'][Math.floor(Math.random() * 4)] as SimulationEvent['severity'],
      };

      setEvents((prev) => {
        const updated = [newEvent, ...prev.slice(0, maxEvents - 1)];
        return updated;
      });
      setNewEventId(newEvent.cycle);

      setTimeout(() => setNewEventId(null), 1000);
    }, 8000);

    return () => clearInterval(interval);
  }, [events, maxEvents]);

  const generateRandomEvent = (): string => {
    const templates = [
      'Saudi Arabia increased energy exports to India.',
      'Ethiopia reported improved water reserves after rainfall.',
      'Tensions rise between Turkey and Iraq over water rights.',
      'Brazil signed new agricultural trade deal with USA.',
      'China announced renewable energy investment program.',
      'Drought conditions worsening in Australian outback.',
      'Germany and Nigeria strengthened energy cooperation.',
      'Egypt requested emergency water aid from neighbors.',
    ];
    return templates[Math.floor(Math.random() * templates.length)];
  };

  const getEventIcon = (type: SimulationEvent['type']) => {
    switch (type) {
      case 'trade':
        return <Handshake className="w-4 h-4 text-[#00E08A]" />;
      case 'conflict':
        return <ShieldAlert className="w-4 h-4 text-[#FF2D55]" />;
      case 'climate':
        return <AlertTriangle className="w-4 h-4 text-[#FFB800]" />;
      case 'collapse':
        return <Skull className="w-4 h-4 text-[#8B5CF6]" />;
      case 'sanction':
        return <ShieldAlert className="w-4 h-4 text-[#FF2D55]" />;
      case 'deal':
        return <Handshake className="w-4 h-4 text-[#00F0FF]" />;
      default:
        return <AlertTriangle className="w-4 h-4 text-[#A7B1C8]" />;
    }
  };

  const getEventClass = (severity?: SimulationEvent['severity']) => {
    switch (severity) {
      case 'critical':
        return 'critical';
      case 'high':
        return 'warning';
      case 'medium':
        return 'success';
      default:
        return '';
    }
  };

  const getSeverityColor = (severity?: SimulationEvent['severity']) => {
    switch (severity) {
      case 'critical':
        return '#FF2D55';
      case 'high':
        return '#FFB800';
      case 'medium':
        return '#00F0FF';
      default:
        return '#00E08A';
    }
  };

  return (
    <div className="w-full h-full flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-[#F4F7FF]">Live Event Feed</h3>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-[#00E08A] animate-pulse" />
          <span className="text-xs text-[#A7B1C8]">Live</span>
        </div>
      </div>

      <div
        ref={feedRef}
        className="flex-1 overflow-y-auto space-y-2 pr-2"
        style={{ maxHeight: '400px' }}
      >
        {events.map((event, index) => (
          <div
            key={`${event.cycle}-${index}`}
            className={`
              event-item rounded-lg cursor-pointer transition-all duration-300
              ${getEventClass(event.severity)}
              ${newEventId === event.cycle ? 'animate-pulse-glow bg-[#00F0FF]/10' : 'bg-[#0B1022]/50'}
            `}
            onClick={() => onEventClick?.(event)}
          >
            <div className="flex items-start gap-3">
              {/* Icon */}
              <div className="mt-0.5">{getEventIcon(event.type)}</div>

              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs font-mono text-[#00F0FF]">
                    Cycle {event.cycle}
                  </span>
                  {event.severity && (
                    <span
                      className="text-[10px] px-1.5 py-0.5 rounded-full font-medium"
                      style={{
                        background: `${getSeverityColor(event.severity)}20`,
                        color: getSeverityColor(event.severity),
                      }}
                    >
                      {event.severity.toUpperCase()}
                    </span>
                  )}
                </div>
                <p className="text-sm text-[#F4F7FF] leading-relaxed">{event.description}</p>
                {event.regions.length > 0 && (
                  <div className="flex items-center gap-1 mt-2">
                    <span className="text-xs text-[#A7B1C8]">Regions:</span>
                    {event.regions.map((region) => (
                      <span
                        key={region}
                        className="text-xs px-2 py-0.5 rounded bg-[#00F0FF]/10 text-[#00F0FF]"
                      >
                        {region.charAt(0).toUpperCase() + region.slice(1)}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Feed stats */}
      <div className="mt-4 grid grid-cols-4 gap-2">
        {[
          { label: 'Total', value: events.length, color: '#00F0FF' },
          { label: 'Critical', value: events.filter((e) => e.severity === 'critical').length, color: '#FF2D55' },
          { label: 'Deals', value: events.filter((e) => e.type === 'deal').length, color: '#00E08A' },
          { label: 'Conflicts', value: events.filter((e) => e.type === 'conflict').length, color: '#FFB800' },
        ].map((stat) => (
          <div key={stat.label} className="panel p-2 text-center">
            <div className="text-lg font-bold font-mono" style={{ color: stat.color }}>
              {stat.value}
            </div>
            <div className="text-[10px] text-[#A7B1C8]">{stat.label}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default EventFeed;
