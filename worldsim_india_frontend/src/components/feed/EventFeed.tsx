import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { events, indiaStates } from '@/data/indiaStates';
import type { Event } from '@/types';
import { ScrollArea } from '@/components/ui/scroll-area';

interface EventFeedProps {
  simulationCycle?: number;
  onEventClick?: (cycle: number) => void;
}

const EventFeed: React.FC<EventFeedProps> = ({ simulationCycle = 0, onEventClick }) => {
  const [displayedEvents, setDisplayedEvents] = useState<Event[]>([]);
  const [newEventId, setNewEventId] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Filter events based on simulation cycle
    const visibleEvents = events.filter(e => e.cycle <= simulationCycle);
    
    // Check if there's a new event
    if (visibleEvents.length > displayedEvents.length) {
      const latestEvent = visibleEvents[visibleEvents.length - 1];
      setNewEventId(latestEvent.id);
      setTimeout(() => setNewEventId(null), 2000);
    }
    
    setDisplayedEvents(visibleEvents);
  }, [simulationCycle, displayedEvents.length]);

  useEffect(() => {
    // Auto-scroll to bottom
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [displayedEvents]);

  const getEventIcon = (type: string) => {
    switch (type) {
      case 'trade': return '💱';
      case 'conflict': return '⚔️';
      case 'alliance': return '🤝';
      case 'crisis': return '⚠️';
      default: return '📰';
    }
  };

  const getEventColor = (type: string) => {
    switch (type) {
      case 'trade': return 'border-cyan-500/50 bg-cyan-500/10';
      case 'conflict': return 'border-red-500/50 bg-red-500/10';
      case 'alliance': return 'border-green-500/50 bg-green-500/10';
      case 'crisis': return 'border-orange-500/50 bg-orange-500/10';
      default: return 'border-gray-500/50 bg-gray-500/10';
    }
  };

  const getStateNames = (stateIds: string[]) => {
    return stateIds.map(id => indiaStates.find(s => s.id === id)?.name || id).join(', ');
  };

  return (
    <div className="relative w-full h-full bg-[#0a0a0f] rounded-xl overflow-hidden">
      {/* Grid Background */}
      <div className="absolute inset-0 grid-bg opacity-50" />
      
      {/* Header */}
      <div className="relative z-10 p-4 border-b border-gray-800">
        <div className="flex justify-between items-center">
          <div>
            <h3 className="text-xl font-bold text-white neon-text">Live Event Feed</h3>
            <p className="text-sm text-gray-400 mt-1">Real-time interstate developments</p>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            <span className="text-xs text-gray-400">Live</span>
          </div>
        </div>
      </div>

      {/* Event List */}
      <ScrollArea className="h-[calc(100%-80px)]" ref={scrollRef}>
        <div className="p-4 space-y-3">
          <AnimatePresence>
            {displayedEvents.length === 0 ? (
              <motion.div 
                className="text-center py-8 text-gray-500"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                <div className="text-4xl mb-2">📰</div>
                <p className="text-sm">No events yet. Start the simulation...</p>
              </motion.div>
            ) : (
              displayedEvents.map((event, index) => (
                <motion.div
                  key={event.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ 
                    opacity: 1, 
                    x: 0,
                    scale: newEventId === event.id ? [1, 1.02, 1] : 1,
                  }}
                  transition={{ 
                    delay: index * 0.05,
                    scale: { duration: 0.3 }
                  }}
                  className={`
                    relative p-3 rounded-lg border cursor-pointer transition-all
                    ${getEventColor(event.type)}
                    ${newEventId === event.id ? 'ring-2 ring-cyan-400' : ''}
                    hover:brightness-110
                  `}
                  onClick={() => onEventClick?.(event.cycle)}
                >
                  {/* New Event Indicator */}
                  {newEventId === event.id && (
                    <motion.div
                      className="absolute -left-1 top-1/2 -translate-y-1/2 w-1 h-8 bg-cyan-400 rounded-full"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: [0, 1, 0] }}
                      transition={{ duration: 1, repeat: 2 }}
                    />
                  )}

                  <div className="flex items-start gap-3">
                    {/* Icon */}
                    <div className="text-2xl">{getEventIcon(event.type)}</div>
                    
                    {/* Content */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-xs font-mono text-cyan-400">
                          Cycle {event.cycle}
                        </span>
                        <span className="text-xs text-gray-500">•</span>
                        <span className="text-xs text-gray-500 capitalize">
                          {event.type}
                        </span>
                      </div>
                      
                      <p className="text-sm text-gray-200 leading-relaxed">
                        {event.description}
                      </p>
                      
                      <div className="mt-2 flex items-center gap-2">
                        <span className="text-xs text-gray-500">States:</span>
                        <span className="text-xs text-cyan-400">
                          {getStateNames(event.states)}
                        </span>
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))
            )}
          </AnimatePresence>
        </div>
      </ScrollArea>

      {/* Event Type Legend */}
      <div className="absolute bottom-4 left-4 z-10 glass rounded-lg p-2 flex gap-3">
        <div className="flex items-center gap-1">
          <span className="text-xs">💱</span>
          <span className="text-xs text-gray-400">Trade</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="text-xs">⚔️</span>
          <span className="text-xs text-gray-400">Conflict</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="text-xs">🤝</span>
          <span className="text-xs text-gray-400">Alliance</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="text-xs">⚠️</span>
          <span className="text-xs text-gray-400">Crisis</span>
        </div>
      </div>

      {/* Stats */}
      <div className="absolute bottom-4 right-4 z-10 glass rounded-lg p-2">
        <div className="text-xs text-gray-400">
          Events: <span className="text-cyan-400 font-semibold">{displayedEvents.length}</span>
        </div>
      </div>
    </div>
  );
};

export default EventFeed;
