import { useEffect, useRef, useState } from 'react';
import { Globe, Cpu, Network, Zap, MessageSquare, ChevronRight, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Sheet, SheetContent, SheetTrigger } from '@/components/ui/sheet';

// Voiceflow Chat Widget Component
const VoiceflowChat = () => {
  useEffect(() => {
    const script = document.createElement('script');
    script.type = 'text/javascript';
    script.src = 'https://cdn.voiceflow.com/widget-next/bundle.mjs';
    script.onload = () => {
      // @ts-ignore
      if (window.voiceflow && window.voiceflow.chat) {
        // @ts-ignore
        window.voiceflow.chat.load({
          verify: { projectID: '69a10ff387c29b7a7f15e3d8' },
          url: 'https://general-runtime.voiceflow.com',
          versionID: 'production',
          voice: {
            url: 'https://runtime-api.voiceflow.com'
          }
        });
      }
    };
    document.body.appendChild(script);

    return () => {
      document.body.removeChild(script);
    };
  }, []);

  return null;
};

// Animated Background Component
const AnimatedBackground = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    // Create particles representing countries/agents
    const particles: Array<{
      x: number;
      y: number;
      vx: number;
      vy: number;
      radius: number;
      color: string;
      connections: number[];
    }> = [];

    const colors = ['#00d4ff', '#7c3aed', '#f59e0b', '#10b981', '#ef4444'];
    const particleCount = 60;

    for (let i = 0; i < particleCount; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
        radius: Math.random() * 3 + 2,
        color: colors[Math.floor(Math.random() * colors.length)],
        connections: []
      });
    }

    let animationId: number;

    const animate = () => {
      ctx.fillStyle = 'rgba(10, 10, 15, 0.1)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Update and draw particles
      particles.forEach((particle, i) => {
        particle.x += particle.vx;
        particle.y += particle.vy;

        // Wrap around edges
        if (particle.x < 0) particle.x = canvas.width;
        if (particle.x > canvas.width) particle.x = 0;
        if (particle.y < 0) particle.y = canvas.height;
        if (particle.y > canvas.height) particle.y = 0;

        // Draw particle
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2);
        ctx.fillStyle = particle.color;
        ctx.fill();

        // Draw glow
        const gradient = ctx.createRadialGradient(
          particle.x, particle.y, 0,
          particle.x, particle.y, particle.radius * 3
        );
        gradient.addColorStop(0, particle.color + '40');
        gradient.addColorStop(1, 'transparent');
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.radius * 3, 0, Math.PI * 2);
        ctx.fill();

        // Draw connections
        particles.forEach((other, j) => {
          if (i === j) return;
          const dx = particle.x - other.x;
          const dy = particle.y - other.y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < 150) {
            ctx.beginPath();
            ctx.moveTo(particle.x, particle.y);
            ctx.lineTo(other.x, other.y);
            ctx.strokeStyle = `rgba(0, 212, 255, ${0.15 * (1 - distance / 150)})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        });
      });

      animationId = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resize);
      cancelAnimationFrame(animationId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 z-0"
      style={{ background: 'linear-gradient(135deg, #0a0a0f 0%, #12121a 50%, #0f0f15 100%)' }}
    />
  );
};

// Navigation Component
const Navigation = () => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 glass">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-400 to-purple-600 flex items-center justify-center animate-pulse-glow">
                <Globe className="w-6 h-6 text-white" />
              </div>
            </div>
            <span className="text-xl font-bold bg-gradient-to-r from-cyan-400 via-purple-500 to-amber-400 bg-clip-text text-transparent">
              WORLDSIM
            </span>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-6">
            <a href="#about" className="text-gray-300 hover:text-cyan-400 transition-colors text-sm font-medium">
              About
            </a>
            <a href="#simulations" className="text-gray-300 hover:text-cyan-400 transition-colors text-sm font-medium">
              Simulations
            </a>
            <Sheet open={isOpen} onOpenChange={setIsOpen}>
              <SheetTrigger asChild>
                <Button
                  variant="outline"
                  className="border-cyan-500/50 text-cyan-400 hover:bg-cyan-500/10 hover:border-cyan-400"
                >
                  <MessageSquare className="w-4 h-4 mr-2" />
                  AI Assistant
                </Button>
              </SheetTrigger>
              <SheetContent side="right" className="w-[400px] bg-[#0f0f15] border-l border-[#2a2a3a]">
                <div className="h-full flex flex-col">
                  <div className="flex items-center gap-3 mb-6">
                    <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-400 to-purple-600 flex items-center justify-center">
                      <MessageSquare className="w-5 h-5 text-white" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-white">AI Assistant</h3>
                      <p className="text-xs text-gray-400">Powered by Voiceflow</p>
                    </div>
                  </div>
                  <div className="flex-1 bg-[#1a1a25] rounded-xl p-4 overflow-hidden">
                    <div className="h-full flex items-center justify-center text-gray-400 text-sm text-center">
                      <div>
                        <Sparkles className="w-8 h-8 mx-auto mb-2 text-cyan-400" />
                        <p>Chat widget loading...</p>
                        <p className="text-xs mt-1">Ask me anything about WORLDSIM!</p>
                      </div>
                    </div>
                  </div>
                </div>
              </SheetContent>
            </Sheet>
          </div>
        </div>
      </div>
    </nav>
  );
};

// Hero Section
const HeroSection = () => {
  return (
    <section className="relative min-h-screen flex items-center justify-center pt-16">
      <div className="relative z-10 max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        {/* Badge */}
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-gradient-to-r from-cyan-500/10 to-purple-500/10 border border-cyan-500/30 mb-8 animate-float">
          <Cpu className="w-4 h-4 text-cyan-400" />
          <span className="text-sm font-medium text-cyan-300">AI-Powered Global Simulations</span>
        </div>

        {/* Main Title */}
        <h1 className="text-5xl sm:text-6xl md:text-7xl lg:text-8xl font-bold mb-6">
          <span className="bg-gradient-to-r from-cyan-400 via-purple-500 to-amber-400 bg-clip-text text-transparent animate-gradient-shift">
            WORLDSIM
          </span>
        </h1>

        {/* Subtitle */}
        <p className="text-xl sm:text-2xl md:text-3xl text-gray-300 mb-4 font-light">
          Where AI Agents Become Nations
        </p>

        {/* Description */}
        <p className="max-w-3xl mx-auto text-gray-400 text-base sm:text-lg mb-12 leading-relaxed">
          Experience the future of geopolitical simulation. Our advanced AI agents, powered by 
          <span className="text-cyan-400 font-medium"> Reinforcement Learning</span>, act as sovereign nations, 
          making strategic decisions, forming alliances, and shaping the world order in real-time.
        </p>

        {/* Feature Icons */}
        <div className="flex flex-wrap justify-center gap-6 mb-16">
          {[
            { icon: Network, label: 'Multi-Agent Systems' },
            { icon: Zap, label: 'Real-Time Decisions' },
            { icon: Cpu, label: 'Neural Networks' },
            { icon: Globe, label: 'Global Scale' },
          ].map((feature, index) => (
            <div
              key={index}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white/5 border border-white/10"
            >
              <feature.icon className="w-4 h-4 text-purple-400" />
              <span className="text-sm text-gray-300">{feature.label}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

// Simulations Section with Buttons
const SimulationsSection = () => {
  const simulations = [
    {
      title: 'Global Simulations',
      description: 'Watch AI agents representing major world powers compete and cooperate on the global stage.',
      url: 'https://world-mappo.vercel.app/',
      gradient: 'from-cyan-500 via-blue-500 to-purple-600',
      icon: Globe,
      delay: 0
    },
    {
      title: 'India Simulations',
      description: 'Explore India\'s strategic position with AI-driven regional and domestic scenario modeling.',
      url: 'https://sitnovate-2-0.vercel.app/',
      gradient: 'from-orange-500 via-amber-500 to-yellow-500',
      icon: Network,
      delay: 0.1
    },
    {
      title: 'Agentic Recent Simulations',
      description: 'Experience our latest breakthrough in autonomous agent behavior and conflict resolution.',
      url: 'https://worldweave-conflicts.lovable.app/',
      gradient: 'from-purple-500 via-pink-500 to-rose-500',
      icon: Cpu,
      delay: 0.2
    }
  ];

  return (
    <section id="simulations" className="relative z-10 py-20">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center mb-16">
          <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold mb-4">
            <span className="bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
              Choose Your Simulation
            </span>
          </h2>
          <p className="text-gray-400 text-lg max-w-2xl mx-auto">
            Select from our suite of AI-powered simulation environments
          </p>
        </div>

        {/* Buttons Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-5xl mx-auto">
          {simulations.map((sim, index) => (
            <a
              key={index}
              href={sim.url}
              target="_blank"
              rel="noopener noreferrer"
              className="group relative"
              style={{ animationDelay: `${sim.delay}s` }}
            >
              <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-[#1a1a25] to-[#12121a] border border-white/10 p-1 transition-all duration-500 hover:scale-105 hover:border-white/20">
                {/* Glow effect */}
                <div className={`absolute inset-0 bg-gradient-to-r ${sim.gradient} opacity-0 group-hover:opacity-20 transition-opacity duration-500 blur-xl`} />
                
                <div className="relative p-6 sm:p-8">
                  {/* Icon */}
                  <div className={`w-14 h-14 rounded-xl bg-gradient-to-br ${sim.gradient} flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300`}>
                    <sim.icon className="w-7 h-7 text-white" />
                  </div>

                  {/* Title */}
                  <h3 className="text-xl sm:text-2xl font-bold text-white mb-3 group-hover:text-transparent group-hover:bg-clip-text group-hover:bg-gradient-to-r group-hover:from-white group-hover:to-gray-300 transition-all">
                    {sim.title}
                  </h3>

                  {/* Description */}
                  <p className="text-gray-400 text-sm leading-relaxed mb-6">
                    {sim.description}
                  </p>

                  {/* Button */}
                  <div className={`inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-gradient-to-r ${sim.gradient} text-white font-semibold text-sm btn-glow shadow-lg group-hover:shadow-xl transition-all duration-300`}>
                    <span>Launch Simulation</span>
                    <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                  </div>
                </div>

                {/* Corner decoration */}
                <div className={`absolute top-0 right-0 w-20 h-20 bg-gradient-to-br ${sim.gradient} opacity-10 rounded-bl-full`} />
              </div>
            </a>
          ))}
        </div>
      </div>
    </section>
  );
};

// About Section
const AboutSection = () => {
  return (
    <section id="about" className="relative z-10 py-20">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          {/* Left Content */}
          <div>
            <h2 className="text-3xl sm:text-4xl font-bold mb-6">
              <span className="bg-gradient-to-r from-cyan-400 to-purple-500 bg-clip-text text-transparent">
                The Future of Simulation
              </span>
            </h2>
            <p className="text-gray-400 text-lg leading-relaxed mb-6">
              WORLDSIM leverages cutting-edge reinforcement learning algorithms to create 
              autonomous AI agents that behave like real-world nations. Each agent learns, 
              adapts, and evolves its strategies based on complex reward systems and 
              environmental feedback.
            </p>
            <div className="space-y-4">
              {[
                'Deep Q-Networks for strategic decision making',
                'Multi-agent reinforcement learning frameworks',
                'Real-time policy adaptation and evolution',
                'Complex geopolitical scenario modeling'
              ].map((item, index) => (
                <div key={index} className="flex items-center gap-3">
                  <div className="w-2 h-2 rounded-full bg-cyan-400" />
                  <span className="text-gray-300">{item}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Right Visual */}
          <div className="relative">
            <div className="relative aspect-square max-w-md mx-auto">
              {/* Orbital rings */}
              <div className="absolute inset-0 border-2 border-cyan-500/20 rounded-full animate-rotate-slow" />
              <div className="absolute inset-8 border-2 border-purple-500/20 rounded-full animate-rotate-slow" style={{ animationDirection: 'reverse', animationDuration: '30s' }} />
              <div className="absolute inset-16 border-2 border-amber-500/20 rounded-full animate-rotate-slow" style={{ animationDuration: '25s' }} />
              
              {/* Center globe */}
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="w-32 h-32 rounded-full bg-gradient-to-br from-cyan-500 via-purple-500 to-amber-500 flex items-center justify-center animate-pulse-glow">
                  <Globe className="w-16 h-16 text-white" />
                </div>
              </div>

              {/* Orbiting dots */}
              {[0, 1, 2].map((i) => (
                <div
                  key={i}
                  className="absolute inset-0"
                  style={{ animation: `rotate-slow ${15 + i * 5}s linear infinite` }}
                >
                  <div
                    className={`absolute w-4 h-4 rounded-full ${
                      i === 0 ? 'bg-cyan-400' : i === 1 ? 'bg-purple-400' : 'bg-amber-400'
                    }`}
                    style={{
                      top: '50%',
                      left: i === 0 ? '0%' : i === 1 ? '100%' : '50%',
                      transform: 'translate(-50%, -50%)'
                    }}
                  />
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

// Footer
const Footer = () => {
  return (
    <footer className="relative z-10 py-12 border-t border-white/10">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-400 to-purple-600 flex items-center justify-center">
              <Globe className="w-5 h-5 text-white" />
            </div>
            <span className="text-lg font-bold bg-gradient-to-r from-cyan-400 to-purple-500 bg-clip-text text-transparent">
              WORLDSIM
            </span>
          </div>

          {/* Copyright */}
          <p className="text-gray-500 text-sm">
            © 2025 WORLDSIM. AI Agents powered by Reinforcement Learning.
          </p>

          {/* Links */}
          <div className="flex items-center gap-6">
            <a href="#" className="text-gray-400 hover:text-cyan-400 transition-colors text-sm">
              Privacy
            </a>
            <a href="#" className="text-gray-400 hover:text-cyan-400 transition-colors text-sm">
              Terms
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
};

// Main App Component
function App() {
  return (
    <div className="relative min-h-screen overflow-x-hidden">
      <AnimatedBackground />
      <Navigation />
      <main>
        <HeroSection />
        <SimulationsSection />
        <AboutSection />
      </main>
      <Footer />
      <VoiceflowChat />
    </div>
  );
}

export default App;
