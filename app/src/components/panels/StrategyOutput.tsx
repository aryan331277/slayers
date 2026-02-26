import { useRef } from 'react';
import type { AllocationStrategy } from '@/types';
import {
  Printer,
  Share2,
  CheckCircle2,
  AlertCircle,
  Clock,
  TrendingUp,
  Shield,
  Droplets,
  Wheat,
  Zap,
  Handshake,
} from 'lucide-react';
import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';

interface StrategyOutputProps {
  strategy: AllocationStrategy;
  onClose?: () => void;
}

const StrategyOutput = ({ strategy, onClose }: StrategyOutputProps) => {
  const printRef = useRef<HTMLDivElement>(null);

  const handlePrint = async () => {
    if (!printRef.current) return;

    const canvas = await html2canvas(printRef.current, {
      backgroundColor: '#070A12',
      scale: 2,
    });

    const imgData = canvas.toDataURL('image/png');
    const pdf = new jsPDF('p', 'mm', 'a4');
    const pdfWidth = pdf.internal.pageSize.getWidth();
    const pdfHeight = (canvas.height * pdfWidth) / canvas.width;

    pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
    pdf.save(`MAPPO-Strategy-${strategy.region}-${Date.now()}.pdf`);
  };

  const handleShare = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: `MAPPO Strategy: ${strategy.region}`,
          text: `Resource allocation strategy for ${strategy.region}: ${strategy.scenario}`,
        });
      } catch (err) {
        console.log('Share cancelled');
      }
    } else {
      // Fallback: copy to clipboard
      navigator.clipboard.writeText(
        `MAPPO Strategy for ${strategy.region}:\n${strategy.scenario}\n\n${strategy.recommendations
          .map((r) => `- ${r.action} (${r.priority})`)
          .join('\n')}`
      );
      alert('Strategy copied to clipboard!');
    }
  };

  const getResourceIcon = (resource: string) => {
    switch (resource) {
      case 'water':
        return <Droplets className="w-4 h-4 text-[#00F0FF]" />;
      case 'food':
        return <Wheat className="w-4 h-4 text-[#00E08A]" />;
      case 'energy':
        return <Zap className="w-4 h-4 text-[#FFB800]" />;
      default:
        return <TrendingUp className="w-4 h-4 text-[#A7B1C8]" />;
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'immediate':
        return '#FF2D55';
      case 'short-term':
        return '#FFB800';
      case 'long-term':
        return '#00E08A';
      default:
        return '#A7B1C8';
    }
  };

  const getPriorityIcon = (priority: string) => {
    switch (priority) {
      case 'immediate':
        return <AlertCircle className="w-4 h-4" />;
      case 'short-term':
        return <Clock className="w-4 h-4" />;
      case 'long-term':
        return <CheckCircle2 className="w-4 h-4" />;
      default:
        return <TrendingUp className="w-4 h-4" />;
    }
  };

  return (
    <div className="w-full">
      {/* Action buttons */}
      <div className="flex items-center justify-end gap-2 mb-4 no-print">
        <button
          onClick={handlePrint}
          className="flex items-center gap-2 px-4 py-2 bg-[#0B1022] text-[#00F0FF] rounded-lg border border-[#00F0FF]/30 hover:bg-[#00F0FF]/10 transition-all duration-300"
        >
          <Printer className="w-4 h-4" />
          <span className="text-sm">Print PDF</span>
        </button>
        <button
          onClick={handleShare}
          className="flex items-center gap-2 px-4 py-2 bg-[#0B1022] text-[#00E08A] rounded-lg border border-[#00E08A]/30 hover:bg-[#00E08A]/10 transition-all duration-300"
        >
          <Share2 className="w-4 h-4" />
          <span className="text-sm">Share</span>
        </button>
        {onClose && (
          <button
            onClick={onClose}
            className="px-4 py-2 text-[#A7B1C8] hover:text-[#F4F7FF] transition-colors"
          >
            Close
          </button>
        )}
      </div>

      {/* Printable content */}
      <div ref={printRef} className="panel p-6">
        {/* Header */}
        <div className="border-b border-[#00F0FF]/20 pb-4 mb-6">
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center gap-2 mb-2">
                <div className="w-8 h-8 rounded-lg bg-[#00F0FF]/20 flex items-center justify-center">
                  <TrendingUp className="w-4 h-4 text-[#00F0FF]" />
                </div>
                <span className="text-sm text-[#00F0FF] font-medium">MAPPO ALLOCATION STRATEGY</span>
              </div>
              <h2 className="text-2xl font-bold text-[#F4F7FF]">{strategy.region}</h2>
            </div>
            <div className="text-right">
              <div className="text-xs text-[#A7B1C8]">Generated</div>
              <div className="text-sm font-mono text-[#F4F7FF]">
                {new Date().toLocaleDateString()} {new Date().toLocaleTimeString()}
              </div>
            </div>
          </div>
          <div className="mt-3 p-3 bg-[#070A12] rounded-lg">
            <div className="text-xs text-[#A7B1C8] mb-1">Scenario</div>
            <p className="text-sm text-[#F4F7FF]">{strategy.scenario}</p>
          </div>
        </div>

        {/* Recommendations */}
        <div className="mb-6">
          <h3 className="text-lg font-semibold text-[#F4F7FF] mb-4 flex items-center gap-2">
            <CheckCircle2 className="w-5 h-5 text-[#00E08A]" />
            Strategic Recommendations
          </h3>
          <div className="space-y-3">
            {strategy.recommendations.map((rec, index) => (
              <div
                key={index}
                className="p-4 bg-[#070A12] rounded-lg border-l-4"
                style={{ borderLeftColor: getPriorityColor(rec.priority) }}
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-2">
                    {getResourceIcon(rec.resource)}
                    <span className="text-sm font-medium text-[#F4F7FF] capitalize">{rec.resource}</span>
                  </div>
                  <div
                    className="flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium"
                    style={{
                      background: `${getPriorityColor(rec.priority)}20`,
                      color: getPriorityColor(rec.priority),
                    }}
                  >
                    {getPriorityIcon(rec.priority)}
                    <span className="capitalize">{rec.priority}</span>
                  </div>
                </div>
                <p className="text-sm text-[#F4F7FF] mb-1">{rec.action}</p>
                <p className="text-xs text-[#A7B1C8]">{rec.details}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Trade Proposals */}
        <div className="mb-6">
          <h3 className="text-lg font-semibold text-[#F4F7FF] mb-4 flex items-center gap-2">
            <Handshake className="w-5 h-5 text-[#00F0FF]" />
            Proposed Trade Agreements
          </h3>
          <div className="space-y-3">
            {strategy.tradeProposals.map((proposal, index) => (
              <div key={index} className="p-4 bg-[#070A12] rounded-lg border border-[#00F0FF]/20">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    {getResourceIcon(proposal.resource)}
                    <span className="text-sm font-medium text-[#F4F7FF] capitalize">
                      {proposal.resource} Supply
                    </span>
                  </div>
                  <span className="text-xs font-mono text-[#00F0FF]">
                    {proposal.amount.toLocaleString()} units
                  </span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-[#A7B1C8]">Partner: <span className="text-[#F4F7FF]">{proposal.partner}</span></span>
                </div>
                <p className="text-xs text-[#A7B1C8] mt-2">{proposal.terms}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Risk Mitigation */}
        <div>
          <h3 className="text-lg font-semibold text-[#F4F7FF] mb-4 flex items-center gap-2">
            <Shield className="w-5 h-5 text-[#FFB800]" />
            Risk Mitigation Measures
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {strategy.riskMitigation.map((measure, index) => (
              <div key={index} className="flex items-start gap-2 p-3 bg-[#070A12] rounded-lg">
                <div className="w-5 h-5 rounded-full bg-[#FFB800]/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <span className="text-[10px] text-[#FFB800] font-bold">{index + 1}</span>
                </div>
                <span className="text-sm text-[#A7B1C8]">{measure}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="mt-6 pt-4 border-t border-[#00F0FF]/10 text-center">
          <p className="text-xs text-[#A7B1C8]">
            Generated by MAPPO Multi-Agent Resource Allocation System
          </p>
          <p className="text-[10px] text-[#A7B1C8]/60 mt-1">
            This strategy is based on simulation data and should be reviewed by domain experts.
          </p>
        </div>
      </div>
    </div>
  );
};

export default StrategyOutput;
