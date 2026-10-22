import React, { useEffect } from 'react';
import { useLocation } from 'wouter';
import { StationId, STATIONS } from './types';
import { 
  Scan, 
  Layers, 
  Activity, 
  Network, 
  Cpu, 
  ArrowRight, 
  RotateCcw, 
  ExternalLink,
  ShieldCheck,
  Sparkles
} from 'lucide-react';

interface LabHudOverlayProps {
  activeStationId: StationId;
  onSelectStation: (id: StationId) => void;
  hoveredStationId: StationId | null;
}

const STATION_ICONS: Record<Exclude<StationId, 'overview'>, React.ReactNode> = {
  spatial: <Scan className="w-4 h-4 text-cyan-400" />,
  temporal: <Layers className="w-4 h-4 text-blue-400" />,
  audio: <Activity className="w-4 h-4 text-emerald-400" />,
  context: <Network className="w-4 h-4 text-purple-400" />,
  fusion: <Cpu className="w-4 h-4 text-amber-400" />,
};

export const LabHudOverlay: React.FC<LabHudOverlayProps> = ({
  activeStationId,
  onSelectStation,
  hoveredStationId,
}) => {
  const [, setLocation] = useLocation();

  // Keyboard shortcut support
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onSelectStation('overview');
      } else if (e.key === '0') {
        onSelectStation('overview');
      } else if (e.key === '1') {
        onSelectStation('spatial');
      } else if (e.key === '2') {
        onSelectStation('temporal');
      } else if (e.key === '3') {
        onSelectStation('audio');
      } else if (e.key === '4') {
        onSelectStation('context');
      } else if (e.key === '5') {
        onSelectStation('fusion');
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onSelectStation]);

  const activeStation = activeStationId !== 'overview' ? STATIONS[activeStationId] : null;

  return (
    <div className="absolute inset-0 pointer-events-none z-10 flex flex-col justify-between pt-24 pb-6 px-4 md:px-8">
      {/* 1. TOP STATUS & NAVIGATION BAR (Positioned below main NavDock) */}
      <div className="flex flex-wrap items-center justify-between gap-4 pointer-events-auto">
        {/* System telemetry pill */}
        <div className="flex items-center gap-3 px-3.5 py-1.5 rounded-full bg-zinc-950/85 border border-zinc-800/80 backdrop-blur-md shadow-lg shadow-black/50">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500" />
          </span>
          <span className="text-xs font-mono tracking-wider text-zinc-300 uppercase">
            3D Forensics Lab
          </span>
          <span className="text-zinc-600 font-mono text-xs">|</span>
          <span className="text-xs font-mono text-cyan-400/90 hidden sm:inline">
            Production: corrected_fixed
          </span>
        </div>

        {/* Station Navigation Quick Selector Tabs */}
        <div className="flex items-center gap-1.5 p-1 rounded-xl bg-zinc-950/85 border border-zinc-800/80 backdrop-blur-md shadow-lg">
          <button
            onClick={() => onSelectStation('overview')}
            className={`px-3 py-1 rounded-lg text-xs font-mono font-medium transition-all cursor-pointer ${
              activeStationId === 'overview'
                ? 'bg-zinc-800 text-white shadow-inner'
                : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900/60'
            }`}
          >
            Overview
          </button>

          {(['spatial', 'temporal', 'audio', 'context', 'fusion'] as const).map((id) => {
            const s = STATIONS[id];
            const isActive = activeStationId === id;
            const isHovered = hoveredStationId === id;

            return (
              <button
                key={id}
                onClick={() => onSelectStation(id)}
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-mono transition-all cursor-pointer ${
                  isActive
                    ? 'bg-cyan-950/60 border border-cyan-500/40 text-cyan-300 shadow-sm shadow-cyan-500/20'
                    : isHovered
                    ? 'bg-zinc-900 text-zinc-100 border border-zinc-700'
                    : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900/50'
                }`}
              >
                {STATION_ICONS[id]}
                <span className="hidden md:inline">{s.name.replace(' Analyst', '')}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* 2. STATION INSPECTOR PANEL (When station is active) */}
      {activeStation && (
        <div className="max-w-xl w-full my-auto ml-0 md:ml-4 p-5 md:p-6 rounded-2xl bg-zinc-950/90 border border-zinc-800/90 backdrop-blur-xl shadow-2xl shadow-black/80 pointer-events-auto animate-in fade-in slide-in-from-left-4 duration-300">
          {/* Badge & Close Button */}
          <div className="flex items-center justify-between pb-3 border-b border-zinc-800/80">
            <div className="flex items-center gap-2.5">
              <div className="p-1.5 rounded-lg bg-zinc-900 border border-zinc-800">
                {STATION_ICONS[activeStation.id as Exclude<StationId, 'overview'>]}
              </div>
              <div>
                <span className="text-[10px] font-mono tracking-widest uppercase text-cyan-400 font-semibold">
                  {activeStation.badge}
                </span>
                <h3 className="text-lg font-bold text-white tracking-tight">
                  {activeStation.name}
                </h3>
              </div>
            </div>

            <button
              onClick={() => onSelectStation('overview')}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-mono text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900 border border-transparent hover:border-zinc-800 transition-all cursor-pointer"
              title="Return to Lab Overview (ESC)"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Overview</span>
            </button>
          </div>

          {/* Model Specification Pill */}
          <div className="my-3 px-3 py-2 rounded-lg bg-zinc-900/60 border border-zinc-800 text-xs font-mono flex items-center justify-between">
            <span className="text-zinc-400">Architecture:</span>
            <span className="text-cyan-300 font-medium">{activeStation.modelName}</span>
          </div>

          {/* Narrative Summary */}
          <p className="text-xs text-zinc-300 leading-relaxed mb-4">
            {activeStation.summary}
          </p>

          {/* Technical Details Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-[11px] font-mono mb-4">
            <div className="p-2.5 rounded-lg bg-zinc-900/40 border border-zinc-800/60">
              <span className="text-zinc-500 block mb-0.5">Input Specification:</span>
              <span className="text-zinc-200">{activeStation.techDetails.inputSpec}</span>
            </div>
            <div className="p-2.5 rounded-lg bg-zinc-900/40 border border-zinc-800/60">
              <span className="text-zinc-500 block mb-0.5">Target Artifacts:</span>
              <span className="text-zinc-200">{activeStation.techDetails.targetArtifacts}</span>
            </div>
          </div>

          {/* Key Principles Checklist */}
          <div className="space-y-1.5 mb-5">
            <span className="text-[10px] font-mono tracking-wider uppercase text-zinc-500 block">
              Forensic Operational Principles:
            </span>
            {activeStation.detectionPrinciples.map((principle, idx) => (
              <div key={idx} className="flex items-start gap-2 text-xs text-zinc-400">
                <ShieldCheck className="w-3.5 h-3.5 text-cyan-400 shrink-0 mt-0.5" />
                <span>{principle}</span>
              </div>
            ))}
          </div>

          {/* Action Row */}
          <div className="flex items-center gap-3 pt-3 border-t border-zinc-800/80">
            <button
              onClick={() => setLocation('/console')}
              className="flex-1 flex items-center justify-center gap-2 py-2 px-4 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-black text-xs font-bold tracking-wider uppercase transition-all shadow-md shadow-cyan-500/20 cursor-pointer"
            >
              <span>Test in Verification Console</span>
              <ExternalLink className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => onSelectStation('overview')}
              className="py-2 px-4 rounded-xl bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-xs text-zinc-300 font-mono transition-all cursor-pointer"
            >
              Back
            </button>
          </div>
        </div>
      )}

      {/* 3. BOTTOM OVERVIEW GUIDE & HINTS */}
      {activeStationId === 'overview' && (
        <div className="flex flex-col sm:flex-row items-center justify-between gap-3 p-3 rounded-2xl bg-zinc-950/80 border border-zinc-800/80 backdrop-blur-md shadow-xl pointer-events-auto">
          <div className="flex items-center gap-2.5 text-xs text-zinc-400">
            <Sparkles className="w-4 h-4 text-cyan-400 shrink-0" />
            <span>
              Click any of the <strong className="text-zinc-200">four agent stations</strong> or the <strong className="text-amber-300">Central Fusion Core</strong> to inspect forensic methodology.
            </span>
          </div>

          <div className="flex items-center gap-2 text-[11px] font-mono text-zinc-500">
            <kbd className="px-1.5 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-zinc-300">1-5</kbd>
            <span>Jump to Station</span>
            <span className="text-zinc-700">|</span>
            <kbd className="px-1.5 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-zinc-300">ESC</kbd>
            <span>Overview</span>
          </div>
        </div>
      )}
    </div>
  );
};
