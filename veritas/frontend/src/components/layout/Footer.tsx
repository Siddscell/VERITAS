import React from 'react';
import { Shield, ArrowUp } from 'lucide-react';
import { useLenis } from '@/components/layout/SmoothScroll';
import { usePageTransition } from '@/context/TransitionContext';

export function Footer() {
  const { scrollTo } = useLenis();
  const { navigateTo } = usePageTransition();

  return (
    <footer className="border-t border-white/[0.06] bg-[#07090e] py-14 px-6 text-xs font-mono text-zinc-500">
      <div className="max-w-6xl mx-auto flex flex-col md:flex-row justify-between items-start md:items-center gap-8 pb-10 border-b border-white/[0.06]">
        {/* Brand and Description */}
        <div className="space-y-2">
          <div className="flex items-center gap-2.5">
            <Shield className="w-4 h-4 text-blue-400" />
            <span className="font-semibold text-zinc-200 text-sm tracking-wider">VERITAS</span>
            <span className="text-zinc-600">|</span>
            <span className="text-zinc-400">Forensic Intelligence</span>
          </div>
          <p className="text-zinc-500 max-w-sm font-sans text-xs">
            Multi-modal deepfake detection powered by specialized neural agents and dynamic-denominator evidence fusion.
          </p>
        </div>

        {/* Quick Nav Anchors */}
        <div className="flex flex-wrap items-center gap-6 text-zinc-400">
          <button
            onClick={() => scrollTo('#overview')}
            className="hover:text-zinc-200 transition-colors cursor-pointer"
          >
            01 Overview
          </button>
          <button
            onClick={() => scrollTo('#visual')}
            className="hover:text-zinc-200 transition-colors cursor-pointer"
          >
            02 Spatial
          </button>
          <button
            onClick={() => scrollTo('#temporal')}
            className="hover:text-zinc-200 transition-colors cursor-pointer"
          >
            03 Temporal
          </button>
          <button
            onClick={() => scrollTo('#audio')}
            className="hover:text-zinc-200 transition-colors cursor-pointer"
          >
            04 Audio
          </button>
          <button
            onClick={() => scrollTo('#fusion')}
            className="hover:text-zinc-200 transition-colors cursor-pointer"
          >
            05 Fusion
          </button>
          <button
            onClick={() => scrollTo('#console')}
            className="text-blue-400 hover:text-blue-300 font-medium transition-colors cursor-pointer"
          >
            06 Live Console →
          </button>
        </div>
      </div>

      {/* Bottom Metadata & Disclaimer */}
      <div className="max-w-6xl mx-auto pt-6 flex flex-col sm:flex-row justify-between items-center gap-4 text-zinc-600 text-[11px]">
        <div>
          Production Engine: <span className="text-zinc-400">corrected_fixed</span> • PyTorch & FastAPI • React Vite
        </div>
        <div className="flex items-center gap-4">
          <span>Results are analytical assessments for informational verification.</span>
          <button
            onClick={() => scrollTo('#hero')}
            className="hover:text-zinc-400 transition-colors flex items-center gap-1 cursor-pointer"
            title="Return to top"
          >
            <span>Top</span>
            <ArrowUp className="w-3 h-3" />
          </button>
        </div>
      </div>
    </footer>
  );
}
