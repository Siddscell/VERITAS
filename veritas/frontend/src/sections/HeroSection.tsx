import React from 'react';
import { motion } from 'framer-motion';
import { ArrowRight, ChevronDown, Shield, CheckCircle2, Layers, Cpu } from 'lucide-react';
import { usePageTransition } from '@/context/TransitionContext';
import { useLenis } from '@/components/layout/SmoothScroll';

const FADE_IN = {
  initial: { opacity: 0, y: 16 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6, ease: [0.16, 1, 0.3, 1] },
};

export function HeroSection() {
  const { navigateTo } = usePageTransition();
  const { scrollTo } = useLenis();

  return (
    <section
      id="hero"
      className="relative min-h-[92vh] flex flex-col justify-center pt-28 pb-20 px-6 max-w-6xl mx-auto border-b border-white/[0.06]"
    >
      {/* Subtle background gradient highlight */}
      <div 
        className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[350px] bg-blue-600/[0.04] rounded-full blur-[140px] pointer-events-none" 
        aria-hidden="true"
      />

      <div className="relative z-10 max-w-4xl">
        {/* Status indicator badge */}
        <motion.div
          {...FADE_IN}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="inline-flex items-center gap-2.5 px-3 py-1 rounded-full border border-white/[0.1] bg-white/[0.03] backdrop-blur-sm text-xs font-mono text-zinc-400 mb-8"
        >
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-60" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500" />
          </span>
          <span className="text-zinc-300 font-medium tracking-wide">VERITAS 1.0</span>
          <span className="text-zinc-600">|</span>
          <span className="text-zinc-400">Multi-Agent Synthetic Media Forensics</span>
        </motion.div>

        {/* Main Headline */}
        <motion.h1
          {...FADE_IN}
          transition={{ duration: 0.6, delay: 0.15 }}
          className="text-4xl sm:text-6xl md:text-7xl font-sans font-semibold tracking-tight text-white leading-[1.08] mb-6"
        >
          Deepfake detection driven by <span className="text-zinc-400">multi-modal evidence</span>.
        </motion.h1>

        {/* Subheading */}
        <motion.p
          {...FADE_IN}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="text-lg sm:text-xl text-zinc-400 max-w-2xl font-normal leading-relaxed mb-10"
        >
          VERITAS coordinates four specialized neural agents across spatial, temporal, acoustic, and cross-modal dimensions to produce explainable authenticity assessments on digital media.
        </motion.p>

        {/* Primary Action Buttons */}
        <motion.div
          {...FADE_IN}
          transition={{ duration: 0.6, delay: 0.25 }}
          className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3.5 mb-16"
        >
          <button
            onClick={() => navigateTo('/console')}
            data-testid="button-hero-launch-console"
            className="inline-flex items-center justify-center gap-2.5 px-6 py-3.5 rounded-lg bg-zinc-100 hover:bg-white text-zinc-950 font-medium text-sm tracking-tight transition-all duration-200 shadow-sm hover:shadow-md cursor-pointer"
          >
            <span>Launch Verification Console</span>
            <ArrowRight className="w-4 h-4 text-zinc-950" />
          </button>

          <button
            onClick={() => scrollTo('#how-it-works')}
            data-testid="button-hero-learn-more"
            className="inline-flex items-center justify-center gap-2 px-5 py-3.5 rounded-lg bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.08] text-zinc-300 hover:text-white font-medium text-sm tracking-tight transition-all duration-200 cursor-pointer"
          >
            <span>Explore Architecture</span>
            <ChevronDown className="w-4 h-4 text-zinc-400" />
          </button>
        </motion.div>

        {/* Technical Specification Bar */}
        <motion.div
          {...FADE_IN}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-8 border-t border-white/[0.08]"
        >
          <div>
            <div className="text-[11px] font-mono uppercase tracking-wider text-zinc-500 mb-1">Spatial Agent</div>
            <div className="text-sm font-medium text-zinc-200">EfficientNet-B0</div>
            <div className="text-xs text-zinc-500 font-mono mt-0.5">GradCAM Heatmaps</div>
          </div>
          <div>
            <div className="text-[11px] font-mono uppercase tracking-wider text-zinc-500 mb-1">Temporal Agent</div>
            <div className="text-sm font-medium text-zinc-200">R3D-18 Conv3D</div>
            <div className="text-xs text-zinc-500 font-mono mt-0.5">16-Frame Tensor Window</div>
          </div>
          <div>
            <div className="text-[11px] font-mono uppercase tracking-wider text-zinc-500 mb-1">Audio Agent</div>
            <div className="text-sm font-medium text-zinc-200">Mel-Spectrogram CNN</div>
            <div className="text-xs text-zinc-500 font-mono mt-0.5">128 Mel-Band Analysis</div>
          </div>
          <div>
            <div className="text-[11px] font-mono uppercase tracking-wider text-zinc-500 mb-1">Production Fusion</div>
            <div className="text-sm font-medium text-blue-400 font-mono">corrected_fixed</div>
            <div className="text-xs text-zinc-500 font-mono mt-0.5">Dynamic-Denominator</div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
