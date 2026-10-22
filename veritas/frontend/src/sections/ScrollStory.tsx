import React, { useEffect, useRef, useState } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { Shield, Eye, Clock, Radio, Network, ArrowDown, ArrowRight, ChevronRight, Activity, Terminal } from 'lucide-react';
import { usePageTransition } from '@/context/TransitionContext';

gsap.registerPlugin(ScrollTrigger);

export function ScrollStory() {
  const { navigateTo } = usePageTransition();
  const containerRef = useRef<HTMLDivElement>(null);
  const viewportRef = useRef<HTMLDivElement>(null);
  const [reducedMotion, setReducedMotion] = useState(false);
  const [activeChapter, setActiveChapter] = useState(1);

  // Chapter DOM Container Refs
  const ch1Ref = useRef<HTMLDivElement>(null);
  const ch2Ref = useRef<HTMLDivElement>(null);
  const ch3Ref = useRef<HTMLDivElement>(null);
  const ch4Ref = useRef<HTMLDivElement>(null);
  const ch5Ref = useRef<HTMLDivElement>(null);

  // Check reduced motion
  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    setReducedMotion(mq.matches);
    const handler = (e: MediaQueryListEvent) => setReducedMotion(e.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);

  // Set up scrubbed ScrollTrigger timeline
  useEffect(() => {
    if (reducedMotion || !containerRef.current || !viewportRef.current) return;

    const ctx = gsap.context(() => {
      // Master scrubbed timeline across 100 units of scroll progress
      const tl = gsap.timeline({
        scrollTrigger: {
          trigger: containerRef.current,
          start: 'top top',
          end: 'bottom bottom',
          scrub: 0.8,
          onUpdate: (self) => {
            const p = self.progress;
            if (p < 0.20) setActiveChapter(1);
            else if (p < 0.40) setActiveChapter(2);
            else if (p < 0.60) setActiveChapter(3);
            else if (p < 0.80) setActiveChapter(4);
            else setActiveChapter(5);
          },
        },
      });

      // Initial visibility states
      gsap.set(ch1Ref.current, { opacity: 1, y: 0, pointerEvents: 'auto' });
      gsap.set([ch2Ref.current, ch3Ref.current, ch4Ref.current, ch5Ref.current], {
        opacity: 0,
        y: 30,
        pointerEvents: 'none',
      });

      // ── Clean Non-overlapping Sequence (Total = 100 Units) ───────────────────────

      // Ch1 -> Ch2 (Exit Ch1: 16-19, Enter Ch2: 19-22)
      tl.to(ch1Ref.current, { opacity: 0, y: -24, pointerEvents: 'none', ease: 'power2.in', duration: 3 }, 16)
        .to(ch2Ref.current, { opacity: 1, y: 0, pointerEvents: 'auto', ease: 'power2.out', duration: 3 }, 19);

      // Ch2 -> Ch3 (Exit Ch2: 36-39, Enter Ch3: 39-42)
      tl.to(ch2Ref.current, { opacity: 0, y: -24, pointerEvents: 'none', ease: 'power2.in', duration: 3 }, 36)
        .to(ch3Ref.current, { opacity: 1, y: 0, pointerEvents: 'auto', ease: 'power2.out', duration: 3 }, 39);

      // Ch3 -> Ch4 (Exit Ch3: 56-59, Enter Ch4: 59-62)
      tl.to(ch3Ref.current, { opacity: 0, y: -24, pointerEvents: 'none', ease: 'power2.in', duration: 3 }, 56)
        .to(ch4Ref.current, { opacity: 1, y: 0, pointerEvents: 'auto', ease: 'power2.out', duration: 3 }, 59);

      // Ch4 -> Ch5 (Exit Ch4: 76-79, Enter Ch5: 79-82)
      tl.to(ch4Ref.current, { opacity: 0, y: -24, pointerEvents: 'none', ease: 'power2.in', duration: 3 }, 76)
        .to(ch5Ref.current, { opacity: 1, y: 0, pointerEvents: 'auto', ease: 'power2.out', duration: 3 }, 79);

      // Ch5 transition toward Console reveal (96-100)
      tl.to(ch5Ref.current, { opacity: 0.1, y: -20, ease: 'power2.inOut', duration: 4 }, 96);

      // Ensure geometry calculation is fresh
      ScrollTrigger.refresh();
    }, containerRef);

    // Refresh ScrollTrigger when DOM is fully settled
    const timer = setTimeout(() => {
      ScrollTrigger.refresh();
    }, 150);

    return () => {
      clearTimeout(timer);
      ctx.revert();
    };
  }, [reducedMotion]);

  if (reducedMotion) {
    // Accessible fallback for users requesting reduced motion
    return (
      <div className="w-full bg-[#07090e] text-white">
        <section id="overview" className="min-h-screen py-24 px-6 max-w-6xl mx-auto flex flex-col justify-center">
          <OpeningStatementContent />
          <div className="mt-12"><OpeningGraphic /></div>
        </section>
        <section id="visual" className="min-h-screen py-24 px-6 max-w-6xl mx-auto flex flex-col justify-center border-t border-white/[0.08]">
          <VisualEvidenceContent />
          <div className="mt-12"><VisualEvidenceGraphic /></div>
        </section>
        <section id="temporal" className="min-h-screen py-24 px-6 max-w-6xl mx-auto flex flex-col justify-center border-t border-white/[0.08]">
          <TemporalContent />
          <div className="mt-12"><TemporalGraphic /></div>
        </section>
        <section id="audio" className="min-h-screen py-24 px-6 max-w-6xl mx-auto flex flex-col justify-center border-t border-white/[0.08]">
          <AudioContent />
          <div className="mt-12"><AudioGraphic /></div>
        </section>
        <section id="fusion" className="min-h-screen py-24 px-6 max-w-6xl mx-auto flex flex-col justify-center border-t border-white/[0.08]">
          <FusionContent />
          <div className="mt-12"><FusionGraphic /></div>
        </section>
        <section id="console" className="min-h-screen py-24 px-6 max-w-6xl mx-auto flex flex-col justify-center border-t border-white/[0.08]">
          <Chapter06CtaContent onOpenConsole={() => navigateTo('/console')} />
        </section>
      </div>
    );
  }

  return (
    <div className="relative w-full bg-[#07090e] text-white">
      {/* ── Scroll Track with Navigation Anchor Markers ───────────────────────────── */}
      <div ref={containerRef} className="relative w-full h-[600vh]">
        <div id="overview" className="absolute top-0 w-full h-[1px] pointer-events-none" />
        <div id="visual" className="absolute top-[28%] w-full h-[1px] pointer-events-none" />
        <div id="temporal" className="absolute top-[48%] w-full h-[1px] pointer-events-none" />
        <div id="audio" className="absolute top-[68%] w-full h-[1px] pointer-events-none" />
        <div id="fusion" className="absolute top-[84%] w-full h-[1px] pointer-events-none" />

        {/* ── Pinned Viewport Container ────────────────────────────────────────── */}
        <div
          ref={viewportRef}
          className="sticky top-0 h-screen w-full overflow-hidden flex flex-col justify-between pt-24 pb-10 px-6 sm:px-10 lg:px-16"
        >
          {/* Subtle Chapter Status Indicator */}
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-4 z-20">
            <div className="flex items-center gap-3">
              <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
              <span className="font-mono text-xs uppercase tracking-widest text-zinc-400">
                VERITAS Forensics · Chapter 0{activeChapter} of 05
              </span>
            </div>
            <div className="hidden sm:flex items-center gap-6 font-mono text-[11px] text-zinc-500">
              <span className={activeChapter === 1 ? 'text-blue-400 font-semibold' : ''}>01 Intro</span>
              <span className={activeChapter === 2 ? 'text-blue-400 font-semibold' : ''}>02 Spatial</span>
              <span className={activeChapter === 3 ? 'text-blue-400 font-semibold' : ''}>03 Temporal</span>
              <span className={activeChapter === 4 ? 'text-blue-400 font-semibold' : ''}>04 Audio</span>
              <span className={activeChapter === 5 ? 'text-blue-400 font-semibold' : ''}>05 Fusion</span>
            </div>
          </div>

          {/* Central Stage: Text Narrative + Interactive Graphic Transform */}
          <div className="relative flex-1 w-full max-w-7xl mx-auto flex items-center justify-center my-auto">
            {/* ── Chapter 01: Opening Statement ── */}
            <div
              ref={ch1Ref}
              className="absolute inset-0 flex flex-col lg:flex-row items-center justify-between gap-10 lg:gap-16 my-auto"
            >
              <div className="w-full lg:w-1/2">
                <OpeningStatementContent />
              </div>
              <div className="w-full lg:w-1/2 flex justify-center">
                <OpeningGraphic />
              </div>
            </div>

            {/* ── Chapter 02: Visual Evidence (Spatial) ── */}
            <div
              ref={ch2Ref}
              className="absolute inset-0 flex flex-col lg:flex-row items-center justify-between gap-10 lg:gap-16 my-auto"
            >
              <div className="w-full lg:w-1/2">
                <VisualEvidenceContent />
              </div>
              <div className="w-full lg:w-1/2 flex justify-center">
                <VisualEvidenceGraphic />
              </div>
            </div>

            {/* ── Chapter 03: Time & Motion (Temporal) ── */}
            <div
              ref={ch3Ref}
              className="absolute inset-0 flex flex-col lg:flex-row items-center justify-between gap-10 lg:gap-16 my-auto"
            >
              <div className="w-full lg:w-1/2">
                <TemporalContent />
              </div>
              <div className="w-full lg:w-1/2 flex justify-center">
                <TemporalGraphic />
              </div>
            </div>

            {/* ── Chapter 04: Audio Evidence (Acoustic) ── */}
            <div
              ref={ch4Ref}
              className="absolute inset-0 flex flex-col lg:flex-row items-center justify-between gap-10 lg:gap-16 my-auto"
            >
              <div className="w-full lg:w-1/2">
                <AudioContent />
              </div>
              <div className="w-full lg:w-1/2 flex justify-center">
                <AudioGraphic />
              </div>
            </div>

            {/* ── Chapter 05: Context & Evidence Fusion ── */}
            <div
              ref={ch5Ref}
              className="absolute inset-0 flex flex-col lg:flex-row items-center justify-between gap-10 lg:gap-16 my-auto"
            >
              <div className="w-full lg:w-1/2">
                <FusionContent />
              </div>
              <div className="w-full lg:w-1/2 flex justify-center">
                <FusionGraphic />
              </div>
            </div>
          </div>

          {/* Bottom Scroll Cue */}
          <div className="flex items-center justify-between border-t border-white/[0.06] pt-3 text-zinc-500 font-mono text-[11px] z-20">
            <div className="flex items-center gap-2">
              <ArrowDown className="w-3.5 h-3.5 text-blue-400 animate-bounce" />
              <span>Scroll to scrub forensic narrative</span>
            </div>
            <div className="hidden sm:block text-zinc-600">
              Bi-directional scrub · Chapter 06 verification CTA below
            </div>
          </div>
        </div>
      </div>

      {/* ── Chapter 06: Final CTA Section ────────────────────────────────────────── */}
      <section
        id="console"
        className="relative w-full py-28 px-4 sm:px-6 lg:px-12 bg-[#07090e] border-t border-white/[0.08] z-30"
      >
        <Chapter06CtaContent onOpenConsole={() => navigateTo('/console')} />
      </section>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Content Components
// ─────────────────────────────────────────────────────────────────────────────

function OpeningStatementContent() {
  return (
    <div className="space-y-6">
      <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 font-mono text-xs">
        <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
        CHAPTER 01 / ARCHITECTURE
      </div>
      <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-white leading-[1.08]">
        Deepfake detection driven by <span className="text-blue-400">multi-modal evidence</span>.
      </h1>
      <p className="text-zinc-400 text-base sm:text-lg leading-relaxed max-w-xl font-sans">
        Single-detector models fail when confronted with multi-generator synthetic media. VERITAS coordinates four
        specialized neural analysts to evaluate spatial boundaries, temporal continuity, vocoder acoustics, and cross-modal
        synchrony.
      </p>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 pt-4 border-t border-white/[0.08] font-mono text-xs">
        <div>
          <span className="text-zinc-500 block">SPATIAL</span>
          <span className="text-zinc-200 font-medium">EfficientNet-B0</span>
        </div>
        <div>
          <span className="text-zinc-500 block">TEMPORAL</span>
          <span className="text-zinc-200 font-medium">R3D-18 Conv3D</span>
        </div>
        <div>
          <span className="text-zinc-500 block">FUSION</span>
          <span className="text-blue-400 font-medium">corrected_fixed</span>
        </div>
      </div>
    </div>
  );
}

function VisualEvidenceContent() {
  return (
    <div className="space-y-6">
      <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 font-mono text-xs">
        <Eye className="w-3.5 h-3.5" />
        CHAPTER 02 / SPATIAL ANALYST
      </div>
      <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight text-white leading-tight">
        Sub-pixel artifact & boundary inspection.
      </h2>
      <p className="text-zinc-400 text-sm sm:text-base leading-relaxed max-w-xl">
        The Spatial Agent extracts high-entropy face crops at 224×224 resolution. Evaluated through an EfficientNet-B0
        backbone, it detects sub-pixel warping, GAN blending seams, and generative noise patterns.
      </p>
      <ul className="space-y-2.5 font-mono text-xs text-zinc-400 border-l border-blue-500/40 pl-4">
        <li className="flex items-center gap-2">
          <ChevronRight className="w-3.5 h-3.5 text-blue-400" />
          <span>Isolates face-boundary boundary blending seams</span>
        </li>
        <li className="flex items-center gap-2">
          <ChevronRight className="w-3.5 h-3.5 text-blue-400" />
          <span>Computes gradient-weighted class activation heatmaps (GradCAM)</span>
        </li>
        <li className="flex items-center gap-2">
          <ChevronRight className="w-3.5 h-3.5 text-blue-400" />
          <span>Frozen production weight: <strong className="text-zinc-200">0.40</strong></span>
        </li>
      </ul>
    </div>
  );
}

function TemporalContent() {
  return (
    <div className="space-y-6">
      <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 font-mono text-xs">
        <Clock className="w-3.5 h-3.5" />
        CHAPTER 03 / TEMPORAL ANALYST
      </div>
      <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight text-white leading-tight">
        Spatio-temporal motion across 16 frames.
      </h2>
      <p className="text-zinc-400 text-sm sm:text-base leading-relaxed max-w-xl">
        Synthetic face swaps that deceive static inspections frequently collapse in motion. The Temporal Agent samples a
        16-frame tensor sequence through 3D ResNet convolutions (R3D-18), exposing inter-frame micro-flickering and facial landmark jitter.
      </p>
      <ul className="space-y-2.5 font-mono text-xs text-zinc-400 border-l border-blue-500/40 pl-4">
        <li className="flex items-center gap-2">
          <ChevronRight className="w-3.5 h-3.5 text-blue-400" />
          <span>Evaluates 3D spatio-temporal video tensors</span>
        </li>
        <li className="flex items-center gap-2">
          <ChevronRight className="w-3.5 h-3.5 text-blue-400" />
          <span>Catches expression desynchronization and unnatural blinking</span>
        </li>
        <li className="flex items-center gap-2">
          <ChevronRight className="w-3.5 h-3.5 text-blue-400" />
          <span>Frozen production weight: <strong className="text-zinc-200">0.30</strong></span>
        </li>
      </ul>
    </div>
  );
}

function AudioContent() {
  return (
    <div className="space-y-6">
      <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 font-mono text-xs">
        <Radio className="w-3.5 h-3.5" />
        CHAPTER 04 / AUDIO ANALYST
      </div>
      <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight text-white leading-tight">
        Mel-spectrogram synthesis & vocoder detection.
      </h2>
      <p className="text-zinc-400 text-sm sm:text-base leading-relaxed max-w-xl">
        Modern deception marries modified visuals with AI-cloned audio. The Audio Agent converts 16 kHz audio streams into
        128 log-mel spectrogram bands to identify neural vocoder synthesis artifacts. Media lacking audio is cleanly bypassed.
      </p>
      <ul className="space-y-2.5 font-mono text-xs text-zinc-400 border-l border-blue-500/40 pl-4">
        <li className="flex items-center gap-2">
          <ChevronRight className="w-3.5 h-3.5 text-blue-400" />
          <span>128-band log-mel spectrogram frequency decomposition</span>
        </li>
        <li className="flex items-center gap-2">
          <ChevronRight className="w-3.5 h-3.5 text-blue-400" />
          <span>Detects synthetic phase dispersion in high-frequency formants</span>
        </li>
        <li className="flex items-center gap-2">
          <ChevronRight className="w-3.5 h-3.5 text-blue-400" />
          <span>Frozen production weight: <strong className="text-zinc-200">0.20</strong> (Dynamic exclusion)</span>
        </li>
      </ul>
    </div>
  );
}

function FusionContent() {
  return (
    <div className="space-y-6">
      <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 font-mono text-xs">
        <Network className="w-3.5 h-3.5" />
        CHAPTER 05 / DYNAMIC EVIDENCE FUSION
      </div>
      <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight text-white leading-tight">
        Dynamic-denominator synthesis & tripartite verdict.
      </h2>
      <p className="text-zinc-400 text-sm sm:text-base leading-relaxed max-w-xl">
        Raw model outputs cannot simply be averaged. The production fusion engine (<code>corrected_fixed</code>) rescales
        active weights dynamically over available modalities. A strict tripartite contract governs forensic decisions.
      </p>
      <div className="p-4 rounded-xl bg-zinc-900/80 border border-white/[0.08] font-mono text-xs space-y-2 text-zinc-300">
        <div className="text-blue-400 font-semibold">FORMULA: Score = ∑(w_i · s_i) / ∑(w_active)</div>
        <p className="text-zinc-400 text-[11px] leading-relaxed">
          When audio is absent, denominator adjusts from 1.00 to 0.80, preventing false-authentic score dilution.
        </p>
        <div className="pt-2 flex flex-wrap gap-2 text-[11px]">
          <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">AUTHENTIC &lt; 0.35</span>
          <span className="px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">UNCERTAIN [0.35 - 0.65]</span>
          <span className="px-2 py-0.5 rounded bg-rose-500/10 text-rose-400 border border-rose-500/20">DEEPFAKE &gt; 0.65</span>
        </div>
      </div>
    </div>
  );
}

function Chapter06CtaContent({ onOpenConsole }: { onOpenConsole: () => void }) {
  return (
    <div className="max-w-4xl mx-auto text-center space-y-8">
      <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 font-mono text-xs">
        <Terminal className="w-3.5 h-3.5" />
        CHAPTER 06 / OPERATIONAL VERIFICATION
      </div>

      <h2 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-white leading-tight">
        Ready to verify a video?
      </h2>

      <p className="text-zinc-400 text-base sm:text-lg max-w-2xl mx-auto leading-relaxed font-sans">
        Upload suspect video containers, static portraits, or voice streams for real multi-agent analysis.
        VERITAS executes live inference across Spatial, Temporal, Audio, and Context neural networks with dynamic-denominator evidence fusion (<code className="text-blue-400 bg-blue-500/10 px-1 rounded">corrected_fixed</code>).
      </p>

      <div className="pt-2 flex flex-col sm:flex-row items-center justify-center gap-4">
        <button
          onClick={onOpenConsole}
          data-testid="button-open-live-console"
          className="inline-flex items-center gap-2.5 px-8 py-4 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-base transition-all duration-200 shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 hover:-translate-y-0.5 cursor-pointer"
        >
          <span>Open Live Console</span>
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>

      <div className="pt-8 grid grid-cols-2 sm:grid-cols-4 gap-4 border-t border-white/[0.06] font-mono text-xs text-zinc-500 text-center">
        <div>
          <span className="text-zinc-300 font-medium block">DRAG & DROP</span>
          <span>MP4, AVI, MKV, WAV</span>
        </div>
        <div>
          <span className="text-zinc-300 font-medium block">STREAMING TELEMETRY</span>
          <span>Live SSE Agent Log</span>
        </div>
        <div>
          <span className="text-zinc-300 font-medium block">EXPLAINABLE GRADCAM</span>
          <span>Visual Heatmap Loci</span>
        </div>
        <div>
          <span className="text-zinc-300 font-medium block">FROZEN RUNTIME</span>
          <span>corrected_fixed</span>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Custom Architectural Graphics (Zero external 3D, Pure SVG/CSS Art Direction)
// ─────────────────────────────────────────────────────────────────────────────

function OpeningGraphic() {
  return (
    <div className="relative w-full max-w-md aspect-square rounded-2xl border border-white/[0.08] bg-[#0c1017] p-6 flex flex-col justify-between overflow-hidden shadow-2xl">
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#ffffff05_1px,transparent_1px),linear-gradient(to_bottom,#ffffff05_1px,transparent_1px)] bg-[size:24px_24px] pointer-events-none" />

      <div className="relative flex items-center justify-between font-mono text-[11px] text-zinc-500 z-10 border-b border-white/[0.06] pb-3">
        <span className="flex items-center gap-1.5 text-zinc-300">
          <Shield className="w-3.5 h-3.5 text-blue-400" />
          SYS_APERTURE_01
        </span>
        <span className="text-blue-400">READY</span>
      </div>

      <div className="relative my-auto flex flex-col items-center justify-center p-8 z-10">
        <div className="relative w-44 h-44 rounded-full border border-blue-500/20 flex items-center justify-center">
          <div className="absolute inset-2 rounded-full border border-dashed border-white/10 animate-[spin_40s_linear_infinite]" />
          <div className="w-24 h-24 rounded-lg border border-blue-500/40 bg-blue-500/5 flex flex-col items-center justify-center p-3 text-center">
            <Activity className="w-6 h-6 text-blue-400 mb-1" />
            <span className="font-mono text-[10px] text-zinc-300 uppercase tracking-wider">Awaiting Stream</span>
          </div>
          <div className="absolute -top-1 w-2 h-1 bg-blue-400" />
          <div className="absolute -bottom-1 w-2 h-1 bg-blue-400" />
          <div className="absolute -left-1 w-1 h-2 bg-blue-400" />
          <div className="absolute -right-1 w-1 h-2 bg-blue-400" />
        </div>
      </div>

      <div className="relative flex justify-between items-center font-mono text-[10px] text-zinc-500 z-10 border-t border-white/[0.06] pt-3">
        <span>IN: MP4/MKV/WAV/PNG</span>
        <span className="text-zinc-400">AGENTS: 4 COORD</span>
      </div>
    </div>
  );
}

function VisualEvidenceGraphic() {
  return (
    <div className="relative w-full max-w-md aspect-square rounded-2xl border border-white/[0.08] bg-[#0c1017] p-6 flex flex-col justify-between overflow-hidden shadow-2xl">
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#ffffff05_1px,transparent_1px),linear-gradient(to_bottom,#ffffff05_1px,transparent_1px)] bg-[size:20px_20px] pointer-events-none" />

      <div className="relative flex items-center justify-between font-mono text-[11px] text-zinc-500 z-10 border-b border-white/[0.06] pb-3">
        <span className="text-zinc-300 font-semibold">TENSOR: 224×224 RGB</span>
        <span className="text-emerald-400">GRADCAM_ACTIVE</span>
      </div>

      <div className="relative my-auto flex items-center justify-center z-10">
        <div className="relative w-48 h-56 rounded-xl border border-white/20 bg-zinc-950 overflow-hidden shadow-inner flex items-center justify-center">
          <svg className="w-32 h-44 text-zinc-700" viewBox="0 0 100 120" fill="none" stroke="currentColor" strokeWidth="1.2">
            <ellipse cx="50" cy="55" rx="32" ry="42" />
            <path d="M35 50 Q40 46 45 50" />
            <path d="M55 50 Q60 46 65 50" />
            <path d="M50 56 L50 68 L46 72 L54 72" />
            <path d="M40 84 Q50 90 60 84" />
          </svg>

          <div className="absolute top-12 left-8 w-16 h-16 rounded-full bg-rose-500/30 blur-md pointer-events-none" />
          <div className="absolute top-24 right-10 w-20 h-14 rounded-full bg-amber-500/25 blur-lg pointer-events-none" />
          <div className="absolute bottom-6 left-12 w-24 h-12 rounded-full bg-blue-500/20 blur-md pointer-events-none" />

          <div className="absolute inset-4 border border-rose-500/40 rounded-lg pointer-events-none">
            <span className="absolute -top-2 left-2 px-1 text-[9px] font-mono bg-rose-500 text-white rounded">
              SEAM: 0.884
            </span>
          </div>
        </div>
      </div>

      <div className="relative flex justify-between font-mono text-[10px] text-zinc-400 z-10 border-t border-white/[0.06] pt-3">
        <span>LOCI: PERI-ORBITAL BLENDING</span>
        <span className="text-blue-400">EfficientNet-B0</span>
      </div>
    </div>
  );
}

function TemporalGraphic() {
  return (
    <div className="relative w-full max-w-md aspect-square rounded-2xl border border-white/[0.08] bg-[#0c1017] p-6 flex flex-col justify-between overflow-hidden shadow-2xl">
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#ffffff05_1px,transparent_1px),linear-gradient(to_bottom,#ffffff05_1px,transparent_1px)] bg-[size:20px_20px] pointer-events-none" />

      <div className="relative flex items-center justify-between font-mono text-[11px] text-zinc-500 z-10 border-b border-white/[0.06] pb-3">
        <span className="text-zinc-300 font-semibold">WINDOW: 16 FRAMES</span>
        <span className="text-blue-400">R3D-18 CONV3D</span>
      </div>

      <div className="relative my-auto flex flex-col gap-3 z-10 w-full">
        <div className="grid grid-cols-4 gap-2">
          {['T_01', 'T_04', 'T_08', 'T_12'].map((t, idx) => (
            <div
              key={t}
              className={`h-24 rounded-lg border p-2 flex flex-col justify-between ${
                idx === 2 ? 'border-amber-500/60 bg-amber-500/10' : 'border-white/10 bg-zinc-950/80'
              }`}
            >
              <div className="flex justify-between font-mono text-[9px] text-zinc-400">
                <span>{t}</span>
                {idx === 2 && <span className="text-amber-400 font-bold">JITTER</span>}
              </div>
              <div className="w-full flex justify-center py-1">
                <div
                  className={`w-7 h-9 rounded border ${
                    idx === 2 ? 'border-amber-400 bg-amber-400/20' : 'border-zinc-700 bg-zinc-800'
                  }`}
                />
              </div>
              <div className="w-full bg-zinc-800 h-1 rounded-full overflow-hidden">
                <div
                  className={`h-full ${idx === 2 ? 'bg-amber-400 w-4/5' : 'bg-blue-400 w-1/3'}`}
                />
              </div>
            </div>
          ))}
        </div>

        <div className="relative h-6 w-full flex items-center px-2">
          <div className="h-[1px] w-full bg-zinc-700" />
          <div className="absolute left-[62%] -top-1 w-2 h-2 rounded-full bg-amber-400 animate-ping" />
          <div className="absolute left-[62%] top-0 w-2 h-2 rounded-full bg-amber-400" />
        </div>
      </div>

      <div className="relative flex justify-between font-mono text-[10px] text-zinc-400 z-10 border-t border-white/[0.06] pt-3">
        <span>INTER-FRAME JITTER DETECTED</span>
        <span className="text-amber-400">Δt: 0.16s DISCONTINUITY</span>
      </div>
    </div>
  );
}

function AudioGraphic() {
  return (
    <div className="relative w-full max-w-md aspect-square rounded-2xl border border-white/[0.08] bg-[#0c1017] p-6 flex flex-col justify-between overflow-hidden shadow-2xl">
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#ffffff05_1px,transparent_1px),linear-gradient(to_bottom,#ffffff05_1px,transparent_1px)] bg-[size:20px_20px] pointer-events-none" />

      <div className="relative flex items-center justify-between font-mono text-[11px] text-zinc-500 z-10 border-b border-white/[0.06] pb-3">
        <span className="text-zinc-300 font-semibold">SPECTRUM: 128 MEL-BANDS</span>
        <span className="text-blue-400">16 kHz MONO</span>
      </div>

      <div className="relative my-auto flex flex-col gap-1.5 z-10 w-full p-2">
        {[
          { label: '8.0 kHz', bars: [20, 35, 60, 85, 45, 90, 75, 30, 95, 60] },
          { label: '4.0 kHz', bars: [40, 70, 90, 50, 80, 60, 85, 70, 40, 80] },
          { label: '2.0 kHz', bars: [60, 85, 40, 75, 90, 65, 50, 85, 70, 90] },
          { label: '1.0 kHz', bars: [80, 50, 70, 90, 60, 85, 95, 60, 80, 50] },
          { label: '250 Hz',  bars: [95, 90, 85, 80, 75, 70, 80, 85, 90, 95] },
        ].map((band, idx) => (
          <div key={band.label} className="flex items-center gap-2">
            <span className="font-mono text-[9px] text-zinc-500 w-12 text-right">{band.label}</span>
            <div className="flex-1 flex gap-1 h-3 items-end">
              {band.bars.map((h, i) => (
                <div
                  key={i}
                  style={{ height: `${h}%` }}
                  className={`flex-1 rounded-sm ${
                    idx === 0 && i >= 6 ? 'bg-rose-500/80' : 'bg-blue-500/60'
                  }`}
                />
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="relative flex justify-between font-mono text-[10px] text-zinc-400 z-10 border-t border-white/[0.06] pt-3">
        <span>VOCODER DISPERSION AT 8.0 kHz</span>
        <span className="text-rose-400">PHASE ANOMALY</span>
      </div>
    </div>
  );
}

function FusionGraphic() {
  return (
    <div className="relative w-full max-w-md aspect-square rounded-2xl border border-white/[0.08] bg-[#0c1017] p-6 flex flex-col justify-between overflow-hidden shadow-2xl">
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#ffffff05_1px,transparent_1px),linear-gradient(to_bottom,#ffffff05_1px,transparent_1px)] bg-[size:20px_20px] pointer-events-none" />

      <div className="relative flex items-center justify-between font-mono text-[11px] text-zinc-500 z-10 border-b border-white/[0.06] pb-3">
        <span className="text-zinc-300 font-semibold">SYNTHESIS: corrected_fixed</span>
        <span className="text-emerald-400">FROZEN_PROD</span>
      </div>

      <div className="relative my-auto flex flex-col items-center justify-center gap-3 z-10 w-full py-2">
        <div className="grid grid-cols-4 gap-2 w-full text-center font-mono text-[10px]">
          <div className="p-2 rounded bg-zinc-900 border border-white/10">
            <span className="text-zinc-400 block">SPATIAL</span>
            <span className="text-blue-400 font-bold">0.40</span>
          </div>
          <div className="p-2 rounded bg-zinc-900 border border-white/10">
            <span className="text-zinc-400 block">TEMPORAL</span>
            <span className="text-blue-400 font-bold">0.30</span>
          </div>
          <div className="p-2 rounded bg-zinc-900 border border-white/10">
            <span className="text-zinc-400 block">AUDIO</span>
            <span className="text-blue-400 font-bold">0.20</span>
          </div>
          <div className="p-2 rounded bg-zinc-900 border border-white/10">
            <span className="text-zinc-400 block">CONTEXT</span>
            <span className="text-blue-400 font-bold">0.10</span>
          </div>
        </div>

        <div className="w-full p-3 rounded-xl border border-blue-500/40 bg-blue-500/10 text-center font-mono space-y-1">
          <div className="text-xs text-white font-bold tracking-wider">DYNAMIC-DENOMINATOR NORMALIZATION</div>
          <div className="text-[11px] text-blue-300">Score = ∑(w_i · s_i) / ∑(w_active)</div>
        </div>

        <div className="grid grid-cols-3 gap-2 w-full text-center font-mono text-[9px]">
          <div className="p-1.5 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
            AUTHENTIC
          </div>
          <div className="p-1.5 rounded bg-amber-500/10 border border-amber-500/30 text-amber-400">
            UNCERTAIN
          </div>
          <div className="p-1.5 rounded bg-rose-500/10 border border-rose-500/30 text-rose-400">
            DEEPFAKE
          </div>
        </div>
      </div>

      <div className="relative flex justify-between font-mono text-[10px] text-zinc-400 z-10 border-t border-white/[0.06] pt-3">
        <span>CONTRACT: TRIPARTITE DEADBAND</span>
        <span className="text-emerald-400">PPO v3: DISABLED</span>
      </div>
    </div>
  );
}
