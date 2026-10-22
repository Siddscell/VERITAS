import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  ArrowRight, 
  UploadCloud, 
  Video, 
  Image as ImageIcon, 
  Mic, 
  Link as LinkIcon, 
  CheckCircle2, 
  Terminal,
  Shield,
  ExternalLink
} from 'lucide-react';
import { usePageTransition } from '@/context/TransitionContext';

export function ProductPreviewSection() {
  const { navigateTo } = usePageTransition();
  const [selectedTab, setSelectedTab] = useState<'video' | 'image' | 'audio' | 'url'>('video');

  return (
    <section id="preview" className="py-28 px-6 max-w-6xl mx-auto border-b border-white/[0.06]">
      {/* Section Header */}
      <div className="max-w-3xl mb-14">
        <div className="text-xs font-mono uppercase tracking-widest text-blue-400 mb-3">
          PRODUCT INTERFACE
        </div>
        <h2 className="text-3xl sm:text-4xl font-sans font-semibold tracking-tight text-white mb-4">
          The Verification Console
        </h2>
        <p className="text-base sm:text-lg text-zinc-400 font-normal leading-relaxed">
          The live product interface provides streaming multi-agent telemetry, drag-and-drop media ingestion, real-time reasoning logs, and GradCAM visual heatmaps.
        </p>
      </div>

      {/* Realistic Console Interface Card */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: '-40px' }}
        transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
        className="rounded-xl border border-white/[0.1] bg-[#0b0e14] overflow-hidden shadow-2xl"
      >
        {/* Top Window Bar */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-white/[0.08] bg-[#0e121a]">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-zinc-700" />
              <span className="w-2.5 h-2.5 rounded-full bg-zinc-700" />
              <span className="w-2.5 h-2.5 rounded-full bg-zinc-700" />
            </div>
            <span className="text-xs font-mono text-zinc-400 hidden sm:inline">
              veritas-console.local:5173/console
            </span>
          </div>

          <div className="flex items-center gap-3">
            <span className="inline-flex items-center gap-1.5 text-[11px] font-mono text-emerald-400 bg-emerald-950/40 border border-emerald-500/20 px-2 py-0.5 rounded">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
              <span>Engine Online</span>
            </span>
            <button
              onClick={() => navigateTo('/console')}
              className="inline-flex items-center gap-1.5 px-3 py-1 rounded bg-white/10 hover:bg-white/20 text-white text-xs font-mono transition-colors cursor-pointer"
            >
              <span>Open Console</span>
              <ExternalLink className="w-3 h-3 text-zinc-300" />
            </button>
          </div>
        </div>

        {/* Console Workspace Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 min-h-[460px]">
          {/* Left Panel: Ingestion Controls */}
          <div className="lg:col-span-5 p-6 border-b lg:border-b-0 lg:border-r border-white/[0.08] flex flex-col justify-between">
            <div>
              <div className="text-xs font-mono uppercase text-zinc-400 mb-2">Analyse Content</div>
              <p className="text-xs text-zinc-500 mb-5">
                Upload suspect media container or provide an online stream link.
              </p>

              {/* Mode Tabs */}
              <div className="grid grid-cols-4 gap-1.5 p-1 rounded-lg bg-white/[0.03] border border-white/[0.06] mb-5">
                {[
                  { id: 'video', label: 'Video', icon: <Video className="w-3.5 h-3.5" /> },
                  { id: 'image', label: 'Image', icon: <ImageIcon className="w-3.5 h-3.5" /> },
                  { id: 'audio', label: 'Audio', icon: <Mic className="w-3.5 h-3.5" /> },
                  { id: 'url', label: 'URL', icon: <LinkIcon className="w-3.5 h-3.5" /> },
                ].map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setSelectedTab(tab.id as typeof selectedTab)}
                    className={`flex items-center justify-center gap-1.5 py-1.5 rounded text-xs font-mono transition-all cursor-pointer ${
                      selectedTab === tab.id
                        ? 'bg-white/10 text-white font-medium'
                        : 'text-zinc-400 hover:text-zinc-200'
                    }`}
                  >
                    {tab.icon}
                    <span>{tab.label}</span>
                  </button>
                ))}
              </div>

              {/* Simulated Dropzone */}
              <div 
                onClick={() => navigateTo('/console')}
                className="border border-dashed border-white/[0.15] hover:border-blue-400/40 rounded-xl p-8 text-center bg-white/[0.01] hover:bg-white/[0.02] transition-colors cursor-pointer"
              >
                <div className="w-10 h-10 rounded-full bg-white/[0.04] flex items-center justify-center mx-auto mb-3">
                  <UploadCloud className="w-5 h-5 text-blue-400" />
                </div>
                <div className="text-sm font-medium text-zinc-200 mb-1">
                  Drop media file here, or click to browse
                </div>
                <div className="text-xs font-mono text-zinc-500">
                  MP4, MOV, AVI, MKV — Maximum 50MB
                </div>
              </div>
            </div>

            <button
              onClick={() => navigateTo('/console')}
              className="w-full mt-6 py-2.5 px-4 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-300 hover:text-white font-mono text-xs font-medium tracking-wide transition-colors cursor-pointer"
            >
              Analyse with VERITAS
            </button>
          </div>

          {/* Right Panel: Telemetry & Multi-Agent Engine Readout */}
          <div className="lg:col-span-7 p-6 bg-[#090b10] flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between pb-3 mb-4 border-b border-white/[0.06]">
                <div className="flex items-center gap-2 text-xs font-mono text-zinc-400">
                  <Terminal className="w-4 h-4 text-blue-400" />
                  <span>Agent Initialization & Reasoning Log</span>
                </div>
                <span className="text-[11px] font-mono text-zinc-500">SSE Protocol 2.0</span>
              </div>

              {/* Boot Log Snippet */}
              <div className="font-mono text-xs space-y-1.5 mb-6 text-zinc-400">
                <div className="text-zinc-500">[INIT] Initializing multi-agent forensics runtime...</div>
                <div className="text-emerald-400/90">[OK] Spatial Analysis Module loaded — EfficientNet-B0</div>
                <div className="text-emerald-400/90">[OK] Temporal Analysis Module loaded — R3D-18</div>
                <div className="text-emerald-400/90">[OK] Audio Analysis Module loaded — MelSpectrogram CNN</div>
                <div className="text-emerald-400/90">[OK] Context Analysis Module loaded</div>
                <div className="text-emerald-400/90">[OK] GradCAM heatmapping engine initialized</div>
                <div className="text-blue-400">[READY] VERITAS v1.0 — awaiting media input</div>
              </div>

              {/* 4 Agent Status Cards */}
              <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                <div className="p-3 rounded-lg bg-white/[0.02] border border-white/[0.06]">
                  <div className="text-zinc-500 mb-0.5">Spatial Analyst</div>
                  <div className="text-zinc-300 font-medium">EfficientNet-B0</div>
                  <div className="text-[10px] text-blue-400 mt-1">Pending Input</div>
                </div>
                <div className="p-3 rounded-lg bg-white/[0.02] border border-white/[0.06]">
                  <div className="text-zinc-500 mb-0.5">Temporal Analyst</div>
                  <div className="text-zinc-300 font-medium">R3D-18 (Conv3D)</div>
                  <div className="text-[10px] text-blue-400 mt-1">Pending Input</div>
                </div>
                <div className="p-3 rounded-lg bg-white/[0.02] border border-white/[0.06]">
                  <div className="text-zinc-500 mb-0.5">Audio Analyst</div>
                  <div className="text-zinc-300 font-medium">MelSpectrogram CNN</div>
                  <div className="text-[10px] text-blue-400 mt-1">Pending Input</div>
                </div>
                <div className="p-3 rounded-lg bg-white/[0.02] border border-white/[0.06]">
                  <div className="text-zinc-500 mb-0.5">Context Analyst</div>
                  <div className="text-zinc-300 font-medium">Synchrony Engine</div>
                  <div className="text-[10px] text-blue-400 mt-1">Pending Input</div>
                </div>
              </div>
            </div>

            <div className="mt-6 pt-4 border-t border-white/[0.06] flex items-center justify-between">
              <span className="text-xs text-zinc-500 font-mono">
                Production Default: <span className="text-zinc-300 font-semibold">corrected_fixed</span>
              </span>
              <button
                onClick={() => navigateTo('/console')}
                className="inline-flex items-center gap-1.5 text-xs font-mono font-medium text-blue-400 hover:text-blue-300 transition-colors cursor-pointer"
              >
                <span>Launch Live Session</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </div>
      </motion.div>
    </section>
  );
}
