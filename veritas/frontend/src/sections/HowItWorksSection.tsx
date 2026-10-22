import React from 'react';
import { motion } from 'framer-motion';
import { Scan, Film, Radio, Layers, Check } from 'lucide-react';

const AGENTS = [
  {
    id: 'spatial',
    name: 'Spatial Analyst',
    badge: 'Spatial Agent',
    icon: <Scan className="w-5 h-5 text-blue-400" />,
    model: 'EfficientNet-B0 + GradCAM',
    input: '224 × 224 RGB Frame',
    target: 'Boundary warping, blending seams, pixel noise anomalies',
    description:
      'Extracts highest-entropy middle frames to detect sub-pixel manipulation seams, GAN boundary blending, and facial warping. Computes gradient-weighted class activation heatmaps (GradCAM) to pinpoint spatial loci of manipulation.',
    checkpoint: 'models/spatial_agent.pth',
  },
  {
    id: 'temporal',
    name: 'Temporal Analyst',
    badge: 'Temporal Agent',
    icon: <Film className="w-5 h-5 text-blue-400" />,
    model: 'R3D-18 (3D Video ResNet)',
    input: '16 sampled frames across 2.0s window',
    target: 'Inter-frame flickering, motion jitter, expression phase glitches',
    description:
      'Evaluates continuous 3D video tensors across 16 sequential frames using 3D spatio-temporal convolutions to detect temporal micro-flickering, unnatural biometric transitions, and motion discontinuities between facial keypoints.',
    checkpoint: 'models/temporal_agent_checkpoint.pth',
  },
  {
    id: 'audio',
    name: 'Audio Analyst',
    badge: 'Audio Agent',
    icon: <Radio className="w-5 h-5 text-blue-400" />,
    model: 'Mel-Spectrogram CNN (EfficientNet-B0)',
    input: '16 kHz mono WAV, 128 mel-bands',
    target: 'Phase vocoder artifacts, synthetic timbre, robotic harmonics',
    description:
      'Converts extracted voice streams into high-resolution log-mel spectrogram tensors to identify synthetic speech vocoder artifacts. Media containers without audio tracks are safely marked as no_audio and excluded from fusion.',
    checkpoint: 'models/audio_agent_checkpoint.pth',
  },
  {
    id: 'context',
    name: 'Context Analyst',
    badge: 'Context Agent',
    icon: <Layers className="w-5 h-5 text-blue-400" />,
    model: 'Cross-Modal Synchrony Engine',
    input: 'Audio-visual stream alignment & metadata',
    target: 'Mixed-modality attacks (e.g. authentic face + cloned voice)',
    description:
      'Evaluates the inter-modal relationship between visual and auditory evidence. If a video appears authentic but contains synthetic cloned speech (or vice versa), the Context Analyst flags cross-modal discrepancy.',
    checkpoint: 'Deterministic Multi-Modal Verification',
  },
];

export function HowItWorksSection() {
  return (
    <section id="how-it-works" className="py-28 px-6 max-w-6xl mx-auto border-b border-white/[0.06]">
      {/* Section Header */}
      <div className="max-w-3xl mb-16">
        <div className="text-xs font-mono uppercase tracking-widest text-blue-400 mb-3">
          SYSTEM ARCHITECTURE
        </div>
        <h2 className="text-3xl sm:text-4xl font-sans font-semibold tracking-tight text-white mb-4">
          Four Specialized Forensic Agents
        </h2>
        <p className="text-base sm:text-lg text-zinc-400 font-normal leading-relaxed">
          Single-modality detectors fail against modern multi-generator synthesis. VERITAS decomposes digital media into independent artifact domains, deploying dedicated neural agents before evidence synthesis.
        </p>
      </div>

      {/* 2x2 Agent Card Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {AGENTS.map((agent, index) => (
          <motion.div
            key={agent.id}
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-40px' }}
            transition={{ duration: 0.5, delay: index * 0.08, ease: [0.16, 1, 0.3, 1] }}
            className="p-6 sm:p-7 rounded-xl bg-[#0d1017] border border-white/[0.08] hover:border-blue-500/30 transition-all duration-200 flex flex-col justify-between"
          >
            <div>
              {/* Header: Icon, Name & Model */}
              <div className="flex items-start justify-between gap-4 mb-4">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-blue-500/10 border border-blue-500/20">
                    {agent.icon}
                  </div>
                  <div>
                    <span className="text-[11px] font-mono tracking-wider text-blue-400 uppercase block">
                      {agent.badge}
                    </span>
                    <h3 className="text-lg font-semibold text-white tracking-tight">
                      {agent.name}
                    </h3>
                  </div>
                </div>
                <span className="px-2.5 py-1 rounded-md bg-white/[0.04] border border-white/[0.08] text-[11px] font-mono text-zinc-400">
                  {agent.model}
                </span>
              </div>

              {/* Description */}
              <p className="text-sm text-zinc-400 leading-relaxed mb-6 font-normal">
                {agent.description}
              </p>
            </div>

            {/* Technical Parameters Spec Box */}
            <div className="pt-4 border-t border-white/[0.06] space-y-2 text-xs font-mono">
              <div className="flex items-center justify-between">
                <span className="text-zinc-500">Input Specification:</span>
                <span className="text-zinc-300 font-medium">{agent.input}</span>
              </div>
              <div className="flex items-start justify-between gap-2">
                <span className="text-zinc-500 shrink-0">Target Artifacts:</span>
                <span className="text-zinc-300 text-right line-clamp-1">{agent.target}</span>
              </div>
              <div className="flex items-center justify-between pt-1">
                <span className="text-zinc-500">Checkpoint Status:</span>
                <span className="text-emerald-400/90 font-medium flex items-center gap-1">
                  <Check className="w-3 h-3 text-emerald-400" />
                  <span>Loaded & Frozen</span>
                </span>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
