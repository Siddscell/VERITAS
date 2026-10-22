import React from 'react';
import { motion } from 'framer-motion';
import { Eye, AlertTriangle, ShieldCheck } from 'lucide-react';

export function TrustSection() {
  return (
    <section id="trust" className="py-28 px-6 max-w-6xl mx-auto border-b border-white/[0.06]">
      {/* Section Header */}
      <div className="max-w-3xl mb-16">
        <div className="text-xs font-mono uppercase tracking-widest text-blue-400 mb-3">
          FORENSIC INTEGRITY
        </div>
        <h2 className="text-3xl sm:text-4xl font-sans font-semibold tracking-tight text-white mb-4">
          Limitations & Responsible Forensics
        </h2>
        <p className="text-base sm:text-lg text-zinc-400 font-normal leading-relaxed">
          Credible artificial intelligence forensics requires scientific humility. VERITAS is engineered with explicit operational boundaries to prevent false accusations and over-confident generalizations.
        </p>
      </div>

      {/* 3 Pillars Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Pillar 1: Probabilistic Assessment */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-40px' }}
          transition={{ duration: 0.5, delay: 0.05, ease: [0.16, 1, 0.3, 1] }}
          className="p-7 rounded-xl bg-[#0d1017] border border-white/[0.08]"
        >
          <div className="p-2.5 rounded-lg bg-blue-500/10 border border-blue-500/20 w-fit mb-5">
            <ShieldCheck className="w-5 h-5 text-blue-400" />
          </div>
          <h3 className="text-base font-semibold text-white tracking-tight mb-2">
            Probabilistic Assessment
          </h3>
          <p className="text-sm text-zinc-400 leading-relaxed font-normal">
            Model predictions reflect statistical likelihoods of synthetic manipulation based on empirical training distributions. They serve as investigative indicators for forensic analysts, not absolute legal certainty.
          </p>
        </motion.div>

        {/* Pillar 2: Abstention Deadband */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-40px' }}
          transition={{ duration: 0.5, delay: 0.12, ease: [0.16, 1, 0.3, 1] }}
          className="p-7 rounded-xl bg-[#0d1017] border border-white/[0.08]"
        >
          <div className="p-2.5 rounded-lg bg-blue-500/10 border border-blue-500/20 w-fit mb-5">
            <AlertTriangle className="w-5 h-5 text-amber-400" />
          </div>
          <h3 className="text-base font-semibold text-white tracking-tight mb-2">
            Refusal to Guess
          </h3>
          <p className="text-sm text-zinc-400 leading-relaxed font-normal">
            Low-bitrate transcoding, severe social media compression, and contradictory modality cues trigger an explicit <code className="text-amber-300 font-mono text-xs">UNCERTAIN</code> verdict rather than forcing a misleading binary call.
          </p>
        </motion.div>

        {/* Pillar 3: Explainable Attribution */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-40px' }}
          transition={{ duration: 0.5, delay: 0.19, ease: [0.16, 1, 0.3, 1] }}
          className="p-7 rounded-xl bg-[#0d1017] border border-white/[0.08]"
        >
          <div className="p-2.5 rounded-lg bg-blue-500/10 border border-blue-500/20 w-fit mb-5">
            <Eye className="w-5 h-5 text-blue-400" />
          </div>
          <h3 className="text-base font-semibold text-white tracking-tight mb-2">
            Inspectable Attribution
          </h3>
          <p className="text-sm text-zinc-400 leading-relaxed font-normal">
            Rather than delivering an opaque score, VERITAS exposes GradCAM visual activation heatmaps and modality contribution weights so practitioners can independently review the physical evidence trail.
          </p>
        </motion.div>
      </div>
    </section>
  );
}
