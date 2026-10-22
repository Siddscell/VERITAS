import React from 'react';
import { motion } from 'framer-motion';
import { GitMerge, ShieldCheck, AlertCircle, Scale } from 'lucide-react';

export function FusionSection() {
  return (
    <section id="fusion" className="py-28 px-6 max-w-6xl mx-auto border-b border-white/[0.06]">
      {/* Section Header */}
      <div className="max-w-3xl mb-16">
        <div className="text-xs font-mono uppercase tracking-widest text-blue-400 mb-3">
          EVIDENCE AGGREGATION
        </div>
        <h2 className="text-3xl sm:text-4xl font-sans font-semibold tracking-tight text-white mb-4">
          Dynamic-Denominator Evidence Fusion
        </h2>
        <p className="text-base sm:text-lg text-zinc-400 font-normal leading-relaxed">
          Raw model outputs cannot simply be averaged. VERITAS normalizes evidence dynamically across available modalities, ensuring missing audio or video streams never artificially dilute manipulation scores.
        </p>
      </div>

      {/* 3 Structural Columns */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
        {/* Column 1: Dynamic-Denominator Normalization */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-40px' }}
          transition={{ duration: 0.5, delay: 0.05, ease: [0.16, 1, 0.3, 1] }}
          className="p-7 rounded-xl bg-[#0d1017] border border-white/[0.08] flex flex-col justify-between"
        >
          <div>
            <div className="p-2.5 rounded-lg bg-blue-500/10 border border-blue-500/20 w-fit mb-5">
              <GitMerge className="w-5 h-5 text-blue-400" />
            </div>
            <h3 className="text-lg font-semibold text-white tracking-tight mb-2">
              Dynamic-Denominator Normalization
            </h3>
            <p className="text-sm text-zinc-400 leading-relaxed mb-6">
              In the production engine (<code className="text-blue-300 font-mono text-xs">corrected_fixed</code>), base weights are allocated across active agents: Spatial (0.40), Temporal (0.30), Audio (0.20), and Context (0.10).
            </p>
            <p className="text-sm text-zinc-400 leading-relaxed">
              When a container lacks audio, the denominator automatically rescales over remaining visual agents rather than dividing by 1.0, eliminating false-authentic dilution.
            </p>
          </div>

          <div className="mt-8 pt-4 border-t border-white/[0.06] text-xs font-mono text-zinc-500">
            Formula: <span className="text-zinc-300">Score = ∑(w_i · s_i) / ∑(w_active)</span>
          </div>
        </motion.div>

        {/* Column 2: 3-State Verdict Contract */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-40px' }}
          transition={{ duration: 0.5, delay: 0.12, ease: [0.16, 1, 0.3, 1] }}
          className="p-7 rounded-xl bg-[#0d1017] border border-white/[0.08] flex flex-col justify-between"
        >
          <div>
            <div className="p-2.5 rounded-lg bg-blue-500/10 border border-blue-500/20 w-fit mb-5">
              <Scale className="w-5 h-5 text-blue-400" />
            </div>
            <h3 className="text-lg font-semibold text-white tracking-tight mb-2">
              Three-State Decision Contract
            </h3>
            <p className="text-sm text-zinc-400 leading-relaxed mb-6">
              Binary detectors force ambiguous media into high-stakes binary guesses. VERITAS operates a strict tripartite contract:
            </p>
            <ul className="space-y-2.5 text-xs font-mono mb-6">
              <li className="flex items-center gap-2 text-emerald-400">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                <strong className="text-zinc-200">AUTHENTIC:</strong> Low composite anomaly with clear agent consensus.
              </li>
              <li className="flex items-center gap-2 text-rose-400">
                <span className="w-1.5 h-1.5 rounded-full bg-rose-400" />
                <strong className="text-zinc-200">DEEPFAKE:</strong> Elevated anomaly score exceeding calibrated threshold.
              </li>
              <li className="flex items-center gap-2 text-amber-400">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                <strong className="text-zinc-200">UNCERTAIN:</strong> Abstention deadband on borderline or conflicting evidence.
              </li>
            </ul>
          </div>

          <div className="mt-8 pt-4 border-t border-white/[0.06] text-xs font-mono text-zinc-500">
            Contract: <span className="text-zinc-300">Calibrated Deadband Abstention</span>
          </div>
        </motion.div>

        {/* Column 3: Production Determinism */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-40px' }}
          transition={{ duration: 0.5, delay: 0.19, ease: [0.16, 1, 0.3, 1] }}
          className="p-7 rounded-xl bg-[#0d1017] border border-white/[0.08] flex flex-col justify-between"
        >
          <div>
            <div className="p-2.5 rounded-lg bg-blue-500/10 border border-blue-500/20 w-fit mb-5">
              <ShieldCheck className="w-5 h-5 text-blue-400" />
            </div>
            <h3 className="text-lg font-semibold text-white tracking-tight mb-2">
              Auditable Production Stability
            </h3>
            <p className="text-sm text-zinc-400 leading-relaxed mb-6">
              Deterministic behavior is paramount for forensic integrity. The production deployment strictly locks <code className="text-zinc-300 font-mono text-xs">corrected_fixed</code> as default.
            </p>
            <p className="text-sm text-zinc-400 leading-relaxed">
              Experimental reinforcement-learning models (PPO v3) remain isolated in evaluation research and are never active in live verification pipelines.
            </p>
          </div>

          <div className="mt-8 pt-4 border-t border-white/[0.06] text-xs font-mono text-zinc-500">
            Status: <span className="text-emerald-400">Production Frozen & Verified</span>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
