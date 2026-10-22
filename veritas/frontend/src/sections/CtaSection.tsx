import React from 'react';
import { motion } from 'framer-motion';
import { ArrowRight, Shield } from 'lucide-react';
import { usePageTransition } from '@/context/TransitionContext';

export function CtaSection() {
  const { navigateTo } = usePageTransition();

  return (
    <section id="cta" className="py-28 px-6 max-w-6xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: '-40px' }}
        transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
        className="p-10 sm:p-14 rounded-2xl bg-gradient-to-b from-[#0f131d] to-[#0a0d14] border border-white/[0.08] text-center relative overflow-hidden"
      >
        <div className="relative z-10 max-w-2xl mx-auto">
          <div className="w-12 h-12 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center mx-auto mb-6">
            <Shield className="w-6 h-6 text-blue-400" />
          </div>

          <h2 className="text-3xl sm:text-4xl font-sans font-semibold tracking-tight text-white mb-4">
            Verify Digital Content With Multi-Modal Precision
          </h2>

          <p className="text-base text-zinc-400 leading-relaxed font-normal mb-8">
            Access the live VERITAS verification console to evaluate images, video files, and audio streams against trained multi-agent forensic models.
          </p>

          <button
            onClick={() => navigateTo('/console')}
            className="inline-flex items-center justify-center gap-2.5 px-7 py-3.5 rounded-lg bg-zinc-100 hover:bg-white text-zinc-950 font-medium text-sm tracking-tight transition-all duration-200 shadow-md hover:shadow-lg cursor-pointer"
          >
            <span>Launch Verification Console</span>
            <ArrowRight className="w-4 h-4 text-zinc-950" />
          </button>
        </div>
      </motion.div>
    </section>
  );
}
