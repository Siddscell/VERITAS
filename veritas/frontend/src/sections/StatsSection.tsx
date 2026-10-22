import { motion, useInView } from 'framer-motion';
import { useRef } from 'react';
import { useCountUp } from '@/hooks/useCountUp';

const stats = [
  { label: 'Accuracy', value: 96.7, suffix: '%', decimals: 1 },
  { label: 'Precision', value: 94.3, suffix: '%', decimals: 1 },
  { label: 'Recall', value: 95.8, suffix: '%', decimals: 1 },
  { label: 'F1 Score', value: 95.0, suffix: '%', decimals: 1 },
  { label: 'ROC-AUC', value: 0.987, suffix: '', decimals: 3 },
  { label: 'Samples Tested', value: 847, suffix: 'K', decimals: 0 }
];

export function StatsSection() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <section id="stats" className="py-32 relative bg-white/[0.02] border-t border-[#e6dfd8]">
      <div className="max-w-6xl mx-auto px-6">
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <h2 className="text-3xl md:text-5xl font-serif font-bold text-[#141413] mb-4">How VERITAS Performs</h2>
          <p className="text-[#6c6a64] max-w-2xl mx-auto">
            VERITAS has been evaluated on thousands of real and AI-generated samples. Here are the results.
          </p>
        </motion.div>

        <div ref={ref} className="grid grid-cols-2 md:grid-cols-3 gap-6 lg:gap-8">
          {stats.map((stat, i) => (
            <motion.div
              key={stat.label}
              className="bg-[#efe9de] border border-[#e6dfd8] rounded-xl shadow-sm p-8 rounded-3xl text-center group hover:bg-[#efe9de] transition-colors"
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1, duration: 0.5 }}
            >
              <div className="text-4xl md:text-5xl lg:text-6xl font-serif font-bold text-[#141413] text-gradient mb-2 drop-shadow-sm">
                <CountUp value={stat.value} decimals={stat.decimals} trigger={isInView} />
                <span className="text-2xl md:text-4xl">{stat.suffix}</span>
              </div>
              <div className="text-sm md:text-base font-medium text-[#6c6a64] uppercase tracking-widest">
                {stat.label}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

function CountUp({ value, decimals, trigger }: { value: number, decimals: number, trigger: boolean }) {
  const count = useCountUp(value, 2000, decimals, trigger);
  return <span>{count}</span>;
}
