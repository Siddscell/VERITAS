import React, { useState, useEffect } from 'react';
import { Shield, ArrowRight } from 'lucide-react';
import { motion } from 'framer-motion';
import { usePageTransition } from '@/context/TransitionContext';
import { useLenis } from '@/components/layout/SmoothScroll';

const NAV_ITEMS = [
  { label: '01 Overview', href: '#overview' },
  { label: '02 Spatial', href: '#visual' },
  { label: '03 Temporal', href: '#temporal' },
  { label: '04 Audio', href: '#audio' },
  { label: '05 Fusion', href: '#fusion' },
  { label: '06 Verify', href: '#console' },
];

export function NavDock() {
  const [activeId, setActiveId] = useState('');
  const [scrollProgress, setScrollProgress] = useState(0);
  const { navigateTo } = usePageTransition();
  const { scrollTo } = useLenis();

  useEffect(() => {
    const handleScroll = () => {
      // Calculate scroll progress percentage (0 - 100)
      const totalScroll = document.documentElement.scrollHeight - window.innerHeight;
      if (totalScroll > 0) {
        setScrollProgress((window.scrollY / totalScroll) * 100);
      }

      // Highlight active section based on true document scroll position
      const scrollPosition = window.scrollY + 200;
      for (let i = NAV_ITEMS.length - 1; i >= 0; i--) {
        const item = NAV_ITEMS[i];
        const el = document.getElementById(item.href.substring(1));
        if (el) {
          const docTop = el.getBoundingClientRect().top + window.scrollY;
          if (scrollPosition >= docTop) {
            setActiveId(item.href.substring(1));
            return;
          }
        }
      }
      setActiveId('overview');
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <>
      {/* Restrained Top Hairline Scroll Progress Bar */}
      <div className="fixed top-0 left-0 right-0 h-[2px] bg-transparent z-50 pointer-events-none">
        <div
          className="h-full bg-blue-500/80 transition-all duration-100 ease-out"
          style={{ width: `${scrollProgress}%` }}
        />
      </div>

      {/* Floating Top Navigation Capsule */}
      <motion.header
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        className="fixed top-5 left-0 right-0 z-40 flex justify-center px-4 pointer-events-none"
      >
        <div className="bg-[#0b0e14]/90 border border-white/[0.08] backdrop-blur-md rounded-full shadow-xl px-5 py-2 flex items-center gap-6 sm:gap-8 pointer-events-auto">
          {/* Logo Brand */}
          <button
            onClick={() => scrollTo('#hero')}
            className="flex items-center gap-2 text-zinc-100 hover:text-white transition-colors cursor-pointer"
          >
            <Shield className="w-4 h-4 text-blue-400" />
            <span className="font-sans font-semibold tracking-tight text-sm text-white">
              VERITAS
            </span>
          </button>

          {/* Navigation Anchors */}
          <nav className="hidden md:flex items-center gap-1">
            {NAV_ITEMS.map((item) => {
              const isActive = activeId === item.href.substring(1);
              return (
                <button
                  key={item.href}
                  onClick={() => scrollTo(item.href)}
                  className={`px-3 py-1 rounded-full text-xs font-sans tracking-tight transition-all duration-200 cursor-pointer ${
                    isActive
                      ? 'bg-white/10 text-white font-medium'
                      : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.04]'
                  }`}
                >
                  {item.label}
                </button>
              );
            })}
          </nav>

          {/* Console CTA Button */}
          <button
            onClick={() => navigateTo('/console')}
            data-testid="button-launch-console"
            className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full bg-zinc-100 hover:bg-white text-zinc-950 font-medium text-xs tracking-tight transition-all duration-200 shadow-sm cursor-pointer"
          >
            <span>Live Console</span>
            <ArrowRight className="w-3.5 h-3.5 text-zinc-950" />
          </button>
        </div>
      </motion.header>
    </>
  );
}
