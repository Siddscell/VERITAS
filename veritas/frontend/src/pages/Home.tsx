import React from 'react';
import { NavDock } from '@/components/layout/NavDock';
import { Footer } from '@/components/layout/Footer';
import { SmoothScroll } from '@/components/layout/SmoothScroll';
import { ScrollStory } from '@/sections/ScrollStory';

export default function Home() {
  return (
    <SmoothScroll>
      <div className="min-h-screen w-full bg-[#07090e] text-white selection:bg-blue-500/30 selection:text-white">
        <NavDock />
        <main>
          <ScrollStory />
        </main>
        <Footer />
      </div>
    </SmoothScroll>
  );
}
