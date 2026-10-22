import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { UploadCloud, Image as ImageIcon, Video, Mic, Link as LinkIcon, AlertTriangle, CheckCircle2, Copy, User, Shield } from 'lucide-react';
import { useTypingEffect } from '@/hooks/useTypingEffect';
import { useLocation } from 'wouter';

const MOCK_FLOW = [
  { role: 'user', text: "I've uploaded a video. Can you check whether it's authentic?" },
  { role: 'system', text: "I've received your content. I'm now examining the visuals, movement, audio, and available context. This usually takes a few moments." },
  { role: 'agent', agent: 'Visual Analysis', color: 'cyan', text: 'Several visual inconsistencies were detected around the face — areas that appear edited or artificially generated.' },
  { role: 'agent', agent: 'Motion Analysis', color: 'blue', text: 'Unusual movement patterns were identified between frames. The motion does not behave as naturally recorded video normally would.' },
  { role: 'agent', agent: 'Audio Analysis', color: 'emerald', text: 'The audio shows characteristics commonly found in synthesised speech. Some frequencies appear inconsistent with natural recording conditions.' },
  { role: 'agent', agent: 'Context Verification', color: 'purple', text: 'Limited online context was found for this content. No matching source could be verified.' },
  { role: 'summary', text: 'Based on the available evidence, this content shows several indicators commonly associated with AI-generated media. Visual inconsistencies, unusual movement, and audio irregularities collectively suggest the content is likely manipulated. VERITAS has high confidence that this media has been artificially generated.' }
];

export function DemoSection() {
  const [activeTab, setActiveTab] = useState('video');
  const [isUploading, setIsUploading] = useState(false);
  const [hasFile, setHasFile] = useState(false);
  const [chatSteps, setChatSteps] = useState<number>(0);
  const scrollRef = useRef<HTMLDivElement>(null);
  const [, setLocation] = useLocation();

  const startAnalysis = () => {
    if (!hasFile) return;
    setChatSteps(1);
    let step = 1;
    const interval = setInterval(() => {
      step++;
      setChatSteps(step);
      if (step >= MOCK_FLOW.length) clearInterval(interval);
    }, 2000);
  };

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [chatSteps]);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsUploading(true);
    setTimeout(() => { setIsUploading(false); setHasFile(true); }, 1000);
  };

  const visibleMessages = MOCK_FLOW.slice(0, chatSteps);

  return (
    <section id="demo" className="py-32 relative">
      <div className="max-w-7xl mx-auto px-6">
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#cc785c]/10 text-[#cc785c] text-xs font-mono font-medium mb-3">
            <span>[ILLUSTRATIVE WALKTHROUGH — SIMULATED DEMO]</span>
          </div>
          <h2 className="text-3xl md:text-5xl font-serif font-bold text-[#141413] mb-4">Interactive Detection Walkthrough</h2>
          <p className="text-[#6c6a64] max-w-2xl mx-auto">
            This illustrative preview demonstrates the multi-agent review sequence with sample media. For live inference using real trained neural networks, open the <strong>Verification Console</strong>.
          </p>
        </motion.div>

        <div className="flex flex-col lg:flex-row gap-8 h-[700px]">
          {/* Left Panel */}
          <motion.div
            className="w-full lg:w-1/3 flex flex-col gap-6"
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
          >
            <div className="flex bg-[#efe9de] rounded-lg p-1 gap-1">
              {[
                { id: 'image', icon: ImageIcon, label: 'Image' },
                { id: 'video', icon: Video, label: 'Video' },
                { id: 'audio', icon: Mic, label: 'Audio' },
                { id: 'url', icon: LinkIcon, label: 'URL' }
              ].map(tab => (
                <button
                  key={tab.id}
                  onClick={() => { setActiveTab(tab.id); setHasFile(false); setChatSteps(0); }}
                  data-testid={`tab-${tab.id}`}
                  className={`flex-1 flex items-center justify-center gap-2 py-2 px-3 text-sm font-medium rounded-md transition-colors ${
                    activeTab === tab.id ? 'bg-[#efe9de] text-[#141413]' : 'text-[#6c6a64] hover:text-[#141413] hover:bg-[#efe9de]'
                  }`}
                >
                  <tab.icon className="w-4 h-4" />
                  <span className="hidden sm:inline">{tab.label}</span>
                </button>
              ))}
            </div>

            <div
              onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
              onDrop={handleDrop}
              data-testid="dropzone"
              className={`flex-1 border-2 border-dashed rounded-2xl flex flex-col items-center justify-center p-8 transition-colors group cursor-pointer ${
                hasFile ? 'border-[#cc785c]/50 bg-[#cc785c]/5' : 'border-[#e6dfd8] hover:border-[#cc785c]/50 bg-[#efe9de] hover:bg-[#efe9de]'
              }`}
              onClick={() => {
                if (!hasFile) {
                  setIsUploading(true);
                  setTimeout(() => { setIsUploading(false); setHasFile(true); }, 1000);
                }
              }}
            >
              {isUploading ? (
                <div className="flex flex-col items-center gap-4">
                  <div className="w-12 h-12 rounded-full border-t-2 border-[#cc785c] animate-spin" />
                  <p className="text-[#6c6a64]">Loading content...</p>
                </div>
              ) : hasFile ? (
                <div className="flex flex-col items-center gap-4">
                  <div className="w-16 h-16 rounded-xl bg-[#cc785c]/20 flex items-center justify-center text-[#cc785c]">
                    <CheckCircle2 className="w-8 h-8" />
                  </div>
                  <p className="text-[#141413] font-medium">Content loaded successfully.</p>
                  <p className="text-sm text-[#6c6a64] text-center">sample_video_042.mp4<br />(14.2 MB)</p>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-4 text-[#6c6a64] group-hover:text-[#141413] transition-colors">
                  <UploadCloud className="w-12 h-12" />
                  <div className="text-center">
                    <p className="font-medium">Drag &amp; drop or click to upload</p>
                    <p className="text-sm mt-1 opacity-70">Supports MP4, MOV, WAV, JPG, PNG (Max 50MB)</p>
                  </div>
                </div>
              )}
            </div>

            <button
              disabled={!hasFile || chatSteps > 0}
              onClick={startAnalysis}
              data-testid="button-analyse"
              className={`w-full py-4 rounded-xl font-serif font-bold text-[#141413] text-lg transition-all ${
                hasFile && chatSteps === 0
                  ? 'bg-gradient-to-r from-[#cc785c] to-[#a9583e] text-[#141413] shadow-sm hover:shadow-sm'
                  : 'bg-[#efe9de] text-[#6c6a64] cursor-not-allowed'
              }`}
            >
              {chatSteps > 0 ? 'Analysing...' : 'Analyse with VERITAS'}
            </button>
          </motion.div>

          {/* Right Panel */}
          <motion.div
            className="w-full lg:w-2/3 bg-[#efe9de] border border-[#e6dfd8] rounded-2xl shadow-sm flex flex-col overflow-hidden border-[#e6dfd8]"
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
          >
            <div className="border-b border-[#e6dfd8] px-6 py-4 flex items-center justify-between bg-[#faf9f5]/50 backdrop-blur-md z-10">
              <div className="flex items-center gap-3">
                <div className="w-3 h-3 rounded-full bg-[#cc785c] shadow-sm animate-pulse" />
                <span className="font-serif text-[#141413] font-semibold">VERITAS Multi-Agent Simulation</span>
              </div>
              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-[#efe9de] text-[#6c6a64] uppercase tracking-wider">
                Simulated Preview
              </span>
            </div>

            <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-6" ref={scrollRef}>
              {chatSteps === 0 && (
                <div className="flex-1 flex items-center justify-center text-[#6c6a64] text-sm">
                  VERITAS is ready. Select a content type and click Analyse.
                </div>
              )}

              <AnimatePresence>
                {visibleMessages.map((msg, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    {msg.role !== 'user' && (
                      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-[#cc785c] to-[#a9583e] flex items-center justify-center shrink-0">
                        <Shield className="w-4 h-4 text-[#141413]" />
                      </div>
                    )}

                    <div className={`max-w-[85%] rounded-2xl p-4 ${
                      msg.role === 'user' ? 'bg-[#efe9de] text-[#141413]' :
                      msg.role === 'agent' ? 'bg-[#faf9f5] border border-[#e6dfd8]' :
                      msg.role === 'summary' ? 'bg-red-500/10 border border-red-500/20' :
                      'bg-transparent text-[#6c6a64] text-sm'
                    }`}>
                      {msg.role === 'agent' && (
                        <div className="flex items-center gap-2 mb-2">
                          <span className={`px-2 py-0.5 rounded text-xs font-bold uppercase tracking-wider bg-${msg.color}-500/20 text-${msg.color}-400`}>
                            {msg.agent}
                          </span>
                        </div>
                      )}

                      {msg.role === 'summary' ? (
                        <div>
                          <div className="flex items-center gap-2 mb-3 text-red-400 font-serif font-bold text-[#141413]">
                            <AlertTriangle className="w-5 h-5" />
                            Likely AI Generated — High Confidence
                          </div>
                          <TypewriterText text={msg.text} />
                        </div>
                      ) : (
                        <div className="text-sm md:text-base text-[#3d3d3a] leading-relaxed">{msg.text}</div>
                      )}

                      {msg.role !== 'user' && msg.role !== 'system' && (
                        <div className="mt-3 flex items-center gap-2">
                          <button className="text-[#6c6a64] hover:text-[#141413] transition-colors" data-testid="button-copy">
                            <Copy className="w-4 h-4" />
                          </button>
                        </div>
                      )}
                    </div>

                    {msg.role === 'user' && (
                      <div className="w-8 h-8 rounded-full bg-[#e8e0d2] flex items-center justify-center shrink-0">
                        <User className="w-4 h-4 text-[#141413]" />
                      </div>
                    )}
                  </motion.div>
                ))}

                {chatSteps > 0 && chatSteps < MOCK_FLOW.length && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex gap-4">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-[#cc785c] to-[#a9583e] flex items-center justify-center shrink-0">
                      <Shield className="w-4 h-4 text-[#141413]" />
                    </div>
                    <div className="bg-transparent p-2 flex gap-1 items-center">
                      <div className="w-2 h-2 rounded-full bg-[#cc785c] animate-bounce" style={{ animationDelay: '0ms' }} />
                      <div className="w-2 h-2 rounded-full bg-[#cc785c] animate-bounce" style={{ animationDelay: '150ms' }} />
                      <div className="w-2 h-2 rounded-full bg-[#cc785c] animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            <div className="border-t border-[#e6dfd8] px-6 py-3 flex items-center justify-between bg-[#faf9f5]">
              <span className="text-xs text-[#6c6a64]">Simulated walkthrough with sample data. Live neural inference runs in Console.</span>
              <button
                onClick={() => setLocation('/console')}
                data-testid="button-try-own-content"
                className="text-xs font-medium text-[#cc785c] hover:underline transition-colors flex items-center gap-1 cursor-pointer"
              >
                Launch Live Verification Console →
              </button>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}

function TypewriterText({ text }: { text: string }) {
  const { displayedText } = useTypingEffect(text, 20);
  return <div className="text-sm md:text-base text-[#3d3d3a] leading-relaxed">{displayedText}</div>;
}
