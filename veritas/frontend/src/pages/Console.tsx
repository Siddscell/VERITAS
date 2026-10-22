import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Shield, ArrowLeft, ArrowRight, UploadCloud, Image as ImageIcon, Video, Mic,
  Link as LinkIcon, CheckCircle2, AlertTriangle, HelpCircle, User,
  ArrowUp, ServerCrash, RefreshCcw
} from 'lucide-react';
import { usePageTransition } from '@/context/TransitionContext';

// ----------------------------------------------------------------------
// Types
// ----------------------------------------------------------------------
export type AgentResult = {
  agent: string;
  confidence: number;
  finding: string;
  status: 'pending' | 'streaming' | 'done';
  gradcam_b64?: string | null;
  backend_status?: string;
  is_fake?: boolean | null;
};

export type AppState = 'idle' | 'uploading' | 'analysing' | 'done' | 'error';

// ----------------------------------------------------------------------
// Initial Setup
// ----------------------------------------------------------------------
const INITIAL_AGENTS: AgentResult[] = [
  { agent: 'Spatial Analyst',   confidence: 0, finding: '', status: 'pending', gradcam_b64: null },
  { agent: 'Temporal Analyst',  confidence: 0, finding: '', status: 'pending' },
  { agent: 'Audio Analyst',     confidence: 0, finding: '', status: 'pending' },
  { agent: 'Context Analyst',   confidence: 0, finding: '', status: 'pending' },
];

const STAGGER = {
  header: 0.18,
  sidebar: 0.28,
  chat: 0.38,
};

const BOOT_LINES = [
  "[OK] Spatial Analysis Module loaded — EfficientNet-B0",
  "[OK] Temporal Analysis Module loaded — R3D-18",
  "[OK] Audio Analysis Module loaded — MelSpectrogram CNN",
  "[OK] Context Analysis Module loaded",
  "[OK] GradCAM engine initialized",
  "[READY] VERITAS v1.0 — awaiting media input"
];

function BootSequence({ onComplete }: { onComplete: () => void }) {
  useEffect(() => {
    const timer = setTimeout(() => {
      onComplete();
    }, BOOT_LINES.length * 120 + 200);
    return () => clearTimeout(timer);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <motion.div
      className="flex flex-col gap-2 text-sm w-full"
      style={{ fontFamily: 'JetBrains Mono, ui-monospace, monospace' }}
      initial="hidden"
      animate="visible"
      variants={{
        visible: {
          transition: {
            staggerChildren: 0.12,
          },
        },
      }}
    >
      {BOOT_LINES.map((line, i) => (
        <motion.div
          key={i}
          variants={{
            hidden: { opacity: 0 },
            visible: { opacity: 1, transition: { duration: 0.05 } },
          }}
          className={
            i === BOOT_LINES.length - 1
              ? "text-[#cc785c] text-base mt-2"
              : "text-[#6c6a64]"
          }
        >
          {line}
        </motion.div>
      ))}
    </motion.div>
  );
}

export default function Console() {
  const { navigateTo } = usePageTransition();
  const [activeTab, setActiveTab] = useState('video');
  const [appState, setAppState] = useState<AppState>('idle');
  const [errorMessage, setErrorMessage] = useState('');
  const [bootComplete, setBootComplete] = useState(false);
  
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  
  const [agents, setAgents] = useState<AgentResult[]>(INITIAL_AGENTS);
  const [verdict, setVerdict] = useState<{ 
    verdict: string, 
    confidence: number,
    contributing: string[],
    unavailable: string[],
    fusion_method?: string,
    predicted_category?: string,
    category_probabilities?: Record<string, number>,
    dynamic_weights?: Record<string, number>,
    decision_threshold?: number,
    fake_probability?: number,
    decision_confidence?: number,
    verdict_class?: string,
    is_abstention?: boolean,
    leading_category_is_hypothesis?: boolean,
  } | null>(null);
  
  const chatScrollRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll chat area when content updates
  useEffect(() => {
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  }, [agents, verdict, appState]);

  // ----------------------------------------------------------------------
  // Actions
  // ----------------------------------------------------------------------
  const reset = () => {
    setAppState('idle');
    setAgents(INITIAL_AGENTS);
    setVerdict(null);
    setErrorMessage('');
    setSelectedFile(null);
  };

  const handleFileDrop = (file: File) => {
    setAppState('uploading');
    setTimeout(() => {
      setSelectedFile(file);
      setAppState('idle');
    }, 800);
  };

  const startAnalysis = async () => {
    if (!selectedFile || appState === 'analysing') return;

    setAppState('analysing');
    setAgents(INITIAL_AGENTS);
    setVerdict(null);
    setErrorMessage('');

    let hasError = false;
    let verdictReceived = false;

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      // We read the stream using fetch + ReadableStream
      const response = await fetch('/api/analyse/stream', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok || !response.body) {
        throw new Error(`Server returned ${response.status}: ${response.statusText || 'Unable to connect'}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      // SSE parser state
      let buffer = '';
      let pendingEvent = '';

      outer: while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // SSE messages are separated by blank lines (\n\n).
        // Split on \n so we can process line by line.
        const lines = buffer.split('\n');
        // Keep last (potentially incomplete) line in buffer
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          const trimmed = line.trim();

          // Blank line = end of one SSE message; reset event accumulator
          if (trimmed === '') {
            pendingEvent = '';
            continue;
          }

          // Capture the named event
          if (trimmed.startsWith('event:')) {
            pendingEvent = trimmed.slice('event:'.length).trim();
            continue;
          }

          // Parse the data payload
          if (trimmed.startsWith('data:')) {
            const dataStr = trimmed.slice('data:'.length).trim();
            if (!dataStr) continue;

            let data: Record<string, unknown>;
            try {
              data = JSON.parse(dataStr);
            } catch (parseErr) {
              console.warn('SSE JSON parse error:', parseErr, '| raw:', dataStr);
              continue;
            }

            const eventName = pendingEvent;

            if (eventName === 'agent_update') {
              if (!data.finding) continue;
              const agentName = ((data.agent || data.agent_name) as string) ?? '';
              let idx = 0;
              if (agentName.toLowerCase().includes('spatial'))  idx = 0;
              else if (agentName.toLowerCase().includes('temporal')) idx = 1;
              else if (agentName.toLowerCase().includes('audio'))    idx = 2;
              else if (agentName.toLowerCase().includes('context'))  idx = 3;

              setAgents(prev => {
                const copy = [...prev];
                copy[idx] = {
                  agent: INITIAL_AGENTS[idx].agent,
                  confidence: data.confidence as number,
                  finding: data.finding as string,
                  status: 'streaming',
                  backend_status: data.status as string | undefined,
                  is_fake: (data.is_fake !== undefined) ? (data.is_fake as boolean | null) : undefined,
                  // Carry gradcam_b64 only for the spatial slot (idx 0)
                  ...(idx === 0 && { gradcam_b64: (data.gradcam_b64 as string | null) ?? null }),
                };
                return copy;
              });

              // Transition to 'done' after the confidence bar has animated (600 ms)
              const capturedIdx = idx;
              setTimeout(() => {
                setAgents(prev => {
                  const copy = [...prev];
                  copy[capturedIdx] = { ...copy[capturedIdx], status: 'done' };
                  return copy;
                });
              }, 650);

            } else if (eventName === 'final_verdict') {
              verdictReceived = true;
              const rawVerdict = (data.verdict as string).toUpperCase();
              let normalised = rawVerdict;
              if (rawVerdict === 'FAKE' || rawVerdict === 'DEEPFAKE') normalised = 'DEEPFAKE';
              else if (rawVerdict === 'REAL' || rawVerdict === 'AUTHENTIC') normalised = 'AUTHENTIC';
              else if (rawVerdict.includes('UNCERTAIN') || rawVerdict.includes('INSUFFICIENT')) {
                normalised = 'UNCERTAIN';
              } else if (rawVerdict.includes('UNVERIFIED')) {
                normalised = 'UNVERIFIED';
              }

              const verdictClass = (data.verdict_class as string | undefined)
                ?? (normalised === 'DEEPFAKE' || normalised === 'AUTHENTIC'
                  || normalised === 'UNCERTAIN' || normalised === 'UNVERIFIED'
                  ? normalised
                  : 'UNCERTAIN');
              const isAbstention = Boolean(
                data.is_abstention
                ?? (verdictClass === 'UNCERTAIN' || verdictClass === 'UNVERIFIED')
              );

              setVerdict({
                verdict: normalised,
                confidence: (data.overall_confidence ?? data.confidence) as number,
                contributing: (data.contributing_agents as string[]) || [],
                unavailable: (data.unavailable_agents as string[]) || [],
                fusion_method: data.fusion_method as string | undefined,
                predicted_category: data.predicted_category as string | undefined,
                category_probabilities: data.category_probabilities as Record<string, number> | undefined,
                dynamic_weights: data.dynamic_weights as Record<string, number> | undefined,
                decision_threshold: data.decision_threshold as number | undefined,
                fake_probability: data.fake_probability as number | undefined,
                decision_confidence: data.decision_confidence as number | undefined,
                verdict_class: verdictClass,
                is_abstention: isAbstention,
                leading_category_is_hypothesis: Boolean(
                  data.leading_category_is_hypothesis ?? isAbstention
                ),
              });

            } else if (eventName === 'done') {
              break outer;

            } else if (eventName === 'error') {
              hasError = true;
              setErrorMessage(
                (data.message ?? data.error ?? 'An unexpected error occurred during inference') as string
              );
              setAppState('error');
              break outer;
            }
          }
        }
      }

      if (!hasError) {
        if (verdictReceived) {
          setAppState('done');
        } else {
          setErrorMessage('Connection closed before analysis verdict was received.');
          setAppState('error');
        }
      }

    } catch (err: any) {
      console.error(err);
      setErrorMessage(err.message || 'Failed to connect to analysis stream');
      setAppState('error');
    }
  };

  // ----------------------------------------------------------------------
  // UI Renderers
  // ----------------------------------------------------------------------
  const tabs = [
    { id: 'image', icon: ImageIcon, label: 'Image', accept: 'image/jpeg,image/png,image/webp,image/bmp' },
    { id: 'video', icon: Video, label: 'Video', accept: 'video/mp4,video/quicktime,video/avi,video/mkv,video/webm' },
    { id: 'audio', icon: Mic, label: 'Audio', accept: 'audio/wav,audio/mpeg,audio/mp4,audio/aac,audio/ogg,audio/flac' },
    { id: 'url', icon: LinkIcon, label: 'URL', accept: '' },
  ];

  return (
    <motion.div
      className="h-screen w-full bg-[#faf9f5] text-[#141413] flex flex-col overflow-hidden"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.35, delay: 0.15, ease: 'easeOut' }}
    >
      {/* ── Top Header ─────────────────────────────────────────────────────── */}
      <motion.div
        className="shrink-0 h-[60px] bg-[#faf9f5] border-b border-[#e6dfd8] flex items-center justify-between px-6 z-20"
        initial={{ opacity: 0, y: -12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: STAGGER.header, ease: 'easeOut' }}
      >
        <div className="flex items-center gap-3">
          <Shield className="w-6 h-6 text-[#cc785c]" />
          <span className="font-serif font-bold tracking-wide">VERITAS</span>
          <span className="text-xs text-[#6c6a64] hidden sm:block">AI Content Verification Console</span>
        </div>
        <button
          onClick={() => navigateTo('/')}
          className="flex items-center gap-2 text-[#6c6a64] hover:text-[#141413] transition-colors text-sm cursor-pointer"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Home
        </button>
      </motion.div>

      {/* ── Two-column body ──────────────────────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden">
        {/* LEFT SIDEBAR */}
        <motion.div
          className="w-full max-w-[320px] shrink-0 border-r border-[#e6dfd8] flex flex-col bg-[#f5f0e8] overflow-y-auto"
          initial={{ opacity: 0, x: -18 }}
          animate={{ opacity: bootComplete ? 1 : 0, x: bootComplete ? 0 : -18, pointerEvents: bootComplete ? 'auto' : 'none' }}
          transition={{ duration: 0.45, ease: 'easeOut' }}
        >
          <div className="p-6 flex flex-col gap-5 flex-1">
            <div>
              <h2 className="text-lg font-serif font-bold text-[#141413] mb-1">Analyse Content</h2>
              <p className="text-xs text-[#6c6a64]">Upload a file or paste a link below.</p>
            </div>

            {/* Type tabs */}
            <div className="flex bg-[#efe9de] rounded-lg p-1 gap-1">
              {tabs.map(tab => (
                <button
                  key={tab.id}
                  onClick={() => { setActiveTab(tab.id); reset(); }}
                  className={`flex-1 flex flex-col items-center gap-1 py-2 px-1 text-xs font-medium rounded-md transition-colors ${
                    activeTab === tab.id ? 'bg-[#faf9f5] text-[#141413] shadow-sm' : 'text-[#6c6a64] hover:text-[#141413] hover:bg-[#faf9f5]/50'
                  }`}
                >
                  <tab.icon className="w-4 h-4" />
                  {tab.label}
                </button>
              ))}
            </div>

            {/* Drop zone */}
            {activeTab === 'url' ? (
              <div className="rounded-2xl border-2 border-[#e6dfd8] bg-[#faf9f5] p-6 min-h-[160px] flex flex-col gap-3 justify-center text-center">
                <LinkIcon className="w-8 h-8 text-[#cc785c] mx-auto" />
                <p className="text-sm text-[#141413] font-medium">Direct File Ingestion</p>
                <p className="text-xs text-[#6c6a64] leading-relaxed">
                  For reliable container verification, please download the media file and upload it using the <strong>Image</strong>, <strong>Video</strong>, or <strong>Audio</strong> tabs above.
                </p>
              </div>
            ) : (
              <div
                className={`rounded-2xl border-2 border-dashed flex flex-col items-center justify-center p-6 cursor-pointer transition-all duration-300 min-h-[160px] ${
                  appState === 'analysing' 
                    ? 'opacity-50 cursor-not-allowed border-[#e6dfd8] bg-[#faf9f5]'
                    : selectedFile
                      ? 'border-[#cc785c]/50 bg-[#fff8f5]'
                      : 'border-[#e6dfd8] bg-[#faf9f5] hover:border-[#cc785c] hover:bg-[#f5f0e8]'
                }`}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  if (appState === 'analysing') return;
                  const file = e.dataTransfer.files[0];
                  if (!file) return;
                  
                  const currentTab = activeTab;
                  const name = file.name.toLowerCase();
                  const isImageExt = /\.(jpe?g|png|webp|bmp|tiff?)$/i.test(name);
                  const isVideoExt = /\.(mp4|mov|avi|mkv|webm|flv)$/i.test(name);
                  const isAudioExt = /\.(wav|mp3|m4a|aac|ogg|flac)$/i.test(name);

                  const isValid = 
                    (currentTab === 'image' && (file.type.startsWith('image/') || isImageExt)) ||
                    (currentTab === 'video' && (file.type.startsWith('video/') || isVideoExt)) ||
                    (currentTab === 'audio' && (file.type.startsWith('audio/') || isAudioExt)) ||
                    currentTab === 'url';
                  
                  if (!isValid) {
                    alert(`Please upload a valid ${currentTab} file. Supported: ${
                      currentTab === 'video' ? 'MP4, MOV, AVI, MKV, WEBM' :
                      currentTab === 'audio' ? 'WAV, MP3, M4A, AAC, FLAC' :
                      'JPG, PNG, WEBP, BMP'
                    }`);
                    return;
                  }

                  if (file.size > 50 * 1024 * 1024) {
                    alert('File exceeds the 50MB limit. Please choose a smaller file.');
                    return;
                  }
                  
                  handleFileDrop(file);
                }}
                onClick={() => {
                  if (appState === 'idle' || appState === 'done') fileInputRef.current?.click();
                }}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  disabled={appState === 'analysing'}
                  accept={tabs.find(t => t.id === activeTab)?.accept || 'image/*,video/*,audio/*'}
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (appState !== 'analysing' && f) {
                      if (f.size > 50 * 1024 * 1024) {
                        alert('File exceeds the 50MB limit. Please choose a smaller file.');
                      } else {
                        handleFileDrop(f);
                      }
                    }
                    e.target.value = '';
                  }}
                />
                {appState === 'uploading' ? (
                  <div className="flex flex-col items-center gap-3">
                    <div className="w-10 h-10 rounded-full border-t-2 border-[#cc785c] animate-spin" />
                    <p className="text-[#6c6a64] text-sm">Loading...</p>
                  </div>
                ) : selectedFile ? (
                  <div className="flex flex-col items-center gap-3 text-center">
                    <div className="w-12 h-12 rounded-xl bg-[#cc785c]/20 flex items-center justify-center text-[#cc785c]">
                      <CheckCircle2 className="w-6 h-6" />
                    </div>
                    <p className="text-[#141413] text-sm font-medium">Content loaded</p>
                    <p className="text-xs text-[#6c6a64]">
                      {selectedFile.name}<br />
                      ({(selectedFile.size / (1024 * 1024)).toFixed(1)} MB)
                    </p>
                  </div>
                ) : (
                  <div className="flex flex-col items-center gap-3 text-center text-[#6c6a64]">
                    <UploadCloud className="w-10 h-10" />
                    <div>
                      <p className="text-sm font-medium">Drop your file here, or click to browse</p>
                      <p className="text-xs mt-1 opacity-60">
                        {activeTab === 'image' && 'JPG, PNG, WEBP — Max 50MB'}
                        {activeTab === 'video' && 'MP4, MOV, AVI, MKV — Max 50MB'}
                        {activeTab === 'audio' && 'WAV, MP3, M4A, AAC — Max 50MB'}
                        {activeTab === 'url' && 'Select media tab to upload file'}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Analyse button */}
            <button
              disabled={!selectedFile || appState !== 'idle' || activeTab === 'url'}
              onClick={startAnalysis}
              title={appState === 'analysing' ? "Analysis in progress..." : ""}
              className={`w-full py-3.5 rounded-xl font-serif font-bold transition-all text-sm ${
                selectedFile && appState === 'idle' && activeTab !== 'url'
                  ? 'bg-[#cc785c] text-white shadow-md hover:bg-[#a9583e] hover:shadow-sm cursor-pointer'
                  : 'bg-[#efe9de] text-[#6c6a64] cursor-not-allowed'
              }`}
            >
              {appState === 'analysing' ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="w-4 h-4 rounded-full border-2 border-white/60 border-t-white animate-spin inline-block" />
                  Analysing with Multi-Agent Pipeline...
                </span>
              ) : appState === 'uploading' ? (
                'Loading file...'
              ) : (
                'Analyse with VERITAS'
              )}
            </button>

            {appState === 'done' && (
              <button
                onClick={reset}
                className="w-full py-3 rounded-xl border border-[#e6dfd8] bg-[#faf9f5] hover:bg-[#efe9de] text-[#141413] text-sm font-medium transition-colors flex items-center justify-center gap-2 cursor-pointer shadow-sm"
              >
                <RefreshCcw className="w-4 h-4 text-[#cc785c]" />
                Verify Another Sample
              </button>
            )}

            <div className="flex-1" />

            <div className="flex items-start gap-3 pt-4 border-t border-[#e6dfd8]">
              <Shield className="w-4 h-4 text-[#cc785c]/50 shrink-0 mt-0.5" />
              <p className="text-xs text-[#6c6a64] leading-relaxed">
                VERITAS combines insights from four specialised systems to determine whether content is authentic or AI-generated.
              </p>
            </div>
          </div>
        </motion.div>

        {/* RIGHT CHAT AREA */}
        <motion.div
          className="flex-1 flex flex-col overflow-hidden bg-[#faf9f5]"
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, delay: STAGGER.chat, ease: 'easeOut' }}
        >
          <div className="flex-1 overflow-y-auto p-6" ref={chatScrollRef}>
            
            {/* INITIAL STATE */}
            {appState === 'idle' && (
              <div className="h-full flex flex-col items-start justify-start p-6 max-w-3xl mx-auto w-full pt-12">
                <BootSequence onComplete={() => setBootComplete(true)} />
              </div>
            )}

            {/* USER MESSAGE */}
            {(appState === 'analysing' || appState === 'done') && selectedFile && (
              <div className="max-w-4xl mx-auto flex flex-col gap-6">
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex gap-4 justify-end"
                >
                  <div className="max-w-[80%] bg-[#efe9de] text-[#141413] rounded-2xl rounded-tr-sm px-5 py-4 shadow-sm">
                    <p className="text-sm md:text-base leading-relaxed text-[#3d3d3a]">
                      I've uploaded <strong>{selectedFile.name}</strong>. Can you verify its authenticity?
                    </p>
                  </div>
                  <div className="w-9 h-9 rounded-full bg-[#f5f0e8] border border-[#e6dfd8] flex items-center justify-center shrink-0">
                    <User className="w-4 h-4 text-[#141413]" />
                  </div>
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex gap-4 justify-start mb-4"
                >
                  <div className="w-9 h-9 rounded-full bg-[#cc785c] flex items-center justify-center shrink-0 shadow-sm">
                    <Shield className="w-4 h-4 text-white" />
                  </div>
                  <div className="max-w-[80%] bg-[#faf9f5] border border-[#e6dfd8] shadow-sm rounded-2xl rounded-tl-sm px-5 py-4">
                    <p className="text-sm md:text-base leading-relaxed text-[#3d3d3a]">
                      Analysing content. Connecting to multi-agent framework...
                    </p>
                  </div>
                </motion.div>

                {/* AGENT CARDS */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <AnimatePresence>
                    {agents.map((ag, idx) => (
                      <AgentCard key={ag.agent} agent={ag} index={idx} />
                    ))}
                  </AnimatePresence>
                </div>

                {/* GRADCAM PANEL — shown below the grid when spatial result has heatmap */}
                <AnimatePresence>
                  {agents[0].gradcam_b64 && (
                    <motion.div
                      key="gradcam-panel"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ duration: 0.8, ease: 'easeOut' }}
                      className="bg-[#faf9f5] border border-[#e6dfd8] rounded-xl shadow-sm overflow-hidden"
                    >
                      <div className="px-5 py-3 border-b border-[#e6dfd8] flex items-center gap-2">
                        <span className="font-mono text-xs font-semibold text-[#cc785c] uppercase tracking-widest"
                          style={{ fontFamily: 'JetBrains Mono, ui-monospace, monospace' }}>
                          Region Analysis
                        </span>
                        <span className="text-xs text-[#6c6a64]">&mdash; Spatial Gradient Activation Map</span>
                      </div>
                      <div className="p-4">
                        <img
                          src={`data:image/png;base64,${agents[0].gradcam_b64}`}
                          alt="GradCAM heatmap showing regions of highest manipulation probability"
                          className="rounded-lg w-full object-cover"
                          style={{ maxHeight: '300px' }}
                        />
                        <p className="text-xs text-[#6c6a64] mt-3 leading-relaxed">
                          Highlighted regions indicate areas of highest manipulation probability. Warmer colours (red/orange) correspond to the features most influential in the FAKE classification decision.
                        </p>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* FINAL VERDICT */}
                <AnimatePresence>
                  {verdict && (
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ type: 'spring', damping: 20 }}
                      className="mt-6"
                    >
                      {(() => {
                        const isAuthentic = verdict.verdict === 'AUTHENTIC';
                        const isDeepfake = verdict.verdict === 'DEEPFAKE';
                        const isUncertain = verdict.verdict === 'UNCERTAIN'
                          || verdict.verdict_class === 'UNCERTAIN'
                          || Boolean(verdict.is_abstention && !isDeepfake && !isAuthentic);
                        const isUnverified = verdict.verdict === 'UNVERIFIED'
                          || verdict.verdict_class === 'UNVERIFIED';
                        const showHypothesis = Boolean(
                          verdict.leading_category_is_hypothesis
                          || isUncertain
                          || isUnverified
                        );
                        const fakeScore = verdict.fake_probability ?? verdict.confidence;

                        const verdictColor = isDeepfake
                          ? { bg: 'bg-[#181715]', border: 'border-red-500/30', icon: 'bg-red-500/20 text-[#c64545]', text: 'text-[#c64545]', sub: 'text-[#faf9f5]' }
                          : isAuthentic
                            ? { bg: 'bg-[#f0fdf4]', border: 'border-green-500/30', icon: 'bg-green-500/20 text-[#5db872]', text: 'text-[#5db872]', sub: 'text-[#6c6a64]' }
                            : { bg: 'bg-[#fefaf0]', border: 'border-[#e8a55a]/50', icon: 'bg-[#e8a55a]/20 text-[#e8a55a]', text: 'text-[#e8a55a]', sub: 'text-[#6c6a64]' };

                        const headline = isDeepfake
                          ? 'DEEPFAKE DETECTED'
                          : isAuthentic
                            ? 'AUTHENTIC MEDIA'
                            : isUnverified
                              ? 'UNVERIFIED MODALITY'
                              : 'INCONCLUSIVE — UNCERTAIN';

                        let subtitle: string;
                        if (isUnverified) {
                          subtitle = 'No usable modality evidence; authenticity cannot be claimed.';
                        } else if (isUncertain) {
                          const hyp = verdict.predicted_category && verdict.predicted_category !== 'UNVERIFIED'
                            ? ` Leading category hypothesis: ${verdict.predicted_category.replace(/_/g, ' ').toLowerCase()} (not definitive).`
                            : '';
                          subtitle = `Evidence is insufficient for a definitive verdict.${hyp}`;
                          if (verdict.unavailable.length > 0) {
                            subtitle += ` Unavailable: ${verdict.unavailable.map(a => a.split(' ')[0]).join(', ')}.`;
                          }
                        } else if (verdict.unavailable.length === 0) {
                          subtitle = verdict.predicted_category && verdict.predicted_category !== 'AUTHENTIC'
                            ? `Fusion identifies ${verdict.predicted_category.replace(/_/g, ' ').toLowerCase()}`
                            : 'Multi-agent consensus reached across all modalities';
                        } else {
                          subtitle = `Based on active modalities: ${verdict.contributing.map(a => a.split(' ')[0]).join(', ')} (Unavailable: ${verdict.unavailable.map(a => a.split(' ')[0]).join(', ')})`;
                        }
                        
                        return (
                          <div className={`p-6 rounded-2xl border shadow-lg flex flex-col md:flex-row gap-4 items-start md:items-center justify-between ${verdictColor.bg} ${verdictColor.border}`}>
                            <div className="flex items-start gap-4">
                              <div className={`w-12 h-12 rounded-full flex items-center justify-center shrink-0 mt-0.5 ${verdictColor.icon}`}>
                                {isDeepfake
                                  ? <AlertTriangle className="w-6 h-6" />
                                  : isAuthentic
                                    ? <CheckCircle2 className="w-6 h-6" />
                                    : <HelpCircle className="w-6 h-6" />}
                              </div>
                              <div>
                                <div className="flex items-center gap-2 flex-wrap mb-1">
                                  <h3 className={`text-xl font-bold ${verdictColor.text}`}>
                                    {headline}
                                  </h3>
                                  {verdict.predicted_category && verdict.predicted_category !== 'AUTHENTIC' && verdict.predicted_category !== 'UNVERIFIED' && (
                                    <span className="px-2 py-0.5 rounded text-[11px] font-mono font-bold tracking-wider uppercase bg-[#cc785c]/20 text-[#cc785c] border border-[#cc785c]/30">
                                      {showHypothesis ? 'HYPOTHESIS: ' : ''}
                                      {verdict.predicted_category.replace(/_/g, ' ')}
                                    </span>
                                  )}
                                  {verdict.fusion_method && (
                                    <span className="px-2 py-0.5 rounded text-[10px] font-mono tracking-wider uppercase bg-[#efe9de] text-[#6c6a64]">
                                      {verdict.fusion_method.toUpperCase()}
                                    </span>
                                  )}
                                </div>
                                <p className={`text-sm ${verdictColor.sub}`}>
                                  {subtitle}
                                </p>
                                {verdict.dynamic_weights && (
                                  <div className="flex items-center gap-2 flex-wrap mt-2 text-[11px] font-mono text-[#6c6a64]">
                                    <span className="font-semibold text-[#141413]">Dynamic Weights:</span>
                                    {Object.entries(verdict.dynamic_weights).map(([k, w]) => (
                                      <span key={k} className="px-1.5 py-0.5 bg-[#efe9de] rounded border border-[#e6dfd8]">
                                        {k}: {(w * 100).toFixed(1)}%
                                      </span>
                                    ))}
                                    {verdict.decision_threshold !== undefined && (
                                      <span className="px-1.5 py-0.5 bg-[#efe9de] rounded border border-[#e6dfd8]">
                                        Threshold: {(verdict.decision_threshold * 100).toFixed(1)}%
                                      </span>
                                    )}
                                    {verdict.decision_confidence !== undefined && (
                                      <span className="px-1.5 py-0.5 bg-[#efe9de] rounded border border-[#e6dfd8]">
                                        Decision conf: {(verdict.decision_confidence * 100).toFixed(1)}%
                                      </span>
                                    )}
                                  </div>
                                )}
                                {verdict.category_probabilities && (
                                  <div className="flex items-center gap-2 flex-wrap mt-2 text-[11px] font-mono text-[#6c6a64]">
                                    <span className="font-semibold text-[#141413]">Category Attribution:</span>
                                    {Object.entries(verdict.category_probabilities).map(([cat, prob]) => (
                                      <span key={cat} className="px-1.5 py-0.5 bg-[#efe9de] rounded border border-[#e6dfd8]">
                                        {cat.replace(/_/g, ' ')}: {(prob * 100).toFixed(1)}%
                                      </span>
                                    ))}
                                  </div>
                                )}
                              </div>
                            </div>
                            <div className="text-right ml-auto shrink-0">
                              <p className={`font-mono text-4xl font-bold ${verdictColor.text}`}>
                                {(fakeScore * 100).toFixed(1)}%
                              </p>
                              <p className={`text-xs ${verdictColor.sub}`}>
                                FUSED FAKE SCORE
                              </p>
                              {isUncertain && (
                                <p className={`text-[10px] mt-1 font-mono ${verdictColor.sub}`}>
                                  ABSTENTION — NOT AUTHENTIC
                                </p>
                              )}
                            </div>
                          </div>
                        );
                      })()}
                    </motion.div>
                  )}
                </AnimatePresence>

              </div>
            )}

            {/* ERROR STATE */}
            {appState === 'error' && (
              <div className="h-full flex flex-col items-center justify-center gap-6 text-center max-w-md mx-auto">
                <motion.div
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  className="w-20 h-20 rounded-2xl bg-red-50 border border-red-100 flex items-center justify-center shadow-sm"
                >
                  <ServerCrash className="w-10 h-10 text-red-500" />
                </motion.div>
                <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
                  <h3 className="text-2xl font-serif font-bold text-red-600 mb-2">Analysis Failed</h3>
                  <p className="text-[#6c6a64] text-sm leading-relaxed mb-6">
                    {errorMessage}
                  </p>
                  <button
                    onClick={reset}
                    className="flex items-center gap-2 mx-auto px-6 py-2 bg-[#efe9de] text-[#141413] rounded-lg hover:bg-[#e6dfd8] transition-colors font-medium text-sm"
                  >
                    <RefreshCcw className="w-4 h-4" />
                    Try Again
                  </button>
                </motion.div>
              </div>
            )}
          </div>

          {/* Bottom input bar */}
          <div className="shrink-0 border-t border-[#e6dfd8] p-4 bg-[#faf9f5]/30 backdrop-blur-sm">
            <div className="max-w-3xl mx-auto">
              <div className="flex items-center gap-3 bg-[#efe9de] border border-[#e6dfd8] rounded-xl shadow-sm px-4 py-3">
                <input
                  type="text"
                  placeholder="Ask VERITAS anything about your content..."
                  readOnly
                  className="flex-1 bg-transparent text-sm text-[#6c6a64] placeholder-slate-500 outline-none cursor-not-allowed"
                />
                <button
                  disabled
                  className="w-8 h-8 rounded-lg bg-[#cc785c]/10 text-[#cc785c] flex items-center justify-center opacity-50 cursor-not-allowed"
                >
                  <ArrowUp className="w-4 h-4" />
                </button>
              </div>
              <p className="text-center text-xs text-[#6c6a64] mt-2">
                VERITAS analyses content only. Results are informational and not legal evidence.
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    </motion.div>
  );
}

// ----------------------------------------------------------------------
// Agent Card Component
// ----------------------------------------------------------------------
function AgentCard({ agent, index }: { agent: AgentResult; index: number }) {
  const isPending = agent.status === 'pending';
  const unavailableStatuses = ['untrained', 'mocked', 'no_audio', 'not_applicable', 'unavailable', 'error', 'failed'];
  const isUnavailable = !isPending && (agent.is_fake === null || agent.is_fake === undefined || unavailableStatuses.includes(agent.backend_status ?? ''));
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1 }}
      className={`border shadow-sm rounded-xl p-5 flex flex-col gap-4 ${isUnavailable ? 'bg-[#f5f0e8] border-[#e6dfd8]/50 opacity-80' : 'bg-[#faf9f5] border-[#e6dfd8]'}`}
    >
      <div className="flex items-center justify-between">
        <span className={`inline-block px-3 py-1 rounded text-white text-xs font-bold tracking-wide uppercase shadow-sm ${
          isPending
            ? 'bg-[#8c887b]'
            : isUnavailable
              ? 'bg-[#a39f98]'
              : agent.confidence > 0.75 
                ? 'bg-[#c64545]' 
                : agent.confidence > 0.45 
                  ? 'bg-[#e8a55a]' 
                  : 'bg-[#5db872]'
        }`}>
          {agent.agent}
        </span>
        <span className="text-xs text-[#6c6a64] font-medium">
          {isPending ? 'Waiting...' : isUnavailable ? 'Unavailable' : 'Complete'}
        </span>
      </div>

      <div className="flex-1">
        {isPending ? (
          <div className="space-y-2">
            <div className="h-3 bg-[#efe9de] rounded animate-pulse w-full" />
            <div className="h-3 bg-[#efe9de] rounded animate-pulse w-5/6" />
            <div className="h-3 bg-[#efe9de] rounded animate-pulse w-4/6" />
          </div>
        ) : (
          <ConsoleTypewriterText text={agent.finding ?? ''} />
        )}
      </div>

      <div>
        <div className="flex justify-between text-xs mb-1.5 font-medium">
          <span className="text-[#6c6a64]">{isUnavailable ? 'Status' : 'Fake Prob'}</span>
          <span className={`font-mono ${
            isPending
              ? 'text-[#6c6a64]'
              : isUnavailable
                ? 'text-[#6c6a64]'
                : agent.confidence > 0.75 
                  ? 'text-[#c64545]' 
                  : agent.confidence > 0.45 
                    ? 'text-[#e8a55a]' 
                    : 'text-[#5db872]'
          }`}>
            {isPending ? '--%' : isUnavailable ? 'N/A' : `${(agent.confidence * 100).toFixed(1)}%`}
          </span>
        </div>
        <div className="h-1.5 w-full bg-[#efe9de] rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: isPending || isUnavailable ? '0%' : `${agent.confidence * 100}%` }}
            transition={{ duration: 0.6, ease: 'easeOut' }}
            className={`h-full rounded-full ${
              isPending
                ? 'bg-transparent'
                : isUnavailable
                  ? 'bg-[#a39f98]'
                  : agent.confidence > 0.75 
                    ? 'bg-[#c64545]' 
                    : agent.confidence > 0.45 
                      ? 'bg-[#e8a55a]' 
                      : 'bg-[#5db872]'
            }`}
          />
        </div>
      </div>
    </motion.div>
  );
}

// ----------------------------------------------------------------------
// Typewriter
// ----------------------------------------------------------------------
function ConsoleTypewriterText({ text }: { text: string }) {
  const [displayed, setDisplayed] = useState('');

  useEffect(() => {
    if (!text) {
      setDisplayed('');
      return;
    }
    setDisplayed('');
    let i = 0;
    const interval = setInterval(() => {
      if (i < text.length) {
        setDisplayed(text.slice(0, i + 1));
        i++;
      } else {
        clearInterval(interval);
      }
    }, 12);
    return () => clearInterval(interval);
  }, [text]);

  return <p className="text-sm leading-relaxed text-[#3d3d3a] min-h-[4rem]">{displayed}</p>;
}
