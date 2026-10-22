// Simulate the exact Console.tsx SSE parser with real SSE output from Vite :5173
// to confirm whether the parser correctly extracts confidence values.

const sseText = 'event: agent_update\ndata: {"agent":"Spatial Analysis","color":"cyan","confidence":0.486,"is_fake":false,"finding":"No significant visual inconsistencies detected.","gradcam_b64":null}\n\nevent: agent_update\ndata: {"agent":"Temporal Analysis","confidence":0.0183,"is_fake":false,"finding":"No significant temporal inconsistencies across 16 sampled frames.","gradcam_b64":null}\n\nevent: agent_update\ndata: {"agent":"Audio Analysis","status":"untrained","has_audio":true,"codec":"mp3","sample_rate":44100,"channels":2,"duration":5,"is_fake":null,"confidence":0,"finding":"Audio stream detected."}\n\nevent: agent_update\ndata: {"agent":"Context Verification","status":"mocked","color":"purple","confidence":0,"is_fake":null,"finding":"Context agent is not yet integrated."}\n\nevent: final_verdict\ndata: {"verdict":"AUTHENTIC","overall_confidence":0.2856,"contributing_agents":["Spatial Analysis","Temporal Analysis"],"unavailable_agents":["Audio Analysis","Context Verification"]}\n\nevent: done\ndata: {}\n\n';

const INITIAL_AGENTS = [
  { agent: 'Spatial Analyst',   confidence: 0, finding: '', status: 'pending' },
  { agent: 'Temporal Analyst',  confidence: 0, finding: '', status: 'pending' },
  { agent: 'Audio Analyst',     confidence: 0, finding: '', status: 'pending' },
  { agent: 'Context Analyst',   confidence: 0, finding: '', status: 'pending' },
];

let agents = INITIAL_AGENTS.map(a => ({...a}));
let verdict = null;
let pendingEvent = '';

const lines = sseText.split('\n');

for (const line of lines) {
  const trimmed = line.trim();
  if (trimmed === '') {
    // Blank line = end of one SSE message; reset event accumulator
    pendingEvent = '';
    continue;
  }
  if (trimmed.startsWith('event:')) {
    pendingEvent = trimmed.slice('event:'.length).trim();
    continue;
  }
  if (trimmed.startsWith('data:')) {
    const dataStr = trimmed.slice('data:'.length).trim();
    if (!dataStr) continue;
    let data;
    try { data = JSON.parse(dataStr); } catch(e) { continue; }

    const eventName = pendingEvent;

    if (eventName === 'agent_update') {
      // Guard: if (!data.finding) continue;
      if (!data.finding) {
        console.log(`  [SKIP] agent_update skipped - no finding. agent=${data.agent}`);
        continue;
      }
      const agentName = (data.agent || data.agent_name) || '';
      let idx = 0;
      if (agentName.toLowerCase().includes('spatial')) idx = 0;
      else if (agentName.toLowerCase().includes('temporal')) idx = 1;
      else if (agentName.toLowerCase().includes('audio')) idx = 2;
      else if (agentName.toLowerCase().includes('context')) idx = 3;

      agents[idx] = {
        agent: INITIAL_AGENTS[idx].agent,
        confidence: data.confidence,
        finding: data.finding,
        status: 'streaming',
        backend_status: data.status,
        is_fake: (data.is_fake !== undefined) ? data.is_fake : undefined,
      };
      console.log(`agent_update[${idx}] ${agentName}: raw confidence=${data.confidence}, stored=${agents[idx].confidence}, renders="${(agents[idx].confidence * 100).toFixed(1)}%"`);
    } else if (eventName === 'final_verdict') {
      const rawVerdict = (data.verdict || '').toUpperCase();
      let normalised = rawVerdict;
      if (rawVerdict === 'FAKE' || rawVerdict === 'DEEPFAKE') normalised = 'DEEPFAKE';
      else if (rawVerdict === 'REAL' || rawVerdict === 'AUTHENTIC') normalised = 'AUTHENTIC';
      verdict = {
        verdict: normalised,
        confidence: (data.overall_confidence !== undefined) ? data.overall_confidence : data.confidence,
        contributing: data.contributing_agents || [],
        unavailable: data.unavailable_agents || [],
      };
      console.log(`final_verdict: overall_confidence=${data.overall_confidence}, verdict.confidence=${verdict.confidence}, renders="${(verdict.confidence * 100).toFixed(1)}%"`);
    }
  }
}

console.log('\n=== SIMULATED FINAL REACT STATE ===');
agents.forEach((a, i) => {
  const isUnavailable = a.is_fake === null || a.backend_status === 'untrained' || a.backend_status === 'mocked' || a.backend_status === 'no_audio';
  const display = a.status === 'pending' ? '--%' : isUnavailable ? 'N/A' : `${(a.confidence * 100).toFixed(1)}%`;
  console.log(`  agents[${i}] "${a.agent}": confidence=${a.confidence}, isUnavailable=${isUnavailable}, status=${a.status}, displays="${display}"`);
});
if (verdict) {
  console.log(`  verdict: confidence=${verdict.confidence}, displays="${(verdict.confidence * 100).toFixed(1)}%", verdict="${verdict.verdict}"`);
} else {
  console.log('  verdict: null (not set)');
}
