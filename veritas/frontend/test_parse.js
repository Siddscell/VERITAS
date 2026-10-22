const fs = require('fs');

const dataStr = '{"type": "agent_result", "agent": "Spatial Analysis", "color": "cyan", "confidence": 0.486, "is_fake": false, "finding": "...", "gradcam_b64": null}';
const data = JSON.parse(dataStr);

const confidence = data.confidence;
console.log(`confidence: ${confidence}`);
console.log(`formatted: ${(confidence * 100).toFixed(1)}%`);

const verdictStr = '{"type": "final_verdict", "verdict": "AUTHENTIC", "overall_confidence": 0.2856, "contributing_agents": ["Spatial Analysis", "Temporal Analysis"]}';
const verdictData = JSON.parse(verdictStr);
const vConf = verdictData.overall_confidence ?? verdictData.confidence;
console.log(`vConf: ${vConf}`);
console.log(`vFormatted: ${(vConf * 100).toFixed(1)}%`);
