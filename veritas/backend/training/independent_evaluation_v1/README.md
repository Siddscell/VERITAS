# Independent Evaluation v1 — Feasibility Archive

**Status:** Phase 1 complete (feasibility / limitations only)  
**Audit date:** 2026-10-21  
**Protocol document:** [`independent_evaluation_protocol.md`](independent_evaluation_protocol.md)

## Purpose

This directory permanently archives the Phase 1 assessment of whether FakeAVCeleb v1.2 can supply a **new**, leakage-free evaluation cohort for comparing production `corrected_fixed` fusion against experimental PPO v3.

It does **not** contain:

- New model training or checkpoints
- New performance metrics or promotion decisions
- Phases 2–5 evaluation outputs (manifest freeze, paired runs, metrics JSON, failure analysis)

## Contents

| File | Role |
|---|---|
| `independent_evaluation_protocol.md` | Versioned feasibility protocol and limitations record |
| `README.md` | This index |

## Preservation

Existing leakage-controlled manifests, frozen `final_test` predictions/metrics, decision-contract artifacts, and all `.pth` checkpoints remain untouched. Production default remains `corrected_fixed`; PPO v3 remains disabled in production.
