# AI-Assisted Development Log

This log tracks how AI tools (primarily Claude Code) were used throughout this project.
It forms the basis for Section 7 of the final report.

---

## Format

Each entry should capture:

- **Date & Phase**
- **Agent/Tool used**: Claude Code with which agent
- **Task**: brief description of what was delegated
- **Prompt quality**: note the actual prompt pattern used
- **AI output quality**: 1-5, with brief comments
- **Iterations**: how many back-and-forth rounds to get usable output
- **Manual interventions**: what the human had to add/fix
- **Lesson**: what to do differently next time

---

## Phase 1: Project Setup (Apr 17, 2026)

### Entry 1.1 — Project planning

- **Date**: 2026-04-17
- **Tool**: Claude (web chat, Opus 4.7)
- **Task**: Overall project direction for AI PM interview prep
- **Prompt quality**: Started with open-ended question, then iteratively narrowed via structured clarifying questions
- **AI output quality**: 5/5 — structured decision framework, concrete tradeoffs, realistic timeline
- **Iterations**: 4-5 rounds
- **Manual interventions**: Clarified GPU (A40 not H40), cluster partition, time budget
- **Lesson**: Verify hardware details upfront with `scontrol show node` and `nvidia-smi`, don't assume

### Entry 1.2 — Configuration scaffolding

- **Date**: 2026-04-17
- **Tool**: Claude (web chat) for generation, Claude Code for local execution
- **Task**: Generate CLAUDE.md, 4 agent files, 4 skill files, README, requirements.txt
- **Prompt quality**: High — I provided constraints (A40, 6h limit, single user, code-must-validate rule)
- **AI output quality**: _[fill in after use]_
- **Iterations**: _[fill in]_
- **Manual interventions**: _[fill in]_
- **Lesson**: _[fill in]_

---

## Phase 2: Data Preparation (Apr 18-19)

_[Entries to be added as work progresses]_

### Entry 2.1 — EDA notebook

- **Date**: 
- **Agent**: `data-engineer`
- **Task**: 
- **Prompt**: 
- **Output quality**: 
- **Iterations**: 
- **Manual interventions**: 
- **Lesson**: 

### Entry 2.2 — Data cleaning with injected errors

- **Date**: 
- **Agent**: `data-engineer`
- **Task**: 
- **Prompt**: 
- **Output quality**: 
- **Iterations**: 
- **Manual interventions**: 
- **Lesson**: 

---

## Phase 3: Baseline 1 (Apr 20)

### Entry 3.1 — TF-IDF + Logistic Regression

- **Date**: 
- **Agent**: `ml-trainer`
- **Task**: 
- **Prompt**: 
- **Output quality**: 
- **Iterations**: 
- **Manual interventions**: 
- **Lesson**: 

---

## Phase 4: Baseline 2 (Apr 21-22)

### Entry 4.1 — DistilBERT training script

- **Date**: 
- **Agent**: `ml-trainer`
- **Task**: 
- **Prompt**: 
- **Output quality**: 
- **Iterations**: 
- **Manual interventions**: 
- **Lesson**: 

### Entry 4.2 — SLURM submission script

- **Date**: 
- **Agent**: `ml-trainer` (+ `slurm-submit` skill)
- **Task**: 
- **Prompt**: 
- **Output quality**: 
- **Iterations**: 
- **Manual interventions**: 
- **Lesson**: 

---

## Phase 5: Main Model (Apr 22-24)

### Entry 5.1 — Qwen LoRA training script

- **Date**: 
- **Agent**: `ml-trainer` (+ `lora-training` skill)
- **Task**: 
- **Prompt**: 
- **Output quality**: 
- **Iterations**: 
- **Manual interventions**: 
- **Lesson**: 

### Entry 5.2 — Smoke test debugging

- **Date**: 
- **Agent**: `ml-trainer`
- **Task**: 
- **Prompt**: 
- **Output quality**: 
- **Iterations**: 
- **Manual interventions**: 
- **Lesson**: 

---

## Phase 6: Evaluation (Apr 24-25)

### Entry 6.1 — Unified evaluation pipeline

- **Date**: 
- **Agent**: `evaluator` (+ `ml-evaluation` skill)
- **Task**: 
- **Prompt**: 
- **Output quality**: 
- **Iterations**: 
- **Manual interventions**: 
- **Lesson**: 

### Entry 6.2 — Error analysis

- **Date**: 
- **Agent**: `evaluator`
- **Task**: 
- **Prompt**: 
- **Output quality**: 
- **Iterations**: 
- **Manual interventions**: 
- **Lesson**: 

---

## Phase 7: Report Writing (Apr 25-26)

### Entry 7.1 — Report assembly

- **Date**: 
- **Agent**: `report-writer`
- **Task**: 
- **Prompt**: 
- **Output quality**: 
- **Iterations**: 
- **Manual interventions**: 
- **Lesson**: 

---

## Meta-Reflection (fill in at end)

### Quantitative summary

- **Total Claude Code sessions**: 
- **Estimated hours saved vs. solo**: 
- **% of code AI-generated**: 
- **% of AI-generated code accepted without modification**: 
- **Number of times validation caught an AI bug**: 

### Qualitative patterns

- **Where AI excelled**: 
- **Where AI failed or needed heavy steering**: 
- **Prompt patterns that worked well**: 
- **Prompt patterns that wasted time**: 

### Implications for AI PM roles

_How does this experience of working with AI as a development collaborator inform my thinking about building AI products?_

- **On reliability**: 
- **On guardrails**: 
- **On human-AI handoff**: 
- **On evaluation**: 

---

## Prompt Library (reusable patterns I found useful)

_Collect here as you go — these become material for the report's Section 7._

### Pattern: Decision framework
```
I'm deciding between A and B for [context]. My constraints are [X, Y, Z].
Give me a decision framework, not just a recommendation. Include tradeoffs I might not have considered.
```

### Pattern: Code + validation
```
Write [X]. Before we proceed, run Level 1 and Level 2 validation (see code-quality skill) and show me the outputs. Only commit if both pass.
```

### Pattern: Debugging with logs
```
Here's the error output: [paste]. Before guessing, check [K, L, M] and tell me what you find. Don't "try fixes" speculatively.
```

### Pattern: Scope control
```
Do ONLY [X]. Do not touch [Y] or [Z]. If you think you need to modify those, stop and ask me first.
```
