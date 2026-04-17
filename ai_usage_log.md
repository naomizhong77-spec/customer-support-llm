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

### Entry 1.2 — Configuration scaffolding + environment debugging

- **Date**: 2026-04-17
- **Tool**: Claude (web chat, Opus 4.7) for generation, VS Code Remote-SSH for execution
- **Task**: Generate CLAUDE.md, 4 agent files, 4 skill files, README, requirements.txt; then set up conda env with torch 2.4.1 + Unsloth + HF stack on CCDS-TC2 cluster
- **Prompt quality**: High — provided hard constraints upfront (A40 48GB, 6h SLURM limit, single-user, "code must validate before commit" rule)
- **AI output quality**: 4/5 for scaffolding (structure was excellent), 3/5 for the initial install command (used `git+https://unsloth main` which broke on real torch 2.4.1)
- **Iterations**: ~8 rounds through environment debugging
- **Manual interventions**:
  - GPU confirmed as A40 (not H40 as initially assumed)
  - `srun --pty bash` interactive sessions die when VS Code terminal tab loses focus — had to re-request GPU session 3 times
  - Unsloth `git+https` main branch pulled `unsloth_zoo==2026.x` with `torch._inductor.config` and `torch.int1` dependencies requiring torch≥2.6 — cascading `AttributeError` chain
  - Solution: pin `unsloth==2024.12.12` + `unsloth_zoo==2024.12.7` + `--no-deps` for transformers/peft/trl stack
  - `--no-deps` left `tokenizers==0.22.2` and `huggingface-hub==1.11.0` unresolved (transformers 4.46 requires `<0.21` and `<1.0` respectively) — fixed with two follow-up `--no-deps` installs
  - `unsloth==2024.12.12` does not support Qwen2.5-0.5B/1.5B in its model whitelist — had to skip straight to Qwen2.5-7B for smoke test (still works since 4-bit load is only ~5GB)
  - `source ~/.bashrc` in SLURM scripts unreliable on CCDS-TC2 — had to use explicit `$CONDA_BASE/etc/profile.d/conda.sh` sourcing
- **Lesson**:
  - **Never use `git+https` for Python packages in production** — the Unsloth team publishes PyPI releases with compatible pinned sibling deps; the main branch is a moving target
  - `--no-deps` is a useful escape hatch when pip's resolver does the wrong thing, but you **have to manually clean up version tails** that the original package's metadata specified (tokenizers, hf-hub in this case)
  - **Every HPC cluster has environment quirks** that documentation won't tell you — plan for 30-60 min of environment setup pain per new node/user
  - **Model whitelists matter**: Unsloth's 2024.12 version doesn't know about newer small Qwen variants; always verify supported models before choosing smoke-test size
- **PM insight**: This entire ordeal is exactly the kind of integration-layer fragility that makes LLM tooling products hard to ship. The "2x faster training!" value prop from Unsloth is real, but if the installation is a 30-minute debug session for anyone outside their reference environment, the DX tax eats the productivity gain for new users. A product insight for AI infra tooling: **version compatibility matrices + copy-paste install commands that actually work** are a more valuable feature than raw performance.
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
