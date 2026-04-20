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

- **Date**: 2026-04-17
- **Agent**: `data-engineer`
- **Task**: Build `notebooks/01_data_exploration.ipynb` covering assignment specs #1 (source + error detection) and #3 (statistical summary) on the 26,872-row Bitext dump. Read-only — no data mutation.
- **Prompt**: Single delegation with explicit scope fences (read-only, read CLAUDE.md + agent spec + code-quality skill first, structured section list, figures to `outputs/figures/` at 150 dpi).
- **Output quality**: 5/5 — 23 cells, zero errors, full-data run (not smoke), 6 figures, and proactively surfaced a discrepancy in CLAUDE.md (11 categories, not 10).
- **Iterations**: 1 (no rework).
- **Manual interventions**: Decided commit granularity (CLAUDE.md fix and notebook as two separate commits). No content edits to the notebook.
- **Lesson**: Read-only scope fences work — the agent did not touch data even when it would have been trivially easy to "fix" the injected-defect story in-place.

### Entry 2.2 — Data cleaning with injected errors

- **Date**: 2026-04-17
- **Agent**: `data-engineer`
- **Task**: Build `notebooks/02_data_cleaning.ipynb` (inject synthetic defects → deterministic cleaning → stratified splits), plus `scripts/prepare_instruction_data.py`, processed parquet + Qwen-format JSONL, and `data_stats.json` audit log.
- **Prompt**: EDA findings embedded directly as design constraints (stratify on intent not category, preserve all 5 cols for Phase 2, max_len=128, seed=42). Two delegations total: main build + a follow-up to add a cleaning-audit table.
- **Output quality**: 4/5 — correct first pass, including the non-obvious ordering fix (NFKC *after* mojibake repair because NFKC decomposes `â€™` into `â€TM`). Dropped 0.5 pt because initial split sizes were smaller than the estimate supplied in the prompt; the agent flagged the reason proactively (natural paraphrase duplicates in pristine data collapsed under strict `(instruction, intent)` dedup) rather than silently diverging.
- **Iterations**: 2 (initial notebook; post-hoc audit-table cell).
- **Manual interventions**:
  - Chose strict `(instruction, intent)` dedup over instruction-only after the agent surfaced the trade-off.
  - Requested a cleaning-audit markdown table for report Section 2.
  - Overrode `.gitignore` policy: track `test.parquet` + `test.jsonl` only; train/val regenerated deterministically.
  - Pre-commit hook blocked `train.parquet` (5.04 MB > 5 MB hook threshold); user picked the drop-from-tracking option rather than `--no-verify` or git-lfs.
  - Requested README update documenting the regeneration workflow.
- **Lesson**: Front-loading EDA findings into the next-stage prompt as explicit design constraints (not "look at the notebook") produced correct first-try design decisions. Pre-commit hook thresholds should be checked against artefact sizes *before* staging, not discovered at commit time.

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

### Entry 5.3 — Phase A-3 debugging retro: BANKING77 Qwen QLoRA silent parser failure

- **Date**: 2026-04-20
- **Agent**: `ml-trainer`
- **Task**: Recover JOBID 21690 (Qwen2.5-7B + QLoRA on BANKING77, 46 min train, final train loss 0.1132). Training succeeded but post-training eval reported 100% parse errors (3079/3079) — the `validate_results()` guard aborted at `assert acc > 0.05`. No retrain permitted; diagnose and re-evaluate the existing adapter only.
- **Prompt**: High-specificity recovery brief with full root-cause context: "response_template serves two roles — loss-mask anchor AND parser contract — and they diverged". Prompt included the exact line numbers of the parser regex and the generation-prompt builder, evidence from the old `errors.csv` (pred_raw="card_arrival", gold="Intent: card_arrival"), a spec'd 3-level parser cascade (regex → bare-label exact match → fuzzy difflib), a sanity-check-before-full-eval requirement, and explicit "do not retrain, do not commit, do not overwrite buggy artifacts" scope fences.
- **Output quality**: 5/5 for the diagnosis itself (the prompt arrived pre-diagnosed — the hard work was already done). 4/5 for the fix: one-shot parser cascade passed static + smoke, and when applied offline to the existing 3079-row `errors.csv` predicted 91.65% correctness — matching the eventual re-eval's 91.65% test accuracy exactly. The fresh re-eval (val+test, 4278 generations over 22.5 min on A40) produced test acc=0.9165 / macro-F1=0.9050 / parse_errors=7 (vs. the buggy run's 3079). Path breakdown on test: regex=0 / bare=2972 / fuzzy=100 / none=7 — i.e. the adapter never emits the "Intent:" anchor, as predicted.
- **Iterations**: 1 fix round; 1 retry on job submission (srun background task killed by harness during Unsloth patch; switched to sbatch and it ran clean).
- **Manual interventions**:
  - Changed parser return type from 2-tuple `(intent, parse_error)` to 3-tuple `(intent, parse_error, parse_path)` to get per-row telemetry on which cascade branch hit — lets future debug sessions distinguish "model emits anchor" from "model emits bare label" from "fuzzy-rescued typo" without re-inspecting `pred_raw`.
  - Relaxed the `acc > 0.05` guard to `acc > 0.5` — the 0.05 floor existed to catch exactly this bug but only fires post-generation, i.e. it costs a full test pass to discover. The real fix would be a pre-train contract test (see Lesson).
  - Preserved the buggy artifacts under `_buggy` suffix rather than overwriting — the 3079-row errors.csv is now report material for the debugging narrative.
- **Root cause (for the report)**: `DataCollatorForCompletionOnlyLM(response_template="Intent:")` masks -100 on every token up to and including the anchor, so training loss only shapes the tokens AFTER "Intent:". The model therefore learns to emit only the bare label (e.g. `"card_arrival"`), not the full `"Intent: card_arrival"` that the gold string contains. At inference the chat template ends the prompt at `<|im_start|>assistant\n` — no "Intent:" anchor to echo — and the original parser regex `r"[Ii]ntent\s*[:=]\s*([A-Za-z0-9_]+)"` required that anchor. 100% parse-fail. The adapter was fine the entire time.
- **How it was diagnosed from `pred_raw` inspection**: the buggy `errors.csv` contained rows like `text="How do I locate my card?", gold="Intent: card_arrival", pred_raw="card_arrival", parse_error=True`. Five seconds of visual inspection revealed pred_raw *was already the correct label* in plain text — the bug was downstream of the model, in the parser. This is the fastest possible diagnosis path and it only works because we logged `pred_raw` alongside the parsed `pred_intent`. Without the raw column the user would have concluded "the model didn't train" and retrained for another 46 minutes.
- **Lesson (headline for the report)**: **In generative classification, the loss-mask anchor and the parser contract are the same design decision and must be designed together**. Masking up to-and-including "Intent:" means the loss never shapes the emission of "Intent:" itself — so the parser cannot assume the model will echo the anchor. Either (a) mask up-to-but-not-including the anchor so loss covers it, (b) put the anchor in the prompt so the model never has to generate it, or (c) write the parser for bare labels from day one. We did (c) post-hoc but should have designed for it up-front.
- **Secondary lesson**: **Log `pred_raw` alongside parsed predictions in every generative-classification pipeline.** The debugging asymmetry is huge — with raw logs we diagnosed in seconds; without them the cheapest diagnosis path is another training run.
- **Tertiary lesson**: **Sanity guards should fail pre-train, not post-eval.** The `acc > 0.05` guard served no purpose — it told us the pipeline was broken only *after* we had already paid for training + eval. A pre-train contract test ("run the parser over one rendered gold string and one rendered prompt-only input; verify both parse successfully") would have caught this in the smoke test.
- **PM insight**: This failure mode — "the model is right; the glue code is wrong" — is the single most common silent-failure class I've hit in LLM product work. The metric looked catastrophic (0% accuracy) but the underlying model was 91.7% accurate. In a production monitoring context this is a false-P1 incident waiting to happen: every metric dashboard says "model is broken", but the raw model outputs say "model is fine, your extraction layer is misconfigured". Product implication: **ship generative classifiers with raw-output sampling + parser-path telemetry by default**, not just aggregated accuracy.
- **Artifacts**:
  - Fix: `scripts/train_qwen_lora_banking77.py` (parse_qwen_output cascade, guard relaxed), `scripts/eval_qwen_lora_banking77.py` (new; re-eval helper), `slurm/eval_qwen_lora_banking77.sh` (new; sbatch wrapper).
  - Recovered metrics: `outputs/metrics/qwen_lora_banking77_20260420_3eb1a21_reval/results.json` (test acc=0.9165, macro-F1=0.9050, parse_errors=7/3079; val acc=0.9124).
  - Preserved buggy evidence: `outputs/error_analysis/qwen_lora_banking77_errors_buggy.csv`, `outputs/figures/10_qwen_lora_banking77_confusion_matrix_buggy.png`.
  - Re-eval artifacts: `outputs/error_analysis/qwen_lora_banking77_errors_reval.csv`, `outputs/figures/10_qwen_lora_banking77_confusion_matrix_reval.png`.

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
