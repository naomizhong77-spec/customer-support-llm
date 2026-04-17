---
name: code-quality
description: Use this skill whenever writing, modifying, or about to commit any code (Python script, shell script, notebook, SLURM script, config file). Enforces the three-level validation protocol (static checks, smoke tests, full validation) that is required before any git commit. Blocks commits of broken code.
---

# Code Quality & Validation Protocol

This skill defines the validation checks that MUST pass before any code is committed or considered "done". It is invoked automatically by the data-engineer, ml-trainer, and evaluator agents.

## The Three-Level Validation Protocol

### Level 1: Static Checks (always required, ~10 seconds)

For Python files:
```bash
# Syntax check
python -m py_compile <file>.py
# Should produce no output if OK

# Linting (catches undefined names, unused imports, basic issues)
ruff check <file>.py
# Or: pyflakes <file>.py

# Type check (optional but recommended for scripts)
# mypy <file>.py --ignore-missing-imports
```

For shell scripts:
```bash
# Syntax check
bash -n <file>.sh

# Linting
shellcheck <file>.sh
```

For SLURM scripts specifically:
```bash
# Validate without actually submitting
sbatch --test-only <file>.sh
```

For YAML configs:
```bash
python -c "import yaml; yaml.safe_load(open('<file>.yaml'))"
```

For JSON files:
```bash
python -c "import json; json.load(open('<file>.json'))"
```

### Level 2: Smoke Tests (required for runnable code, ~1-5 minutes)

Smoke test = run the thing on tiny input and verify it completes.

**For data processing scripts**:
```bash
python scripts/prepare_instruction_data.py \
  --input data/processed/train.parquet \
  --output /tmp/smoke_out.jsonl \
  --limit 10
```
Verify: output file exists, has expected format, row count matches.

**For training scripts**:
```bash
python scripts/train_xxx.py --config configs/xxx.yaml --smoke-test
```
Smoke-test mode should:
- Use ≤100 samples
- Run 1 epoch or 50 steps, whichever is smaller
- Save a checkpoint
- Print at least one loss value
- Exit 0

**For evaluation scripts**:
```bash
python scripts/eval.py \
  --checkpoint checkpoints/smoke_test_model \
  --test-set data/processed/test.parquet \
  --limit 50 \
  --output /tmp/smoke_results.json
```
Verify: valid JSON output, expected schema.

**For notebook cells**:
Run "Restart Kernel & Run All" end to end. If it errors, fix before committing.

### Level 3: Full Validation (required before merging to main)

- Training: full training completes, metrics within expected range
- Evaluation: runs on full test set, produces complete results JSON
- Data pipeline: processes full dataset, output row counts match expectations

## Pre-Commit Checklist

Before running `git commit`, verify:

- [ ] Level 1 passes on all modified files
- [ ] Level 2 passes on all modified runnable files (scripts, notebooks)
- [ ] No secrets in diff (`git diff --cached | grep -iE "api.?key|token|password|secret"`)
- [ ] No large files (`git diff --cached --stat` — flag anything >1MB)
- [ ] `.gitignore` appropriately lists: `checkpoints/`, `logs/`, `data/raw/`, `*.ipynb_checkpoints`, `__pycache__/`, `.env`
- [ ] Commit message follows Conventional Commits format: `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `exp:`
- [ ] If AI was used: added entry to `ai_usage_log.md`

## Automated Pre-Commit Hook

Install this once in the repo:

```bash
# Create .git/hooks/pre-commit
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
set -e

# Get staged Python files
PY_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$' || true)

if [ -n "$PY_FILES" ]; then
    echo "🔍 Running Level 1 checks on Python files..."
    for f in $PY_FILES; do
        python -m py_compile "$f" || { echo "❌ Syntax error in $f"; exit 1; }
    done
    # Run ruff if available
    if command -v ruff &> /dev/null; then
        ruff check $PY_FILES || { echo "❌ ruff check failed"; exit 1; }
    fi
    echo "✅ Level 1 passed"
fi

# Check for large files
LARGE_FILES=$(git diff --cached --name-only --diff-filter=A | xargs -I{} find {} -size +1M 2>/dev/null || true)
if [ -n "$LARGE_FILES" ]; then
    echo "⚠️  Large files detected:"
    echo "$LARGE_FILES"
    echo "Consider adding to .gitignore or using git-lfs"
    exit 1
fi

# Check for secrets
if git diff --cached | grep -iE "api[_-]?key|token|password|secret" | grep -v "^#" > /dev/null; then
    echo "⚠️  Possible secret in diff. Review carefully:"
    git diff --cached | grep -iE "api[_-]?key|token|password|secret" | head -5
    read -p "Continue? [y/N] " -n 1 -r
    echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi

echo "✅ All pre-commit checks passed"
EOF
chmod +x .git/hooks/pre-commit
```

## How Agents Use This Skill

When any agent completes code, they must:

1. Run Level 1 checks themselves
2. Show the output to the user
3. Run Level 2 smoke test themselves
4. Show the output to the user
5. Only then suggest `git commit`

If user asks "commit this", the agent should respond:
> "Before committing, let me run validation.  
> [Level 1 output]  
> [Level 2 output]  
> ✅ Both checks passed. Here's the suggested commit command: `git add ... && git commit -m '...'`"

If validation fails, the agent must fix the issues before proceeding.

## When Validation Can Be Skipped

**Almost never.** Exceptions:
- Pure markdown docs (`.md`): skip Level 2, Level 1 can be `markdownlint` if available
- Config-only changes: Level 1 is just YAML/JSON syntax check
- Deleting files: no validation needed
- Work-in-progress branches (`wip/*`): user may choose to push broken code, but MUST flag in commit message as `wip:` and not merge to main

**Never skip validation for:**
- Training scripts (could waste GPU hours)
- Data pipelines (could corrupt the dataset)
- Evaluation scripts (could produce misleading results)
- Anything touching `main` branch

## Common Failure Modes and Fixes

**"py_compile" fails**: Syntax error. Look at exact line number, usually indentation or missing colon.

**"ruff check" fails with F401**: Unused import. Either use it or remove it.

**"ruff check" fails with F821**: Undefined name. Usually missing import or typo.

**Smoke test OOM**: Smoke config too large. Reduce batch size to 1, samples to 10.

**Smoke test loss is NaN**: Bug in loss computation or data. Don't proceed to full run.

**SLURM `--test-only` fails**: Syntax error in sbatch directives or missing required fields.
