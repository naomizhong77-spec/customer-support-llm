---
name: lora-training
description: Use this skill whenever working on LoRA or QLoRA fine-tuning of LLMs, especially Qwen2.5-7B on A40 (48GB). Includes memory budget calculations, framework comparisons (Unsloth vs Transformers+PEFT), hyperparameter recommendations, instruction data format templates, inference with adapters, and A40-specific gotchas.
---

# LoRA / QLoRA Training Guide (Qwen2.5-7B on A40)

## Framework Decision

| Framework | Speed | Memory | Ease | Verdict for this project |
|-----------|-------|--------|------|--------------------------|
| **Unsloth** | 2x faster | 30% less | Easy | ✅ **Primary choice** |
| Transformers + PEFT | baseline | baseline | Medium | ✅ Fallback if Unsloth breaks |
| LLaMA-Factory | varied | varied | Easy CLI | ❌ Too opinionated, hard to customize |
| Axolotl | similar to Unsloth | similar | Medium | ❌ Overkill for single task |

**Decision**: Use Unsloth. If it has compatibility issues with our specific torch/CUDA version, fall back to Transformers+PEFT with identical LoRA config.

## Installation

```bash
# Primary (Unsloth)
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
pip install transformers==4.46.0 datasets==3.0.0 accelerate==1.0.0 peft==0.13.0 trl==0.11.0 bitsandbytes==0.44.0

# If Unsloth fails, fallback stays functional (already installed above):
# just use from peft import LoraConfig, get_peft_model
```

## Memory Budget for A40 (48GB)

Target: stay under 40GB peak to leave headroom for CUDA context and measurement spikes.

**Qwen2.5-7B with QLoRA (4-bit)**:

| Component | Size |
|-----------|------|
| Base model (4-bit quantized) | ~5 GB |
| LoRA adapters (r=16, all linear targets) | ~100 MB |
| Activations (bs=4, seq=512, w/ grad_checkpoint) | ~12 GB |
| Gradients | ~100 MB |
| Optimizer state (8-bit AdamW) | ~200 MB |
| KV cache during training | ~3 GB |
| CUDA context + misc | ~2 GB |
| **Total target** | **~22 GB peak** |

**Qwen2.5-7B with LoRA (bf16, no quantization)** — NOT RECOMMENDED for A40:

| Component | Size |
|-----------|------|
| Base model (bf16) | ~14 GB |
| ... activations with bs=4 | ~20 GB |
| ... optimizer state | ~500 MB |
| **Total** | **~40+ GB** — too close to limit |

**Conclusion**: Use QLoRA (4-bit) for Qwen-7B on A40. It's not just about memory — the 4-bit quantization also speeds up training due to less memory bandwidth.

## Recommended Hyperparameters

For Qwen2.5-7B QLoRA on intent classification (27 classes):

```yaml
# configs/qwen_lora_config.yaml

model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  load_in_4bit: true
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_use_double_quant: true

lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  bias: "none"
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  task_type: "CAUSAL_LM"

training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 8
  gradient_accumulation_steps: 4      # effective batch = 16
  learning_rate: 2.0e-4               # higher than full FT; LoRA standard
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.03
  weight_decay: 0.01
  max_grad_norm: 1.0
  max_seq_length: 512
  
  # Precision & memory
  bf16: true
  fp16: false                          # NEVER use fp16 on this setup
  gradient_checkpointing: true
  optim: "paged_adamw_8bit"
  
  # Logging & saving
  logging_steps: 20
  eval_strategy: "steps"
  eval_steps: 200
  save_strategy: "steps"
  save_steps: 200
  save_total_limit: 2
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"
  greater_is_better: false
  
  # Reproducibility
  seed: 42
  data_seed: 42
  
  # Output
  output_dir: "checkpoints/qwen_lora"
  report_to: "none"                    # change to "wandb" if you want tracking
```

## Instruction Data Format

The Bitext dataset is classification; convert to instruction format for generative fine-tuning.

**Per-sample JSONL structure**:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a customer support intent classifier. Given a customer message, output the category and intent."
    },
    {
      "role": "user",
      "content": "Classify this customer message:\n\nI want to cancel my recent order because it arrived damaged."
    },
    {
      "role": "assistant",
      "content": "Category: ORDER\nIntent: CANCEL_ORDER"
    }
  ]
}
```

Then in training, apply Qwen2.5 chat template:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Apply chat template
text = tokenizer.apply_chat_template(
    sample["messages"],
    tokenize=False,
    add_generation_prompt=False  # include assistant response for training
)
```

## Critical Setup Code Snippets

### Loading Qwen with Unsloth + 4-bit

```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    max_seq_length=512,
    dtype=None,              # auto-detect bf16
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth's optimized version
    random_state=42,
)

# CRITICAL: Qwen tokenizer might not have pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

### Fallback: Transformers + PEFT

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

### Training Loop with TRL's SFTTrainer

```python
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

train_ds = load_dataset("json", data_files="data/instruction/train.jsonl")["train"]
val_ds = load_dataset("json", data_files="data/instruction/val.jsonl")["train"]

# Format function — applies chat template
def formatting_func(example):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

training_args = SFTConfig(
    output_dir="checkpoints/qwen_lora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    max_seq_length=512,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=200,
    save_steps=200,
    save_total_limit=2,
    load_best_model_at_end=True,
    seed=42,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    formatting_func=formatting_func,
    args=training_args,
)

trainer.train()
trainer.save_model()  # saves LoRA adapter only, ~100MB
```

### Inference with Trained Adapter

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="checkpoints/qwen_lora/checkpoint-XXXX",
    max_seq_length=512,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)  # 2x faster inference

messages = [
    {"role": "system", "content": "You are a customer support intent classifier..."},
    {"role": "user", "content": "Classify this customer message: ..."},
]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")

outputs = model.generate(inputs, max_new_tokens=32, do_sample=False, temperature=0.0)
response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
print(response)
# Expected: "Category: ORDER\nIntent: CANCEL_ORDER"
```

## Gotchas and Fixes

### `tokenizer.pad_token is None`
```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
```

### Labels not masked on prompt (model learns to repeat prompt)
Use `DataCollatorForCompletionOnlyLM`:
```python
from trl import DataCollatorForCompletionOnlyLM

# For Qwen2.5 chat template, the assistant response starts after <|im_start|>assistant\n
response_template = "<|im_start|>assistant\n"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
)
trainer = SFTTrainer(..., data_collator=collator)
```

### bitsandbytes version conflict
```bash
pip install bitsandbytes==0.44.0 --force-reinstall --no-deps
```

### Unsloth import errors
Check torch version matches exactly. Reinstall:
```bash
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git" --force-reinstall --no-deps
```

### `TypeError: dispatch_model() got unexpected keyword argument`
accelerate version mismatch. Try `pip install accelerate==1.0.0`.

### LoRA merged weights are huge (15GB)
Don't merge unless deploying. Keep separate adapter (100MB) for iteration.

### Very slow training
Check:
- Is `gradient_checkpointing=True`? (required for 7B on 48GB)
- Is `bf16=True`?
- Is effective batch size reasonable? (16-32)
- Is `optim="paged_adamw_8bit"`?
- Check GPU utilization with `nvidia-smi` — should be >80% during train step

## Smoke Test Protocol

Before submitting 2-3h training, always smoke test:

```bash
srun -p MGPU-TC2 --gres=gpu:1 --time=00:15:00 --pty bash
conda activate customer-support-llm
cd ~/customer-support-llm
python scripts/train_qwen_lora.py --config configs/qwen_lora_config.yaml --smoke-test
# Should:
# 1. Load model successfully (~2 min)
# 2. Print trainable params: ~40-60M
# 3. Run 50 steps with decreasing loss
# 4. Save checkpoint to checkpoints/qwen_lora/smoke/
# 5. Exit cleanly
exit  # leave interactive session
```

If smoke passes, submit full job with `sbatch`.
