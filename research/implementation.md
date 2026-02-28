# Implementation Details: GRPO Trainer

## Project Structure

```
remote_trainer/
├── pyproject.toml              # uv package config
├── configs/
│   ├── default.yaml            # Default (unsupervised)
│   ├── unsupervised.yaml       # Self-distillation config
│   └── supervised.yaml         # Supervised GRPO config
├── data/
│   ├── train_unsupervised.jsonl
│   └── train_supervised.jsonl
├── scripts/
│   └── train.py                # CLI entry point
└── src/
    ├── config.py               # RewardConfig with supervised_mode
    ├── trainer.py              # GRPO trainer wrapper
    └── rewards/
        ├── __init__.py         # Factory function
        ├── inverse_loss.py     # Unsupervised reward
        └── supervised.py       # Coverage-weighted loss
```

---

## 1. Unsupervised Reward: Inverse Loss

Located in `src/rewards/inverse_loss.py`.

```python
def create_inverse_loss_reward(model, tokenizer):
    """
    Reward = -perplexity(completion)
    
    Lower perplexity = higher reward.
    Only calculates loss on completion tokens.
    """
    def inverse_loss_reward(prompts, completions, **kwargs):
        rewards = []
        for prompt, completion in zip(prompts, completions):
            text = prompt + completion
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            prompt_length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            
            with torch.no_grad():
                outputs = model(**inputs)
                shift_logits = outputs.logits[..., :-1, :]
                shift_labels = inputs["input_ids"][..., 1:]
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                token_losses = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                completion_losses = token_losses[prompt_length - 1:]
                if len(completion_losses) == 0:
                    rewards.append(0.0)
                else:
                    rewards.append(-completion_losses.mean().item())
        
        return rewards
    return inverse_loss_reward
```

---

## 2. Supervised Reward: Coverage-Weighted Loss

Located in `src/rewards/supervised.py`.

### Why Not Simple Perplexity on Answer?

If we calculate `ppl(prompt + answer)` for all K completions, they ALL get the SAME reward because the answer is fixed. GRPO requires rewards to vary per completion for advantage calculation.

### Coverage-Weighted Cross-Entropy Loss

```python
def _compute_completion_cross_entropy(model, tokenizer, answer, completion):
    """
    reward = -CE(completion_logits, answer_tokens) * coverage_ratio
    
    Measures how well the completion matches the expected answer.
    """
    answer_tokens = tokenizer(answer, add_special_tokens=False)["input_ids"]
    completion_tokens = tokenizer(completion, add_special_tokens=False)["input_ids"]
    
    # Get logits for completion
    inputs = tokenizer(completion, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[..., :-1, :]  # Shift for next-token prediction
    
    # Calculate CE against answer tokens (up to min length)
    min_len = min(logits.shape[1], len(answer_tokens))
    if min_len == 0:
        return 0.0
    
    target = torch.tensor(answer_tokens[:min_len]).to(model.device)
    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
    ce_loss = loss_fct(logits[0, :min_len], target).item()
    
    coverage = min(1.0, len(completion_tokens) / len(answer_tokens))
    return -ce_loss * coverage
```

---

## 3. Configuration

### RewardConfig (`src/config.py`)

```python
class RewardConfig(BaseModel):
    type: Literal["inverse_loss", "supervised"] = "inverse_loss"
    answer_field: str = "answer"  # Field name for expected answer
```

### Config Files

**unsupervised.yaml:**
```yaml
reward:
  type: inverse_loss
```

**supervised.yaml:**
```yaml
reward:
  type: supervised
  answer_field: answer
```

---

## 4. Training with Unsloth + vLLM

```python
import os
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"  # 30% VRAM savings

from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

# Load model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="model_name",
    max_seq_length=2048,
    load_in_4bit=True,
    fast_inference=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
)

# Create reward function based on config
from src.rewards import create_reward_function
reward_func = create_reward_function(
    model, tokenizer, 
    reward_type="supervised",
)

# Configure GRPO
training_args = GRPOConfig(
    output_dir="output",
    learning_rate=5e-6,
    num_generations=8,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_prompt_length=512,
    max_completion_length=512,
    beta=0.04,  # KL penalty
)

# Train
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_func],
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

---

## 5. Key Implementation Notes

### VRAM Safety
- Reward calculation wrapped in `torch.no_grad()`
- Process completions sequentially or in micro-batches
- vLLM standby mode for inference memory savings

### Mode Selection
- **Unsupervised**: For domain adaptation without labeled data
- **Supervised**: When you have prompt-answer pairs

### Coverage Weighting
The coverage ratio `min(1.0, len(completion)/len(answer))` ensures:
- Short completions get penalized
- Model is incentivized to generate complete answers
- Prevents "gaming" by generating minimal text

---

## 6. Output Formats

The trainer supports multiple output formats:
- LoRA adapters (default)
- Merged 16-bit model
- Merged 4-bit model
- GGUF for llama.cpp
