# Architecture Overview: GRPO Trainer with Unsupervised & Supervised Modes

## Abstract

This document outlines a GRPO-based fine-tuning system supporting **two modes**:
1. **Unsupervised Self-Distillation**: Uses inverse loss (negative perplexity) as reward
2. **Supervised GRPO**: Uses coverage-weighted loss comparing completions to expected answers

Both approaches leverage Group-Relative Policy Optimization (GRPO) to eliminate the need for a separate Value Model.

---

## 1. Unsupervised Mode: Internal Confidence as Reward

In this self-distillation setup, the policy model acts as its own critic.

### Reward Metric: Inverse Loss (Negative Log-Likelihood)

- When a model generates a sequence of tokens, it computes the probability of each token.
- A lower cross-entropy loss indicates higher internal confidence (lower perplexity).
- By converting this loss into a reward (multiplying by -1), we instruct the model to favor completions it finds most coherent.

### Formula

```
reward = -avg_loss(completion_tokens)
```

Where loss is calculated only on completion tokens (not the prompt).

---

## 2. Supervised Mode: Coverage-Weighted Loss

Supervised GRPO requires rewards that **vary per completion** to enable advantage calculation. We explored several approaches before settling on coverage-weighted loss.

### Rejected Approaches

1. **Discrete Rewards (exact_match, contains, fuzzy)** - Too sparse, doesn't provide gradient signal
2. **Perplexity on Expected Answer** - All K completions get the SAME reward (doesn't vary), which breaks GRPO's advantage calculation: $A_i = \frac{R_i - \mu}{\sigma}$

### Final Approach: Coverage-Weighted Loss

Two modes available:

#### Mode 1: `answer_perplexity`
Calculates perplexity of the expected answer given the prompt:
```
reward = -ppl(prompt + answer, mask=answer_tokens) * coverage_ratio
```

#### Mode 2: `completion_cross_entropy`
Calculates cross-entropy between completion logits and answer tokens:
```
reward = -CE(completion_logits, answer_tokens) * coverage_ratio
```

### Coverage Ratio

Both modes apply a coverage penalty to discourage short completions:
```
coverage_ratio = min(1.0, len(completion_tokens) / len(answer_tokens))
```

This ensures the model is incentivized to generate complete answers.

---

## 3. The GRPO Mechanism

GRPO eliminates the need for a Value Model by evaluating multiple completions against each other.

### The Training Loop

1. **Generation Phase**: For a given prompt, generate $K$ completions (e.g., $G_1, \dots, G_8$)
2. **Scoring Phase**: Calculate reward for each completion using the selected mode
3. **Advantage Calculation**: Normalize rewards within the group:
   $$A_i = \frac{R_i - \mu}{\sigma}$$
4. **Policy Update**: Update weights to increase likelihood of high-advantage completions

---

## 4. Preventing Mode Collapse (KL Penalty)

The KL Divergence penalty prevents the model from generating low-quality text just to game the reward:

- A frozen Reference Model is kept in memory
- During updates, the policy model's distribution is compared against the reference
- Deviations are penalized, forcing coherent outputs

---

## 5. System Constraints & Memory Management

- **vLLM Backend**: Fast inference with standby mode (30% VRAM savings via `UNSLOTH_VLLM_STANDBY=1`)
- **4-bit Quantization**: LoRA adapters for memory-efficient training
- **No-Gradient Evaluation**: Scoring phase drops gradients to prevent OOM
- **Target Hardware**: Single GPU with 24GB VRAM

---

## 6. Data Formats

### Unsupervised (JSONL)
```json
{"text": "Your unlabeled text here..."}
```

### Supervised (JSONL)
```json
{"prompt": "Question or instruction", "answer": "Expected response"}
```

---

## 7. Usage

```bash
# Unsupervised self-distillation
uv run python scripts/train.py --config configs/unsupervised.yaml

# Supervised GRPO
uv run python scripts/train.py --config configs/supervised.yaml
```
