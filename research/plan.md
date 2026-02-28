# Architecture Overview: Self-Distillation via Group-Relative Policy Optimization (GRPO)

## Abstract

This document outlines a method for fine-tuning Large Language Models (LLMs) using an unsupervised self-distillation approach. By leveraging Group-Relative Policy Optimization (GRPO) and replacing the traditional external Reward Model with the policy model's own internal confidence metrics (inverse loss/perplexity), we create a self-reinforcing feedback loop. This approach is highly effective for Unsupervised Domain Adaptation, allowing a model to align its generation style to a new domain without requiring labeled data or human preference feedback.

---

## 1. The Core Concept: Internal Confidence as a Reward Signal

In standard Reinforcement Learning from Human Feedback (RLHF), an external Reward Model is trained to score the policy model's outputs. In this self-distillation setup, the policy model acts as its own critic.

The primary metric for the reward is **Inverse Loss** (Negative Log-Likelihood).

* When a model generates a sequence of tokens, it computes the probability of each token.
* A lower cross-entropy loss indicates higher internal confidence (lower perplexity) for that specific completion.
* By converting this loss into a reward (e.g., multiplying by -1), we instruct the model to favor the completions it inherently finds the most coherent and mathematically probable based on its pre-trained or progressively fine-tuned weights.

---

## 2. The GRPO Mechanism

Standard policy gradient methods rely on a separate Value Model to estimate a baseline for advantage calculation, doubling the memory footprint. GRPO eliminates the need for a Value Model by evaluating multiple completions against each other.

Instead of estimating an absolute baseline, GRPO computes a relative baseline directly from a group of generations produced from the same prompt.

### The Training Loop (Step-by-Step)

1. **Exploration (Generation Phase):** For a given unlabeled prompt, the active policy model generates a group of $N$ distinct completions (e.g., $G_1, G_2, \dots, G_8$).
2. **Evaluation (Scoring Phase):**
The model performs a frozen forward pass over each generated completion to calculate the token-wise cross-entropy loss. The average loss of the completion tokens becomes the raw score.
3. **Advantage Calculation (Relative Baseline):**
The raw scores (inverse loss) are normalized within the group. The advantage $A_i$ for a specific generation $G_i$ is calculated using the mean ($\mu$) and standard deviation ($\sigma$) of the group's rewards:

$$A_i = \frac{R_i - \mu}{\sigma}$$



Completions with lower-than-average loss get a positive advantage; those with higher-than-average loss get a negative advantage.
4. **Policy Update (Optimization Phase):**
The model updates its weights to increase the likelihood of the generations with positive advantages, effectively pulling its "average" performance toward its "best" performance.

---

## 3. Preventing Mode Collapse (The KL Penalty)

A significant risk in self-distillation is mode collapse. If a model is rewarded purely for low loss, it will rapidly converge on generating highly repetitive, low-entropy text (e.g., repeating the word "the" infinitely).

To counteract this, the optimization step includes a **Kullback-Leibler (KL) Divergence Penalty**.

* A frozen copy of the base model (the Reference Model) is kept in memory.
* During the update step, the probability distribution of the active policy model is compared against the Reference Model.
* If the policy model's outputs deviate too drastically from the Reference Model's expected distribution, the reward is heavily penalized.
* This forces the model to remain coherent and structurally sound while it hunts for the most confident variations of its domain-specific outputs.

---

## 4. Why This Works for Unsupervised Domain Adaptation

When adapting a model to a massive corpus of highly specialized text (e.g., legal documents, medical records, or proprietary code) where labeled QA pairs do not exist, this method shines:

* **Zero Human Annotation:** It requires only raw text prompts.
* **Self-Correction:** It allows the model to explore different ways of structuring a response in the new domain and naturally biases toward the phrasing that aligns best with the underlying statistical patterns of the domain.
* **Hallucination Reduction:** Because it penalizes high-perplexity outputs relative to a group, it actively filters out anomalous, low-confidence "hallucinated" strings during the adaptation phase.

---

## 5. System Constraints & Memory Management

Because this approach requires generating multiple outputs and then calculating their losses before a backward pass, it is highly memory-intensive.

* **No-Gradient Evaluation:** The scoring phase must strictly drop gradients to prevent Out-Of-Memory (OOM) failures.
* **Quantization and Kernels:** Implementations usually rely on 4-bit quantization and fused attention kernels to make holding the policy model, the reference model, and the generation batches viable on single-node hardware.
