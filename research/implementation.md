Ah, my mistake! I got ahead of myself and thought you were already holding the code. Let’s build this out from scratch.

Since you are dealing with Unsloth and large language models, the primary enemy here is VRAM exhaustion. Calculating the log-likelihood of 8 different completions per prompt *while* holding gradients for the main model update is a recipe for an Out-Of-Memory (OOM) error if not structured perfectly.

Here is the step-by-step implementation plan for your Self-Distillation GRPO loop.

### Phase 1: The Core Logic – The Reward Function

The secret sauce is the reward function. TRL’s `GRPOTrainer` allows you to pass custom reward functions that take lists of `prompts` and `completions` and return a list of float rewards.

Because you want the model to evaluate *itself*, we need to pass a reference to the model into the reward function, ensure we are running in `torch.no_grad()`, and strictly calculate the loss on the *completion* tokens (ignoring the prompt).

```python
import torch

def create_inverse_loss_reward(eval_model, tokenizer):
    """
    Creates a reward function that calculates the negative log-likelihood 
    (inverse loss) of the completions using the provided model.
    """
    def inverse_loss_reward(prompts, completions, **kwargs):
        rewards = []
        
        # Unsloth is fast, but we still want to avoid massive padding matrices.
        # Looping through the group (e.g., 8 completions) sequentially or in micro-batches
        # is often safer for VRAM than one giant batched forward pass.
        for prompt, completion in zip(prompts, completions):
            text = prompt + completion
            
            # Tokenize text
            inputs = tokenizer(text, return_tensors="pt").to(eval_model.device)
            
            # Calculate where the prompt ends so we don't penalize the prompt text
            prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            prompt_length = len(prompt_tokens)
            
            with torch.no_grad(): # CRITICAL: Drop gradients for evaluation
                outputs = eval_model(**inputs)
                logits = outputs.logits
                
                # Standard next-token prediction shift
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs["input_ids"][..., 1:].contiguous()
                
                # Calculate token-wise Cross Entropy Loss
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                token_losses = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1)
                )
                
                # Slice the tensor to only calculate loss on the COMPLETION tokens
                # We subtract 1 because of the shift
                completion_losses = token_losses[prompt_length - 1:]
                
                if len(completion_losses) == 0:
                    rewards.append(0.0)
                    continue
                    
                # Average the loss over the completion
                avg_loss = completion_losses.mean().item()
                
                # Reward is INVERSE loss (lower loss = higher reward)
                rewards.append(-avg_loss)
                
        return rewards
        
    return inverse_loss_reward

```

### Phase 2: Preventing Mode Collapse (The "Teacher" vs "Student" Problem)

If the exact same model weights are used to both generate the text and score the text, the model will quickly discover a "cheat code": it will start generating highly repetitive, overly safe text (like "the the the") because repetitive tokens have incredibly low perplexity/loss.

**The Solution:** You need a Reference Model.
In GRPO, you generally maintain a frozen snapshot of the base model.

1. **The Policy Model (Student):** Generates the $G_1 ... G_8$ completions and gets updated.
2. **The Reference Model (Teacher):** Used to calculate a KL-divergence penalty. This penalizes the Student if its outputs drift too far from the base model's distribution, forcing it to remain coherent while it hunts for lower loss.

TRL's `GRPOTrainer` handles the KL divergence automatically under the hood, but you need to decide *which* model calculates the reward. For true self-distillation, you usually use the **Reference Model** (the frozen base weights) as the `eval_model` in the reward function above, not the actively training policy model.

### Phase 3: Wiring it up with Unsloth and TRL

Here is how you piece it all together using Unsloth's optimized pipeline and Hugging Face's TRL.

```python
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

# 1. Load the model and tokenizer via Unsloth (4-bit quantization for VRAM savings)
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "your-base-model",
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    fast_inference = True, 
)

# 2. Add LoRA adapters (This creates the "Student" policy model)
# The base frozen weights effectively act as your "Teacher" / Reference Model
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
)

# 3. Instantiate the Reward Function
# We pass the model in inference mode. In a LoRA setup, you can optionally 
# disable the adapters during the reward calculation if you strictly want 
# the base model to act as the evaluator.
inverse_loss_reward_func = create_inverse_loss_reward(model, tokenizer)

# 4. Configure GRPO
training_args = GRPOConfig(
    output_dir = "grpo_self_distill_output",
    learning_rate = 5e-6,
    num_generations = 8, # The 'G' in GRPO
    per_device_train_batch_size = 1, # Keep this small!
    gradient_accumulation_steps = 4,
    max_prompt_length = 512,
    max_completion_length = 512,
    # KL Divergence coefficient - crucial for preventing mode collapse
    beta = 0.04, 
    logging_steps = 10,
)

# 5. Initialize Trainer
trainer = GRPOTrainer(
    model = model,
    reward_funcs = [inverse_loss_reward_func],
    args = training_args,
    train_dataset = your_unsupervised_dataset, # Dataset needs a "prompt" column
)

# 6. Train
trainer.train()

```

### Key Takeaways for this Architecture

* **VRAM Safety:** By wrapping the reward calculation in `torch.no_grad()` and processing the 8 completions sequentially inside the reward function, we keep the VRAM spike strictly constrained.
* **Granular Slicing:** Slicing the `token_losses` tensor ensures the model isn't being rewarded for the prompt, only its own generated continuation.
* **The KL Penalty (`beta`):** This is your safety net against the model outputting gibberish just to game its own loss function.

Would you like me to walk through how to format your raw, unlabeled text dataset into the specific prompt structure that TRL's `GRPOTrainer` expects for this loop?
