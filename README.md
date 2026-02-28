# Self-Distillation GRPO Trainer

A self-distillation trainer using Group-Relative Policy Optimization (GRPO) for unsupervised domain adaptation of Large Language Models.

## Overview

This trainer implements the self-distillation approach described in the research documentation:

- Uses the model's own **inverse loss** (negative log-likelihood) as the reward signal
- Eliminates the need for labeled data or human preference feedback
- Applies **KL divergence penalty** to prevent mode collapse
- Optimized for single GPU (24GB VRAM) with vLLM acceleration

## Installation

```bash
# Install uv (if not already)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install dependencies
uv venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

uv pip install -e .
```

## Data Format

Prepare your training data as a JSONL file with each line containing a prompt:

```jsonl
{"text": "Your first training text here..."}
{"text": "Your second training text here..."}
```

## Usage

### Basic Training

```bash
python scripts/train.py --config configs/default.yaml
```

### With Overrides

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --data.path data/my_data.jsonl \
    --training.max_steps 500 \
    --output_dir outputs/my_run
```

### Using uv run

```bash
uv run python scripts/train.py --config configs/default.yaml
```

## Configuration

Edit `configs/default.yaml` to customize training:

### Model Settings

```yaml
model:
  name: "unsloth/LFM2.5-1.2B-Instruct"  # Model to fine-tune
  max_seq_length: 4096
  lora_rank: 32
  load_in_4bit: false
  fast_inference: true  # Enable vLLM
```

### vLLM Settings

```yaml
vllm:
  standby_mode: true   # 30% VRAM savings
  min_p: 0.1
  temperature: 1.0
```

### Training Settings

```yaml
training:
  learning_rate: 5e-6
  num_generations: 4   # Completions per prompt
  beta: 0.04           # KL penalty coefficient
  max_steps: 100
```

### Output Settings

```yaml
output:
  save_lora: true
  save_merged_16bit: false
  save_merged_4bit: false
  save_gguf: false
  gguf_quantization: "q4_k_m"
```

## Chat Templates

The trainer supports configurable chat templates:

- `default` - Standard conversation format
- `none` - Raw text without template
- Custom path - Provide path to template file

```yaml
data:
  chat_template: "default"
```

## Memory Optimization

For limited VRAM (24GB), the following optimizations are applied:

| Technique | Setting |
|-----------|---------|
| vLLM Standby | `UNSLOTH_VLLM_STANDBY=1` |
| Gradient Accumulation | `gradient_accumulation_steps: 4` |
| Small Batch | `per_device_train_batch_size: 1` |
| Fewer Generations | `num_generations: 4` |

## Output Formats

After training, you can save in multiple formats:

- **LoRA adapters** - Lightweight, recommended
- **Merged 16-bit** - Full precision merged model
- **Merged 4-bit** - Quantized merged model
- **GGUF** - For llama.cpp / Ollama deployment

## Architecture

```
remote_trainer/
├── src/
│   ├── config.py      # Configuration management
│   ├── model.py       # Model loading & LoRA setup
│   ├── rewards.py     # Inverse loss reward function
│   ├── data.py        # Dataset loading
│   ├── templates.py   # Chat templates
│   ├── trainer.py     # GRPO training logic
│   └── utils.py       # Memory & logging utilities
├── configs/
│   └── default.yaml   # Default configuration
├── scripts/
│   └── train.py       # Entry point
└── data/              # Your training data
```

## How It Works

1. **Generation**: For each prompt, generate N completions
2. **Scoring**: Calculate inverse loss (negative perplexity) for each completion
3. **Normalization**: Compute relative advantage within each group
4. **Update**: Adjust model to favor high-advantage completions
5. **KL Penalty**: Prevent deviation from base model distribution

## References

- [Unsloth Documentation](https://docs.unsloth.ai/)
- [TRL GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer)
- [vLLM](https://github.com/vllm-project/vllm)

## License

MIT
