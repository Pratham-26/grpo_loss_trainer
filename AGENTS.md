# AGENTS.md - Guidelines for Coding Agents

This document provides guidelines for coding agents working in this repository.

## Project Overview

Self-distillation GRPO (Group-Relative Policy Optimization) trainer for unsupervised domain adaptation of LLMs. Uses inverse loss as reward signal, eliminating need for labeled data. Built on Unsloth, TRL, and vLLM.

## Build/Lint/Test Commands

```bash
# Install dependencies (using uv)
uv venv && source .venv/bin/activate  # Linux/Mac
uv venv && .venv\Scripts\activate     # Windows
uv pip install -e ".[dev]"

# Lint with ruff
uv run ruff check .

# Format with ruff
uv run ruff format .

# Run tests (when available)
uv run pytest
uv run pytest tests/test_specific.py -v           # Single test file
uv run pytest tests/test_specific.py::test_name  # Single test

# Run training
uv run python scripts/train.py --config configs/default.yaml
```

## Code Style Guidelines

### Imports

```python
# Order: standard library, third-party, local imports
# Separate groups with blank lines
import os
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel
from unsloth import FastLanguageModel

from src.config import Config
from src.utils import clear_cuda_cache
```

### Type Hints

- Use modern Python 3.10+ syntax: `list[str]`, `dict[str, Any]`, `int | None`
- Use `Any` for external library objects (model, tokenizer)
- Use `Literal` for enum-like parameters
- Always annotate function parameters and return types

```python
def load_jsonl_dataset(
    path: str,
    prompt_field: str,
    answer_field: str,
    chat_template_name: str,
    mode: Literal["unsupervised", "supervised"] = "unsupervised",
    max_samples: int | None = None,
) -> Dataset:
    ...

def create_trainer(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    config: Any,
) -> GRPOTrainer:
    ...
```

### Naming Conventions

- **Functions/variables**: `snake_case` - `load_model`, `clear_cuda_cache`, `max_steps`
- **Classes**: `PascalCase` - `ModelConfig`, `TrainingConfig`
- **Private functions**: Prefix with underscore - `_extract_text`, `_compute_loss`
- **Constants**: `UPPER_SNAKE_CASE` - `DEFAULT_CHAT_TEMPLATE`
- **Config fields**: `snake_case` matching YAML keys

### Function Design

- Keep functions small and focused (single responsibility)
- Use factory pattern for creating components: `create_trainer()`, `create_supervised_reward()`
- Return early for guard clauses
- Validate inputs early and fail fast

```python
def load_config(config_path: str) -> Config:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path) as f:
        data = yaml.safe_load(f)
    
    return Config(**data)
```

### Error Handling

- Raise specific exceptions with descriptive messages
- Validate list/array lengths before processing
- Use Pydantic for config validation
- Never silently ignore invalid inputs

```python
# Good: Validate and raise
if len(prompts) != len(completions):
    raise ValueError(
        f"Mismatched list lengths: prompts={len(prompts)}, "
        f"completions={len(completions)}"
    )

# Good: Check for None explicitly
if answers is None:
    answers = kwargs.get(answer_field, [])

# Bad: Silent ignore
if key in config_dict:
    config_dict[key] = value  # Unknown keys silently ignored
```

### Configuration

- Use Pydantic models with default values
- Group related settings into nested models
- YAML config files in `configs/` directory
- Support CLI overrides with `key.subkey` syntax

```python
class ModelConfig(BaseModel):
    name: str = "unsloth/LFM2.5-1.2B-Instruct"
    max_seq_length: int = 4096
    lora_rank: int = 32
```

### Comments

- **Do not add comments** unless explicitly requested
- Code should be self-documenting through clear naming
- Docstrings only for public APIs that need usage explanation

### Formatting

- Line length: 100 characters (enforced by ruff)
- Use double quotes for strings
- Blank line before `return` in short functions is optional
- Trailing commas in multi-line structures

```python
training_args = GRPOConfig(
    learning_rate=config.training.learning_rate,
    weight_decay=config.training.weight_decay,
    max_steps=config.training.max_steps,  # Trailing comma
)
```

## Project Structure

```
remote_trainer/
├── src/
│   ├── __init__.py
│   ├── config.py       # Pydantic config models + loading
│   ├── model.py        # Model loading, LoRA setup, memory cleanup
│   ├── data.py         # Dataset loading and validation
│   ├── templates.py    # Chat template management
│   ├── trainer.py      # GRPO trainer creation and training loop
│   ├── utils.py        # Memory utilities, formatting helpers
│   └── rewards/
│       ├── __init__.py         # Reward factory function
│       ├── inverse_loss.py     # Unsupervised reward
│       └── supervised.py       # Supervised reward with coverage
├── configs/
│   ├── default.yaml
│   ├── supervised.yaml
│   └── unsupervised.yaml
├── scripts/
│   └── train.py        # Entry point
└── data/               # Training data (JSONL)
```

## Key Dependencies

- **unsloth**: Fast LLM fine-tuning with LoRA
- **trl**: GRPO trainer implementation
- **vllm**: Fast inference for generation
- **pydantic**: Configuration validation
- **torch**: Deep learning framework

## Common Patterns

### Creating Reward Functions

```python
def create_supervised_reward(model, tokenizer, answer_field="answer"):
    def supervised_reward(prompts, completions, answers=None, **kwargs):
        if answers is None:
            answers = kwargs.get(answer_field, [])

        rewards = []
        for prompt, completion, answer in zip(prompts, completions, answers):
            reward = _compute_completion_cross_entropy(
                model, tokenizer, completion, answer
            )
            rewards.append(reward)

        return rewards

    return supervised_reward
```

### Config with CLI Overrides

```python
overrides = {}
if args.data_path:
    overrides["data.path"] = args.data_path
if args.max_steps:
    overrides["training.max_steps"] = args.max_steps

config = merge_cli_overrides(config, overrides)
```

## Testing Notes

- No test directory exists yet
- When adding tests, place in `tests/` directory
- Use pytest with fixtures for model/tokenizer mocks
- Test config validation, data loading, and reward computation
