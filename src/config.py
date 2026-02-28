from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str = "unsloth/LFM2.5-1.2B-Instruct"
    max_seq_length: int = 4096
    lora_rank: int = 32
    load_in_4bit: bool = False
    gpu_memory_utilization: float = 0.9
    fast_inference: bool = True
    lora_alpha_multiplier: int = 2
    random_state: int = 3407


class VLLMConfig(BaseModel):
    standby_mode: bool = True
    min_p: float = 0.1
    top_p: float = 1.0
    top_k: int = -1
    seed: int = 3407
    temperature: float = 1.0


class RewardConfig(BaseModel):
    type: Literal["inverse_loss", "supervised"] = "inverse_loss"
    answer_field: str = "answer"
    supervised_mode: Literal["answer_perplexity", "completion_cross_entropy"] = "answer_perplexity"


class DataConfig(BaseModel):
    path: str = "data/train.jsonl"
    prompt_field: str = "text"
    answer_field: str = "answer"
    chat_template: str = "default"
    max_samples: int | None = None


class TrainingConfig(BaseModel):
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_8bit"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_generations: int = 4
    max_prompt_length: int = 512
    max_completion_length: int = 512
    beta: float = 0.04
    max_steps: int = 100
    num_train_epochs: int | None = None
    output_dir: str = "outputs"
    logging_steps: int = 1


class OutputConfig(BaseModel):
    save_lora: bool = True
    lora_dir: str = "lora_adapter"
    save_merged_16bit: bool = False
    save_merged_4bit: bool = False
    save_gguf: bool = False
    gguf_quantization: Literal["q8_0", "q4_k_m", "q5_k_m", "f16"] = "q4_k_m"
    push_to_hub: bool = False
    hub_repo_id: str | None = None
    hub_token: str | None = None


class LoggingConfig(BaseModel):
    log_every: int = 10


class Config(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    vllm: VLLMConfig = Field(default_factory=VLLMConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(config_path: str) -> Config:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    return Config(**data)


def merge_cli_overrides(config: Config, overrides: dict) -> Config:
    if not overrides:
        return config

    config_dict = config.model_dump()
    unknown_keys = []

    for key, value in overrides.items():
        if "." in key:
            section, param = key.split(".", 1)
            if section in config_dict and param in config_dict[section]:
                config_dict[section][param] = value
            else:
                unknown_keys.append(key)
        elif key in config_dict:
            config_dict[key] = value
        else:
            unknown_keys.append(key)

    if unknown_keys:
        raise ValueError(f"Unknown config keys: {', '.join(unknown_keys)}")

    return Config(**config_dict)
