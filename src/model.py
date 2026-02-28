from typing import Any

from unsloth import FastLanguageModel

from src.utils import clear_cuda_cache


def load_model(config: Any) -> tuple[Any, Any]:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model.name,
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=config.model.load_in_4bit,
        fast_inference=config.model.fast_inference,
        max_lora_rank=config.model.lora_rank,
        gpu_memory_utilization=config.model.gpu_memory_utilization,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.model.lora_rank,
        target_modules=_get_target_modules(config.model.name),
        lora_alpha=config.model.lora_rank * config.model.lora_alpha_multiplier,
        use_gradient_checkpointing="unsloth",
        random_state=config.model.random_state,
    )

    return model, tokenizer


def _get_target_modules(model_name: str) -> list[str]:
    model_name_lower = model_name.lower()

    if "lfm" in model_name_lower:
        return ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"]
    elif "qwen" in model_name_lower:
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    else:
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]


def apply_chat_template(tokenizer: Any, template: str | None) -> Any:
    if template is not None:
        tokenizer.chat_template = template
    return tokenizer


def cleanup_memory() -> None:
    clear_cuda_cache()
