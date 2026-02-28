import os
from typing import Any

from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

from src.rewards import get_reward_function


def create_trainer(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    config: Any,
) -> GRPOTrainer:
    reward_func = get_reward_function(
        reward_type=config.reward.type,
        model=model,
        tokenizer=tokenizer,
        answer_field=config.data.answer_field,
    )

    vllm_sampling_params = SamplingParams(
        min_p=config.vllm.min_p,
        top_p=config.vllm.top_p,
        top_k=config.vllm.top_k,
        seed=config.vllm.seed,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )

    training_args = GRPOConfig(
        vllm_sampling_params=vllm_sampling_params,
        temperature=config.vllm.temperature,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        optim=config.training.optim,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        num_generations=config.training.num_generations,
        max_prompt_length=config.training.max_prompt_length,
        max_completion_length=config.training.max_completion_length,
        beta=config.training.beta,
        max_steps=config.training.max_steps,
        output_dir=config.training.output_dir,
        logging_steps=config.training.logging_steps,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=dataset,
    )

    return trainer


def train(trainer: GRPOTrainer) -> None:
    trainer.train()


def save_outputs(model: Any, tokenizer: Any, config: Any) -> None:
    output_dir = config.training.output_dir

    if config.output.save_lora:
        lora_path = os.path.join(output_dir, config.output.lora_dir)
        model.save_lora(lora_path)
        print(f"LoRA adapters saved to: {lora_path}")

    if config.output.save_merged_16bit:
        merged_path = os.path.join(output_dir, "merged_16bit")
        model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
        print(f"Merged 16-bit model saved to: {merged_path}")

    if config.output.save_merged_4bit:
        merged_path = os.path.join(output_dir, "merged_4bit")
        model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_4bit")
        print(f"Merged 4-bit model saved to: {merged_path}")

    if config.output.save_gguf:
        gguf_path = os.path.join(output_dir, "gguf")
        model.save_pretrained_gguf(
            gguf_path,
            tokenizer,
            quantization_method=config.output.gguf_quantization,
        )
        print(f"GGUF model saved to: {gguf_path}")

    if config.output.push_to_hub and config.output.hub_repo_id:
        token = config.output.hub_token or os.environ.get("HF_TOKEN")
        if token:
            if config.output.save_lora:
                model.push_to_hub(config.output.hub_repo_id, token=token)
                tokenizer.push_to_hub(config.output.hub_repo_id, token=token)
                print(f"Model pushed to Hub: {config.output.hub_repo_id}")
        else:
            print("Warning: No HuggingFace token provided. Skipping hub push.")
