import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data import load_jsonl_dataset, validate_dataset
from src.model import apply_chat_template, cleanup_memory, load_model
from src.templates import get_template
from src.trainer import save_outputs, train
from src.utils import clear_cuda_cache, print_memory_usage


def parse_args():
    parser = argparse.ArgumentParser(description="Self-distillation GRPO trainer")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data.path",
        type=str,
        dest="data_path",
        help="Override data path",
    )
    parser.add_argument(
        "--training.max_steps",
        type=int,
        dest="max_steps",
        help="Override max training steps",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        dest="output_dir",
        help="Override output directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Self-Distillation GRPO Trainer")
    print("=" * 60)

    print(f"\nLoading config from: {args.config}")
    config = load_config(args.config)

    overrides = {}
    if args.data_path:
        overrides["data.path"] = args.data_path
    if args.max_steps:
        overrides["training.max_steps"] = args.max_steps
    if args.output_dir:
        overrides["training.output_dir"] = args.output_dir

    if overrides:
        print(f"Applying overrides: {overrides}")

    if config.vllm.standby_mode:
        os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
        print("vLLM standby mode enabled (30% VRAM savings)")

    print(f"\nModel: {config.model.name}")
    print(f"Dataset: {config.data.path}")
    print(f"Max steps: {config.training.max_steps}")
    print(f"Num generations: {config.training.num_generations}")

    print("\n" + "-" * 60)
    print("Loading model...")
    print("-" * 60)

    model, tokenizer = load_model(config)

    template = get_template(config.data.chat_template)
    tokenizer = apply_chat_template(tokenizer, template)
    print(f"Chat template: {config.data.chat_template}")

    clear_cuda_cache()
    print_memory_usage()

    print("\n" + "-" * 60)
    print("Loading dataset...")
    print("-" * 60)

    dataset = load_jsonl_dataset(
        path=config.data.path,
        prompt_field=config.data.prompt_field,
        chat_template_name=config.data.chat_template,
        max_samples=config.data.max_samples,
    )
    validate_dataset(dataset)
    print(f"Dataset size: {len(dataset)} samples")

    print("\n" + "-" * 60)
    print("Creating trainer...")
    print("-" * 60)

    from src.trainer import create_trainer

    trainer = create_trainer(model, tokenizer, dataset, config)

    print("\n" + "-" * 60)
    print("Starting training...")
    print("-" * 60)

    train(trainer)

    print("\n" + "-" * 60)
    print("Saving outputs...")
    print("-" * 60)

    save_outputs(model, tokenizer, config)

    cleanup_memory()
    print_memory_usage()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
