from pathlib import Path

from datasets import Dataset

from src.templates import get_template


def load_jsonl_dataset(
    path: str,
    prompt_field: str,
    chat_template_name: str,
    max_samples: int | None = None,
) -> Dataset:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            import json

            item = json.loads(line)
            prompt_text = item.get(prompt_field, "")

            if not prompt_text:
                continue

            data.append({"text": prompt_text, "raw_prompt": prompt_text})

    if max_samples is not None and max_samples < len(data):
        data = data[:max_samples]

    dataset = Dataset.from_list(data)

    template = get_template(chat_template_name)
    if template is None:
        dataset = dataset.map(
            lambda x: {"prompt": [{"role": "user", "content": x["text"]}]},
            remove_columns=["text", "raw_prompt"],
        )
    else:
        dataset = dataset.map(
            lambda x: {"prompt": [{"role": "user", "content": x["text"]}]},
            remove_columns=["text", "raw_prompt"],
        )

    return dataset


def validate_dataset(dataset: Dataset) -> None:
    if "prompt" not in dataset.column_names:
        raise ValueError("Dataset must have a 'prompt' column")

    sample = dataset[0]
    prompt = sample["prompt"]

    if not isinstance(prompt, list):
        raise ValueError("Prompt must be a list of message dictionaries")

    if len(prompt) == 0:
        raise ValueError("Prompt list cannot be empty")

    for msg in prompt:
        if not isinstance(msg, dict):
            raise ValueError("Each prompt message must be a dictionary")
        if "role" not in msg or "content" not in msg:
            raise ValueError("Each prompt message must have 'role' and 'content' keys")
