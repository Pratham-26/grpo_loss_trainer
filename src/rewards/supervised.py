"""
Supervised GRPO reward using coverage-weighted cross-entropy loss.

reward = -CE(completion_logits, answer_tokens) * coverage
"""

from typing import Any, Callable

import torch


def create_supervised_reward(
    model: Any,
    tokenizer: Any,
    answer_field: str = "answer",
) -> Callable:
    """
    Create a supervised reward using coverage-weighted cross-entropy loss.

    Args:
        model: Language model
        tokenizer: Tokenizer
        answer_field: Field name for expected answers in dataset

    Returns:
        Reward function compatible with GRPO trainer
    """

    def supervised_reward(
        prompts: list, completions: list, answers: list | None = None, **kwargs
    ) -> list[float]:
        rewards = []
        if answers is None:
            answers = kwargs.get(answer_field, [])

        if len(prompts) != len(completions) or len(prompts) != len(answers):
            raise ValueError(
                f"Mismatched list lengths: prompts={len(prompts)}, "
                f"completions={len(completions)}, answers={len(answers)}"
            )

        for prompt, completion, expected_answer in zip(prompts, completions, answers):
            if not expected_answer:
                rewards.append(0.0)
                continue

            completion_text = _extract_text(completion)

            if not completion_text or not expected_answer:
                rewards.append(0.0)
                continue

            completion_token_ids = tokenizer(completion_text, add_special_tokens=False).get(
                "input_ids", []
            )
            answer_token_ids = tokenizer(expected_answer, add_special_tokens=False).get(
                "input_ids", []
            )

            completion_tokens = completion_token_ids if completion_token_ids else []
            answer_tokens = answer_token_ids if answer_token_ids else []

            coverage = min(1.0, len(completion_tokens) / max(1, len(answer_tokens)))

            avg_loss = _compute_completion_cross_entropy(
                model, tokenizer, completion_text, expected_answer
            )

            reward = -avg_loss * coverage
            rewards.append(reward)

        return rewards

    return supervised_reward


def _compute_completion_cross_entropy(
    model: Any,
    tokenizer: Any,
    completion_text: str,
    expected_answer: str,
) -> float:
    """
    Compute cross-entropy between completion logits and answer tokens.

    Measures how well the completion's probability distribution matches the answer.
    """
    completion_inputs = tokenizer(completion_text, return_tensors="pt").to(model.device)
    answer_inputs = tokenizer(expected_answer, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**completion_inputs)
        logits = outputs.logits[..., :-1, :]

        compare_len = min(logits.size(1), answer_inputs["input_ids"].size(1))

        if compare_len == 0:
            return 0.0

        completion_logits = logits[:, :compare_len, :]
        answer_labels = answer_inputs["input_ids"][:, :compare_len]

        loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
        return loss_fct(
            completion_logits.view(-1, completion_logits.size(-1)),
            answer_labels.view(-1),
        ).item()


def _extract_text(data: Any) -> str:
    """Extract text from various formats (string, list of dicts, etc.)."""
    if isinstance(data, str):
        return data
    elif isinstance(data, list):
        parts = []
        for item in data:
            if isinstance(item, dict):
                parts.append(item.get("content", ""))
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts)
    return str(data)
