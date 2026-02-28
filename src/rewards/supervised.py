"""
Supervised GRPO reward using coverage-weighted loss.

Two modes available:
- answer_perplexity (default): reward = -ppl(prompt + answer, mask=answer) * coverage
- completion_cross_entropy: reward = -CE(completion_logits, answer_tokens) * coverage
"""

import torch
from typing import Any, Callable, Literal


def create_supervised_reward(
    model: Any,
    tokenizer: Any,
    answer_field: str = "answer",
    mode: Literal["answer_perplexity", "completion_cross_entropy"] = "answer_perplexity",
) -> Callable:
    """
    Create a supervised reward using coverage-weighted loss.

    Args:
        model: Language model
        tokenizer: Tokenizer
        answer_field: Field name for expected answers in dataset
        mode: "answer_perplexity" or "completion_cross_entropy"
            - answer_perplexity: Measures how probable the model finds the expected answer
            - completion_cross_entropy: Cross-entropy between completion logits and answer tokens

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

            prompt_text = _extract_text(prompt)
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

            if mode == "answer_perplexity":
                avg_loss = _compute_answer_perplexity(
                    model, tokenizer, prompt_text, expected_answer
                )
            elif mode == "completion_cross_entropy":
                avg_loss = _compute_completion_cross_entropy(
                    model, tokenizer, completion_text, expected_answer
                )
            else:
                avg_loss = 0.0

            reward = -avg_loss * coverage
            rewards.append(reward)

        return rewards

    return supervised_reward


def _compute_answer_perplexity(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    expected_answer: str,
) -> float:
    """
    Option B: Compute perplexity on prompt + answer, with loss only on answer tokens.

    Measures how probable the model finds the expected answer given the prompt.
    """
    text = prompt_text + expected_answer
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    prompt_token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    prompt_length = len(prompt_token_ids)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        answer_start = max(0, prompt_length - 1)
        answer_losses = token_losses[answer_start:]

        if len(answer_losses) == 0:
            return 0.0

        return answer_losses.mean().item()


def _compute_completion_cross_entropy(
    model: Any,
    tokenizer: Any,
    completion_text: str,
    expected_answer: str,
) -> float:
    """
    Option A: Compute cross-entropy between completion logits and answer tokens.

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
