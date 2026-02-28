from typing import Any

import torch


def create_inverse_loss_reward(model: Any, tokenizer: Any) -> Any:
    def inverse_loss_reward(prompts: list, completions: list, **kwargs) -> list[float]:
        rewards = []

        for prompt, completion in zip(prompts, completions):
            prompt_text = _extract_prompt_text(prompt)
            completion_text = _extract_completion_text(completion)

            if not completion_text:
                rewards.append(0.0)
                continue

            text = prompt_text + completion_text
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            prompt_length = len(prompt_tokens)

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

                completion_start = max(0, prompt_length - 1)
                completion_losses = token_losses[completion_start:]

                if len(completion_losses) == 0:
                    rewards.append(0.0)
                    continue

                avg_loss = completion_losses.mean().item()
                rewards.append(-avg_loss)

        return rewards

    return inverse_loss_reward


def _extract_prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    elif isinstance(prompt, list):
        parts = []
        for msg in prompt:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    parts.append(content)
                elif role == "user":
                    parts.append(content)
                elif role == "assistant":
                    parts.append(content)
            elif isinstance(msg, str):
                parts.append(msg)
        return " ".join(parts)
    return str(prompt)


def _extract_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    elif isinstance(completion, list):
        parts = []
        for msg in completion:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                parts.append(content)
            elif isinstance(msg, str):
                parts.append(msg)
        return " ".join(parts)
    return str(completion)
