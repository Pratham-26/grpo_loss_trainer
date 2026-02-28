from typing import Any, Callable, Literal

from src.rewards.inverse_loss import create_inverse_loss_reward
from src.rewards.supervised import create_supervised_reward


def get_reward_function(
    reward_type: Literal["inverse_loss", "supervised"],
    model: Any,
    tokenizer: Any,
    answer_field: str = "answer",
    supervised_mode: Literal["answer_perplexity", "completion_cross_entropy"] = "answer_perplexity",
) -> Callable:
    """
    Factory function to create reward function.

    Args:
        reward_type: "inverse_loss" (unsupervised) or "supervised"
        model: Language model
        tokenizer: Tokenizer
        answer_field: Field name for expected answers in dataset (supervised mode)
        supervised_mode: "answer_perplexity" or "completion_cross_entropy"
            - answer_perplexity: -ppl(prompt + answer, mask=answer) * coverage
            - completion_cross_entropy: -CE(completion_logits, answer_tokens) * coverage

    Returns:
        Reward function compatible with GRPO trainer
    """
    if reward_type == "inverse_loss":
        return create_inverse_loss_reward(model, tokenizer)
    elif reward_type == "supervised":
        return create_supervised_reward(
            model=model,
            tokenizer=tokenizer,
            answer_field=answer_field,
            mode=supervised_mode,
        )
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")


__all__ = [
    "create_inverse_loss_reward",
    "create_supervised_reward",
    "get_reward_function",
]
