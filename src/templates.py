from pathlib import Path


DEFAULT_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{{ messages[0]['content'] }}"
    "{% set loop_messages = messages[1:] %}"
    "{% else %}"
    "{{ messages[0]['content'] }}"
    "{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ message['content'] }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
)

TEMPLATES = {
    "default": DEFAULT_CHAT_TEMPLATE,
    "none": None,
}


def get_template(name: str) -> str | None:
    if name in TEMPLATES:
        return TEMPLATES[name]

    path = Path(name)
    if path.exists():
        return path.read_text().strip()

    raise ValueError(
        f"Unknown template: {name}. Use 'default', 'none', or provide a path to a custom template file."
    )
