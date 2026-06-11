from src.generation.constants import SYSTEM_PROMPT, USER_MESSAGE_TEMPLATE


def generate_messages(context: str, question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_MESSAGE_TEMPLATE.format(context=context, question=question)},
    ]
