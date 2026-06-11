from dataclasses import dataclass


SYSTEM_PROMPT = "Answer the question using the provided context."
USER_MESSAGE_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}"


@dataclass(frozen=True)
class SamplingParams:
    max_new_tokens: int = 8192
    temperature: float = 0.7
    do_sample: bool = True


DEFAULT_SAMPLING_PARAMS = SamplingParams()
