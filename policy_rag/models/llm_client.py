from transformers import pipeline
from ..config import LLM_GENERATION_MODEL, LLM_JUDGE_MODEL

_gen_pipe = None
_judge_pipe = None

def _get_gen_pipe():
    global _gen_pipe
    if _gen_pipe is None:
        _gen_pipe = pipeline("text-generation", model=LLM_GENERATION_MODEL)
    return _gen_pipe

def _get_judge_pipe():
    global _judge_pipe
    if _judge_pipe is None:
        _judge_pipe = pipeline("text-generation", model=LLM_JUDGE_MODEL)
    return _judge_pipe

def call_llm(system_prompt: str, user_prompt: str, max_new_tokens: int = 256) -> str:
    """
    We build a structured prompt and let a causal LM continue it.
    We then capture only the assistant portion after [ASSISTANT].
    """
    pipe = _get_gen_pipe()
    full_prompt = (
        "[SYSTEM]\n" + system_prompt +
        "\n[USER]\n" + user_prompt +
        "\n[ASSISTANT]\n"
    )
    out = pipe(
        full_prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )[0]["generated_text"]

    return out.split("[ASSISTANT]")[-1].strip()

def call_judge(system_prompt: str, user_prompt: str, max_new_tokens: int = 256) -> str:
    pipe = _get_judge_pipe()
    full_prompt = (
        "[SYSTEM]\n" + system_prompt +
        "\n[USER]\n" + user_prompt +
        "\n[ASSISTANT]\n"
    )
    out = pipe(
        full_prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )[0]["generated_text"]

    return out.split("[ASSISTANT]")[-1].strip()
