"""LLM client and edge scoring."""

from openai import OpenAI

from mas.prompts import score_prompt

_client: OpenAI | None = None
_model: str = "deepseek-chat"
_record_path: str = "outputs/run.log"


def init_llm(base_url: str, api_key: str, model: str, record_path: str) -> None:
    global _client, _model, _record_path
    _client = OpenAI(base_url=base_url, api_key=api_key)
    _model = model
    _record_path = record_path


def log_message(text: str) -> None:
    print(text)
    with open(_record_path, "a", encoding="utf-8") as f:
        f.write(str(text) + "\n")


def ask(prompt: str, user_input: str, write_log: bool = False) -> str:
    if _client is None:
        raise RuntimeError("Call init_llm() before ask().")
    if write_log:
        log_message("input:")
        log_message(prompt)
        log_message(user_input)
    response = _client.chat.completions.create(
        model=_model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
        ],
        stream=False,
    )
    msg = response.choices[0].message
    text = msg.content if msg.content is not None else msg.reasoning_content
    if write_log:
        log_message("response:")
        log_message(text)
    return text


def parse_score(raw: str) -> int | None:
    tail = raw.split("[score]")[-1].strip()
    if tail.startswith("-1"):
        return -1
    if tail.startswith("0"):
        return 0
    if tail.startswith("1"):
        return 1
    return None


def get_edge_score(advisor_answer: str, respondent_answer: str, retries: int = 5) -> int:
    prompt = score_prompt()
    user_input = (
        f"The conversation:\nAssistant:\n{respondent_answer}\nAdvisor:\n{advisor_answer}\n"
    )
    for _ in range(retries):
        output = ask(prompt, user_input)
        score = parse_score(output)
        if score is not None:
            return score
    return 0
