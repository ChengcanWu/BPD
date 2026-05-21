"""Flat and hierarchy MAS runners with BPD repair."""

from typing import Literal

import pandas as pd

from mas.graph import ChatTurn, EdgeGraph, run_round
from mas.prompts import (
    CHOICE_DICT,
    flat_peer_suggestion_prompt,
    flat_summary_prompt,
    start_prompt,
    suggestion_prompt,
    summary_prompt,
)

MASStructure = Literal["flat", "hierarchy"]


def build_question(df: pd.DataFrame, question_id: int) -> dict[str, str]:
    choices = "".join(
        f"{CHOICE_DICT[str(k)]}. {df['choices'][question_id][k]}\n" for k in range(4)
    )
    return {"Question": df["question"][question_id], "Choices": choices}


def _init_answers(agents_per_round: list[int]) -> list[dict[int, list[str]]]:
    answers: list[dict[int, list[str]]] = [{} for _ in range(len(agents_per_round))]
    for r, count in enumerate(agents_per_round):
        for i in range(1, count + 1):
            answers[r][i] = []
    return answers


def _first_answer(answers: list[dict[int, list[str]]], round_idx: int, agent_id: int) -> str:
    return answers[round_idx][agent_id][0]


def _is_malicious(
    agent_id: int,
    malicious_agent_id: int,
    repair_mode: bool,
    blocked_agents: set[int],
) -> bool:
    if repair_mode and agent_id in blocked_agents:
        return False
    return agent_id == malicious_agent_id


def _clear_blocked_edges(edges: EdgeGraph, blocked_agents: set[int]) -> None:
    for sender_idx in range(edges.total_nodes):
        agent_id = _agent_id_from_node(edges, sender_idx)
        if agent_id in blocked_agents:
            for row in range(edges.total_nodes):
                edges.connections[row][sender_idx] = 0


def _agent_id_from_node(edges: EdgeGraph, node_idx: int) -> int | None:
    offset = 0
    for count in edges.agents_per_round:
        if node_idx < offset + count:
            return node_idx - offset + 1
        offset += count
    return None


# --- Hierarchy: 5 respondents -> 2 advisors -> 5 summaries (5+2+5) ---


def _hierarchy_advisor_feedback(
    answers: list[dict[int, list[str]]], advisor_id: int, respondent_id: int
) -> str:
    return answers[1][advisor_id][respondent_id - 1]


def communicate_hierarchy(
    question_id: int,
    df: pd.DataFrame,
    malicious_agent_id: int = 1,
    num_respondents: int = 5,
    num_advisors: int = 2,
    repair_mode: bool = False,
    blocked_agents: set[int] | None = None,
) -> tuple[list[dict[int, list[str]]], EdgeGraph]:
    blocked_agents = blocked_agents or set()
    agents_per_round = [num_respondents, num_advisors, num_respondents]
    answers = _init_answers(agents_per_round)
    edges = EdgeGraph(agents_per_round)
    question = build_question(df, question_id)

    # Round 0: initial answers
    round0 = [
        ChatTurn(
            agent_id,
            start_prompt(
                agent_id,
                malicious=_is_malicious(agent_id, malicious_agent_id, repair_mode, blocked_agents),
                df=df,
                question_id=question_id,
            ),
            dict(question),
            {},
        )
        for agent_id in range(1, num_respondents + 1)
    ]
    run_round(round0, answers, edges, 0, sender_round=-1)

    # Round 1: each advisor comments on each respondent
    for respondent_id in range(1, num_respondents + 1):
        respondent_answer = _first_answer(answers, 0, respondent_id)
        round1 = [
            ChatTurn(
                advisor_id,
                suggestion_prompt(advisor_id, respondent_id, malicious=False),
                {**question, f"assistant {respondent_id}'s answer": respondent_answer},
                {respondent_id: respondent_answer},
            )
            for advisor_id in range(1, num_advisors + 1)
        ]
        run_round(round1, answers, edges, 1, sender_round=0)

    # Round 2: respondents summarize advisor feedback
    round2 = []
    for agent_id in range(1, num_respondents + 1):
        extra = {
            f"advisor {aid}'s suggestion": _hierarchy_advisor_feedback(answers, aid, agent_id)
            for aid in range(1, num_advisors + 1)
        }
        score_inputs = {
            aid: _hierarchy_advisor_feedback(answers, aid, agent_id)
            for aid in range(1, num_advisors + 1)
        }
        round2.append(
            ChatTurn(
                agent_id,
                summary_prompt(
                    agent_id,
                    list(range(1, num_advisors + 1)),
                    malicious=_is_malicious(agent_id, malicious_agent_id, repair_mode, blocked_agents),
                    df=df,
                    question_id=question_id,
                ),
                {**question, "your answer": _first_answer(answers, 0, agent_id), **extra},
                score_inputs,
            )
        )
    run_round(round2, answers, edges, 2, sender_round=1)
    _clear_blocked_edges(edges, blocked_agents)
    return answers, edges


# --- Flat: 5 agents answer -> mutual peer suggestions -> summaries (5+5+5) ---


def _flat_peer_feedback(
    answers: list[dict[int, list[str]]], from_id: int, to_id: int
) -> str:
    """Feedback from assistant from_id to assistant to_id (round 1 storage order)."""
    if from_id == to_id:
        raise ValueError("No self-feedback in flat MAS.")
    items = answers[1][from_id]
    if to_id < from_id:
        return items[to_id - 1]
    return items[to_id - 2]


def communicate_flat(
    question_id: int,
    df: pd.DataFrame,
    malicious_agent_id: int = 1,
    num_agents: int = 5,
    repair_mode: bool = False,
    blocked_agents: set[int] | None = None,
) -> tuple[list[dict[int, list[str]]], EdgeGraph]:
    blocked_agents = blocked_agents or set()
    agents_per_round = [num_agents, num_agents, num_agents]
    answers = _init_answers(agents_per_round)
    edges = EdgeGraph(agents_per_round)
    question = build_question(df, question_id)

    # Round 0: initial answers
    round0 = [
        ChatTurn(
            agent_id,
            start_prompt(
                agent_id,
                malicious=_is_malicious(agent_id, malicious_agent_id, repair_mode, blocked_agents),
                df=df,
                question_id=question_id,
            ),
            dict(question),
            {},
        )
        for agent_id in range(1, num_agents + 1)
    ]
    run_round(round0, answers, edges, 0, sender_round=-1)

    # Round 1: each agent suggests on every other agent's answer
    for target_id in range(1, num_agents + 1):
        target_answer = _first_answer(answers, 0, target_id)
        round1 = []
        for suggester_id in range(1, num_agents + 1):
            if suggester_id == target_id:
                continue
            round1.append(
                ChatTurn(
                    suggester_id,
                    flat_peer_suggestion_prompt(
                        suggester_id,
                        target_id,
                        malicious=_is_malicious(
                            suggester_id, malicious_agent_id, repair_mode, blocked_agents
                        ),
                        df=df,
                        question_id=question_id,
                    ),
                    {**question, f"assistant {target_id}'s answer": target_answer},
                    {target_id: target_answer},
                )
            )
        run_round(round1, answers, edges, 1, sender_round=0)

    # Round 2: each agent synthesizes peer suggestions
    round2 = []
    for agent_id in range(1, num_agents + 1):
        peer_ids = [p for p in range(1, num_agents + 1) if p != agent_id]
        extra = {
            f"assistant {pid}'s suggestion": _flat_peer_feedback(answers, pid, agent_id)
            for pid in peer_ids
        }
        score_inputs = {pid: _flat_peer_feedback(answers, pid, agent_id) for pid in peer_ids}
        round2.append(
            ChatTurn(
                agent_id,
                flat_summary_prompt(
                    agent_id,
                    peer_ids,
                    malicious=_is_malicious(agent_id, malicious_agent_id, repair_mode, blocked_agents),
                    df=df,
                    question_id=question_id,
                ),
                {**question, "your answer": _first_answer(answers, 0, agent_id), **extra},
                score_inputs,
            )
        )
    run_round(round2, answers, edges, 2, sender_round=1)
    _clear_blocked_edges(edges, blocked_agents)
    return answers, edges


def communicate(
    question_id: int,
    df: pd.DataFrame,
    structure: MASStructure = "hierarchy",
    malicious_agent_id: int = 1,
    num_respondents: int = 5,
    num_advisors: int = 2,
    repair_mode: bool = False,
    blocked_agents: set[int] | None = None,
) -> tuple[list[dict[int, list[str]]], EdgeGraph]:
    if structure == "flat":
        return communicate_flat(
            question_id,
            df,
            malicious_agent_id=malicious_agent_id,
            num_agents=num_respondents,
            repair_mode=repair_mode,
            blocked_agents=blocked_agents,
        )
    return communicate_hierarchy(
        question_id,
        df,
        malicious_agent_id=malicious_agent_id,
        num_respondents=num_respondents,
        num_advisors=num_advisors,
        repair_mode=repair_mode,
        blocked_agents=blocked_agents,
    )


def communicate_with_bpd(
    question_id: int,
    df: pd.DataFrame,
    structure: MASStructure = "hierarchy",
    epsilon: float = 1.5,
    malicious_agent_id: int = 1,
    num_respondents: int = 5,
    num_advisors: int = 2,
) -> dict:
    """Full pipeline: attacked MAS -> BPD -> repaired MAS."""
    from mas.bpd import run_bpd

    agent_ids = list(range(1, num_respondents + 1))
    common = dict(
        question_id=question_id,
        df=df,
        structure=structure,
        malicious_agent_id=malicious_agent_id,
        num_respondents=num_respondents,
        num_advisors=num_advisors,
    )

    attacked_answers, attacked_edges = communicate(**common)
    bpd_result = run_bpd(attacked_edges, attacked_answers, agent_ids, epsilon)
    bpd_result["structure"] = structure

    blocked = set()
    if bpd_result["detected_agent"] is not None:
        blocked.add(bpd_result["detected_agent"])

    repaired_answers, repaired_edges = communicate(
        **common, repair_mode=True, blocked_agents=blocked
    )

    return {
        "structure": structure,
        "attacked_answers": attacked_answers,
        "attacked_edges": attacked_edges,
        "repaired_answers": repaired_answers,
        "repaired_edges": repaired_edges,
        **bpd_result,
    }
