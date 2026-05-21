"""Backward Propagation Detection (BPD)."""

import re

from mas.graph import EdgeGraph

CHOICE_RE = re.compile(r"the correct answer is\s*([A-D])", re.IGNORECASE)


def extract_choice(text: str) -> str | None:
    match = CHOICE_RE.search(text or "")
    return match.group(1).upper() if match else None


def majority_choice(texts: list[str]) -> str | None:
    counts: dict[str, int] = {}
    for text in texts:
        choice = extract_choice(text)
        if choice:
            counts[choice] = counts.get(choice, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def backward_propagate(
    edges: EdgeGraph,
    final_round_answers: dict[int, str],
    final_choice: str | None,
) -> list[float]:
    """Eq. (1): propagate scores from the last round backward on the signed DAG."""
    n = edges.total_nodes
    scores = [0.0] * n
    last_round = len(edges.agents_per_round) - 1

    for agent_id in range(1, edges.agents_per_round[last_round] + 1):
        idx = edges.node_index(last_round, agent_id)
        ans = extract_choice(final_round_answers.get(agent_id, ""))
        scores[idx] = 1.0 if ans == final_choice else -1.0

    for round_idx in range(last_round, 0, -1):
        for agent_id in range(1, edges.agents_per_round[round_idx - 1] + 1):
            from_idx = edges.node_index(round_idx - 1, agent_id)
            outgoing = edges.outgoing_edges(from_idx)
            if not outgoing:
                continue
            scores[from_idx] = sum(weight * scores[target] for target, weight in outgoing) / len(outgoing)

    return scores


def aggregate_agent_scores(
    node_scores: list[float],
    edges: EdgeGraph,
    respondent_ids: list[int],
) -> dict[int, float]:
    """Average temporal node scores for each respondent agent."""
    agent_scores: dict[int, float] = {}
    last_round = len(edges.agents_per_round) - 1
    for agent_id in respondent_ids:
        indices = [
            edges.node_index(0, agent_id),
            edges.node_index(last_round, agent_id),
        ]
        agent_scores[agent_id] = sum(node_scores[i] for i in indices) / len(indices)
    return agent_scores


def detect_malicious(agent_scores: dict[int, float], epsilon: float = 1.5) -> tuple[int | None, float]:
    """Eq. (2): flag the agent with the largest mean score deviation."""
    agents = list(agent_scores.keys())
    if len(agents) < 2:
        return None, 0.0

    best_agent = None
    best_gap = 0.0
    for i in agents:
        gap = sum(abs(agent_scores[i] - agent_scores[j]) for j in agents if j != i) / (len(agents) - 1)
        if gap >= epsilon and gap > best_gap:
            best_gap = gap
            best_agent = i
    return best_agent, best_gap


def run_bpd(
    edges: EdgeGraph,
    answers: list[dict[int, list[str]]],
    respondent_ids: list[int],
    epsilon: float,
) -> dict:
    last_round = len(answers) - 1
    final_texts = [answers[last_round][i][-1] for i in respondent_ids if answers[last_round][i]]
    final_choice = majority_choice(final_texts)
    final_round_answers = {i: answers[last_round][i][-1] for i in respondent_ids if answers[last_round][i]}

    node_scores = backward_propagate(edges, final_round_answers, final_choice)
    agent_scores = aggregate_agent_scores(node_scores, edges, respondent_ids)
    detected_agent, deviation = detect_malicious(agent_scores, epsilon)

    return {
        "final_choice": final_choice,
        "node_scores": node_scores,
        "agent_scores": agent_scores,
        "detected_agent": detected_agent,
        "deviation": deviation,
    }
