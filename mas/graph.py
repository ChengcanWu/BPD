"""Signed DAG for MAS communication."""

from mas.llm import ask, get_edge_score


class ChatTurn:
    def __init__(self, agent_id: int, prompt: str, context: dict, score_inputs: dict):
        self.agent_id = agent_id
        self.prompt = prompt
        self.context = context
        self.score_inputs = score_inputs

    def run(self) -> tuple[str, dict[int, int]]:
        user_text = "".join(f"{key}:\n{self.context[key]}\n" for key in self.context)
        answer = ask(self.prompt, user_text)
        scores = {}
        for sender_id, sender_answer in self.score_inputs.items():
            scores[sender_id] = get_edge_score(answer, sender_answer)
        return answer, scores


class EdgeGraph:
    def __init__(self, agents_per_round: list[int]):
        self.agents_per_round = agents_per_round
        self.total_nodes = sum(agents_per_round)
        # connections[receiver][sender] = edge score
        self.connections = [
            [0 for _ in range(self.total_nodes)] for _ in range(self.total_nodes)
        ]

    def node_index(self, round_idx: int, agent_id: int) -> int:
        return sum(self.agents_per_round[:round_idx]) + agent_id - 1

    def update_edge(self, sender_round: int, sender_id: int, receiver_round: int, receiver_id: int, score: int):
        if sender_round < 0:
            return
        row = self.node_index(receiver_round, receiver_id)
        col = self.node_index(sender_round, sender_id)
        self.connections[row][col] = score

    def outgoing_edges(self, sender_idx: int) -> list[tuple[int, int]]:
        pairs = []
        for row in range(self.total_nodes):
            score = self.connections[row][sender_idx]
            if score != 0:
                pairs.append((row, score))
        return pairs


def run_round(
    discussion: list[ChatTurn],
    answers: list[dict[int, list[str]]],
    edges: EdgeGraph,
    round_idx: int,
    sender_round: int,
) -> None:
    for turn in discussion:
        answer, edge_scores = turn.run()
        answers[round_idx][turn.agent_id].append(answer)
        for sender_id, score in edge_scores.items():
            edges.update_edge(sender_round, sender_id, round_idx, turn.agent_id, score)
