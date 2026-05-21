"""BPD multi-agent system package."""

from mas.bpd import backward_propagate, aggregate_agent_scores, detect_malicious
from mas.runner import communicate, communicate_flat, communicate_hierarchy, communicate_with_bpd

__all__ = [
    "backward_propagate",
    "aggregate_agent_scores",
    "detect_malicious",
    "communicate",
    "communicate_flat",
    "communicate_hierarchy",
    "communicate_with_bpd",
]
