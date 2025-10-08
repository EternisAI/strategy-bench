"""Agent implementations for Social Deduction Bench."""

from sdb.agents.llm.openrouter_agent import OpenRouterAgent
from sdb.agents.baselines.random_agent import RandomAgent

__all__ = [
    "OpenRouterAgent",
    "RandomAgent",
]

