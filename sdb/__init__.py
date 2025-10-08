"""Social Deduction Bench - Unified benchmark suite for LLMs in social deduction games."""

__version__ = "0.1.0"

from sdb.core.base_env import BaseEnvironment
from sdb.core.base_agent import BaseAgent
from sdb.core.base_state import BaseState
from sdb.core.types import GamePhase, PlayerRole, ActionType
from sdb.core.exceptions import (
    SDBException,
    InvalidActionError,
    InvalidStateError,
    AgentError,
)

__all__ = [
    "__version__",
    "BaseEnvironment",
    "BaseAgent",
    "BaseState",
    "GamePhase",
    "PlayerRole",
    "ActionType",
    "SDBException",
    "InvalidActionError",
    "InvalidStateError",
    "AgentError",
]

