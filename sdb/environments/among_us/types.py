"""Type definitions for simplified Among Us game."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class PlayerRole(Enum):
    """Player roles in Among Us."""
    CREWMATE = "crewmate"
    IMPOSTOR = "impostor"


class Phase(Enum):
    """Game phases."""
    TASK = "task"  # Task completion and impostor actions
    EMERGENCY = "emergency"  # Emergency meeting called
    DISCUSSION = "discussion"  # Discussing after body report or emergency
    VOTING = "voting"  # Voting to eject
    GAME_END = "game_end"


@dataclass
class PlayerState:
    """State of a single player."""
    player_id: int
    name: str
    role: PlayerRole
    is_alive: bool = True
    tasks_completed: int = 0
    total_tasks: int = 3
    has_called_emergency: bool = False
    
    # Spatial attributes
    location: str = "Cafeteria"  # Current room name
    assigned_tasks: List[tuple[str, str]] = field(default_factory=list)  # [(task_name, room_name), ...]
    
    def task_progress(self) -> float:
        """Get task completion progress (0.0 to 1.0)."""
        return self.tasks_completed / self.total_tasks if self.total_tasks > 0 else 0.0


@dataclass
class TaskRoundResult:
    """Result of a task round."""
    tasks_completed: List[int] = field(default_factory=list)  # Player IDs who completed tasks
    kills: List[tuple[int, int]] = field(default_factory=list)  # (killer_id, victim_id)
    body_reported: Optional[tuple[int, int]] = None  # (reporter_id, victim_id) if body found
    emergency_called: Optional[int] = None  # Player ID who called emergency


@dataclass
class MeetingResult:
    """Result of a meeting (discussion + voting)."""
    discussion_statements: List[tuple[int, str]] = field(default_factory=list)  # (player_id, statement)
    votes: dict[int, int] = field(default_factory=dict)  # voter_id -> voted_for_id
    ejected: Optional[int] = None  # Player ID ejected (if any)
    skipped: bool = False  # True if vote skipped

