"""Common types and enums used across all games."""

from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime


class GamePhase(Enum):
    """Standard game phases across all social deduction games."""
    
    SETUP = auto()
    DAY = auto()
    NIGHT = auto()
    DISCUSSION = auto()
    VOTING = auto()
    ACCUSATION = auto()
    TASK = auto()
    MEETING = auto()
    NOMINATION = auto()
    TEAM_SELECTION = auto()
    QUEST = auto()
    ASSASSINATION = auto()
    POLICY_SELECTION = auto()
    PRESIDENTIAL_POWER = auto()
    TERMINAL = auto()


class PlayerRole(Enum):
    """Common player roles (game-specific roles are defined in each environment)."""
    
    INNOCENT = auto()
    GUILTY = auto()
    GOOD = auto()
    EVIL = auto()
    CREWMATE = auto()
    IMPOSTOR = auto()
    LIBERAL = auto()
    FASCIST = auto()
    HITLER = auto()
    SPY = auto()
    NON_SPY = auto()
    VILLAGER = auto()
    WEREWOLF = auto()
    SEER = auto()
    DOCTOR = auto()


class ActionType(Enum):
    """Common action types across games."""
    
    # Universal actions
    SPEAK = auto()
    VOTE = auto()
    NOMINATE = auto()
    ACCUSE = auto()
    
    # Game-specific common patterns
    MOVE = auto()
    COMPLETE_TASK = auto()
    KILL = auto()
    SABOTAGE = auto()
    REPORT = auto()
    INVESTIGATE = auto()
    PROTECT = auto()
    ELIMINATE = auto()
    ASK_QUESTION = auto()
    ANSWER_QUESTION = auto()
    PROPOSE_TEAM = auto()
    APPROVE_TEAM = auto()
    REJECT_TEAM = auto()
    PASS_POLICY = auto()
    VETO = auto()
    GUESS_LOCATION = auto()
    STOP_CLOCK = auto()


class ObservationType(Enum):
    """Types of observations agents can receive."""
    
    PUBLIC = auto()  # Visible to all players
    PRIVATE = auto()  # Only visible to specific player
    TEAM = auto()  # Visible to team members
    ROLE_SPECIFIC = auto()  # Visible based on role


@dataclass
class Action:
    """Represents a player action in the game."""
    
    player_id: int
    action_type: ActionType
    target: Optional[int] = None
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary."""
        return {
            "player_id": self.player_id,
            "action_type": self.action_type.name,
            "target": self.target,
            "data": self.data,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        """Create action from dictionary."""
        data = data.copy()
        data["action_type"] = ActionType[data["action_type"]]
        if data.get("timestamp"):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class Observation:
    """Represents what a player observes."""
    
    player_id: int
    obs_type: ObservationType
    phase: GamePhase
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert observation to dictionary."""
        return {
            "player_id": self.player_id,
            "obs_type": self.obs_type.name,
            "phase": self.phase.name,
            "data": self.data,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observation":
        """Create observation from dictionary."""
        data = data.copy()
        data["obs_type"] = ObservationType[data["obs_type"]]
        data["phase"] = GamePhase[data["phase"]]
        if data.get("timestamp"):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class GameResult:
    """Represents the outcome of a game."""
    
    game_id: str
    winner: Union[str, List[int]]  # Team name or list of player IDs
    win_reason: str
    num_rounds: int
    duration_seconds: float
    player_stats: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "game_id": self.game_id,
            "winner": self.winner,
            "win_reason": self.win_reason,
            "num_rounds": self.num_rounds,
            "duration_seconds": self.duration_seconds,
            "player_stats": self.player_stats,
            "metadata": self.metadata,
        }


# Type aliases for common patterns
PlayerID = int
TeamID = Union[str, int]
Score = float
Probability = float

