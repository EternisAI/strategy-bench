"""Log formats and data structures."""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from datetime import datetime

from sdb.core.utils import safe_json_dumps


class EventType(Enum):
    """Types of loggable events."""
    
    # Game lifecycle
    GAME_START = auto()
    GAME_END = auto()
    PHASE_CHANGE = auto()
    ROUND_START = auto()
    ROUND_END = auto()
    
    # Player actions
    PLAYER_ACTION = auto()
    PLAYER_SPEAK = auto()
    PLAYER_VOTE = auto()
    PLAYER_NOMINATE = auto()
    VOTE_CAST = auto()
    ELECTION_RESULT = auto()
    
    # Game events
    PLAYER_ELIMINATED = auto()
    PLAYER_INVESTIGATED = auto()
    INVESTIGATION_RESULT = auto()
    POLICY_ENACTED = auto()
    PRESIDENTIAL_POWER = auto()
    QUEST_RESULT = auto()
    ACCUSATION = auto()
    DISCUSSION = auto()
    VETO_PROPOSED = auto()
    VETO_RESPONSE = auto()
    
    # Agent events
    AGENT_REASONING = auto()
    AGENT_ERROR = auto()
    LLM_CALL = auto()
    
    # System events
    ERROR = auto()
    WARNING = auto()
    INFO = auto()


@dataclass
class LogEntry:
    """Single log entry."""
    
    timestamp: datetime
    event_type: EventType
    game_id: str
    round_number: int
    data: Dict[str, Any] = field(default_factory=dict)
    player_id: Optional[int] = None
    is_private: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.name,
            "game_id": self.game_id,
            "round_number": self.round_number,
            "data": self.data,
            "player_id": self.player_id,
            "is_private": self.is_private,
            "metadata": self.metadata,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return safe_json_dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogEntry":
        """Create from dictionary."""
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        data["event_type"] = EventType[data["event_type"]]
        return cls(**data)

