"""Type definitions for Spyfall game."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class Phase(Enum):
    """Game phases in Spyfall."""
    QANDA = "qanda"  # Question and answer phase
    ACCUSATION_VOTE = "accusation_vote"  # Voting on an accusation
    FINAL_VOTE = "final_vote"  # Final round voting
    SPY_GUESS = "spy_guess"  # Spy is guessing the location
    GAME_END = "game_end"


@dataclass
class PlayerCard:
    """A player's secret card."""
    is_spy: bool
    location: Optional[str] = None  # None if spy
    role: Optional[str] = None  # None if spy
    
    def to_dict(self) -> Dict:
        return {
            "is_spy": self.is_spy,
            "location": self.location,
            "role": self.role
        }


@dataclass
class QAPair:
    """A question-answer exchange."""
    turn: int
    asker: int
    answerer: int
    question: str
    answer: str


@dataclass
class AccusationState:
    """State of an ongoing accusation vote."""
    accuser: int
    suspect: int
    voters: List[int]  # All players except suspect
    votes: Dict[int, bool] = field(default_factory=dict)  # voter_id -> yes/no
    
    def is_complete(self) -> bool:
        """Check if all voters have voted."""
        return len(self.votes) >= len(self.voters)
    
    def is_successful(self) -> bool:
        """Check if accusation passed (unanimous yes)."""
        return self.is_complete() and all(self.votes.values())


@dataclass
class FinalVoteState:
    """State of final voting phase."""
    current_nominator: int
    current_suspect: Optional[int] = None
    votes: Dict[int, bool] = field(default_factory=dict)
    nominators_tried: set = field(default_factory=set)
    
    def is_vote_complete(self) -> bool:
        """Check if current vote is complete."""
        if self.current_suspect is None:
            return False
        # All players except suspect should vote
        return len(self.votes) >= (len(self.nominators_tried) + 1)
    
    def is_vote_successful(self) -> bool:
        """Check if current vote passed (unanimous yes)."""
        return self.is_vote_complete() and all(self.votes.values())

