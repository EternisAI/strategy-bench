"""Game state for Spyfall."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# from sdb.core.base_state import BaseState  # Not needed - using plain dataclass
from sdb.environments.spyfall.types import (
    AccusationState,
    FinalVoteState,
    Phase,
    PlayerCard,
    QAPair,
)


@dataclass
class SpyfallState:
    """Complete state of a Spyfall game.
    
    Attributes:
        phase: Current game phase
        turn: Current turn number
        location: The secret location (hidden from spy)
        spy_index: Index of the spy player
        cards: Player cards (role/location assignments)
        qa_history: History of questions and answers
        current_asker: Player who is currently asking
        awaiting_answer_from: Player who needs to answer (if any)
        cannot_ask_back: Player who cannot be asked (to prevent immediate back-and-forth)
        stops_used: Track if each player has used their stop-the-clock
        spy_guess_allowed: Whether spy can still guess location
        accusation: Current accusation state (if any)
        final_vote: Final vote state (if in final voting)
        winner: Winner of the game ("spy" or "non_spy")
        scores: Points earned by each player
    """
    phase: Phase = Phase.QANDA
    turn: int = 0
    location: str = ""
    spy_index: int = 0
    cards: Dict[int, PlayerCard] = field(default_factory=dict)
    qa_history: List[QAPair] = field(default_factory=list)
    
    # Q&A phase tracking
    current_asker: int = 0
    awaiting_answer_from: Optional[int] = None
    cannot_ask_back: Optional[int] = None
    
    # Action tracking
    stops_used: Dict[int, bool] = field(default_factory=dict)
    spy_guess_allowed: bool = True
    
    # Voting state
    accusation: Optional[AccusationState] = None
    final_vote: Optional[FinalVoteState] = None
    
    # Game outcome
    winner: Optional[str] = None  # "spy" or "non_spy"
    scores: Dict[int, int] = field(default_factory=dict)
    win_reason: str = ""
    
    def is_spy(self, player_id: int) -> bool:
        """Check if a player is the spy."""
        return player_id == self.spy_index
    
    def get_player_location(self, player_id: int) -> Optional[str]:
        """Get a player's location (None if spy)."""
        if player_id in self.cards:
            return self.cards[player_id].location
        return None
    
    def get_player_role(self, player_id: int) -> Optional[str]:
        """Get a player's role (None if spy)."""
        if player_id in self.cards:
            return self.cards[player_id].role
        return None
    
    def add_qa(self, asker: int, answerer: int, question: str, answer: str):
        """Add a Q&A pair to history."""
        self.qa_history.append(
            QAPair(
                turn=self.turn,
                asker=asker,
                answerer=answerer,
                question=question,
                answer=answer
            )
        )
    
    def can_stop_clock(self, player_id: int) -> bool:
        """Check if a player can stop the clock."""
        return player_id in self.stops_used and not self.stops_used[player_id]
    
    def can_spy_guess(self) -> bool:
        """Check if spy can guess location."""
        return self.spy_guess_allowed and not any(self.stops_used.values())

