"""Sheriff of Nottingham game state."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import random

from .config import SheriffConfig
from .types import CardDef, Phase, PlayerState, Offer


@dataclass
class SheriffState:
    """Complete game state for Sheriff of Nottingham.
    
    Attributes:
        config: Game configuration
        rng: Random number generator
        deck: List of card IDs in draw pile
        discard_left: Left discard pile (face-up)
        discard_right: Right discard pile (face-up)
        card_defs: List of all CardDef objects
        players: List of player states
        sheriff_idx: Index of current sheriff
        rotation_counts: Number of times each player has been sheriff
        phase: Current game phase
        round_step: Which merchant is active in current phase (index offset from sheriff)
        offers: Bribe offers by merchant player ID
        negotiation_round: Current negotiation round (0-indexed)
        inspected_merchants: Set of merchant IDs already inspected this round
        game_over: Whether the game has ended
        winner: Player ID of the winner (or None)
    """

    config: SheriffConfig = field(default_factory=SheriffConfig)
    rng: Optional[random.Random] = None
    
    # Deck and cards
    deck: List[int] = field(default_factory=list)
    discard_left: List[int] = field(default_factory=list)
    discard_right: List[int] = field(default_factory=list)
    card_defs: List[CardDef] = field(default_factory=list)
    
    # Players
    players: List[PlayerState] = field(default_factory=list)
    
    # Game state
    sheriff_idx: int = 0
    rotation_counts: List[int] = field(default_factory=list)
    round_number: int = 0  # Increments with each full sheriff rotation cycle
    phase: Phase = Phase.MARKET
    round_step: int = 0  # Merchant offset from sheriff
    
    # Negotiation state
    offers: Dict[int, Offer] = field(default_factory=dict)
    negotiation_round: int = 0
    sheriff_responses: Dict[int, Dict] = field(default_factory=dict)  # merchant_id -> response dict
    
    # Merchant queue system (replaces offset math)
    merchant_queue: List[int] = field(default_factory=list)
    
    # Inspection state
    inspected_merchants: set = field(default_factory=set)
    
    # Refund tracking (for idempotent refunds)
    refunded: set = field(default_factory=set)  # (sheriff_id, merchant_id) tuples
    
    # Timeout tracking
    phase_start_time: float = field(default_factory=lambda: 0.0)
    
    # Game history tracking
    game_history: List[str] = field(default_factory=list)
    
    # Inspect phase tracking
    inspect_queue: List[int] = field(default_factory=list)  # Separate queue for inspection phase
    
    # End game
    game_over: bool = False
    winner: Optional[int] = None
    
    def top_discard_choices(self) -> Dict[str, Optional[int]]:
        """Get the top cards from each discard pile."""
        return {
            "left": self.discard_left[-1] if self.discard_left else None,
            "right": self.discard_right[-1] if self.discard_right else None,
        }
    
    def get_player(self, pid: int) -> PlayerState:
        """Get player state by ID."""
        return self.players[pid]
    
    def get_merchant_idx(self, offset: int = 0) -> int:
        """Get merchant index at given offset from sheriff."""
        return (self.sheriff_idx + 1 + offset) % self.config.n_players
    
    def is_sheriff(self, pid: int) -> bool:
        """Check if player is the current sheriff."""
        return pid == self.sheriff_idx
    
    def get_all_merchants(self) -> List[int]:
        """Get list of all merchant player IDs (excluding sheriff)."""
        return [i for i in range(self.config.n_players) if i != self.sheriff_idx]
    
    def start_merchant_cycle(self):
        """Initialize the merchant queue for the current round."""
        self.merchant_queue = [(self.sheriff_idx + i) % self.config.n_players
                               for i in range(1, self.config.n_players)]
        # Clear refund tracking for new round
        self.refunded.clear()
    
    def next_merchant(self) -> Optional[int]:
        """Get the next merchant in the queue."""
        return self.merchant_queue[0] if self.merchant_queue else None
    
    def finish_current_merchant(self):
        """Finish the current merchant and remove from queue."""
        if self.merchant_queue:
            m = self.merchant_queue.pop(0)
            # Clean up responses for this merchant
            self.sheriff_responses.pop(m, None)
            return m
        return None
    
    def start_inspect_cycle(self):
        """Initialize inspect queue at the start of inspection phase."""
        self.inspect_queue = [(self.sheriff_idx + i) % self.config.n_players
                              for i in range(1, self.config.n_players)]
        self.inspected_merchants.clear()
    
    def current_inspect_merchant(self) -> Optional[int]:
        """Get the current merchant to inspect (doesn't remove from queue)."""
        return self.inspect_queue[0] if self.inspect_queue else None
    
    def finish_inspect_merchant(self):
        """Mark current merchant as inspected and move to next."""
        if self.inspect_queue:
            m = self.inspect_queue.pop(0)
            self.inspected_merchants.add(m)
            return m
        return None
    
    def should_rotate_sheriff(self) -> bool:
        """Check if sheriff should rotate to next player."""
        return all(count >= 1 for count in self.rotation_counts)
    
    def rotate_sheriff(self):
        """Rotate sheriff to next player."""
        self.rotation_counts[self.sheriff_idx] += 1
        self.sheriff_idx = (self.sheriff_idx + 1) % self.config.n_players
        
        # Check if game should end
        if self.rotation_counts[self.sheriff_idx] >= self.config.sheriff_rotations:
            self.game_over = True
    
    def get_card_def(self, card_id: int) -> CardDef:
        """Get card definition by ID."""
        if card_id is None:
            raise ValueError(f"Cannot get card definition for None card_id")
        if not isinstance(card_id, int):
            raise TypeError(f"Card ID must be int, got {type(card_id)}: {card_id}")
        if card_id < 0 or card_id >= len(self.card_defs):
            raise IndexError(f"Card ID {card_id} out of range (deck has {len(self.card_defs)} cards)")
        return self.card_defs[card_id]
    
    def get_formatted_history(self) -> str:
        """Format game history for display.
        
        Returns:
            Formatted string of game history events
        """
        if not self.game_history:
            return "   (No history yet - first round)"
        
        return "\n".join(f"   {i+1}. {event}" for i, event in enumerate(self.game_history))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "config": {
                "n_players": self.config.n_players,
                "seed": self.config.seed,
                "include_royal": self.config.include_royal,
                "hand_size": self.config.hand_size,
                "bag_limit": self.config.bag_limit,
                "sheriff_rotations": self.config.sheriff_rotations,
            },
            "deck_size": len(self.deck),
            "discard_left_size": len(self.discard_left),
            "discard_right_size": len(self.discard_right),
            "players": [
                {
                    "pid": p.pid,
                    "gold": p.gold,
                    "hand_size": len(p.hand),
                    "bag_size": len(p.bag),
                    "declared_type": p.declared_type.value if p.declared_type else None,
                    "declared_count": p.declared_count,
                }
                for p in self.players
            ],
            "sheriff_idx": self.sheriff_idx,
            "rotation_counts": self.rotation_counts,
            "phase": self.phase.value,
            "round_step": self.round_step,
            "game_over": self.game_over,
            "winner": self.winner,
        }

