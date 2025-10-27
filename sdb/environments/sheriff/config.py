"""Sheriff of Nottingham configuration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SheriffConfig:
    """Configuration for Sheriff of Nottingham game.
    
    Args:
        n_players: Number of players (3-5)
        seed: Random seed for reproducibility
        include_royal: Whether to include Royal Goods expansion
        hand_size: Starting hand size
        bag_limit: Maximum cards per bag
        sheriff_rotations: Number of times each player is sheriff (2 for 4-5P, 3 for 3P)
        max_negotiation_rounds: Maximum negotiation rounds per merchant
        max_rounds: Maximum rounds (same as sheriff_rotations)
        starting_coins: Starting gold for each player (not used, default gold is 50)
        legal_goods_payment: Whether to pay for legal goods (always true)
        contraband_penalties: Whether contraband incurs penalties (always true)
        enable_bribes: Whether bribing is enabled (always true)
    """

    n_players: int = 4
    seed: Optional[int] = None
    include_royal: bool = False
    hand_size: int = 6
    bag_limit: int = 5
    sheriff_rotations: int = 2
    max_negotiation_rounds: int = 1
    max_phase_seconds: int = 300  # 5 minutes max per phase
    
    # Additional parameters for tournament compatibility (not actively used)
    max_rounds: Optional[int] = None
    starting_coins: int = 50
    legal_goods_payment: bool = True
    contraband_penalties: bool = True
    enable_bribes: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.n_players < 3 or self.n_players > 5:
            raise ValueError(f"Sheriff supports 3-5 players, got {self.n_players}")
        if self.hand_size < 1:
            raise ValueError(f"hand_size must be positive, got {self.hand_size}")
        if self.bag_limit < 1 or self.bag_limit > 5:
            raise ValueError(f"bag_limit must be 1-5, got {self.bag_limit}")
        if self.sheriff_rotations < 1:
            raise ValueError(f"sheriff_rotations must be positive, got {self.sheriff_rotations}")
        
        # Sync max_rounds with sheriff_rotations if provided
        if self.max_rounds is not None and self.max_rounds != self.sheriff_rotations:
            self.sheriff_rotations = self.max_rounds
        
        # Default sheriff rotations based on player count
        if self.n_players == 3 and self.sheriff_rotations == 2:
            self.sheriff_rotations = 3  # 3 players need 3 rotations

