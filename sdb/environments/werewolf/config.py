"""Configuration for Werewolf game."""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class WerewolfConfig:
    """Configuration for Werewolf game.
    
    Attributes:
        n_players: Number of players (5-12 recommended)
        n_werewolves: Number of werewolves (default: ~25% of players)
        include_seer: Whether to include the Seer role
        include_doctor: Whether to include the Doctor role
        max_debate_turns: Maximum number of debate statements per day
        max_rounds: Maximum number of rounds before draw
        debate_retries: Number of retries for failed debate generation
    """
    n_players: int = 7
    n_werewolves: Optional[int] = None  # Auto-calculated if None
    include_seer: bool = True
    include_doctor: bool = True
    max_debate_turns: int = 10
    max_rounds: int = 50
    debate_retries: int = 3
    
    def __post_init__(self):
        """Validate and set default values."""
        if self.n_players < 5:
            raise ValueError("Werewolf requires at least 5 players")
        if self.n_players > 20:
            raise ValueError("Werewolf supports max 20 players")
        
        # Auto-calculate werewolves if not specified
        if self.n_werewolves is None:
            self.n_werewolves = max(1, self.n_players // 4)
        
        # Validate werewolf count
        if self.n_werewolves < 1:
            raise ValueError("Need at least 1 werewolf")
        if self.n_werewolves >= self.n_players:
            raise ValueError("Too many werewolves")
        
        # Validate special roles
        special_roles = int(self.include_seer) + int(self.include_doctor)
        if special_roles + self.n_werewolves >= self.n_players:
            raise ValueError(
                f"Not enough players for {self.n_werewolves} werewolves "
                f"and {special_roles} special roles"
            )

