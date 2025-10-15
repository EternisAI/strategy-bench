"""Avalon configuration."""

from dataclasses import dataclass
from typing import Optional, List
from .types import Role


@dataclass
class AvalonConfig:
    """Configuration for Avalon game.
    
    Args:
        n_players: Number of players (5-10)
        seed: Random seed for reproducibility
        roles: Optional list of specific roles to include
        include_merlin: Whether to include Merlin
        include_percival: Whether to include Percival
        include_morgana: Whether to include Morgana
        include_mordred: Whether to include Mordred
        include_oberon: Whether to include Oberon
    """

    n_players: int = 5
    seed: Optional[int] = None
    
    # Role configuration
    roles: Optional[List[Role]] = None
    include_merlin: bool = True
    include_percival: bool = False
    include_morgana: bool = False
    include_mordred: bool = False
    include_oberon: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if self.n_players < 5 or self.n_players > 10:
            raise ValueError(f"Avalon supports 5-10 players, got {self.n_players}")
        
        # If roles are specified, validate them
        if self.roles:
            if len(self.roles) != self.n_players:
                raise ValueError(
                    f"Number of roles ({len(self.roles)}) must match number of players ({self.n_players})"
                )

