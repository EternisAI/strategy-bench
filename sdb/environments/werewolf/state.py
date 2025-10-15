"""Game state for Werewolf."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from sdb.core.base_state import BaseState
from sdb.environments.werewolf.types import (
    DayResult,
    NightResult,
    Phase,
    PlayerState,
    Role,
    Team,
)


@dataclass
class WerewolfState(BaseState):
    """Complete state of a Werewolf game.
    
    Attributes:
        players: Dictionary of player states
        phase: Current game phase
        round_number: Current round (1-indexed)
        current_speaker: Player currently speaking in debate (if any)
        night_results: Results from night phases
        day_results: Results from day phases
        winner: Winning team if game is over
    """
    players: Dict[int, PlayerState] = field(default_factory=dict)
    phase: Phase = Phase.NIGHT_WEREWOLF
    round_number: int = 1
    current_speaker: Optional[int] = None
    
    # Results tracking
    night_results: List[NightResult] = field(default_factory=list)
    day_results: List[DayResult] = field(default_factory=list)
    
    # Current round temp data
    current_night: NightResult = field(default_factory=NightResult)
    current_day: DayResult = field(default_factory=DayResult)
    
    # Game outcome
    winner: Optional[Team] = None
    win_reason: str = ""
    
    def get_alive_players(self) -> List[int]:
        """Return list of alive player IDs."""
        return [pid for pid, p in self.players.items() if p.is_alive]
    
    def get_alive_werewolves(self) -> List[int]:
        """Return list of alive werewolf IDs."""
        return [
            pid for pid, p in self.players.items()
            if p.is_alive and p.role == Role.WEREWOLF
        ]
    
    def get_alive_villagers(self) -> List[int]:
        """Return list of alive village team members."""
        return [
            pid for pid, p in self.players.items()
            if p.is_alive and p.team == Team.VILLAGE
        ]
    
    def get_player_by_role(self, role: Role) -> Optional[int]:
        """Get player ID by role."""
        for pid, p in self.players.items():
            if p.role == role:
                return pid
        return None
    
    def is_player_alive(self, player_id: int) -> bool:
        """Check if a player is alive."""
        return player_id in self.players and self.players[player_id].is_alive
    
    def add_observation(self, player_id: int, observation: str):
        """Add an observation to a player's history."""
        if player_id in self.players:
            obs_text = f"Round {self.round_number}: {observation}"
            self.players[player_id].observations.append(obs_text)
    
    def broadcast_observation(self, observation: str, include_dead: bool = False):
        """Add an observation to all (alive) players."""
        for pid, player in self.players.items():
            if include_dead or player.is_alive:
                self.add_observation(pid, observation)

