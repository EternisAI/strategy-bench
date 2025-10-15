"""Type definitions for Werewolf game."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class Role(Enum):
    """Player roles in Werewolf."""
    VILLAGER = "villager"
    WEREWOLF = "werewolf"
    SEER = "seer"
    DOCTOR = "doctor"


class Team(Enum):
    """Teams in Werewolf."""
    VILLAGE = "village"
    WEREWOLVES = "werewolves"


class Phase(Enum):
    """Game phases."""
    NIGHT_WEREWOLF = "night_werewolf"  # Werewolves choose victim
    NIGHT_DOCTOR = "night_doctor"  # Doctor chooses protection
    NIGHT_SEER = "night_seer"  # Seer investigates a player
    DAY_BIDDING = "day_bidding"  # Players bid to speak
    DAY_DEBATE = "day_debate"  # Selected player speaks
    DAY_VOTING = "day_voting"  # Vote to eliminate
    GAME_END = "game_end"


@dataclass
class PlayerState:
    """State of a single player."""
    player_id: int
    name: str
    role: Role
    team: Team
    is_alive: bool = True
    # Observations this player has accumulated
    observations: List[str] = field(default_factory=list)
    # For bidding to speak
    bidding_rationale: str = ""


@dataclass
class NightResult:
    """Result of night actions."""
    werewolf_target: Optional[int] = None
    doctor_target: Optional[int] = None
    seer_target: Optional[int] = None
    seer_result: Optional[Role] = None  # What the seer learned
    eliminated_player: Optional[int] = None  # Who died (target if not protected)


@dataclass
class DayResult:
    """Result of day phase."""
    debate: List[Tuple[int, str]] = field(default_factory=list)  # (player_id, statement)
    bids: List[Dict[int, int]] = field(default_factory=list)  # Bids per turn
    votes: Dict[int, int] = field(default_factory=dict)  # player_id -> voted_for_player_id
    eliminated_player: Optional[int] = None

