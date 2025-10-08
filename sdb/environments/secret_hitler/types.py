"""Secret Hitler game types and enums."""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional


class Party(Enum):
    """Political party affiliation."""
    LIBERAL = auto()
    FASCIST = auto()


class Role(Enum):
    """Player roles in Secret Hitler."""
    LIBERAL = auto()
    FASCIST = auto()
    HITLER = auto()


class Policy(Enum):
    """Policy types."""
    LIBERAL = auto()
    FASCIST = auto()


class Phase(Enum):
    """Game phases."""
    ELECTION_NOMINATION = auto()
    ELECTION_DISCUSSION = auto()
    ELECTION_VOTING = auto()
    LEGISLATIVE_SESSION = auto()
    PRESIDENTIAL_POWER = auto()
    VETO_DISCUSSION = auto()
    GAME_OVER = auto()


class PresidentialPower(Enum):
    """Presidential powers unlocked by Fascist policies."""
    NONE = auto()
    INVESTIGATE_LOYALTY = auto()
    CALL_SPECIAL_ELECTION = auto()
    POLICY_PEEK = auto()
    EXECUTION = auto()


@dataclass
class PlayerInfo:
    """Information about a player."""
    player_id: int
    role: Role
    party: Party
    is_alive: bool = True
    is_hitler: bool = False
    
    def is_fascist(self) -> bool:
        """Check if player is on fascist team."""
        return self.party == Party.FASCIST
    
    def is_liberal(self) -> bool:
        """Check if player is on liberal team."""
        return self.party == Party.LIBERAL


@dataclass
class Government:
    """Elected government (President + Chancellor)."""
    president: int
    chancellor: int


@dataclass
class Vote:
    """Vote in an election."""
    voter: int
    ja: bool  # True for Ja (yes), False for Nein (no)

