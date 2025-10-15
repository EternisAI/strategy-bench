"""Avalon type definitions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Role(str, Enum):
    """Avalon roles."""
    # Good roles
    MERLIN = "merlin"
    PERCIVAL = "percival"
    SERVANT = "servant"
    
    # Evil roles
    MORGANA = "morgana"
    MORDRED = "mordred"
    OBERON = "oberon"
    ASSASSIN = "assassin"
    MINION = "minion"


class Team(str, Enum):
    """Team affiliations."""
    GOOD = "good"
    EVIL = "evil"


class Phase(str, Enum):
    """Game phases."""
    TEAM_SELECTION = "team_selection"
    TEAM_VOTING = "team_voting"
    QUEST_VOTING = "quest_voting"
    ASSASSINATION = "assassination"
    GAME_END = "game_end"


class VoteChoice(str, Enum):
    """Vote choices."""
    APPROVE = "approve"
    REJECT = "reject"


class QuestChoice(str, Enum):
    """Quest voting choices."""
    SUCCESS = "success"
    FAIL = "fail"


# Role definitions
GOOD_ROLES = {Role.MERLIN, Role.PERCIVAL, Role.SERVANT}
EVIL_ROLES = {Role.MORGANA, Role.MORDRED, Role.OBERON, Role.ASSASSIN, Role.MINION}


def get_team(role: Role) -> Team:
    """Get team for a role."""
    if role in GOOD_ROLES:
        return Team.GOOD
    return Team.EVIL


@dataclass
class PlayerState:
    """State for a single player."""
    pid: int
    role: Role
    team: Team
    is_alive: bool = True


@dataclass
class QuestResult:
    """Result of a quest."""
    quest_num: int  # 0-indexed
    team_members: List[int]
    success_votes: int
    fail_votes: int
    succeeded: bool


@dataclass
class TeamProposal:
    """A proposed quest team."""
    leader: int
    team: List[int]
    approved: Optional[bool] = None
    approve_votes: int = 0
    reject_votes: int = 0


# Quest configuration by player count
# Format: {num_players: [num_good, num_evil]}
TEAM_COMPOSITION = {
    5: [3, 2],
    6: [4, 2],
    7: [4, 3],
    8: [5, 3],
    9: [6, 3],
    10: [6, 4],
}

# Quest team sizes by player count
# Format: {num_players: [quest1_size, quest2_size, quest3_size, quest4_size, quest5_size]}
QUEST_SIZES = {
    5: [2, 3, 2, 3, 3],
    6: [2, 3, 4, 3, 4],
    7: [2, 3, 3, 4, 4],
    8: [3, 4, 4, 5, 5],
    9: [3, 4, 4, 5, 5],
    10: [3, 4, 4, 5, 5],
}

# Number of fails needed for quest to fail
# Format: {num_players: [quest1_fails, quest2_fails, quest3_fails, quest4_fails, quest5_fails]}
QUEST_FAILS_NEEDED = {
    5: [1, 1, 1, 1, 1],
    6: [1, 1, 1, 1, 1],
    7: [1, 1, 1, 2, 1],  # Quest 4 needs 2 fails
    8: [1, 1, 1, 2, 1],
    9: [1, 1, 1, 2, 1],
    10: [1, 1, 1, 2, 1],
}

# Maximum team rejections before force-approve
MAX_REJECTIONS = 5

