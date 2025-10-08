"""Tournament framework for running multi-game experiments."""

from sdb.tournament.base import BaseTournament, TournamentConfig, TournamentResult, GameRecord
from sdb.tournament.round_robin import RoundRobinTournament
from sdb.tournament.swiss import SwissTournament
from sdb.tournament.manager import TournamentManager

__all__ = [
    "BaseTournament",
    "TournamentConfig",
    "TournamentResult",
    "GameRecord",
    "RoundRobinTournament",
    "SwissTournament",
    "TournamentManager",
]

