"""Game rules and logic for Among Us."""

import random
from typing import Dict, List, Tuple

from sdb.environments.among_us.config import AmongUsConfig
from sdb.environments.among_us.types import PlayerRole, PlayerState


def assign_roles(config: AmongUsConfig, rng: random.Random) -> List[PlayerRole]:
    """Assign roles to players.
    
    Args:
        config: Game configuration
        rng: Random number generator
        
    Returns:
        List of roles (one per player)
    """
    roles = []
    
    # Add impostors
    roles.extend([PlayerRole.IMPOSTOR] * config.n_impostors)
    
    # Fill remaining with crewmates
    n_crewmates = config.n_players - config.n_impostors
    roles.extend([PlayerRole.CREWMATE] * n_crewmates)
    
    # Shuffle roles
    rng.shuffle(roles)
    
    return roles


def check_win_condition(
    alive_crewmates: int,
    alive_impostors: int,
    task_completion: float
) -> Tuple[bool, str | None, str]:
    """Check if a team has won.
    
    Args:
        alive_crewmates: Number of alive crewmates
        alive_impostors: Number of alive impostors
        task_completion: Task completion progress (0.0 to 1.0)
        
    Returns:
        Tuple of (game_over, winning_team, reason)
    """
    # Crewmates win if all tasks completed
    if task_completion >= 1.0:
        return True, "crewmates", "All tasks completed"
    
    # Crewmates win if all impostors ejected
    if alive_impostors == 0:
        return True, "crewmates", "All impostors ejected"
    
    # Impostors win if they equal or outnumber crewmates
    if alive_impostors >= alive_crewmates:
        return True, "impostors", "Impostors equal or outnumber crewmates"
    
    # Game continues
    return False, None, ""


def get_vote_result(votes: Dict[int, int]) -> Tuple[int | None, bool]:
    """Determine who gets ejected by vote.
    
    Args:
        votes: Dictionary mapping voter_id -> voted_for_id
        
    Returns:
        Tuple of (ejected_player_id, is_skip)
        Returns (None, True) if majority voted to skip
        Returns (None, False) if there's a tie
    """
    if not votes:
        return None, True
    
    # Count votes
    vote_counts: Dict[int | None, int] = {}
    for voted_for in votes.values():
        vote_counts[voted_for] = vote_counts.get(voted_for, 0) + 1
    
    # Find max votes
    max_votes = max(vote_counts.values())
    players_with_max = [pid for pid, count in vote_counts.items() if count == max_votes]
    
    # Check for skip
    if len(players_with_max) == 1 and players_with_max[0] is None:
        return None, True  # Skipped
    
    # Tie - no ejection
    if len(players_with_max) > 1:
        return None, False  # Tie
    
    ejected = players_with_max[0]
    return ejected, False


def validate_kill(
    killer_id: int,
    target_id: int,
    alive_players: List[int],
    can_kill: bool
) -> Tuple[bool, str]:
    """Validate a kill action.
    
    Args:
        killer_id: ID of impostor attempting kill
        target_id: ID of target player
        alive_players: List of alive player IDs
        can_kill: Whether impostor's cooldown allows killing
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not can_kill:
        return False, "Kill on cooldown"
    
    if target_id not in alive_players:
        return False, f"Target {target_id} is not alive"
    
    if target_id == killer_id:
        return False, "Cannot kill yourself"
    
    return True, ""


def validate_vote(
    voter_id: int,
    target_id: int | None,
    alive_players: List[int]
) -> Tuple[bool, str]:
    """Validate a vote.
    
    Args:
        voter_id: ID of player voting
        target_id: ID of player being voted for (None to skip)
        alive_players: List of alive player IDs
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if voter_id not in alive_players:
        return False, f"Voter {voter_id} is not alive"
    
    if target_id is not None:
        if target_id not in alive_players:
            return False, f"Target {target_id} is not alive"
        
        if voter_id == target_id:
            return False, "Cannot vote for yourself"
    
    return True, ""


def validate_emergency_call(
    caller_id: int,
    player_has_called: bool
) -> Tuple[bool, str]:
    """Validate an emergency meeting call.
    
    Args:
        caller_id: ID of player calling meeting
        player_has_called: Whether this player has already called emergency
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if player_has_called:
        return False, "Already used emergency meeting"
    
    return True, ""

