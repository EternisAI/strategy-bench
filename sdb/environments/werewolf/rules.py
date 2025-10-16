"""Game rules and logic for Werewolf."""

import random
from typing import Dict, List, Tuple

from sdb.environments.werewolf.config import WerewolfConfig
from sdb.environments.werewolf.types import PlayerState, Role, Team


def assign_roles(config: WerewolfConfig) -> List[Role]:
    """Assign roles to players.
    
    Args:
        config: Game configuration
        
    Returns:
        List of roles (one per player)
    """
    roles = []
    
    # Add werewolves
    roles.extend([Role.WEREWOLF] * config.n_werewolves)
    
    # Add special roles
    if config.include_seer:
        roles.append(Role.SEER)
    if config.include_doctor:
        roles.append(Role.DOCTOR)
    
    # Fill remaining with villagers
    n_villagers = config.n_players - len(roles)
    roles.extend([Role.VILLAGER] * n_villagers)
    
    # Shuffle roles
    random.shuffle(roles)
    
    return roles


def get_team_for_role(role: Role) -> Team:
    """Get the team for a given role.
    
    Args:
        role: Player role
        
    Returns:
        Team (VILLAGE or WEREWOLVES)
    """
    if role == Role.WEREWOLF:
        return Team.WEREWOLVES
    return Team.VILLAGE


def check_win_condition(
    alive_werewolves: int,
    alive_villagers: int
) -> Tuple[bool, Team | None, str]:
    """Check if a team has won.
    
    Args:
        alive_werewolves: Number of alive werewolves
        alive_villagers: Number of alive villagers
        
    Returns:
        Tuple of (game_over, winning_team, reason)
    """
    # Werewolves win if they equal or outnumber villagers
    if alive_werewolves >= alive_villagers:
        return True, Team.WEREWOLVES, "Werewolves equal or outnumber villagers"
    
    # Village wins if all werewolves are eliminated
    if alive_werewolves == 0:
        return True, Team.VILLAGE, "All werewolves eliminated"
    
    # Game continues
    return False, None, ""


def get_vote_result(
    votes: Dict[int, int | None],
    n_alive: int,
    require_majority: bool = True
) -> Tuple[int | None, int]:
    """Determine who gets eliminated by vote.
    
    Args:
        votes: Dictionary mapping voter_id -> voted_for_id (None = No-Elimination)
        n_alive: Number of alive players
        require_majority: If True, requires >50% to eliminate (default: True)
        
    Returns:
        Tuple of (eliminated_player_id, vote_count)
        Returns (None, 0) if there's a tie, no majority, or No-Elimination wins
    """
    if not votes:
        return None, 0
    
    # Count votes (None votes count toward No-Elimination)
    vote_counts: Dict[int | None, int] = {}
    for voted_for in votes.values():
        vote_counts[voted_for] = vote_counts.get(voted_for, 0) + 1
    
    # Find max votes
    max_votes = max(vote_counts.values())
    players_with_max = [pid for pid, count in vote_counts.items() if count == max_votes]
    
    # Tie - no elimination
    if len(players_with_max) > 1:
        return None, 0
    
    leader = players_with_max[0]
    
    # Check if No-Elimination won
    if leader is None:
        return None, 0
    
    # Check majority requirement
    if require_majority:
        from math import floor
        required_votes = floor(n_alive / 2) + 1
        if max_votes < required_votes:
            return None, 0  # No elimination - didn't reach majority
    
    return leader, max_votes


def get_max_bidders(bids: Dict[int, int]) -> List[int]:
    """Get all players with the highest bid.
    
    Args:
        bids: Dictionary mapping player_id -> bid_value
        
    Returns:
        List of player IDs with max bid
    """
    if not bids:
        return []
    
    max_bid = max(bids.values())
    return [pid for pid, bid in bids.items() if bid == max_bid]


def validate_night_action(
    action_type: str,
    target: int,
    alive_players: List[int],
    actor_id: int
) -> Tuple[bool, str]:
    """Validate a night action.
    
    Args:
        action_type: Type of action ("eliminate", "protect", "investigate")
        target: Target player ID
        alive_players: List of alive player IDs
        actor_id: ID of the player taking the action
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check target is alive
    if target not in alive_players:
        return False, f"Target {target} is not alive"
    
    # Werewolves can't target themselves
    if action_type == "eliminate" and target == actor_id:
        return False, "Werewolves cannot eliminate themselves"
    
    return True, ""


def validate_vote(
    voter_id: int,
    target_id: int,
    alive_players: List[int]
) -> Tuple[bool, str]:
    """Validate a vote.
    
    Args:
        voter_id: ID of player voting
        target_id: ID of player being voted for
        alive_players: List of alive player IDs
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if voter_id not in alive_players:
        return False, f"Voter {voter_id} is not alive"
    
    if target_id not in alive_players:
        return False, f"Target {target_id} is not alive"
    
    if voter_id == target_id:
        return False, "Cannot vote for yourself"
    
    return True, ""


def validate_bid(bid: int, max_bid: int = 4) -> Tuple[bool, str]:
    """Validate a bid value.
    
    Args:
        bid: Bid value
        max_bid: Maximum allowed bid
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(bid, int):
        return False, f"Bid must be an integer, got {type(bid)}"
    
    if bid < 0:
        return False, f"Bid cannot be negative: {bid}"
    
    if bid > max_bid:
        return False, f"Bid {bid} exceeds maximum {max_bid}"
    
    return True, ""

