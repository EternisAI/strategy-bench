"""Avalon game rules and utilities."""

from typing import List, Tuple
import random

from .types import (
    Role, Team, PlayerState, get_team,
    TEAM_COMPOSITION, GOOD_ROLES, EVIL_ROLES,
)
from .config import AvalonConfig


def assign_roles(config: AvalonConfig, rng: random.Random) -> List[PlayerState]:
    """Assign roles to players based on configuration.
    
    Args:
        config: Game configuration
        rng: Random number generator
        
    Returns:
        List of PlayerState objects with assigned roles
    """
    n_players = config.n_players
    
    # If roles are explicitly specified, use them
    if config.roles:
        return [
            PlayerState(pid=i, role=role, team=get_team(role))
            for i, role in enumerate(config.roles)
        ]
    
    # Otherwise, assign roles based on configuration
    num_good, num_evil = TEAM_COMPOSITION[n_players]
    
    # Build list of roles to assign
    good_roles = []
    evil_roles = []
    
    # Add special good roles
    if config.include_merlin:
        good_roles.append(Role.MERLIN)
    if config.include_percival:
        good_roles.append(Role.PERCIVAL)
    
    # Fill remaining good slots with servants
    while len(good_roles) < num_good:
        good_roles.append(Role.SERVANT)
    
    # Add special evil roles
    # Always include one assassin
    evil_roles.append(Role.ASSASSIN)
    
    if config.include_morgana and len(evil_roles) < num_evil:
        evil_roles.append(Role.MORGANA)
    if config.include_mordred and len(evil_roles) < num_evil:
        evil_roles.append(Role.MORDRED)
    if config.include_oberon and len(evil_roles) < num_evil:
        evil_roles.append(Role.OBERON)
    
    # Fill remaining evil slots with minions
    while len(evil_roles) < num_evil:
        evil_roles.append(Role.MINION)
    
    # Combine and shuffle
    all_roles = good_roles + evil_roles
    rng.shuffle(all_roles)
    
    # Create player states
    players = [
        PlayerState(pid=i, role=role, team=get_team(role))
        for i, role in enumerate(all_roles)
    ]
    
    return players


def check_quest_result(
    fail_votes: int,
    fails_needed: int,
) -> bool:
    """Check if a quest succeeded or failed.
    
    Args:
        fail_votes: Number of fail votes cast
        fails_needed: Number of fails needed for quest to fail
        
    Returns:
        True if quest succeeded, False if failed
    """
    return fail_votes < fails_needed


def check_game_end(quests_succeeded: int, quests_failed: int) -> Tuple[bool, Team]:
    """Check if the game has ended and who won.
    
    Args:
        quests_succeeded: Number of quests that succeeded
        quests_failed: Number of quests that failed
        
    Returns:
        Tuple of (game_over, winner)
        - game_over: True if game ended
        - winner: Team.GOOD or Team.EVIL (or None if not over)
    """
    if quests_succeeded >= 3:
        return True, Team.GOOD
    elif quests_failed >= 3:
        return True, Team.EVIL
    else:
        return False, None


def find_assassin(players: List[PlayerState]) -> int:
    """Find the player with the Assassin role.
    
    Args:
        players: List of player states
        
    Returns:
        Player ID of the assassin
    """
    for player in players:
        if player.role == Role.ASSASSIN:
            return player.pid
    # Fallback: return first evil player
    for player in players:
        if player.team == Team.EVIL:
            return player.pid
    return 0


def find_merlin(players: List[PlayerState]) -> int:
    """Find the player with the Merlin role.
    
    Args:
        players: List of player states
        
    Returns:
        Player ID of Merlin (or -1 if not present)
    """
    for player in players:
        if player.role == Role.MERLIN:
            return player.pid
    return -1


def validate_team_proposal(
    team: List[int],
    required_size: int,
    num_players: int,
) -> bool:
    """Validate a proposed quest team.
    
    Args:
        team: List of player IDs on the team
        required_size: Required team size
        num_players: Total number of players
        
    Returns:
        True if valid, False otherwise
    """
    # Check size
    if len(team) != required_size:
        return False
    
    # Check for duplicates
    if len(set(team)) != len(team):
        return False
    
    # Check all IDs are valid
    if any(pid < 0 or pid >= num_players for pid in team):
        return False
    
    return True


def get_role_info_for_player(pid: int, players: List[PlayerState]) -> str:
    """Get role information string for a player (what they know at game start).
    
    Args:
        pid: Player ID
        players: List of all player states
        
    Returns:
        String describing what the player knows
    """
    player = players[pid]
    info_parts = [f"You are {player.role.value.upper()}."]
    
    if player.role == Role.MERLIN:
        evil_players = [
            p.pid for p in players
            if p.team == Team.EVIL and p.role != Role.MORDRED
        ]
        info_parts.append(f"You see these evil players: {evil_players}")
        info_parts.append("(Note: Mordred is hidden from you)")
    
    elif player.role == Role.PERCIVAL:
        merlin_morgana = [
            p.pid for p in players
            if p.role in (Role.MERLIN, Role.MORGANA)
        ]
        info_parts.append(
            f"You see these players as potential Merlin: {merlin_morgana}"
        )
    
    elif player.team == Team.EVIL and player.role != Role.OBERON:
        other_evil = [
            p.pid for p in players
            if p.team == Team.EVIL and p.pid != pid and p.role != Role.OBERON
        ]
        info_parts.append(f"Your evil teammates are: {other_evil}")
        
        # Show roles if known
        role_info = []
        for p in players:
            if p.team == Team.EVIL and p.pid != pid and p.role != Role.OBERON:
                role_info.append(f"Player {p.pid} is {p.role.value}")
        if role_info:
            info_parts.append(" | ".join(role_info))
    
    elif player.role == Role.OBERON:
        info_parts.append("You are alone and do not know the other evil players.")
    
    else:  # Servant
        info_parts.append("You have no special information.")
    
    return " ".join(info_parts)

