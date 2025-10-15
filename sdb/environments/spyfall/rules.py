"""Game rules and logic for Spyfall."""

import random
from typing import Dict, List, Tuple

from sdb.environments.spyfall.config import SpyfallConfig
from sdb.environments.spyfall.types import PlayerCard


def assign_roles(
    config: SpyfallConfig,
    rng: random.Random
) -> Tuple[str, int, Dict[int, PlayerCard]]:
    """Assign location, spy, and roles to players.
    
    Args:
        config: Game configuration
        rng: Random number generator
        
    Returns:
        Tuple of (location, spy_index, cards)
    """
    # Choose location
    location = rng.choice(config.locations)
    roles = config.roles_by_location[location]
    
    # Choose spy
    spy_index = rng.randrange(config.n_players)
    
    # Assign cards
    cards = {}
    non_spy_indices = [i for i in range(config.n_players) if i != spy_index]
    
    # Sample roles with replacement if needed
    for pid in range(config.n_players):
        if pid == spy_index:
            cards[pid] = PlayerCard(is_spy=True, location=None, role=None)
        else:
            role = rng.choice(roles) if roles else None
            cards[pid] = PlayerCard(is_spy=False, location=location, role=role)
    
    return location, spy_index, cards


def calculate_scores(
    n_players: int,
    spy_index: int,
    winner: str,
    spy_guessed_location: bool = False
) -> Dict[int, int]:
    """Calculate scores based on game outcome.
    
    Scoring (per Spyfall rulebook):
    - If spy identified: Non-spy players get 1 point each, spy gets 0
    - If spy not identified AND guesses location correctly: Spy gets 2 points, others get 0
    - If time runs out without identifying spy: Spy gets 1 point, others get 0
    
    Args:
        n_players: Number of players
        spy_index: Index of spy player
        winner: "spy" or "non_spy"
        spy_guessed_location: Whether spy correctly guessed location
        
    Returns:
        Dictionary mapping player_id to score
    """
    scores = {}
    
    if winner == "spy":
        # Spy wins
        if spy_guessed_location:
            # Spy guessed location correctly
            for pid in range(n_players):
                scores[pid] = 2 if pid == spy_index else 0
        else:
            # Time ran out, spy not identified
            for pid in range(n_players):
                scores[pid] = 1 if pid == spy_index else 0
    else:
        # Non-spy players win
        for pid in range(n_players):
            scores[pid] = 0 if pid == spy_index else 1
    
    return scores


def validate_question_target(
    asker: int,
    target: int,
    n_players: int,
    cannot_ask_back: int = None
) -> Tuple[bool, str]:
    """Validate a question target.
    
    Args:
        asker: Player asking the question
        target: Target player to ask
        n_players: Number of players
        cannot_ask_back: Player who cannot be asked (if any)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not (0 <= target < n_players):
        return False, f"Invalid target {target}"
    
    if target == asker:
        return False, "Cannot ask yourself"
    
    if cannot_ask_back is not None and target == cannot_ask_back:
        return False, "Cannot immediately ask back to previous asker"
    
    return True, ""


def validate_accusation_target(
    accuser: int,
    suspect: int,
    n_players: int
) -> Tuple[bool, str]:
    """Validate an accusation target.
    
    Args:
        accuser: Player making accusation
        suspect: Suspected spy
        n_players: Number of players
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not (0 <= suspect < n_players):
        return False, f"Invalid suspect {suspect}"
    
    if suspect == accuser:
        return False, "Cannot accuse yourself"
    
    return True, ""


def validate_spy_guess(
    guess: str,
    actual_location: str,
    valid_locations: List[str]
) -> Tuple[bool, bool, str]:
    """Validate a spy's location guess.
    
    Args:
        guess: Guessed location
        actual_location: The actual location
        valid_locations: List of valid location names
        
    Returns:
        Tuple of (is_valid_guess, is_correct, error_message)
    """
    if guess not in valid_locations:
        return False, False, f"Invalid location: {guess}"
    
    is_correct = guess == actual_location
    return True, is_correct, ""


def get_voting_result(votes: Dict[int, bool]) -> bool:
    """Determine if a vote passes (unanimous yes).
    
    Args:
        votes: Dictionary mapping voter_id -> yes/no
        
    Returns:
        True if all votes are yes, False otherwise
    """
    if not votes:
        return False
    
    return all(votes.values())

