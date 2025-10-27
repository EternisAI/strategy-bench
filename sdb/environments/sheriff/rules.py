"""Sheriff of Nottingham game rules and utilities."""

from typing import List, Dict, Tuple, Optional
from collections import Counter
from dataclasses import dataclass

from .types import (
    CardDef,
    CardKind,
    LegalType,
    PlayerState,
    LEGAL_DEFAULTS,
    CONTRABAND_DEFAULTS,
    ROYAL_DEFAULTS,
    KING_BONUS,
    QUEEN_BONUS,
)


@dataclass
class InspectionOutcome:
    """Result of inspecting a merchant's bag."""
    truthful: bool
    confiscated: List[int]  # Card IDs confiscated to discard
    delivered: List[int]  # Card IDs delivered to stand
    sheriff_delta: int  # Gold change for sheriff (can be negative)
    merchant_delta: int  # Gold change for merchant (can be negative)


def build_deck(include_royal: bool = False, num_players: int = 4) -> List[CardDef]:
    """Build a deck of cards matching the official rulebook distribution.
    
    Args:
        include_royal: Include Royal Goods variant cards
        num_players: Number of players (3-5); cards marked "4+" are removed for 3P
        
    Returns:
        List of CardDef objects
    """
    deck: List[CardDef] = []
    
    # Add legal goods
    for lt, spec in LEGAL_DEFAULTS.items():
        for _ in range(spec["count"]):
            deck.append(
                CardDef(
                    name=lt.value,
                    kind=CardKind.LEGAL,
                    value=spec["value"],
                    penalty=spec["penalty"],
                )
            )
    
    # Add contraband
    for name, spec in CONTRABAND_DEFAULTS.items():
        for _ in range(spec["count"]):
            deck.append(
                CardDef(
                    name=name,
                    kind=CardKind.CONTRABAND,
                    value=spec["value"],
                    penalty=spec["penalty"],
                )
            )
    
    # Add royal goods if enabled
    if include_royal:
        for name, spec in ROYAL_DEFAULTS.items():
            # Skip 4+ cards in 3-player games
            if num_players < int(spec.get("min_players", 3)):
                continue
            for _ in range(int(spec["count"])):
                deck.append(
                    CardDef(
                        name=name,
                        kind=CardKind.ROYAL,
                        value=int(spec["value"]),
                        penalty=int(spec["penalty"]),
                        counts_as=spec["counts_as"],
                        counts_as_n=int(spec["counts_as_n"]),
                    )
                )
    
    return deck


def is_declaration_truthful(
    bag_card_defs: List[CardDef],
    declared_type: LegalType,
    declared_count: int,
) -> bool:
    """Check if a declaration matches the actual bag contents.
    
    Args:
        bag_card_defs: List of CardDef objects in the bag
        declared_type: The legal type declared
        declared_count: The count declared
        
    Returns:
        True if the declaration is truthful
    """
    # Count cards of the declared type
    count = 0
    for card in bag_card_defs:
        if card.kind == CardKind.LEGAL and card.name == declared_type.value:
            count += 1
    
    # Must match exactly
    return count == len(bag_card_defs) and count == declared_count


def calculate_inspection_penalty(
    bag_card_defs: List[CardDef],
    declared_type: LegalType,
    declared_count: int,
) -> int:
    """Calculate the penalty for an inspection.
    
    If merchant is truthful: Sheriff pays merchant (penalty for each card).
    If merchant is lying: Merchant pays Sheriff (penalty for each contraband/illegal card).
    
    Args:
        bag_card_defs: List of CardDef objects in the bag
        declared_type: The legal type declared
        declared_count: The count declared
        
    Returns:
        Penalty amount (positive means merchant receives, negative means merchant pays)
    """
    truthful = is_declaration_truthful(bag_card_defs, declared_type, declared_count)
    
    if truthful:
        # Sheriff pays merchant: penalty for each card
        return sum(card.penalty for card in bag_card_defs)
    else:
        # Merchant pays Sheriff: penalty for each contraband/illegal card
        penalty = 0
        for card in bag_card_defs:
            # If it's not the declared legal type, it's contraband or wrong legal
            if not (card.kind == CardKind.LEGAL and card.name == declared_type.value):
                penalty += card.penalty
        return -penalty  # Negative because merchant pays


def classify_bag(bag_card_defs: List[CardDef], declared_type: Optional[LegalType] = None, declared_count: Optional[int] = None) -> str:
    """Classify bag with deterministic taxonomy for analytics.
    
    Args:
        bag_card_defs: List of CardDef objects in the bag
        declared_type: The declared legal type (if any)
        declared_count: The declared count (if any)
        
    Returns:
        Classification string:
        - pure_declared: all cards match declaration
        - mixed_legal: legal goods but not all match declaration
        - has_contraband_low: exactly 1 contraband
        - has_contraband_high: 2+ contraband
    """
    contraband_count = 0
    declared_matching_count = 0
    legal_count = 0
    
    for card in bag_card_defs:
        if card.kind == CardKind.LEGAL:
            legal_count += 1
            # Check if card name matches declared type's value (e.g., "apples" == LegalType.APPLES.value)
            if declared_type and card.name == declared_type.value:
                declared_matching_count += 1
        elif card.kind in (CardKind.CONTRABAND, CardKind.ROYAL):
            contraband_count += 1
    
    # Classification logic
    if contraband_count == 0:
        # No contraband - check if all match declaration
        if (declared_type is not None and 
            declared_count is not None and 
            declared_matching_count == len(bag_card_defs) == declared_count):
            return "pure_declared"
        else:
            return "mixed_legal"
    elif contraband_count == 1:
        return "has_contraband_low"
    else:  # contraband_count >= 2
        return "has_contraband_high"


def calculate_king_queen_bonuses(
    players: List[PlayerState], card_defs: List[CardDef]
) -> Dict[int, int]:
    """Calculate King/Queen bonuses for end-game scoring.
    
    Players with the most and 2nd-most of each legal good type get bonuses.
    Royal goods count as their corresponding legal type (with multipliers).
    
    Args:
        players: List of PlayerState objects
        card_defs: List of all CardDef objects
        
    Returns:
        Dict mapping player_id to bonus gold
    """
    bonuses = {p.pid: 0 for p in players}
    
    for legal_type in LegalType:
        # Count each player's goods of this type (including Royal goods)
        counts = {}
        for p in players:
            count = 0
            # Count legal goods on stand
            for card_id in p.stand_legal[legal_type]:
                count += 1
            # Count Royal goods that count as this type
            for card_id in p.stand_contraband:
                card = card_defs[card_id]
                if card.kind == CardKind.ROYAL and card.counts_as == legal_type:
                    count += card.counts_as_n  # Royal goods count as 2 or 3
            # Also check legal stand for Royal goods (shouldn't happen but be safe)
            for lt in LegalType:
                for card_id in p.stand_legal[lt]:
                    card = card_defs[card_id]
                    if card.kind == CardKind.ROYAL and card.counts_as == legal_type:
                        count += card.counts_as_n
            counts[p.pid] = count
        
        # Find top 2 players (handle ties)
        sorted_counts = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        
        if len(sorted_counts) >= 1 and sorted_counts[0][1] > 0:
            # King bonus
            king_pid = sorted_counts[0][0]
            king_count = sorted_counts[0][1]
            bonuses[king_pid] += KING_BONUS[legal_type]
            
            # Queen bonus (must be less than King)
            if len(sorted_counts) >= 2:
                queen_count = sorted_counts[1][1]
                if queen_count > 0 and queen_count < king_count:
                    queen_pid = sorted_counts[1][0]
                    bonuses[queen_pid] += QUEEN_BONUS[legal_type]
    
    return bonuses


def calculate_final_scores(
    players: List[PlayerState], card_defs: List[CardDef]
) -> Dict[int, int]:
    """Calculate final scores for all players.
    
    Score = gold + card values + King/Queen bonuses
    
    Args:
        players: List of PlayerState objects
        card_defs: List of all CardDef objects
        
    Returns:
        Dict mapping player_id to final score
    """
    scores = {}
    
    # Calculate King/Queen bonuses first
    bonuses = calculate_king_queen_bonuses(players, card_defs)
    
    for p in players:
        score = p.gold
        
        # Add card values from stand
        for legal_type in LegalType:
            for card_id in p.stand_legal[legal_type]:
                score += card_defs[card_id].value
        for card_id in p.stand_contraband:
            score += card_defs[card_id].value
        
        # Add bonuses
        score += bonuses[p.pid]
        
        scores[p.pid] = score
    
    return scores


def get_next_merchant_idx(sheriff_idx: int, merchant_offset: int, num_players: int) -> int:
    """Get the index of the next merchant to act.
    
    Args:
        sheriff_idx: Index of the sheriff
        merchant_offset: Offset from sheriff (0 = first merchant)
        num_players: Total number of players
        
    Returns:
        Player index of the merchant
    """
    return (sheriff_idx + 1 + merchant_offset) % num_players


def validate_bag_and_declaration(
    bag: List[int],
    declared_type: Optional[LegalType],
    declared_count: Optional[int],
    bag_limit: int,
) -> None:
    """Validate that a bag and declaration meet game rules.
    
    Raises:
        ValueError: If validation fails
    """
    if not bag:
        raise ValueError("Bag must contain at least 1 card.")
    if len(bag) > bag_limit:
        raise ValueError(f"Bag exceeds limit of {bag_limit}.")
    if declared_type is None:
        raise ValueError("Declared type is required.")
    if declared_count is None or declared_count <= 0:
        raise ValueError("Declared count must be >= 1.")
    if declared_count != len(bag):
        raise ValueError(f"Declared count ({declared_count}) must equal the number of cards in the bag ({len(bag)}).")


def is_bag_truthful(
    bag: List[int],
    declared_type: Optional[LegalType],
    declared_count: Optional[int],
    card_defs: List[CardDef],
) -> bool:
    """Check if a bag declaration is truthful.
    
    Args:
        bag: List of card IDs in the bag
        declared_type: The declared legal type
        declared_count: The declared count
        card_defs: List of all card definitions
        
    Returns:
        True if the declaration is completely truthful
    """
    if declared_type is None or not bag or not declared_count or declared_count != len(bag):
        return False
    
    for cid in bag:
        c = card_defs[cid]
        if c.kind != CardKind.LEGAL or c.name != declared_type.value:
            return False
    return True


def auto_fill_declaration(player_state: PlayerState, card_defs: List[CardDef]) -> None:
    """Auto-fill declaration after auto-load to prevent None declared_type.
    
    Args:
        player_state: Player state to update
        card_defs: List of all card definitions
    """
    if not player_state.bag:
        return  # No bag to declare
    
    # Get legal cards in bag
    legal_cards = []
    for card_id in player_state.bag:
        card = card_defs[card_id]
        if card.kind == CardKind.LEGAL:
            legal_cards.append(card.name)
    
    if legal_cards:
        # Use the first legal card type found
        declared_type_name = legal_cards[0]
        declared_type = LegalType(declared_type_name.lower())
        declared_count = legal_cards.count(declared_type_name)
        
        player_state.declared_type = declared_type
        player_state.declared_count = declared_count
    else:
        # No legal cards - set safe default
        player_state.declared_type = LegalType.BREAD
        player_state.declared_count = 1


def compute_inspection_outcome(
    bag: List[int],
    declared_type: Optional[LegalType],
    declared_count: Optional[int],
    card_defs: List[CardDef],
) -> InspectionOutcome:
    """Compute the outcome of inspecting a merchant's bag.
    
    Args:
        bag: List of card IDs in the bag
        declared_type: The declared legal type
        declared_count: The declared count
        card_defs: List of all card definitions
        
    Returns:
        InspectionOutcome with confiscations, deliveries, and gold deltas
    """
    truthful_flag = is_bag_truthful(bag, declared_type, declared_count, card_defs)
    confiscated, delivered = [], []
    sheriff_delta = 0
    merchant_delta = 0

    if truthful_flag:
        # Sheriff pays penalties for each legal card to the merchant
        for cid in bag:
            p = card_defs[cid].penalty
            sheriff_delta -= p
            merchant_delta += p
            delivered.append(cid)
        return InspectionOutcome(truthful_flag, confiscated, delivered, sheriff_delta, merchant_delta)

    # Not truthful → collect penalties for mismatched legal and contraband
    for cid in bag:
        c = card_defs[cid]
        # Handle case where declared_type is None (invalid declaration)
        if declared_type is not None and c.kind == CardKind.LEGAL and c.name == declared_type.value:
            # matching legal still delivered; no penalty
            delivered.append(cid)
        else:
            # contraband or mismatched legal → confiscate & merchant pays penalty
            confiscated.append(cid)
            p = c.penalty
            sheriff_delta += p
            merchant_delta -= p

    return InspectionOutcome(truthful_flag, confiscated, delivered, sheriff_delta, merchant_delta)

