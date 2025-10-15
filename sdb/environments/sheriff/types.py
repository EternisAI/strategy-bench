"""Sheriff of Nottingham type definitions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict


class Phase(str, Enum):
    """Game phases in Sheriff of Nottingham."""
    MARKET = "market"
    LOAD = "load_bag"
    DECLARE = "declare"
    NEGOTIATE = "negotiate"
    INSPECT = "inspect"
    RESOLVE = "resolve"


class LegalType(str, Enum):
    """Legal goods that can be traded."""
    APPLES = "apples"
    CHEESE = "cheese"
    BREAD = "bread"
    CHICKEN = "chicken"


class CardKind(str, Enum):
    """Types of cards in the game."""
    LEGAL = "legal"
    CONTRABAND = "contraband"
    ROYAL = "royal"  # Optional module


@dataclass(frozen=True)
class CardDef:
    """Card definition from the rulebook."""
    name: str
    kind: CardKind
    value: int  # Gold value for scoring
    penalty: int  # Penalty paid on inspection
    counts_as: Optional[LegalType] = None  # For Royal goods (end-game counts)
    counts_as_n: int = 1  # Royal Goods count as 2 or 3 for King/Queen bonuses


@dataclass
class Offer:
    """Bribe offer from merchant to sheriff."""
    from_pid: int  # Merchant player ID
    to_pid: int  # Sheriff player ID
    gold: int = 0
    stand_goods: List[int] = field(default_factory=list)  # Delivered immediately
    bag_goods: List[int] = field(default_factory=list)  # Delivered after pass
    promises: List[str] = field(default_factory=list)  # Non-binding, logged only
    accepted: Optional[bool] = None


@dataclass
class PlayerState:
    """State for a single player."""
    pid: int
    gold: int = 50  # Starting gold from rulebook p. 5
    hand: List[int] = field(default_factory=list)  # Card IDs in hand
    stand_legal: Dict[LegalType, List[int]] = field(
        default_factory=lambda: {t: [] for t in LegalType}
    )  # Legal goods on stand
    stand_contraband: List[int] = field(default_factory=list)  # Contraband on stand
    bag: List[int] = field(default_factory=list)  # Cards currently in bag
    declared_type: Optional[LegalType] = None  # Declared legal type
    declared_count: Optional[int] = None  # Declared count

    def clear_bag(self):
        """Clear the bag after resolution."""
        self.bag.clear()
        self.declared_type = None
        self.declared_count = None


# Card defaults from rulebook
LEGAL_DEFAULTS: Dict[LegalType, Dict[str, int]] = {
    LegalType.APPLES: {"value": 2, "penalty": 2, "count": 48},
    LegalType.CHEESE: {"value": 3, "penalty": 2, "count": 36},
    LegalType.BREAD: {"value": 3, "penalty": 2, "count": 36},
    LegalType.CHICKEN: {"value": 4, "penalty": 2, "count": 24},
}

CONTRABAND_DEFAULTS: Dict[str, Dict[str, int]] = {
    "pepper": {"value": 4, "penalty": 4, "count": 20},
    "mead": {"value": 7, "penalty": 4, "count": 18},
    "silk": {"value": 8, "penalty": 4, "count": 16},
    "crossbow": {"value": 9, "penalty": 4, "count": 6},
}

# Royal goods are optional (rulebook pp. 14-15)
ROYAL_DEFAULTS: Dict[str, Dict[str, object]] = {
    "green_apples": {
        "value": 4,
        "penalty": 3,
        "count": 2,
        "counts_as": LegalType.APPLES,
        "counts_as_n": 2,
        "min_players": 3,
    },
    "golden_apples": {
        "value": 6,
        "penalty": 4,
        "count": 2,
        "counts_as": LegalType.APPLES,
        "counts_as_n": 3,
        "min_players": 4,
    },
    "gouda_cheese": {
        "value": 6,
        "penalty": 4,
        "count": 2,
        "counts_as": LegalType.CHEESE,
        "counts_as_n": 2,
        "min_players": 3,
    },
    "bleu_cheese": {
        "value": 9,
        "penalty": 5,
        "count": 1,
        "counts_as": LegalType.CHEESE,
        "counts_as_n": 3,
        "min_players": 4,
    },
    "rye_bread": {
        "value": 6,
        "penalty": 4,
        "count": 2,
        "counts_as": LegalType.BREAD,
        "counts_as_n": 2,
        "min_players": 3,
    },
    "pumpernickel_bread": {
        "value": 9,
        "penalty": 5,
        "count": 1,
        "counts_as": LegalType.BREAD,
        "counts_as_n": 3,
        "min_players": 4,
    },
    "royal_rooster": {
        "value": 8,
        "penalty": 4,
        "count": 2,
        "counts_as": LegalType.CHICKEN,
        "counts_as_n": 2,
        "min_players": 3,
    },
}

# King/Queen bonuses from rulebook p. 13
KING_BONUS = {
    LegalType.APPLES: 20,
    LegalType.CHEESE: 15,
    LegalType.BREAD: 15,
    LegalType.CHICKEN: 10,
}

QUEEN_BONUS = {
    LegalType.APPLES: 10,
    LegalType.CHEESE: 10,
    LegalType.BREAD: 10,
    LegalType.CHICKEN: 5,
}

