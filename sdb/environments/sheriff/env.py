"""Sheriff of Nottingham environment implementation."""

import random
from typing import Dict, List, Optional, Any, Tuple
import copy

from sdb.core.base_env import BaseEnvironment
from sdb.core.base_agent import BaseAgent
from sdb.core.types import Action, Observation, GameResult
from sdb.logging.game_logger import GameLogger
from sdb.logging.formats import EventType

from .config import SheriffConfig
from .state import SheriffState
from .types import Phase, LegalType, CardKind, CardDef, PlayerState, Offer
from .rules import (
    build_deck,
    is_declaration_truthful,
    calculate_inspection_penalty,
    classify_bag,
    calculate_final_scores,
    get_next_merchant_idx,
)


class SheriffEnv(BaseEnvironment):
    """Sheriff of Nottingham environment.
    
    A bluffing/negotiation game where merchants try to smuggle contraband past the Sheriff.
    
    Phases:
    1. Market: Draw cards
    2. Load Bag: Put up to 5 cards in bag
    3. Declare: Declare contents (1 legal type + exact count)
    4. Negotiate: Merchants can bribe Sheriff
    5. Inspect: Sheriff inspects or passes each merchant
    6. Resolve: Pay penalties/rewards, goods go to stands
    """

    def __init__(
        self,
        agents: List[BaseAgent],
        config: Optional[SheriffConfig] = None,
        game_id: Optional[str] = None,
        logger: Optional[GameLogger] = None,
    ):
        """Initialize Sheriff of Nottingham environment.
        
        Args:
            agents: List of agents (must match config.n_players)
            config: Game configuration
            game_id: Unique game identifier
            logger: Game logger instance
        """
        config = config or SheriffConfig(n_players=len(agents))
        
        # Set game config before calling super().__init__()
        self.game_config = config
        self.logger = logger
        self.rng = random.Random(config.seed)
        
        super().__init__(agents=agents, config=config.__dict__, game_id=game_id, seed=config.seed)

    def reset(self) -> Dict[int, Observation]:
        """Reset the game to initial state."""
        # Build deck
        card_defs = build_deck(
            include_royal=self.game_config.include_royal, num_players=self.game_config.n_players
        )
        card_ids = list(range(len(card_defs)))
        self.rng.shuffle(card_ids)
        
        # Create players
        players = [PlayerState(pid=i) for i in range(self.game_config.n_players)]
        
        # Deal hands
        for p in players:
            for _ in range(self.game_config.hand_size):
                if card_ids:
                    p.hand.append(card_ids.pop())
        
        # Initialize discard piles
        discard_left = [card_ids.pop()] if card_ids else []
        discard_right = [card_ids.pop()] if card_ids else []
        
        # Initialize state
        self.state = SheriffState(
            config=self.game_config,
            rng=self.rng,
            deck=card_ids,
            discard_left=discard_left,
            discard_right=discard_right,
            card_defs=card_defs,
            players=players,
            sheriff_idx=0,
            rotation_counts=[0] * self.game_config.n_players,
            phase=Phase.MARKET,
            round_step=get_next_merchant_idx(0, 0, self.game_config.n_players),
        )
        
        # Log game start
        if self.logger:
            self.logger.log(
                EventType.GAME_START,
                {
                    "n_players": self.game_config.n_players,
                    "include_royal": self.game_config.include_royal,
                    "sheriff": 0,
                    "deck_size": len(card_ids),
                }
            )
        
        return self._get_observations()

    def get_state(self) -> SheriffState:
        """Get current game state."""
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self.state

    def _format_round_history(self) -> str:
        """Format history of all previous rounds.
        
        Returns:
            Formatted string of declarations and inspections
        """
        if not hasattr(self.state, 'history') or not self.state.history:
            return "   (First round)"
        
        formatted = []
        for event in self.state.history:
            if event.get('type') == 'declaration':
                formatted.append(
                    f"   • Player {event['player']}: Declared {event['quantity']} {event['good']}"
                )
            elif event.get('type') == 'inspection':
                result = "HONEST" if event['honest'] else "LYING"
                formatted.append(
                    f"      → Sheriff inspected: {result} (actual: {event.get('actual', 'N/A')})"
                )
        return "\n".join(formatted) if formatted else "   (No history yet)"
    
    def _format_player_standings(self) -> str:
        """Format current player gold and bonuses.
        
        Returns:
            Formatted string of player standings
        """
        st = self.state
        formatted = []
        for p in st.players:
            formatted.append(
                f"   • Player {p.pid}: {p.gold} gold, "
                f"{len([c for c in p.stand if self.card_defs[c].kind == CardKind.LEGAL])} legal goods, "
                f"{len([c for c in p.stand if self.card_defs[c].kind == CardKind.CONTRABAND])} contraband"
            )
        return "\n".join(formatted)
    
    def _get_observations(self) -> Dict[int, Observation]:
        """Generate observations for all players."""
        st = self.state
        obs = {}
        
        # Map phase to GamePhase enum
        from sdb.core.types import GamePhase, ObservationType
        phase_map = {
            Phase.MARKET: GamePhase.SETUP,
            Phase.LOAD: GamePhase.SETUP,
            Phase.DECLARE: GamePhase.SETUP,
            Phase.NEGOTIATE: GamePhase.DISCUSSION,
            Phase.INSPECT: GamePhase.VOTING,
            Phase.RESOLVE: GamePhase.TERMINAL,
        }
        game_phase = phase_map.get(st.phase, GamePhase.SETUP)
        
        for p in st.players:
            # Build data dictionary with all information
            data = {
                # Public information
                "phase": st.phase.value,
                "sheriff": st.sheriff_idx,
                "is_sheriff": p.pid == st.sheriff_idx,
                "round_step": st.round_step,
                "rotation_counts": st.rotation_counts.copy(),
                "top_discard": st.top_discard_choices(),
                "deck_size": len(st.deck),
                
                # Private information
                "player_id": p.pid,
                "gold": p.gold,
                "hand": self._format_cards([st.get_card_def(cid) for cid in p.hand]),
                "hand_ids": p.hand.copy(),
                "bag": self._format_cards([st.get_card_def(cid) for cid in p.bag]),
                "bag_ids": p.bag.copy(),
                "bag_count": len(p.bag),
                "declared_type": p.declared_type.value if p.declared_type else None,
                "declared_count": p.declared_count,
            }
            
            # Other players' public information
            other_players = []
            for other in st.players:
                if other.pid == p.pid:
                    continue
                other_players.append({
                    "player_id": other.pid,
                    "gold": other.gold,
                    "hand_size": len(other.hand),
                    "bag_size": len(other.bag),
                    "declared_type": other.declared_type.value if other.declared_type else None,
                    "declared_count": other.declared_count,
                    "legal_goods": {lt.value: len(other.stand_legal[lt]) for lt in LegalType},
                    "contraband_count": len(other.stand_contraband),
                })
            data["other_players"] = other_players
            
            # Current offers (if in negotiation phase)
            offers_info = {}
            if st.phase == Phase.NEGOTIATE:
                for mpid, offer in st.offers.items():
                    offers_info[mpid] = {
                        "from": offer.from_pid,
                        "to": offer.to_pid,
                        "gold": offer.gold,
                        "accepted": offer.accepted,
                        "promises": offer.promises.copy(),
                    }
            data["offers"] = offers_info
            
            # Generate phase-specific instructions
            instruction = self._generate_instruction(p.pid, st, data)
            data["instruction"] = instruction
            
            obs[p.pid] = Observation(
                player_id=p.pid,
                obs_type=ObservationType.PRIVATE,
                phase=game_phase,
                data=data,
            )
        
        return obs

    def _generate_instruction(self, pid: int, state: SheriffState, data: Dict[str, Any]) -> str:
        """Generate phase-specific instruction for a player."""
        is_sheriff = (pid == state.sheriff_idx)
        is_active = (pid == state.round_step)
        
        if state.phase == Phase.MARKET:
            if is_active:
                hand_with_ids = self._format_card_list_with_ids(data['hand'], data['hand_ids'])
                return f"""=== MARKET PHASE - YOUR TURN ===

You are a MERCHANT this round (Player {state.sheriff_idx} is Sheriff).

YOUR HAND: {len(data['hand_ids'])} cards
{hand_with_ids}

YOUR GOLD: {data['gold']}

AVAILABLE DISCARDS:
- Left pile: {data['top_discard']['left']}
- Right pile: {data['top_discard']['right']}
- Draw from deck (deck has {data['deck_size']} cards)

⚡ ACTION REQUIRED:
First, you may discard cards from your hand (optional).
Then, draw cards to replenish. Choose ONE:
1. Take cards from left discard pile
2. Take cards from right discard pile  
3. Draw from deck

Respond with JSON: {{"type": "draw", "source": "left"|"right"|"deck", "count": <1-5>, "discard_ids": [<optional card IDs to discard>]}}
Example: {{"type": "draw", "source": "deck", "count": 2, "discard_ids": []}}"""
            else:
                return "Waiting for other merchants to draw cards from the market."
                
        elif state.phase == Phase.LOAD:
            if is_active:
                # Format hand with card IDs
                hand_with_ids = self._format_card_list_with_ids(data['hand'], data['hand_ids'])
                return f"""=== LOAD BAG PHASE - YOUR TURN ===

You are a MERCHANT preparing your bag for inspection.

YOUR HAND ({len(data['hand_ids'])} cards):
{hand_with_ids}

YOUR GOLD: {data['gold']}

⚡ ACTION REQUIRED:
**IMPORTANT**: You MUST place **1-{state.config.bag_limit} cards** into your bag.
Use the card IDs shown above (the numbers in brackets: [ID:XX]).

**EMPTY BAGS ARE NOT ALLOWED** - you will be rejected at declaration!

You can put:
- Legal goods ONLY (you'll declare honestly - safe but lower profit)
- Contraband (you'll lie about them - more valuable but risky!)
- Mix of both (partial lie - balanced strategy)

Strategy tip: More contraband = more profit but higher risk if inspected!

Respond with JSON: {{"type": "load", "card_ids": [<1-{state.config.bag_limit} card IDs from above>]}}
Example if your hand IDs are [45, 23, 89, 12]: {{"type": "load", "card_ids": [45, 89, 12]}}"""
            else:
                return "Waiting for other merchants to load their bags."
                
        elif state.phase == Phase.DECLARE:
            if is_active:
                bag_cards = data['bag']
                bag_ids = data['bag_ids']
                bag_display = self._format_card_list_with_ids(bag_cards, bag_ids) if bag_cards else "  (empty bag - ERROR!)"
                return f"""=== DECLARATION PHASE - YOUR TURN ===

Your bag contains {len(bag_cards)} cards:
{bag_display}

YOUR GOLD: {data['gold']}

⚡ ACTION REQUIRED:
Declare what type of goods are in your bag.

**CRITICAL RULES**:
1. Your `declared_count` MUST equal the number of cards in your bag ({len(bag_cards)})
2. You MUST declare exactly ONE legal good type: apples, cheese, bread, or chicken
3. Empty bags are INVALID and will be rejected

STRATEGY:
- Be HONEST if you have only legal goods of one type (no penalty if inspected)
- LIE if you have contraband or mixed goods (claim they're all one legal type - risky!)

Respond with JSON: {{"type": "declare", "declared_type": "<apples|cheese|bread|chicken>", "declared_count": {len(bag_cards)}}}
Example if bag has 3 cards: {{"type": "declare", "declared_type": "apples", "declared_count": 3}}"""
            else:
                return "Waiting for other merchants to declare their goods."
                
        elif state.phase == Phase.NEGOTIATE:
            if is_sheriff:
                # Show which merchants still need responses
                merchants = state.get_all_merchants()
                pending = [m for m in merchants if m not in state.sheriff_responses]
                responded = [m for m in merchants if m in state.sheriff_responses]
                
                merchant_offers = []
                for mpid in merchants:
                    if mpid in data['offers']:
                        offer_data = data['offers'][mpid]
                        status = "✓ Responded" if mpid in responded else "⏳ Pending"
                        merchant_offers.append(f"  • Player {mpid}: {offer_data['gold']} gold ({status})")
                offers_text = "\n".join(merchant_offers) if merchant_offers else "  (No offers yet)"
                
                if pending:
                    next_merchant = pending[0]
                    action_text = f"""⚡ ACTION REQUIRED:
Respond to Player {next_merchant}'s offer.

Respond with JSON: {{"type": "respond", "merchant": {next_merchant}, "decision": "accept"|"reject"}}
Example: {{"type": "respond", "merchant": {next_merchant}, "decision": "reject"}}

Strategy: Accepting bribes gives you gold but lets contraband through!"""
                else:
                    action_text = "✅ All offers responded to. System will advance to next phase..."
                
                return f"""=== NEGOTIATION ROUND {state.negotiation_round}/{state.config.max_negotiation_rounds} - SHERIFF'S TURN ===

You are the SHERIFF. Merchants have made their offers.

OFFERS RECEIVED:
{offers_text}

{action_text}"""
            else:
                return f"""=== NEGOTIATION ROUND {state.negotiation_round}/{state.config.max_negotiation_rounds} - YOUR TURN ===

You are a MERCHANT. Make an offer to bribe the Sheriff!

YOUR BAG: {len(data['bag'])} cards declared as {data['declared_type']} x{data['declared_count']}
YOUR GOLD: {data['gold']}

⚡ ACTION REQUIRED:
Offer a bribe to the Sheriff. Higher bribes = less likely to be inspected!
If your bag is honest, you can offer little or nothing.
If you're smuggling contraband, consider offering more!

Respond with JSON: {{"type": "offer", "gold": <amount>, "promises": [<optional promises>]}}
Example: {{"type": "offer", "gold": 5, "promises": ["I have only apples"]}}
Or no bribe: {{"type": "offer", "gold": 0, "promises": []}}"""
                
        elif state.phase == Phase.INSPECT:
            if is_sheriff:
                merchants_info = []
                for other in data['other_players']:
                    decl = f"{other['declared_type']} x{other['declared_count']}" if other['declared_type'] else "nothing"
                    merchants_info.append(f"  • Player {other['player_id']}: declared {decl}, bag size {other['bag_size']}")
                merchants_text = "\n".join(merchants_info)
                
                return f"""=== INSPECTION PHASE - SHERIFF'S TURN ===

You are the SHERIFF. Decide who to inspect!

MERCHANTS THIS ROUND:
{merchants_text}

YOUR GOLD: {data['gold']}

⚡ ACTION REQUIRED:
Choose for each merchant:
1. INSPECT: Open their bag. If they lied, they pay penalties. If honest, YOU pay!
2. PASS: Let them through without inspection

To inspect: {{"type": "inspect", "merchant": <player_id>, "choice": "inspect"}}
To pass: {{"type": "inspect", "merchant": <player_id>, "choice": "pass"}}
When done: {{"type": "resolve"}}

Strategy: Inspect suspicious players (big bribes, unusual declarations)!"""
            else:
                return "Waiting for Sheriff to inspect merchants."
                
        elif state.phase == Phase.RESOLVE:
            return "Round is being resolved..."
        
        return "Waiting..."
    
    def _format_card_list(self, cards: List[Dict[str, Any]]) -> str:
        """Format a list of cards for display."""
        if not cards:
            return "  (no cards)"
        lines = []
        for i, card in enumerate(cards, 1):
            lines.append(f"  {i}. {card['name']} ({card['kind']}) - Value: {card['value']}")
        return "\n".join(lines)
    
    def _format_card_list_with_ids(self, cards: List[Dict[str, Any]], card_ids: List[int]) -> str:
        """Format a list of cards with their IDs for display."""
        if not cards:
            return "  (no cards)"
        lines = []
        for i, (card, cid) in enumerate(zip(cards, card_ids), 1):
            lines.append(f"  {i}. [ID:{cid}] {card['name']} ({card['kind']}) - Value: {card['value']}, Penalty: {card['penalty']}")
        return "\n".join(lines)
    
    def _format_cards(self, cards: List[CardDef]) -> List[Dict[str, Any]]:
        """Format card definitions for observation."""
        return [
            {
                "name": card.name,
                "kind": card.kind.value,
                "value": card.value,
                "penalty": card.penalty,
            }
            for card in cards
        ]


    def step(self, actions: Dict[int, Action]) -> Tuple[
        Dict[int, Observation],
        Dict[int, float],
        bool,
        Dict[str, Any],
    ]:
        """Execute actions and advance game state."""
        st = self.state
        
        # Handle RESOLVE phase (system phase, no player actions needed)
        if st.phase == Phase.RESOLVE:
            rewards = self._handle_resolve()
            obs = self._get_observations()
            done = st.game_over
            return obs, rewards, done, {}
        
        # Get active player for other phases
        if st.phase == Phase.INSPECT:
            active_pid = st.sheriff_idx
        else:
            active_pid = st.round_step
        
        # Execute action
        if active_pid not in actions:
            raise ValueError(f"Active player {active_pid} must provide an action in phase {st.phase.value}")
        
        action = actions[active_pid]
        
        # Dispatch by phase
        if st.phase == Phase.MARKET:
            self._handle_market(active_pid, action)
        elif st.phase == Phase.LOAD:
            self._handle_load(active_pid, action)
        elif st.phase == Phase.DECLARE:
            self._handle_declare(active_pid, action)
        elif st.phase == Phase.NEGOTIATE:
            self._handle_negotiate(active_pid, action)
        elif st.phase == Phase.INSPECT:
            self._handle_inspect(active_pid, action)
        
        # Get observations
        obs = self._get_observations()
        rewards = {p.pid: 0.0 for p in st.players}
        done = False
        
        return obs, rewards, done, {}

    def _handle_market(self, pid: int, action: Action):
        """Handle market phase action."""
        st = self.state
        p = st.get_player(pid)
        
        if pid == st.sheriff_idx:
            # Sheriff skips
            self._advance_market()
            return
        
        action_data = action.data
        
        # Discard cards
        discard_ids = action_data.get("discard_ids", [])
        discard_sides = action_data.get("discard_sides", [])
        
        # Default to "left" if not specified
        if len(discard_sides) < len(discard_ids):
            discard_sides += ["left"] * (len(discard_ids) - len(discard_sides))
        
        for cid, side in zip(discard_ids, discard_sides):
            if cid in p.hand:
                p.hand.remove(cid)
                if side == "right":
                    st.discard_right.append(cid)
                else:
                    st.discard_left.append(cid)
        
        # Draw cards
        # Support both "source" (string) and "draw_from" (list) for compatibility
        source = action_data.get("source", "deck")
        count = action_data.get("count", 1)
        draw_from = action_data.get("draw_from", [source] * count)
        
        while len(p.hand) < st.config.hand_size and draw_from:
            src = draw_from.pop(0) if draw_from else "deck"
            
            if src == "left" and st.discard_left:
                p.hand.append(st.discard_left.pop())
            elif src == "right" and st.discard_right:
                p.hand.append(st.discard_right.pop())
            else:
                # Draw from deck
                if not st.deck:
                    self._reshuffle_deck()
                if st.deck:
                    p.hand.append(st.deck.pop())
                else:
                    break  # No cards left
        
        # Log (public summary)
        if self.logger:
            self.logger.log(
                EventType.PLAYER_ACTION,
                {
                    "player_id": pid,
                    "phase": "market",
                    "drew_cards": st.config.hand_size - (len(p.hand) - len(discard_ids)),
                    "final_hand_size": len(p.hand),
                },
                is_private=False,
            )
        
        self._advance_market()

    def _advance_market(self):
        """Advance to next merchant in market phase or move to load phase."""
        st = self.state
        merchants = st.get_all_merchants()
        current_offset = (st.round_step - st.sheriff_idx - 1) % st.config.n_players
        
        if current_offset + 1 < len(merchants):
            # Next merchant
            st.round_step = get_next_merchant_idx(
                st.sheriff_idx, current_offset + 1, st.config.n_players
            )
        else:
            # Move to load phase
            st.phase = Phase.LOAD
            st.round_step = get_next_merchant_idx(st.sheriff_idx, 0, st.config.n_players)
            
            if self.logger:
                self.logger.log(EventType.PHASE_CHANGE, {"new_phase": "load_bag"})

    def _handle_load(self, pid: int, action: Action):
        """Handle load bag phase action."""
        st = self.state
        p = st.get_player(pid)
        
        card_ids = action.data.get("card_ids", [])
        
        # Load cards into bag
        for cid in card_ids:
            if cid in p.hand and len(p.bag) < st.config.bag_limit:
                p.hand.remove(cid)
                p.bag.append(cid)
        
        # Log (public summary)
        if self.logger:
            self.logger.log(
                EventType.PLAYER_ACTION,
                {
                    "player_id": pid,
                    "phase": "load_bag",
                    "bag_size": len(p.bag),
                },
                is_private=False,
            )
        
        self._advance_load()

    def _advance_load(self):
        """Advance to next merchant in load phase or move to declare phase."""
        st = self.state
        merchants = st.get_all_merchants()
        current_offset = (st.round_step - st.sheriff_idx - 1) % st.config.n_players
        
        if current_offset + 1 < len(merchants):
            # Next merchant
            st.round_step = get_next_merchant_idx(
                st.sheriff_idx, current_offset + 1, st.config.n_players
            )
        else:
            # Validate all merchants have loaded bags before moving to declare
            for m_pid in merchants:
                m_player = st.get_player(m_pid)
                if len(m_player.bag) == 0:
                    raise RuntimeError(f"Cannot enter DECLARE: Player {m_pid} has an empty bag.")
            
            # Move to declare phase
            st.phase = Phase.DECLARE
            st.round_step = get_next_merchant_idx(st.sheriff_idx, 0, st.config.n_players)
            
            if self.logger:
                self.logger.log(EventType.PHASE_CHANGE, {"new_phase": "declare"})

    def _handle_declare(self, pid: int, action: Action):
        """Handle declaration phase action."""
        from .rules import validate_bag_and_declaration
        
        st = self.state
        p = st.get_player(pid)
        
        # Get declaration
        declared_type_str = action.data.get("declared_type")
        declared_count = action.data.get("declared_count")
        
        # Convert to LegalType
        declared_type = LegalType(declared_type_str) if declared_type_str else None
        
        # Validate bag and declaration
        try:
            validate_bag_and_declaration(
                bag=p.bag,
                declared_type=declared_type,
                declared_count=declared_count,
                bag_limit=st.config.bag_limit,
            )
        except ValueError as e:
            # Log error and reject the declaration
            if self.logger:
                self.logger.log(
                    EventType.ERROR,
                    {
                        "player_id": pid,
                        "phase": "declare",
                        "error": str(e),
                    },
                    is_private=False,
                )
            raise
        
        # Set declaration
        p.declared_type = declared_type
        p.declared_count = declared_count
        
        # Log PUBLIC info only (no bag contents)
        if self.logger:
            self.logger.log(
                EventType.PLAYER_ACTION,
                {
                    "player_id": pid,
                    "phase": "declare",
                    "declared_type": declared_type_str,
                    "declared_count": declared_count,
                },
                is_private=False,
            )
            
            # Log PRIVATE info (for analytics/debugging)
            bag_cards = [st.get_card_def(cid) for cid in p.bag]
            bag_class = classify_bag(bag_cards)
            self.logger.log(
                EventType.PLAYER_ACTION,
                {
                    "player_id": pid,
                    "phase": "declare",
                    "declared_type": declared_type_str,
                    "declared_count": declared_count,
                    "bag_class": bag_class,
                    "actual_bag": [card.name for card in bag_cards],
                },
                is_private=True,  # Hidden until inspection
            )
        
        self._advance_declare()

    def _advance_declare(self):
        """Advance to next merchant in declare phase or move to negotiate phase."""
        st = self.state
        merchants = st.get_all_merchants()
        current_offset = (st.round_step - st.sheriff_idx - 1) % st.config.n_players
        
        if current_offset + 1 < len(merchants):
            # Next merchant
            st.round_step = get_next_merchant_idx(
                st.sheriff_idx, current_offset + 1, st.config.n_players
            )
        else:
            # Validate all merchants have declarations before moving to negotiate
            for m_pid in merchants:
                m_player = st.get_player(m_pid)
                if m_player.declared_type is None or m_player.declared_count is None:
                    raise RuntimeError(f"Cannot enter NEGOTIATE: Player {m_pid} has not declared.")
            
            # Move to negotiate phase
            st.phase = Phase.NEGOTIATE
            st.round_step = get_next_merchant_idx(st.sheriff_idx, 0, st.config.n_players)
            st.negotiation_round = 1  # Start at round 1
            st.offers = {}
            st.sheriff_responses = set()  # Track which merchants sheriff has responded to
            
            if self.logger:
                self.logger.log(EventType.PHASE_CHANGE, {"new_phase": "negotiate", "negotiation_round": 1})

    def _handle_negotiate(self, pid: int, action: Action):
        """Handle negotiation phase action."""
        st = self.state
        
        if pid == st.sheriff_idx:
            # Sheriff responds or ends negotiation
            action_type = action.data.get("type", "")
            if action_type == "end_negotiate":
                # Move to inspect phase
                st.phase = Phase.INSPECT
                st.inspected_merchants = set()
                
                if self.logger:
                    self.logger.log(EventType.PHASE_CHANGE, {"new_phase": "inspect"})
            
            elif action_type == "respond":
                merchant_pid = action.data.get("merchant")
                decision = action.data.get("decision")  # "accept" or "reject"
                
                # Skip if already responded to this merchant in this round
                if merchant_pid in st.sheriff_responses:
                    return
                
                st.sheriff_responses.add(merchant_pid)
                
                if merchant_pid in st.offers:
                    offer = st.offers[merchant_pid]
                    offer.accepted = (decision == "accept")
                    
                    # Process offer if accepted
                    if offer.accepted:
                        merchant = st.get_player(merchant_pid)
                        sheriff = st.get_player(st.sheriff_idx)
                        
                        # Transfer gold
                        merchant.gold -= offer.gold
                        sheriff.gold += offer.gold
                        
                        # Transfer stand goods (immediate)
                        for cid in offer.stand_goods:
                            for lt in LegalType:
                                if cid in merchant.stand_legal[lt]:
                                    merchant.stand_legal[lt].remove(cid)
                                    sheriff.stand_legal[lt].append(cid)
                                    break
                            if cid in merchant.stand_contraband:
                                merchant.stand_contraband.remove(cid)
                                sheriff.stand_contraband.append(cid)
                    
                    # Log
                    if self.logger:
                        self.logger.log(
                            EventType.PLAYER_ACTION,
                            {
                                "player_id": pid,
                                "phase": "negotiate",
                                "negotiation_round": st.negotiation_round,
                                "merchant": merchant_pid,
                                "decision": decision,
                                "gold_transferred": offer.gold if offer.accepted else 0,
                            }
                        )
                
                # Check if all merchants have been responded to
                merchants = st.get_all_merchants()
                if len(st.sheriff_responses) >= len(merchants):
                    # Sheriff has responded to all merchants
                    if st.negotiation_round < st.config.max_negotiation_rounds:
                        # Start another negotiation round
                        st.negotiation_round += 1
                        st.round_step = get_next_merchant_idx(st.sheriff_idx, 0, st.config.n_players)
                        st.offers = {}
                        st.sheriff_responses = set()
                        
                        if self.logger:
                            self.logger.log(
                                EventType.PHASE_CHANGE,
                                {
                                    "new_phase": "negotiate",
                                    "negotiation_round": st.negotiation_round,
                                }
                            )
                    else:
                        # Max rounds reached, move to inspect
                        st.phase = Phase.INSPECT
                        st.inspected_merchants = set()
                        
                        if self.logger:
                            self.logger.log(EventType.PHASE_CHANGE, {"new_phase": "inspect"})
        else:
            # Merchant makes offer
            offer = Offer(
                from_pid=pid,
                to_pid=st.sheriff_idx,
                gold=action.data.get("gold", 0),
                stand_goods=action.data.get("stand_goods", []),
                bag_goods=action.data.get("bag_goods", []),
                promises=action.data.get("promises", []),
            )
            st.offers[pid] = offer
            
            # Log
            if self.logger:
                self.logger.log(
                    EventType.PLAYER_ACTION,
                    {
                        "player_id": pid,
                        "phase": "negotiate",
                        "offer_gold": offer.gold,
                        "promises": offer.promises,
                    }
                )
            
            # Advance to next merchant or back to sheriff
            self._advance_negotiate()

    def _advance_negotiate(self):
        """Advance negotiation phase."""
        st = self.state
        merchants = st.get_all_merchants()
        current_offset = (st.round_step - st.sheriff_idx - 1) % st.config.n_players
        
        if current_offset + 1 < len(merchants):
            # Next merchant
            st.round_step = get_next_merchant_idx(
                st.sheriff_idx, current_offset + 1, st.config.n_players
            )
        else:
            # All merchants have offered, sheriff's turn
            st.round_step = st.sheriff_idx

    def _handle_inspect(self, pid: int, action: Action):
        """Handle inspection phase action."""
        from .rules import compute_inspection_outcome
        
        st = self.state
        
        action_type = action.data.get("type", "")
        if action_type == "resolve":
            # Move to resolve phase
            st.phase = Phase.RESOLVE
            if self.logger:
                self.logger.log(EventType.PHASE_CHANGE, {"new_phase": "resolve"})
            return
        
        # Inspect or pass a merchant
        merchant_pid = action.data.get("merchant")
        choice = action.data.get("choice")  # "inspect" or "pass"
        
        if merchant_pid in st.inspected_merchants:
            return  # Already inspected
        
        st.inspected_merchants.add(merchant_pid)
        merchant = st.get_player(merchant_pid)
        sheriff = st.get_player(st.sheriff_idx)
        
        if choice == "inspect":
            # Compute inspection outcome using the new rules
            outcome = compute_inspection_outcome(
                bag=merchant.bag,
                declared_type=merchant.declared_type,
                declared_count=merchant.declared_count,
                card_defs=st.card_defs,
            )
            
            # Apply gold deltas
            sheriff.gold += outcome.sheriff_delta
            merchant.gold += outcome.merchant_delta
            
            # Move delivered cards to merchant's stand
            for cid in outcome.delivered:
                c = st.get_card_def(cid)
                if c.kind == CardKind.LEGAL:
                    for lt in LegalType:
                        if c.name == lt.value:
                            merchant.stand_legal[lt].append(cid)
                            break
                else:
                    merchant.stand_contraband.append(cid)
            
            # Move confiscated cards to discard
            for cid in outcome.confiscated:
                st.discard_left.append(cid)
            
            # Log PUBLIC reveal after inspection
            if self.logger:
                # Create a reveal showing what was in the bag
                revealed_cards = []
                for cid in outcome.delivered + outcome.confiscated:
                    revealed_cards.append(st.get_card_def(cid).name)
                
                self.logger.log(
                    EventType.PLAYER_ACTION,
                    {
                        "player_id": pid,
                        "phase": "inspect",
                        "merchant": merchant_pid,
                        "choice": "inspect",
                        "truthful": outcome.truthful,
                        "revealed_cards": revealed_cards,
                        "delivered": outcome.delivered,
                        "confiscated": outcome.confiscated,
                        "penalties": {
                            "sheriff_delta": outcome.sheriff_delta,
                            "merchant_delta": outcome.merchant_delta,
                        },
                    },
                    is_private=False,
                )
        else:
            # Pass - merchant gets all goods
            for cid in merchant.bag:
                card = st.get_card_def(cid)
                if card.kind == CardKind.LEGAL:
                    for lt in LegalType:
                        if card.name == lt.value:
                            merchant.stand_legal[lt].append(cid)
                            break
                else:
                    merchant.stand_contraband.append(cid)
            
            # Transfer bag goods from offer if accepted
            if merchant_pid in st.offers and st.offers[merchant_pid].accepted:
                offer = st.offers[merchant_pid]
                for cid in offer.bag_goods:
                    if cid in merchant.bag:
                        merchant.bag.remove(cid)
                        card = st.get_card_def(cid)
                        if card.kind == CardKind.LEGAL:
                            for lt in LegalType:
                                if card.name == lt.value:
                                    sheriff.stand_legal[lt].append(cid)
                                    break
                        else:
                            sheriff.stand_contraband.append(cid)
            
            # Log
            if self.logger:
                self.logger.log(
                    EventType.PLAYER_ACTION,
                    {
                        "player_id": pid,
                        "phase": "inspect",
                        "merchant": merchant_pid,
                        "choice": "pass",
                    }
                )
        
        # Clear merchant's bag
        merchant.clear_bag()

    def _handle_resolve(self) -> Dict[int, float]:
        """Handle resolve phase - rotate sheriff and check for game end."""
        st = self.state
        
        # Rotate sheriff
        st.rotation_counts[st.sheriff_idx] += 1
        st.sheriff_idx = (st.sheriff_idx + 1) % st.config.n_players
        
        # Check if game should end
        if all(count >= st.config.sheriff_rotations for count in st.rotation_counts):
            st.game_over = True
            
            # Calculate final scores
            scores = calculate_final_scores(st.players, st.card_defs)
            winner_pid = max(scores.items(), key=lambda x: x[1])[0]
            st.winner = winner_pid
            
            # Log game end
            if self.logger:
                self.logger.log(
                    EventType.GAME_END,
                    {
                        "winner": winner_pid,
                        "scores": scores,
                    }
                )
            
            # Return normalized rewards (winner gets 1.0, others get scores/winner_score)
            winner_score = scores[winner_pid]
            rewards = {
                pid: scores[pid] / winner_score if winner_score > 0 else 0.0
                for pid in scores
            }
            return rewards
        else:
            # Reset for next round
            st.phase = Phase.MARKET
            st.round_step = get_next_merchant_idx(st.sheriff_idx, 0, st.config.n_players)
            st.offers = {}
            st.inspected_merchants = set()
            
            if self.logger:
                self.logger.log(
                    EventType.PHASE_CHANGE,
                    {
                        "new_phase": "market",
                        "new_sheriff": st.sheriff_idx,
                    }
                )
            
            return {p.pid: 0.0 for p in st.players}

    def _transfer_penalty(self, payer: PlayerState, receiver: PlayerState, amount: int):
        """Transfer penalty from payer to receiver (gold first, then goods)."""
        st = self.state
        
        # Transfer gold
        pay = min(payer.gold, amount)
        payer.gold -= pay
        receiver.gold += pay
        amount -= pay
        
        if amount <= 0:
            return
        
        # Transfer legal goods
        for lt in LegalType:
            while amount > 0 and payer.stand_legal[lt]:
                cid = payer.stand_legal[lt].pop()
                value = st.get_card_def(cid).value
                receiver.stand_legal[lt].append(cid)
                amount -= value
        
        # Transfer contraband
        while amount > 0 and payer.stand_contraband:
            cid = payer.stand_contraband.pop()
            value = st.get_card_def(cid).value
            receiver.stand_contraband.append(cid)
            amount -= value
        
        # Remainder is forgiven

    def _reshuffle_deck(self):
        """Reshuffle discards into deck, keeping top 5 of each pile."""
        st = self.state
        
        # Keep top 5 of each discard
        left_keep = st.discard_left[-5:] if len(st.discard_left) > 5 else list(st.discard_left)
        right_keep = st.discard_right[-5:] if len(st.discard_right) > 5 else list(st.discard_right)
        
        # Pool everything else
        pool = st.discard_left[:-5] if len(st.discard_left) > 5 else []
        pool += st.discard_right[:-5] if len(st.discard_right) > 5 else []
        
        # Update piles
        st.discard_left[:] = left_keep
        st.discard_right[:] = right_keep
        
        # Shuffle and add to deck
        st.rng.shuffle(pool)
        st.deck.extend(pool)

    def _validate_num_players(self):
        """Validate player count."""
        if len(self.agents) < 3 or len(self.agents) > 5:
            from sdb.core.exceptions import EnvironmentError
            raise EnvironmentError(f"Sheriff requires 3-5 players, got {len(self.agents)}")
    
    def _get_current_player(self) -> int:
        """Get the ID of the current active player."""
        if self.state is None:
            return 0
        
        if self.state.phase in (Phase.INSPECT, Phase.RESOLVE):
            return self.state.sheriff_idx
        else:
            return self.state.round_step
    
    def get_winner(self):
        """Get game winner."""
        if self.state is None or not self.state.game_over:
            return None
        return self.state.winner
    
    def get_win_reason(self):
        """Get the reason for the win."""
        if self.state is None or not self.state.game_over:
            return None
        
        scores = calculate_final_scores(self.state.players, self.state.card_defs)
        winner_score = scores.get(self.state.winner, 0)
        return f"Player {self.state.winner} won with {winner_score} points"

    def play_game(self) -> GameResult:
        """Play a complete game with the configured agents."""
        if not self.agents:
            raise RuntimeError("No agents configured")
        
        # Environment already initialized in __init__ (reset was called there)
        obs = self._get_observations()
        done = False
        num_rounds = 0
        
        while not done:
            st = self.state
            
            # Get active player
            if st.phase in (Phase.INSPECT, Phase.RESOLVE):
                active_pid = st.sheriff_idx
            else:
                active_pid = st.round_step
            
            # Get action from agent
            agent = self.agents[active_pid]
            action = agent.act(obs[active_pid])
            
            # Execute action
            obs, rewards, done, info = self.step({active_pid: action})
            
            # Count rounds
            if st.phase == Phase.MARKET and st.round_step == st.get_merchant_idx(0):
                num_rounds += 1
        
        # Create result
        st = self.state
        scores = calculate_final_scores(st.players, st.card_defs)
        
        # Determine winner (player with highest score)
        winner_pid = max(scores.items(), key=lambda x: x[1])[0] if scores else 0
        winner_score = scores.get(winner_pid, 0)
        
        # Build player stats
        player_stats = {}
        for pid in range(st.config.n_players):
            score = scores.get(pid, 0)
            player_stats[pid] = {
                "score": float(score),
                "is_winner": pid == winner_pid,
                "gold": st.players[pid].gold if pid < len(st.players) else 0,
            }
        
        return GameResult(
            game_id=self.game_id,
            winner=f"Player {winner_pid}",
            win_reason=f"Highest score: {winner_score} gold",
            num_rounds=num_rounds,
            duration_seconds=0.0,
            player_stats=player_stats,
            metadata={
                "final_scores": {str(k): v for k, v in scores.items()},
                "rounds_completed": sum(st.rotation_counts) if hasattr(st, 'rotation_counts') else 0,
            }
        )

