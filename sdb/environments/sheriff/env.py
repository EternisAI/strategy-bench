"""Sheriff of Nottingham environment implementation."""

import random
import time
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
    auto_fill_declaration,
)
from .helpers import ensure_player_idx, safe_get_player


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
        role_assignment: Optional[Dict] = None,
    ):
        """Initialize Sheriff of Nottingham environment.
        
        Args:
            agents: List of agents (must match config.n_players)
            config: Game configuration
            game_id: Unique game identifier
            logger: Game logger instance
            role_assignment: Optional role assignment (not used - Sheriff rotates)
        """
        config = config or SheriffConfig(n_players=len(agents))
        
        # Set game config before calling super().__init__()
        self.game_config = config
        self.logger = logger
        self.rng = random.Random(config.seed)
        # Note: role_assignment not used in Sheriff - sheriff role rotates each round
        
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
            round_number=0,
            phase=Phase.MARKET,
            round_step=0,
        )
        
        # Initialize merchant queue
        self.state.start_merchant_cycle()
        
        # Set initial round_step to first merchant
        self.state.round_step = self.state.next_merchant()
        
        # Start timeout tracking
        self.state.phase_start_time = time.time()
        
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
            
            # Log agent metadata including model information
            agent_metadata = {}
            for i, agent in enumerate(self.agents):
                agent_info = {
                    "name": agent.name,
                    "type": agent.__class__.__name__,
                }
                # Add model information if available
                if hasattr(agent, 'llm_client') and hasattr(agent.llm_client, 'model'):
                    agent_info["model"] = agent.llm_client.model
                elif hasattr(agent, 'model'):
                    agent_info["model"] = agent.model
                elif hasattr(agent, 'config') and hasattr(agent.config, 'model'):
                    agent_info["model"] = agent.config.model
                agent_metadata[str(i)] = agent_info
            
            # Log agent metadata as INFO event, not second GAME_START
            self.logger.log(
                EventType.INFO,
                data={
                    "event": "agent_metadata",
                    "agents": agent_metadata
                },
                is_private=False
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
                "round_number": st.round_number,
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
            sheriff_responses_info = {}
            if st.phase == Phase.NEGOTIATE:
                for mpid, offer in st.offers.items():
                    offers_info[mpid] = {
                        "from": offer.from_pid,
                        "to": offer.to_pid,
                        "gold": offer.gold,
                        "stand_goods": offer.stand_goods.copy(),
                        "bag_goods": offer.bag_goods.copy(),
                        "accepted": offer.accepted,
                        "promises": offer.promises.copy(),
                    }
                # Include sheriff responses so agents can see who has been responded to
                sheriff_responses_info = {k: v.copy() for k, v in st.sheriff_responses.items()}
            data["offers"] = offers_info
            data["sheriff_responses"] = sheriff_responses_info
            
            # Inspection tracking (if in inspect phase)
            if st.phase == Phase.INSPECT:
                data["inspected_merchants"] = list(st.inspected_merchants)
                data["current_inspect_merchant"] = st.current_inspect_merchant()
            else:
                data["inspected_merchants"] = []
                data["current_inspect_merchant"] = None
            
            # Game history (available to all players)
            data["game_history"] = st.game_history.copy()
            data["formatted_history"] = st.get_formatted_history()
            
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
                # Get current merchant from inspect queue
                current_merchant = data.get('current_inspect_merchant')
                
                if current_merchant is None:
                    return """✅ All merchants processed. Waiting for resolve phase..."""
                
                # Get info about current merchant
                current_merchant_info = None
                for other in data['other_players']:
                    if other['player_id'] == current_merchant:
                        current_merchant_info = other
                        break
                
                if not current_merchant_info:
                    return "Waiting for next merchant..."
                
                # Show all merchants with status
                merchants_info = []
                inspected = set(data.get('inspected_merchants', []))
                
                for other in data['other_players']:
                    decl = f"{other['declared_type']} x{other['declared_count']}" if other['declared_type'] else "nothing"
                    if other['player_id'] == current_merchant:
                        status = "👉 CURRENT"
                    elif other['player_id'] in inspected:
                        status = "✓ DONE"
                    else:
                        status = "⏳ PENDING"
                    merchants_info.append(f"  • Player {other['player_id']}: {status} - declared {decl}, bag size {other['bag_size']}")
                
                merchants_text = "\n".join(merchants_info)
                
                # Current merchant details
                decl = f"{current_merchant_info['declared_type']} x{current_merchant_info['declared_count']}"
                
                return f"""=== INSPECTION PHASE - SHERIFF'S TURN ===

🎯 CURRENT MERCHANT: Player {current_merchant}
   Declared: {decl}
   Bag size: {current_merchant_info['bag_size']}

ALL MERCHANTS THIS ROUND:
{merchants_text}

YOUR GOLD: {data['gold']}

⚡ DECISION REQUIRED:

Choose one action for Player {current_merchant}:

1. INSPECT: Open their bag and check contents
   - If lying: Confiscate contraband, collect penalties
   - If truthful: YOU pay them penalties!

2. PASS: Let them through without inspection
   - They keep everything in their bag
   - No penalties either way

Respond with exact JSON (no merchant field needed):

{{"type": "sheriff_decision", "choice": "inspect"}}  ← to INSPECT Player {current_merchant}
{{"type": "sheriff_decision", "choice": "pass"}}     ← to PASS Player {current_merchant}

Strategy: Inspect suspicious declarations, big bribes, or unusual bag sizes!"""
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
        
        # Check for phase timeout
        if self._check_phase_timeout():
            self._handle_phase_timeout()
            obs = self._get_observations()
            rewards = {p.pid: 0.0 for p in st.players}
            done = st.game_over
            return obs, rewards, done, {}
        
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
            if active_pid is None:
                # This should not happen with proper merchant queue management
                raise RuntimeError(f"round_step is None in phase {st.phase.value}")
        
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

    def _check_phase_timeout(self) -> bool:
        """Check if current phase has timed out."""
        st = self.state
        if st.phase_start_time == 0.0:
            return False  # No timeout set
        
        elapsed = time.time() - st.phase_start_time
        return elapsed > st.config.max_phase_seconds

    def _handle_phase_timeout(self):
        """Handle phase timeout by applying default actions."""
        st = self.state
        
        if st.phase == Phase.NEGOTIATE:
            # Default to reject all pending offers
            merchants = st.get_all_merchants()
            for merchant_pid in merchants:
                if merchant_pid not in st.sheriff_responses:
                    st.sheriff_responses[merchant_pid] = {"decision": "reject", "gold": 0}
            
            # Move to inspect phase
            st.phase = Phase.INSPECT
            st.phase_start_time = time.time()  # Start timeout tracking for inspect
            st.start_inspect_cycle()  # Initialize inspect queue
            
            if self.logger:
                self.logger.log(
                    EventType.INFO,
                    {
                        "event": "phase_timeout",
                        "phase": "negotiate",
                        "action": "default_reject_all"
                    },
                    is_private=False
                )
        elif st.phase == Phase.INSPECT:
            # Default to pass all uninspected merchants
            merchants = st.get_all_merchants()
            for merchant_pid in merchants:
                if merchant_pid not in st.inspected_merchants:
                    merchant = st.get_player(merchant_pid)
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
                    merchant.clear_bag()
                    st.inspected_merchants.add(merchant_pid)
            
            # Move to resolve
            st.phase = Phase.RESOLVE
            
            if self.logger:
                self.logger.log(
                    EventType.INFO,
                    {
                        "event": "phase_timeout",
                        "phase": "inspect",
                        "action": "default_pass_all"
                    },
                    is_private=False
                )

    def _handle_market(self, pid: int, action: Action):
        """Handle market phase action."""
        st = self.state
        p = st.get_player(pid)
        
        if pid == st.sheriff_idx:
            # Sheriff skips
            self._advance_market()
            return
        
        action_data = action.data
        
        # Track hand size before any changes
        hand_size_before = len(p.hand)
        
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
        
        # CRITICAL FIX: Ensure hand has exactly hand_size cards after Market
        # If player still needs more cards, draw from deck
        while len(p.hand) < st.config.hand_size:
            if not st.deck:
                self._reshuffle_deck()
            if st.deck:
                p.hand.append(st.deck.pop())
            else:
                break  # Truly no cards left in entire game
        
        # INVARIANT CHECK: Market phase must result in hand_size cards
        if len(p.hand) != st.config.hand_size:
            # Log warning if invariant violated (only if deck truly exhausted)
            if self.logger:
                self.logger.log(
                    EventType.INFO,
                    {
                        "event": "market_hand_size_warning",
                        "expected": st.config.hand_size,
                        "actual": len(p.hand),
                        "reason": "deck_exhausted" if not st.deck else "unknown"
                    },
                    player_id=pid,
                    is_private=False,
                )
        
        # Log (public summary)
        if self.logger:
            # Calculate how many cards were actually drawn
            # drew_cards = final_hand_size - (initial_hand_size - discarded)
            cards_actually_drawn = len(p.hand) - (hand_size_before - len(discard_ids))
            
            self.logger.log(
                EventType.PLAYER_ACTION,
                {
                    "phase": "market",
                    "drew_cards": cards_actually_drawn,
                    "final_hand_size": len(p.hand),
                    "discarded": len(discard_ids),  # Added for clarity
                },
                player_id=pid,
                is_private=False,
            )
        
        # Advance to next merchant
        self._advance_market()

    def _advance_market(self):
        """Advance to next merchant in market phase or move to load phase."""
        st = self.state
        
        # Finish current merchant first
        st.finish_current_merchant()
        
        next_merchant = st.next_merchant()
        
        if next_merchant is not None:
            # Next merchant in queue
            st.round_step = next_merchant
        else:
            # All merchants have completed market phase
            # Guardrail: log warning if merchants don't have full hands before LOAD_BAG
            merchants = st.get_all_merchants()
            for merchant_pid in merchants:
                merchant = st.get_player(merchant_pid)
                if len(merchant.hand) != st.config.hand_size:
                    if self.logger:
                        self.logger.log(
                            EventType.INFO,
                            {
                                "event": "market_incomplete_hand",
                                "merchant": merchant_pid,
                                "expected": st.config.hand_size,
                                "actual": len(merchant.hand),
                                "reason": "deck_exhausted" if not st.deck else "unknown"
                            },
                            is_private=False,
                        )
            
            # Move to load phase
            st.phase = Phase.LOAD
            st.phase_start_time = time.time()  # Start timeout tracking
            st.start_merchant_cycle()  # Reset merchant queue for load phase
            st.round_step = st.next_merchant()
            
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
                    "phase": "load_bag",
                    "bag_size": len(p.bag),
                },
                player_id=pid,
                is_private=False,
            )
        
        # Advance to next merchant
        self._advance_load()

    def _advance_load(self):
        """Advance to next merchant in load phase or move to declare phase."""
        st = self.state
        
        # Finish current merchant first
        st.finish_current_merchant()
        
        next_merchant = st.next_merchant()
        
        if next_merchant is not None:
            # Next merchant in queue
            st.round_step = next_merchant
        else:
            # Handle empty bags: Auto-load 1 random card for any empty bags
            merchants = st.get_all_merchants()
            for m_pid in merchants:
                m_player = st.get_player(m_pid)
                if len(m_player.bag) == 0 and len(m_player.hand) > 0:
                    # Force load 1 card from hand
                    card_to_load = m_player.hand[0]
                    m_player.hand.remove(card_to_load)
                    m_player.bag.append(card_to_load)
                    
                    if self.logger:
                        self.logger.log(
                            EventType.INFO,
                            {
                                "event": "auto_load_empty_bag",
                                "reason": "Merchant must load at least 1 card",
                            },
                            player_id=m_pid,
                            is_private=False,
                        )
                elif len(m_player.bag) == 0:
                    # No cards in hand either - log error and skip this merchant
                    if self.logger:
                        self.logger.log(
                            EventType.ERROR,
                            {
                                "event": "empty_bag_empty_hand",
                                "reason": "Merchant has no cards to load",
                            },
                            player_id=m_pid,
                            is_private=False,
                        )
            
            # Move to declare phase
            st.phase = Phase.DECLARE
            st.phase_start_time = time.time()  # Start timeout tracking
            st.start_merchant_cycle()  # Reset merchant queue for declare phase
            st.round_step = st.next_merchant()
            
            if self.logger:
                self.logger.log(EventType.PHASE_CHANGE, {"new_phase": "declare"})

    def _handle_declare(self, pid: int, action: Action):
        """Handle declaration phase action."""
        from .rules import validate_bag_and_declaration
        
        st = self.state
        p = st.get_player(pid)
        
        # Skip if bag is empty (shouldn't happen but defensive check)
        if len(p.bag) == 0:
            if self.logger:
                self.logger.log(
                    EventType.INFO,
                    {
                        "event": "skipped_declare_empty_bag",
                        "reason": "Player has empty bag",
                    },
                    player_id=pid,
                    is_private=False,
                )
            # Finish current merchant and advance
            self.state.finish_current_merchant()
            self._advance_declare()
            return
        
        # Get declaration
        declared_type_str = action.data.get("declared_type")
        declared_count = action.data.get("declared_count")
        
        # Convert to LegalType
        declared_type = LegalType(declared_type_str) if declared_type_str else None
        
        # Auto-fill declaration if None (prevents NoneType errors)
        if declared_type is None or declared_count is None:
            auto_fill_declaration(p, st.card_defs)
            declared_type = p.declared_type
            declared_count = p.declared_count
        
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
                        "phase": "declare",
                        "error": str(e),
                    },
                    player_id=pid,
                    is_private=False,
                )
            raise
        
        # Set declaration
        p.declared_type = declared_type
        p.declared_count = declared_count
        
        # Add to game history
        st.game_history.append(
            f"Round {st.round_number+1}, Sheriff P{st.sheriff_idx}: "
            f"P{pid} declared {declared_count} {declared_type.value}"
        )
        
        # Log PUBLIC info only (no bag contents)
        if self.logger:
            self.logger.log(
                EventType.PLAYER_ACTION,
                {
                    "phase": "declare",
                    "declared_type": declared_type_str,
                    "declared_count": declared_count,
                },
                player_id=pid,
                is_private=False,
            )
            
            # Log PRIVATE info (for analytics/debugging)
            bag_cards = [st.get_card_def(cid) for cid in p.bag]
            # Pass declared_type and declared_count for deterministic classification
            bag_class = classify_bag(bag_cards, declared_type=p.declared_type, declared_count=p.declared_count)
            self.logger.log(
                EventType.PLAYER_ACTION,
                {
                    "phase": "declare",
                    "declared_type": declared_type_str,
                    "declared_count": declared_count,
                    "bag_class": bag_class,
                    "actual_bag": [card.name for card in bag_cards],
                },
                player_id=pid,
                is_private=True,  # Hidden until inspection
            )
        
        # Finish current merchant and advance
        self.state.finish_current_merchant()
        self._advance_declare()

    def _advance_declare(self):
        """Advance to next merchant in declare phase or move to negotiate phase."""
        st = self.state
        next_merchant = st.next_merchant()
        
        if next_merchant is not None:
            # Next merchant in queue
            st.round_step = next_merchant
        else:
            # Validate all merchants have declarations before moving to negotiate
            merchants = st.get_all_merchants()
            for m_pid in merchants:
                m_player = st.get_player(m_pid)
                if m_player.declared_type is None or m_player.declared_count is None:
                    raise RuntimeError(f"Cannot enter NEGOTIATE: Player {m_pid} has not declared.")
                if not isinstance(m_player.declared_count, int):
                    raise RuntimeError(f"Cannot enter NEGOTIATE: Player {m_pid} has invalid declared_count: {m_player.declared_count}")
            
            # Move to negotiate phase
            st.phase = Phase.NEGOTIATE
            st.phase_start_time = time.time()  # Start timeout tracking
            st.start_merchant_cycle()  # Reset merchant queue for negotiation
            st.round_step = st.next_merchant()  # Set first merchant
            st.negotiation_round = 1  # Start at round 1
            st.offers = {}
            st.sheriff_responses = {}  # Track which merchants sheriff has responded to
            
            if self.logger:
                self.logger.log(EventType.PHASE_CHANGE, {"new_phase": "negotiate", "negotiation_round": 1})

    def _handle_negotiate(self, pid: int, action: Action):
        """Handle negotiation phase action."""
        st = self.state
        
        if pid == st.sheriff_idx:
            # Sheriff responds or ends negotiation
            action_type = action.data.get("type", "")
            if action_type == "end_negotiate":
                # RULE: Sheriff may end negotiation anytime; undecided merchants = reject (0g)
                # This allows Sheriff to move to inspection after deciding on some merchants
                # Default to reject (0g) for merchants that never offered
                merchants = st.get_all_merchants()
                for merchant_pid in merchants:
                    if merchant_pid not in st.sheriff_responses:
                        st.sheriff_responses[merchant_pid] = {"decision": "reject", "gold": 0}
                
                # Move to inspect phase
                st.phase = Phase.INSPECT
                st.phase_start_time = time.time()  # Start timeout tracking
                st.start_inspect_cycle()  # Initialize inspect queue
                
                if self.logger:
                    self.logger.log(EventType.PHASE_CHANGE, {"new_phase": "inspect"})
            
            elif action_type == "respond":
                merchant_pid = action.data.get("merchant")
                decision = action.data.get("decision")  # "accept" or "reject"
                
                # Validate merchant ID - assertion for list indexing safety
                try:
                    merchant_pid = ensure_player_idx(merchant_pid, st.config.n_players, "merchant")
                except ValueError:
                    if self.logger:
                        self.logger.log(EventType.ERROR, {
                            "player_id": pid,
                            "error": f"Invalid merchant ID: {merchant_pid}"
                        })
                    return
                
                # Skip if already responded to this merchant in this round
                if merchant_pid in st.sheriff_responses:
                    return
                
                st.sheriff_responses[merchant_pid] = {"decision": decision, "gold": 0}
                
                if merchant_pid in st.offers:
                    offer = st.offers[merchant_pid]
                    offer.accepted = (decision == "accept")
                    
                    # Process offer if accepted
                    if offer.accepted:
                        merchant = st.get_player(merchant_pid)
                        sheriff = st.get_player(st.sheriff_idx)
                        
                        # Transfer gold (ensure non-negative)
                        actual_gold = min(offer.gold, merchant.gold)
                        merchant.gold -= actual_gold
                        sheriff.gold += actual_gold
                        
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
                                "phase": "negotiate",
                                "negotiation_round": st.negotiation_round,
                                "merchant": merchant_pid,
                                "decision": decision,
                                "gold_transferred": offer.gold if offer.accepted else 0,
                            },
                            player_id=pid
                        )
                
                # Check if all merchants have been responded to
                merchants = st.get_all_merchants()
                if len(st.sheriff_responses) >= len(merchants):
                    # Sheriff has responded to all merchants
                    if st.negotiation_round < st.config.max_negotiation_rounds:
                        # Start another negotiation round
                        st.negotiation_round += 1
                        st.offers = {}
                        st.sheriff_responses = {}
                        st.start_merchant_cycle()  # Reset merchant queue for next round
                        st.round_step = st.next_merchant()  # First merchant offers again
                        
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
                        st.phase_start_time = time.time()  # Start timeout tracking
                        st.start_inspect_cycle()  # Initialize inspect queue
                        
                        if self.logger:
                            self.logger.log(EventType.PHASE_CHANGE, {"new_phase": "inspect"})
        else:
            # Merchant makes offer
            merchant = st.get_player(pid)
            
            # Validate gold amount
            offered_gold = action.data.get("gold", 0)
            if offered_gold < 0:
                offered_gold = 0
            if offered_gold > merchant.gold:
                offered_gold = merchant.gold  # Cap at available gold
            
            # Validate stand_goods (must exist in merchant's stand)
            stand_goods = action.data.get("stand_goods", [])
            valid_stand_goods = []
            for cid in stand_goods:
                # Check if card exists in any of merchant's stand piles
                found = False
                for lt in LegalType:
                    if cid in merchant.stand_legal[lt]:
                        found = True
                        break
                if not found and cid in merchant.stand_contraband:
                    found = True
                if found:
                    valid_stand_goods.append(cid)
            
            # Validate bag_goods (must exist in merchant's bag)
            bag_goods = action.data.get("bag_goods", [])
            valid_bag_goods = [cid for cid in bag_goods if cid in merchant.bag]
            
            offer = Offer(
                from_pid=pid,
                to_pid=st.sheriff_idx,
                gold=offered_gold,
                stand_goods=valid_stand_goods,
                bag_goods=valid_bag_goods,
                promises=action.data.get("promises", []),
            )
            st.offers[pid] = offer
            
            # Add bribe offer to history
            if offered_gold > 0:
                st.game_history.append(
                    f"Round {st.round_number+1}: P{pid} offered {offered_gold} gold bribe to Sheriff P{st.sheriff_idx}"
                )
            
            # Log
            if self.logger:
                self.logger.log(
                    EventType.PLAYER_ACTION,
                    {
                        "phase": "negotiate",
                        "offer_gold": offer.gold,
                        "promises": offer.promises,
                    },
                    player_id=pid
                )
            
            # Advance to next merchant or back to sheriff
            self._advance_negotiate()

    def _advance_negotiate(self):
        """Advance negotiation phase."""
        st = self.state
        
        # Finish current merchant first
        st.finish_current_merchant()
        
        next_merchant = st.next_merchant()
        
        if next_merchant is not None:
            # Next merchant in queue
            st.round_step = next_merchant
        else:
            # All merchants have offered, sheriff's turn
            st.round_step = st.sheriff_idx

    def _handle_inspect(self, pid: int, action: Action):
        """Handle inspection phase action using queue system."""
        from .rules import compute_inspection_outcome
        
        st = self.state
        
        # Get current merchant from queue (NO merchant field needed from LLM!)
        merchant_pid = st.current_inspect_merchant()
        
        if merchant_pid is None:
            # Queue empty - all merchants processed, move to resolve
            st.phase = Phase.RESOLVE
            if self.logger:
                self.logger.log(EventType.PHASE_CHANGE, {"new_phase": "resolve"})
            return
        
        # Get choice from action (only field needed!)
        choice = action.data.get("choice")  # "inspect" or "pass"
        
        # Validate choice
        if choice not in ("inspect", "pass"):
            # Default to pass if invalid
            if self.logger:
                self.logger.log(EventType.ERROR, {
                    "player_id": pid,
                    "error": f"Invalid choice: {choice}, defaulting to 'pass'",
                    "merchant": merchant_pid
                }, is_private=False)
            choice = "pass"
        
        # Validate merchant_pid (belt-and-suspenders check)
        assert isinstance(merchant_pid, int), f"merchant_pid must be int, got {type(merchant_pid)}"
        assert 0 <= merchant_pid < st.config.n_players, f"merchant_pid {merchant_pid} out of range"
        assert merchant_pid != st.sheriff_idx, f"Cannot inspect sheriff (pid={merchant_pid})"
        
        # Mark as inspected and remove from queue
        st.finish_inspect_merchant()
        merchant = st.get_player(merchant_pid)
        sheriff = st.get_player(st.sheriff_idx)
        
        # CRITICAL FIX: Check if sheriff accepted a bribe from this merchant
        bribe_accepted = False
        bribe_amount = 0
        if merchant_pid in st.offers:
            offer = st.offers[merchant_pid]
            if offer.accepted:
                bribe_accepted = True
                bribe_amount = offer.gold
        
        if choice == "inspect":
            # If sheriff accepted bribe but still inspects, REFUND the bribe (idempotent)
            if bribe_accepted:
                refund_key = (st.sheriff_idx, merchant_pid)
                if refund_key not in st.refunded:
                    # Refund the bribe money
                    merchant.gold += bribe_amount
                    sheriff.gold -= bribe_amount
                    st.refunded.add(refund_key)
                    
                    # Log the refund
                    if self.logger:
                        self.logger.log(
                            EventType.INFO,
                            {
                                "event": "bribe_refunded",
                                "sheriff": st.sheriff_idx,
                                "merchant": merchant_pid,
                                "amount": bribe_amount,
                                "reason": "sheriff_inspected_after_accepting"
                            },
                            is_private=False,
                        )
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
            
            # Add inspection to history
            truthful_str = "HONEST" if outcome.truthful else "LYING"
            actual_cards = ", ".join([st.get_card_def(cid).name for cid in merchant.bag] if merchant.bag else ["(empty)"])
            st.game_history.append(
                f"Round {st.round_number+1}: Sheriff P{st.sheriff_idx} inspected P{merchant_pid} → {truthful_str} "
                f"(declared {merchant.declared_count} {merchant.declared_type.value if merchant.declared_type else 'nothing'}, "
                f"actual: {actual_cards})"
            )
            
            # Log PUBLIC reveal after inspection
            if self.logger:
                # Create detailed card information for transparency
                revealed_cards = []
                delivered_cards = []
                confiscated_cards = []
                
                for cid in outcome.delivered:
                    card = st.get_card_def(cid)
                    revealed_cards.append(card.name)
                    delivered_cards.append({"card_id": cid, "name": card.name})
                
                for cid in outcome.confiscated:
                    card = st.get_card_def(cid)
                    revealed_cards.append(card.name)
                    confiscated_cards.append({"card_id": cid, "name": card.name})
                
                self.logger.log(
                    EventType.PLAYER_ACTION,
                    {
                        "phase": "inspect",
                        "merchant": merchant_pid,
                        "choice": "inspect",
                        "truthful": outcome.truthful,
                        "revealed_cards": revealed_cards,
                        "delivered_cards": delivered_cards,  # Include both ID and name
                        "confiscated_cards": confiscated_cards,  # Include both ID and name
                        "penalties": {
                            "sheriff_delta": outcome.sheriff_delta,
                            "merchant_delta": outcome.merchant_delta,
                        },
                    },
                    player_id=pid,
                    is_private=False,
                )
        else:
            # Pass - distribute goods based on accepted offer
            cards_to_sheriff = set()
            if merchant_pid in st.offers and st.offers[merchant_pid].accepted:
                offer = st.offers[merchant_pid]
                cards_to_sheriff = set(offer.bag_goods)
            
            # Distribute cards
            for cid in merchant.bag:
                card = st.get_card_def(cid)
                
                # Determine destination
                if cid in cards_to_sheriff:
                    # This card goes to sheriff
                    if card.kind == CardKind.LEGAL:
                        for lt in LegalType:
                            if card.name == lt.value:
                                sheriff.stand_legal[lt].append(cid)
                                break
                    else:
                        sheriff.stand_contraband.append(cid)
                else:
                    # This card goes to merchant
                    if card.kind == CardKind.LEGAL:
                        for lt in LegalType:
                            if card.name == lt.value:
                                merchant.stand_legal[lt].append(cid)
                                break
                    else:
                        merchant.stand_contraband.append(cid)
            
            # Add pass to history
            st.game_history.append(
                f"Round {st.round_number+1}: Sheriff P{st.sheriff_idx} passed P{merchant_pid} without inspection"
            )
            
            # Log
            if self.logger:
                self.logger.log(
                    EventType.PLAYER_ACTION,
                    {
                        "phase": "inspect",
                        "merchant": merchant_pid,
                        "choice": "pass",
                    },
                    player_id=pid
                )
        
        # Clear merchant's bag
        merchant.clear_bag()
        
        # Check if all merchants have been inspected
        merchants = st.get_all_merchants()
        if len(st.inspected_merchants) >= len(merchants):
            # Assertion: merchant_queue should be empty when transitioning to RESOLVE
            assert len(st.merchant_queue) == 0, f"Merchant queue not empty when transitioning to RESOLVE: {st.merchant_queue}"
            
            # All merchants inspected, move to resolve
            st.phase = Phase.RESOLVE
            if self.logger:
                self.logger.log(EventType.PHASE_CHANGE, {"new_phase": "resolve"})

    def _handle_resolve(self) -> Dict[int, float]:
        """Handle resolve phase - rotate sheriff and check for game end."""
        st = self.state
        
        # Rotate sheriff
        st.rotation_counts[st.sheriff_idx] += 1
        old_sheriff_idx = st.sheriff_idx
        st.sheriff_idx = (st.sheriff_idx + 1) % st.config.n_players
        
        # Increment round_number when sheriff completes full cycle (back to player 0)
        if old_sheriff_idx == st.config.n_players - 1 and st.sheriff_idx == 0:
            st.round_number += 1
        
        # Log sheriff rotation
        if self.logger:
            self.logger.log(
                EventType.INFO,
                {
                    "event": "sheriff_rotation",
                    "old_sheriff": old_sheriff_idx,
                    "new_sheriff": st.sheriff_idx,
                    "round_number": st.round_number,
                    "rotation_counts": st.rotation_counts.copy(),
                },
                is_private=False,
            )
        
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
            st.phase_start_time = time.time()  # Start timeout tracking
            st.start_merchant_cycle()  # Initialize merchant queue
            st.round_step = st.next_merchant()  # Set first merchant
            st.offers = {}
            st.sheriff_responses = {}
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

    async def play_game(self) -> GameResult:
        """Play a complete game with the configured agents."""
        if not self.agents:
            raise RuntimeError("No agents configured")
        
        # Environment already initialized in __init__ (reset was called there)
        obs = self._get_observations()
        done = False
        num_rounds = 0
        max_retries = 2
        
        while not done:
            st = self.state
            
            # Get active player
            if st.phase in (Phase.INSPECT, Phase.RESOLVE):
                active_pid = st.sheriff_idx
            else:
                active_pid = st.round_step
            
            # Get action from agent with retry logic
            agent = self.agents[active_pid]
            action = None
            last_error = None
            
            for retry_attempt in range(max_retries + 1):
                try:
                    # Get action from agent
                    action = await agent.act_async(obs[active_pid])
                    
                    # Validate action before executing (quick pre-check)
                    if st.phase == Phase.DECLARE:
                        from .rules import validate_bag_and_declaration
                        p = st.get_player(active_pid)
                        
                        if len(p.bag) > 0:  # Only validate if bag has cards
                            declared_type_str = action.data.get("declared_type")
                            declared_count = action.data.get("declared_count")
                            declared_type = LegalType(declared_type_str) if declared_type_str else None
                            
                            validate_bag_and_declaration(
                                bag=p.bag,
                                declared_type=declared_type,
                                declared_count=declared_count,
                                bag_limit=st.config.bag_limit,
                            )
                    
                    # Action is valid, break out of retry loop
                    break
                    
                except Exception as e:
                    last_error = e
                    if retry_attempt < max_retries:
                        # Log retry attempt
                        if self.logger:
                            self.logger.log(
                                EventType.ERROR,
                                {
                                    "event": "action_retry",
                                    "phase": st.phase.value,
                                    "attempt": retry_attempt + 1,
                                    "max_retries": max_retries,
                                    "error": str(e)
                                },
                                player_id=active_pid,
                                is_private=False,
                            )
                        # Continue to next retry
                        continue
                    else:
                        # All retries exhausted, use fallback
                        if self.logger:
                            self.logger.log(
                                EventType.ERROR,
                                {
                                    "event": "action_retries_exhausted",
                                    "phase": st.phase.value,
                                    "error": str(e)
                                },
                                player_id=active_pid,
                                is_private=False,
                            )
                        
                        # Create fallback action for DECLARE phase
                        if st.phase == Phase.DECLARE:
                            p = st.get_player(active_pid)
                            if len(p.bag) > 0:
                                # AUTO-DECLARE: Find first legal card type or default to apples
                                bag_cards = [st.get_card_def(cid) for cid in p.bag]
                                default_type = None
                                for card in bag_cards:
                                    if card.kind == CardKind.LEGAL:
                                        for lt in LegalType:
                                            if card.name == lt.value:
                                                default_type = lt
                                                break
                                        if default_type:
                                            break
                                
                                if default_type is None:
                                    default_type = LegalType.APPLES
                                
                                action = Action(
                                    player_id=active_pid,
                                    data={
                                        "type": "declare",
                                        "declared_type": default_type.value,
                                        "declared_count": len(p.bag)
                                    }
                                )
                                
                                if self.logger:
                                    self.logger.log(
                                        EventType.INFO,
                                        {
                                            "event": "auto_declare_fallback",
                                            "reason": "All retries failed",
                                            "auto_type": default_type.value,
                                            "auto_count": len(p.bag)
                                        },
                                        player_id=active_pid,
                                        is_private=False,
                                    )
                        
                        # If still no action, re-raise the error
                        if action is None:
                            raise last_error
            
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

