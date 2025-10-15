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
            self.logger.log_event(
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
            
            obs[p.pid] = Observation(
                player_id=p.pid,
                obs_type=ObservationType.PRIVATE,
                phase=game_phase,
                data=data,
            )
        
        return obs

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
        
        # Get active player
        if st.phase in (Phase.INSPECT, Phase.RESOLVE):
            active_pid = st.sheriff_idx
        else:
            active_pid = st.round_step
        
        # Execute action
        if active_pid not in actions:
            raise ValueError(f"Active player {active_pid} must provide an action")
        
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
        elif st.phase == Phase.RESOLVE:
            rewards = self._handle_resolve()
            obs = self._get_observations()
            done = st.game_over
            return obs, rewards, done, {}
        
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
        draw_from = action_data.get("draw_from", [])
        while len(p.hand) < st.config.hand_size:
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
        
        # Log
        if self.logger:
            self.logger.log_event(
                EventType.ACTION,
                {
                    "player_id": pid,
                    "phase": "market",
                    "discarded": len(discard_ids),
                    "hand_size": len(p.hand),
                },
                is_private=True,
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
                self.logger.log_event(EventType.PHASE_CHANGE, {"new_phase": "load_bag"})

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
        
        # Log
        if self.logger:
            self.logger.log_event(
                EventType.ACTION,
                {
                    "player_id": pid,
                    "phase": "load_bag",
                    "bag_size": len(p.bag),
                },
                is_private=True,
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
            # Move to declare phase
            st.phase = Phase.DECLARE
            st.round_step = get_next_merchant_idx(st.sheriff_idx, 0, st.config.n_players)
            
            if self.logger:
                self.logger.log_event(EventType.PHASE_CHANGE, {"new_phase": "declare"})

    def _handle_declare(self, pid: int, action: Action):
        """Handle declaration phase action."""
        st = self.state
        p = st.get_player(pid)
        
        # Set declaration
        declared_type_str = action.data.get("declared_type")
        declared_count = action.data.get("declared_count")
        
        if declared_type_str:
            p.declared_type = LegalType(declared_type_str)
        p.declared_count = declared_count
        
        # Log
        if self.logger:
            bag_cards = [st.get_card_def(cid) for cid in p.bag]
            bag_class = classify_bag(bag_cards)
            
            self.logger.log_event(
                EventType.ACTION,
                {
                    "player_id": pid,
                    "phase": "declare",
                    "declared_type": declared_type_str,
                    "declared_count": declared_count,
                    "bag_class": bag_class,
                    "actual_bag": [card.name for card in bag_cards],
                },
                is_private=False,  # Declaration is public
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
            # Move to negotiate phase
            st.phase = Phase.NEGOTIATE
            st.round_step = get_next_merchant_idx(st.sheriff_idx, 0, st.config.n_players)
            st.negotiation_round = 0
            st.offers = {}
            
            if self.logger:
                self.logger.log_event(EventType.PHASE_CHANGE, {"new_phase": "negotiate"})

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
                    self.logger.log_event(EventType.PHASE_CHANGE, {"new_phase": "inspect"})
            
            elif action_type == "respond":
                merchant_pid = action.data.get("merchant")
                decision = action.data.get("decision")  # "accept" or "reject"
                
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
                        self.logger.log_event(
                            EventType.ACTION,
                            {
                                "player_id": pid,
                                "phase": "negotiate",
                                "merchant": merchant_pid,
                                "decision": decision,
                                "gold_transferred": offer.gold if offer.accepted else 0,
                            }
                        )
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
                self.logger.log_event(
                    EventType.ACTION,
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
        st = self.state
        
        action_type = action.data.get("type", "")
        if action_type == "resolve":
            # Move to resolve phase
            st.phase = Phase.RESOLVE
            if self.logger:
                self.logger.log_event(EventType.PHASE_CHANGE, {"new_phase": "resolve"})
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
            # Check truthfulness
            bag_cards = [st.get_card_def(cid) for cid in merchant.bag]
            truthful = is_declaration_truthful(
                bag_cards, merchant.declared_type, merchant.declared_count
            )
            
            # Calculate penalty
            penalty = calculate_inspection_penalty(
                bag_cards, merchant.declared_type, merchant.declared_count
            )
            
            if truthful:
                # Sheriff pays merchant
                self._transfer_penalty(sheriff, merchant, abs(penalty))
                
                # Merchant keeps goods in bag (go to stand)
                for cid in merchant.bag:
                    card = st.get_card_def(cid)
                    if card.kind == CardKind.LEGAL:
                        # Find matching legal type
                        for lt in LegalType:
                            if card.name == lt.value:
                                merchant.stand_legal[lt].append(cid)
                                break
                    else:
                        merchant.stand_contraband.append(cid)
            else:
                # Merchant pays sheriff
                self._transfer_penalty(merchant, sheriff, abs(penalty))
                
                # Confiscate contraband and illegal goods
                for cid in merchant.bag:
                    card = st.get_card_def(cid)
                    if card.kind == CardKind.LEGAL and card.name == merchant.declared_type.value:
                        # Correct legal goods go to merchant stand
                        merchant.stand_legal[merchant.declared_type].append(cid)
                    else:
                        # Contraband/illegal goods go to sheriff
                        if card.kind == CardKind.LEGAL:
                            for lt in LegalType:
                                if card.name == lt.value:
                                    sheriff.stand_legal[lt].append(cid)
                                    break
                        else:
                            sheriff.stand_contraband.append(cid)
            
            # Log
            if self.logger:
                self.logger.log_event(
                    EventType.ACTION,
                    {
                        "player_id": pid,
                        "phase": "inspect",
                        "merchant": merchant_pid,
                        "choice": "inspect",
                        "truthful": truthful,
                        "penalty": penalty,
                    }
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
                self.logger.log_event(
                    EventType.ACTION,
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
                self.logger.log_event(
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
                self.logger.log_event(
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
        
        obs = self.reset()
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
        
        return GameResult(
            winner=st.winner,
            num_rounds=num_rounds,
            final_scores=scores,
            players=[i for i in range(st.config.n_players)],
        )

