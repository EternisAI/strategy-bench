"""Secret Hitler game environment implementation."""

import asyncio
import random
from typing import List, Optional, Any, Dict

from sdb.core.base_env import BaseEnvironment
from sdb.core.base_agent import BaseAgent
from sdb.core.types import Action, ActionType, GamePhase, Observation
from sdb.core.exceptions import EnvironmentError, InvalidActionError
from sdb.environments.secret_hitler.config import SecretHitlerConfig
from sdb.environments.secret_hitler.state import SecretHitlerState
from sdb.environments.secret_hitler.types import (
    PlayerInfo, Role, Party, Policy, Phase, Vote, Government, PresidentialPower
)
from sdb.environments.secret_hitler.rules import PolicyDeck, GameRules
from sdb.environments.secret_hitler import prompts
from sdb.logging import GameLogger


class SecretHitlerEnv(BaseEnvironment):
    """Secret Hitler game environment.
    
    Implements the full Secret Hitler game with all rules, phases,
    and presidential powers.
    """
    
    def __init__(
        self,
        agents: List[BaseAgent],
        config: Optional[SecretHitlerConfig] = None,
        game_id: Optional[str] = None,
        logger: Optional[GameLogger] = None,
    ):
        """Initialize Secret Hitler environment.
        
        Args:
            agents: List of agents (must match config.n_players)
            config: Game configuration
            game_id: Unique game identifier
            logger: Game logger instance
        """
        config = config or SecretHitlerConfig(n_players=len(agents))
        
        # Set game config before calling super().__init__() because reset() needs it
        self.game_config = config
        self.logger = logger or GameLogger(game_id=game_id or "secret_hitler", enabled=True)
        self.rng = random.Random(config.seed)
        
        super().__init__(agents=agents, config=config.__dict__, game_id=game_id, seed=config.seed)
        
    def reset(self) -> Dict[int, Observation]:
        """Reset environment for new game.
        
        Returns:
            Initial game state
        """
        # Initialize roles
        roles_config = self.game_config.get_roles()
        roles = (
            [Role.LIBERAL] * roles_config["liberals"] +
            [Role.FASCIST] * roles_config["fascists"] +
            [Role.HITLER]
        )
        self.rng.shuffle(roles)
        
        # Create player info
        players = []
        for i, role in enumerate(roles):
            party = Party.FASCIST if role in [Role.FASCIST, Role.HITLER] else Party.LIBERAL
            players.append(PlayerInfo(
                player_id=i,
                role=role,
                party=party,
                is_alive=True,
                is_hitler=(role == Role.HITLER)
            ))
        
        # Initialize state
        self.state = SecretHitlerState(
            game_id=self.game_id,
            num_players=self.num_players,
            current_phase=GamePhase.SETUP,
            alive_players=list(range(self.num_players)),
            players=players,
            policy_deck=PolicyDeck(seed=self.game_config.seed),
            president_idx=self.rng.randint(0, self.num_players - 1)
        )
        
        self.logger.log_game_start({"config": self.config})
        
        # Log role assignments (private)
        if self.logger:
            self.logger.log(
                event_type=EventType.PLAYER_ACTION,
                data={
                    "action": "role_assignment",
                    "roles": [r.name for r in roles],
                    "role_map": {str(i): r.name for i, r in enumerate(roles)}
                },
                is_private=True
            )
            
            # Log agent metadata including model information
            agent_metadata = {}
            for i, agent in enumerate(self.agents):
                agent_info = {
                    "name": agent.name if hasattr(agent, 'name') else f"Agent_{i}",
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
            
            self.logger.log(
                event_type=EventType.GAME_START,
                data={
                    "action": "agent_metadata",
                    "agents": agent_metadata
                },
                is_private=True
            )
        
        # Notify agents of their roles
        for agent in self.agents:
            player = self.state.players[agent.player_id]
            agent.notify("game_start", {
                "role": player.role.name,
                "party": player.party.name,
                "is_hitler": player.is_hitler
            })
        
        # Return initial observations for each player
        observations = {}
        for i in range(self.num_players):
            player = self.state.players[i]
            observations[i] = Observation(
                player_id=i,
                obs_type="role_assignment",
                phase=GamePhase.SETUP,
                data={
                    "role": player.role.value,
                    "party": player.party.value,
                    "is_hitler": player.is_hitler,
                    "instruction": f"You are {player.role.value} ({player.party.value}). The game will begin shortly."
                }
            )
        return observations
    
    def step(self, action: Action) -> tuple[SecretHitlerState, bool]:
        """Execute one action in the game.
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple of (new_state, is_terminal)
        """
        if not self.state.is_action_legal(action):
            raise InvalidActionError(f"Illegal action: {action}")
        
        self.state.add_action(action)
        self.logger.log_action(
            player_id=action.player_id,
            action_type=action.action_type.name,
            target=action.target,
            data=action.data
        )
        
        # Process action based on phase
        # (Simplified - full implementation would handle all phases)
        
        return self.state, self.state.is_terminal
    
    async def play_game(self):
        """Play complete Secret Hitler game.
        
        Returns:
            GameResult with outcome
        """
        # Environment already initialized in __init__ (reset was called there)
        
        while not self.state.is_terminal:
            await self._run_round()
            
            # Check win conditions
            if self._check_game_over():
                break
        
        return self._build_game_result()
    
    async def _run_round(self):
        """Run one complete round of Secret Hitler."""
        self.state.round_number += 1
        self.logger.log_round_start(self.state.round_number)
        
        # 1. Nomination Phase
        await self._nomination_phase()
        
        # 2. Discussion Phase
        await self._discussion_phase()
        
        # 3. Voting Phase
        votes_passed = await self._voting_phase()
        
        if votes_passed:
            # 4. Legislative Session
            await self._legislative_session()
            
            # 5. Presidential Power (if applicable)
            if self.state.pending_power and self.state.pending_power != PresidentialPower.NONE:
                await self._execute_presidential_power()
                
                # Check if game ended (e.g., Hitler executed)
                if self.state.is_terminal:
                    return
            
            # Reset election tracker
            self.state.election_tracker = 0
            
            # Log tracker update
            self.logger.log(EventType.INFO, {
                "event": "election_tracker_reset",
                "new_value": 0,
                "reason": "government_passed"
            })
        else:
            # Election failed
            self._failed_election()
            
            # Log tracker update (after increment in _failed_election)
            self.logger.log(EventType.INFO, {
                "event": "election_tracker_updated",
                "new_value": self.state.election_tracker,
                "reason": "election_failed"
            })
            
            # Check if game ended (e.g., chaos policy triggered win)
            if self.state.is_terminal:
                return
        
        # 6. Advance to next president (unless just did special election)
        self._advance_president()
    
    async def _nomination_phase(self):
        """Handle nomination of chancellor."""
        self.state.phase = Phase.ELECTION_NOMINATION
        
        president = self.agents[self.state.president_idx]
        
        # Get legal candidates
        legal_candidates = self._get_legal_chancellor_candidates()
        
        # For random agents, just pick randomly
        if hasattr(president, '__class__') and 'Random' in president.__class__.__name__:
            nominee = self.rng.choice(legal_candidates)
        else:
            # For smart agents, give them observation
            obs = self.state.get_observation(self.state.president_idx)
            obs.data["legal_candidates"] = legal_candidates
            obs.data["action_required"] = "nominate_chancellor"
            # Add full context instruction
            obs.data["instruction"] = prompts.get_nomination_instruction(legal_candidates, obs.data)
            
            try:
                action = await president.act_async(obs)
                self._log_agent_reasoning(action, self.state.president_idx)
                
                # FIX: Check if target is not None, not just if attribute exists
                nominee = action.target if action.target is not None else action.data.get('nominee')
                
                if nominee not in legal_candidates:
                    self.logger.log_error("nomination", f"Invalid nominee {nominee} from {legal_candidates}")
                    # Let the agent try again with clearer instructions
                    obs.data["error"] = f"Invalid nominee {nominee}. Must be one of {legal_candidates}"
                    action = await president.act_async(obs)
                    self._log_agent_reasoning(action, self.state.president_idx)
                    # FIX: Check if target is not None, not just if attribute exists
                    nominee = action.target if action.target is not None else action.data.get('nominee')
                    
                    # If still invalid, pick first legal as last resort
                    if nominee not in legal_candidates:
                        nominee = legal_candidates[0]
                        self.logger.log_error("nomination", f"Still invalid, using first legal: {nominee}")
            except Exception as e:
                self.logger.log_error("nomination", f"Agent error: {str(e)}")
                raise  # Re-raise to let error handling deal with it
        
        self.state.chancellor_nominee = nominee
        self.logger.log(EventType.PLAYER_NOMINATE, {
            "president": self.state.president_idx,
            "nominee": nominee
        })
        
        # Log to terminal [[memory:7216156]]
        print(f"   👔 President {self.state.president_idx} nominates Player {nominee} for Chancellor")
    
    async def _discussion_phase(self):
        """Handle pre-vote discussion where players can speak."""
        self.state.phase = Phase.ELECTION_DISCUSSION
        self.state.current_discussion = []
        
        # Each alive player can make a statement about the nomination
        discussion_order = self.state.alive_players.copy()
        self.rng.shuffle(discussion_order)
        
        print(f"   💬 Discussion phase...")
        
        for player_id in discussion_order:
            obs = self.state.get_observation(player_id)
            obs.data["action_required"] = "discuss_nomination"
            obs.data["president"] = self.state.president_idx
            obs.data["chancellor_nominee"] = self.state.chancellor_nominee
            obs.data["discussion_is_public"] = True
            obs.data["previous_statements"] = self.state.current_discussion.copy()
            # Use proper prompt with JSON format
            obs.data["instruction"] = prompts.get_discussion_instruction({
                'president': self.state.president_idx,
                'nominee': self.state.chancellor_nominee,
                'previous_statements': self.state.current_discussion.copy()
            })
            
            try:
                # Agent can make a statement (returns Action with statement in data)
                agent = self.agents[player_id]
                action = await agent.act_async(obs)
                self._log_agent_reasoning(action, player_id)
                
                # Extract statement from action (try multiple field names for robustness)
                statement = action.data.get("statement") or action.data.get("text") or action.data.get("parameter") or ""
                if statement and len(statement.strip()) > 0:
                    discussion_entry = {
                        "speaker": player_id,
                        "statement": statement.strip(),
                        "context": "nomination_discussion"
                    }
                    self.state.current_discussion.append(discussion_entry)
                    # Discussions are PUBLIC - all players can hear them
                    self.logger.log(EventType.DISCUSSION, discussion_entry, is_private=False)
                    print(f"      💬 Player {player_id}: \"{statement[:60]}...\"")
                    
                    # Notify ALL agents about this PUBLIC statement (add to their memories)
                    for agent_id in self.state.alive_players:
                        if hasattr(self.agents[agent_id], 'notify'):
                            self.agents[agent_id].notify("discussion_statement", {
                                "speaker": player_id,
                                "statement": statement.strip(),
                                "context": "nomination_discussion",
                                "president": self.state.president_idx,
                                "chancellor_nominee": self.state.chancellor_nominee
                            })
            except Exception as e:
                # If agent fails during discussion, they stay silent
                pass
    
    async def _veto_discussion_phase(self):
        """Handle discussion when chancellor proposes veto."""
        self.state.phase = Phase.VETO_DISCUSSION
        
        print(f"   💬 Veto discussion phase...")
        
        # All alive players can comment on the veto proposal
        discussion_order = self.state.alive_players.copy()
        self.rng.shuffle(discussion_order)
        
        for player_id in discussion_order:
            obs = self.state.get_observation(player_id)
            obs.data["action_required"] = "discuss_veto"
            obs.data["president"] = self.state.president_idx
            obs.data["chancellor"] = self.state.last_government.chancellor
            obs.data["discussion_is_public"] = True
            obs.data["previous_statements"] = self.state.current_discussion.copy()
            # Use proper prompt with JSON format
            obs.data["instruction"] = prompts.get_veto_discussion_instruction({
                'president': self.state.president_idx,
                'chancellor': self.state.last_government.chancellor,
                'previous_statements': self.state.current_discussion.copy()
            })
            
            try:
                agent = self.agents[player_id]
                action = await agent.act_async(obs)
                self._log_agent_reasoning(action, player_id)
                
                # Extract statement (try multiple field names for robustness)
                statement = action.data.get("statement") or action.data.get("text") or action.data.get("parameter") or ""
                if statement and len(statement.strip()) > 0:
                    discussion_entry = {
                        "speaker": player_id,
                        "statement": statement.strip(),
                        "context": "veto_discussion"
                    }
                    self.state.current_discussion.append(discussion_entry)
                    # Veto discussions are PUBLIC - all players can hear them
                    self.logger.log(EventType.DISCUSSION, discussion_entry, is_private=False)
                    print(f"      💬 Player {player_id}: \"{statement[:60]}...\"")
                    
                    # Notify ALL agents about this PUBLIC veto discussion statement
                    for agent_id in self.state.alive_players:
                        if hasattr(self.agents[agent_id], 'notify'):
                            self.agents[agent_id].notify("discussion_statement", {
                                "speaker": player_id,
                                "statement": statement.strip(),
                                "context": "veto_discussion",
                                "president": self.state.president_idx,
                                "chancellor": self.state.last_government.chancellor
                            })
            except Exception:
                # If agent fails, they stay silent
                pass
    
    def _enact_chaos_policy(self):
        """Enact top policy from deck due to 3 failed elections (chaos)."""
        policy = self.state.policy_deck.draw(1)[0]
        
        if policy == Policy.LIBERAL:
            self.state.liberal_policies += 1
        else:
            self.state.fascist_policies += 1
            # Unlock veto if needed
            if self.state.fascist_policies >= 5:
                self.state.veto_unlocked = True
        
        self.logger.log(EventType.POLICY_ENACTED, {
            "policy": policy.name,
            "cause": "chaos",
            "liberal_total": self.state.liberal_policies,
            "fascist_total": self.state.fascist_policies
        })
        
        # Reset election tracker
        self.state.election_tracker = 0
        
        # Log to terminal
        policy_emoji = "🔵" if policy == Policy.LIBERAL else "🔴"
        print(f"   ⚡ CHAOS! Top policy enacted: {policy_emoji} {policy.name}")
        print(f"      Board: 🔵 {self.state.liberal_policies}/5 Liberal, 🔴 {self.state.fascist_policies}/6 Fascist")
    
    async def _voting_phase(self) -> bool:
        """Handle voting on government.
        
        Returns:
            True if election passed
        """
        self.state.phase = Phase.ELECTION_VOTING
        self.state.votes = []
        
        ja_count = 0
        
        # Collect votes from all alive players
        for player_id in self.state.alive_players:
            obs = self.state.get_observation(player_id)
            agent = self.agents[player_id]
            
            try:
                action = await agent.act_async(obs)
                self._log_agent_reasoning(action, player_id)
                # Interpret vote from action data
                vote_ja = action.data.get("vote", self.rng.choice([True, False]))
            except Exception:
                # Random vote on error
                vote_ja = self.rng.choice([True, False])
            
            self.state.votes.append(Vote(voter=player_id, ja=vote_ja))
            if vote_ja:
                ja_count += 1
            
            # Log each vote
            self.logger.log(EventType.VOTE_CAST, {
                "vote": vote_ja
            }, player_id=player_id)
        
        # Check if passed
        passed = GameRules.check_election_passed(ja_count, len(self.state.alive_players))
        
        # Log election result with tracker BEFORE update (for comparison)
        tracker_before = self.state.election_tracker
        
        self.logger.log(EventType.ELECTION_RESULT, {
            "passed": passed,
            "ja_votes": ja_count,
            "nein_votes": len(self.state.alive_players) - ja_count,
            "election_tracker_before": tracker_before,
            "will_reset_tracker": passed,
            "will_increment_tracker": not passed
        })
        
        # Log to terminal
        vote_emoji = "✅" if passed else "❌"
        print(f"   {vote_emoji} Election {'PASSED' if passed else 'FAILED'} ({ja_count} Ja, {len(self.state.alive_players) - ja_count} Nein)")
        print(f"      Election Tracker: {tracker_before} → {'0 (reset)' if passed else tracker_before + 1}")
        
        if passed:
            # Check Hitler election after 3 fascist policies
            if self.state.fascist_policies >= 3:
                nominee = self.state.players[self.state.chancellor_nominee]
                if nominee.is_hitler:
                    self._win(Party.FASCIST, "Hitler elected Chancellor after 3 Fascist policies")
                    return True
                else:
                    self.state.confirmed_not_hitler.add(self.state.chancellor_nominee)
            
            # Set government
            self.state.last_government = Government(
                president=self.state.president_idx,
                chancellor=self.state.chancellor_nominee
            )
        
        return passed
    
    async def _legislative_session(self):
        """Handle legislative session (policy selection)."""
        self.state.phase = Phase.LEGISLATIVE_SESSION
        
        # Draw 3 policies
        policies = self.state.policy_deck.draw(3)
        self.state.president_hand = policies
        
        # President discards 1
        president_obs = self.state.get_observation(self.state.president_idx)
        president_obs.data["policies"] = [p.name for p in policies]
        
        print(f"   🎴 President {self.state.president_idx} draws: {[p.name for p in policies]}")
        
        try:
            action = await self.agents[self.state.president_idx].act_async(president_obs)
            self._log_agent_reasoning(action, self.state.president_idx)
            discard_idx = action.data.get("discard", 0)
            discard_idx = max(0, min(2, discard_idx))  # Clamp to 0-2
            
            # Show president's thinking
            reasoning = action.metadata.get("reasoning", "")
            if reasoning:
                # Extract key reasoning (first 150 chars)
                reasoning_preview = reasoning[:150].replace("\n", " ")
                print(f"      💭 President thinks: {reasoning_preview}...")
        except Exception:
            discard_idx = self.rng.randint(0, 2)
        
        discarded = policies.pop(discard_idx)
        print(f"      ➡️  President discards {discarded.name}, passes {[p.name for p in policies]} to Chancellor")
        self.state.policy_deck.discard_policy(discarded)
        self.state.chancellor_hand = policies
        
        # Chancellor can propose veto if unlocked (5+ Fascist policies)
        veto_proposed = False
        if self.state.veto_unlocked:
            chancellor_obs = self.state.get_observation(self.state.last_government.chancellor)
            chancellor_obs.data["policies"] = [p.name for p in policies]
            chancellor_obs.data["veto_available"] = True
            
            try:
                action = await self.agents[self.state.last_government.chancellor].act_async(chancellor_obs)
                self._log_agent_reasoning(action, self.state.last_government.chancellor)
                
                # Check if chancellor proposes veto
                if action.data.get("propose_veto", False):
                    veto_proposed = True
                    
                    # Show chancellor's veto reasoning
                    reasoning = action.metadata.get("reasoning", "")
                    if reasoning:
                        reasoning_preview = reasoning[:150].replace("\n", " ")
                        print(f"      💭 Chancellor thinks: {reasoning_preview}...")
                    
                    self.logger.log(EventType.VETO_PROPOSED, {
                        "president": self.state.president_idx,
                        "chancellor": self.state.last_government.chancellor
                    })
                    print(f"   🚨 Chancellor {self.state.last_government.chancellor} proposes VETO!")
                    
                    # Conduct veto discussion
                    await self._veto_discussion_phase()
                    
                    # President decides whether to accept veto
                    president_obs = self.state.get_observation(self.state.president_idx)
                    president_obs.data["action_required"] = "veto_response"
                    president_obs.data["chancellor"] = self.state.last_government.chancellor
                    
                    try:
                        pres_action = await self.agents[self.state.president_idx].act_async(president_obs)
                        self._log_agent_reasoning(pres_action, self.state.president_idx)
                        veto_accepted = pres_action.data.get("accept_veto", False)
                        
                        # Show president's veto response reasoning
                        reasoning = pres_action.metadata.get("reasoning", "")
                        if reasoning:
                            reasoning_preview = reasoning[:150].replace("\n", " ")
                            print(f"      💭 President thinks: {reasoning_preview}...")
                    except Exception:
                        veto_accepted = False  # Default: reject veto
                    
                    self.logger.log(EventType.VETO_RESPONSE, {
                        "president": self.state.president_idx,
                        "accepted": veto_accepted
                    })
                    
                    if veto_accepted:
                        # Both policies discarded, advance election tracker
                        self.state.policy_deck.discard_policy(policies[0])
                        self.state.policy_deck.discard_policy(policies[1])
                        self.state.election_tracker += 1
                        print(f"   ✅ President accepts veto - both policies discarded!")
                        
                        # Check for chaos
                        if self.state.election_tracker >= 3:
                            self._enact_chaos_policy()
                        return
                    else:
                        print(f"   ❌ President rejects veto - Chancellor must enact a policy")
                        veto_proposed = False  # Continue to normal enactment
            except Exception:
                pass
        
        # Chancellor enacts 1 policy (either veto was rejected or not available)
        if not veto_proposed:
            chancellor_obs = self.state.get_observation(self.state.last_government.chancellor)
            chancellor_obs.data["policies"] = [p.name for p in policies]
            chancellor_obs.data["veto_available"] = False  # Veto rejected or not available
            
            print(f"   🎴 Chancellor {self.state.last_government.chancellor} receives: {[p.name for p in policies]}")
            
            try:
                action = await self.agents[self.state.last_government.chancellor].act_async(chancellor_obs)
                self._log_agent_reasoning(action, self.state.last_government.chancellor)
                enact_idx = action.data.get("enact", 0)
                enact_idx = max(0, min(1, enact_idx))  # Clamp to 0-1
                
                # Show chancellor's thinking
                reasoning = action.metadata.get("reasoning", "")
                if reasoning:
                    # Extract key reasoning (first 150 chars)
                    reasoning_preview = reasoning[:150].replace("\n", " ")
                    print(f"      💭 Chancellor thinks: {reasoning_preview}...")
            except Exception:
                enact_idx = self.rng.randint(0, 1)
            
            enacted = policies[enact_idx]
            discarded = policies[1 - enact_idx]
            print(f"      ✅ Chancellor enacts {enacted.name}, discards {discarded.name}")
            self.state.policy_deck.discard_policy(discarded)
            
            # Update policy counts
            if enacted == Policy.LIBERAL:
                self.state.liberal_policies += 1
            else:
                self.state.fascist_policies += 1
                # Check for presidential power
                power = self.game_config.get_presidential_power(self.state.fascist_policies)
                if power != PresidentialPower.NONE:
                    self.state.pending_power = power
                
                # Unlock veto after 5 Fascist policies
                if self.state.fascist_policies >= 5:
                    self.state.veto_unlocked = True
                    if self.state.fascist_policies == 5:
                        print(f"   🔓 VETO POWER UNLOCKED!")
            
            self.logger.log(EventType.POLICY_ENACTED, {
                "policy": enacted.name,
                "liberal_total": self.state.liberal_policies,
                "fascist_total": self.state.fascist_policies
            })
            
            # Log to terminal
            policy_emoji = "🔵" if enacted == Policy.LIBERAL else "🔴"
            print(f"   📜 Policy enacted: {policy_emoji} {enacted.name}")
            print(f"      Board: 🔵 {self.state.liberal_policies}/5 Liberal, 🔴 {self.state.fascist_policies}/6 Fascist")
    
    async def _execute_presidential_power(self):
        """Execute presidential power."""
        power = self.state.pending_power
        
        if power == PresidentialPower.INVESTIGATE_LOYALTY:
            await self._power_investigate()
        elif power == PresidentialPower.EXECUTION:
            await self._power_execution()
        elif power == PresidentialPower.POLICY_PEEK:
            self._power_policy_peek()
        elif power == PresidentialPower.CALL_SPECIAL_ELECTION:
            await self._power_special_election()
        
        self.state.pending_power = None
    
    async def _power_investigate(self):
        """Investigate a player's party membership."""
        legal_targets = [p for p in self.state.alive_players 
                        if p != self.state.president_idx and p not in self.state.investigated_players]
        
        if not legal_targets:
            return
        
        # President chooses target
        obs = self.state.get_observation(self.state.president_idx)
        obs.data["power"] = "investigate"
        obs.data["legal_targets"] = legal_targets
        
        try:
            action = await self.agents[self.state.president_idx].act_async(obs)
            self._log_agent_reasoning(action, self.state.president_idx)
            target = action.target if action.target in legal_targets else self.rng.choice(legal_targets)
        except Exception:
            target = self.rng.choice(legal_targets)
        
        # Reveal party to president
        target_party = self.state.players[target].party
        self.agents[self.state.president_idx].notify("investigation_result", {
            "target": target,
            "party": target_party.name
        })
        
        self.state.investigated_players.append(target)
        
        # Log investigation
        self.logger.log(EventType.PRESIDENTIAL_POWER, {
            "power": "INVESTIGATE_LOYALTY",
            "target": target
        }, player_id=self.state.president_idx, is_private=True)
        
        self.logger.log(EventType.INVESTIGATION_RESULT, {
            "target": target,
            "party": target_party.name
        }, player_id=self.state.president_idx, is_private=True)
        
        # Log to terminal
        print(f"   🔍 President {self.state.president_idx} investigates Player {target}")
    
    async def _power_execution(self):
        """Execute a player."""
        legal_targets = [p for p in self.state.alive_players if p != self.state.president_idx]
        
        if not legal_targets:
            return
        
        obs = self.state.get_observation(self.state.president_idx)
        obs.data["power"] = "execution"
        obs.data["legal_targets"] = legal_targets
        
        try:
            action = await self.agents[self.state.president_idx].act_async(obs)
            self._log_agent_reasoning(action, self.state.president_idx)
            target = action.target if action.target in legal_targets else self.rng.choice(legal_targets)
        except Exception:
            target = self.rng.choice(legal_targets)
        
        # Execute player
        self.state.eliminate_player(target)
        
        # Log execution
        was_hitler = self.state.players[target].is_hitler
        self.logger.log(EventType.PLAYER_ELIMINATED, {
            "player_id": target,
            "reason": "executed",
            "was_hitler": was_hitler
        })
        
        # Log to terminal
        print(f"   💀 President {self.state.president_idx} executes Player {target}")
        if was_hitler:
            print(f"      👑 Player {target} was HITLER!")
        
        # Check if Hitler was executed (liberal victory)
        if was_hitler:
            self._win(Party.LIBERAL, "Hitler was executed")
        
        # Notify all players
        for agent in self.agents:
            agent.notify("player_executed", {"player_id": target})
    
    def _power_policy_peek(self):
        """Peek at top 3 policies."""
        top_policies = self.state.policy_deck.peek_top(3)
        self.agents[self.state.president_idx].notify("policy_peek", {
            "policies": [p.name for p in top_policies]
        })
    
    async def _power_special_election(self):
        """Call special election."""
        legal_targets = [p for p in self.state.alive_players if p != self.state.president_idx]
        
        obs = self.state.get_observation(self.state.president_idx)
        obs.data["power"] = "special_election"
        obs.data["legal_targets"] = legal_targets
        
        try:
            action = await self.agents[self.state.president_idx].act_async(obs)
            self._log_agent_reasoning(action, self.state.president_idx)
            target = action.target if action.target in legal_targets else self.rng.choice(legal_targets)
        except Exception:
            target = self.rng.choice(legal_targets)
        
        # Set up special election
        self.state.special_election_return_to = self.state.president_idx
        self.state.president_idx = target
    
    def _log_agent_reasoning(self, action: Action, player_id: int):
        """Log agent reasoning if present in action metadata."""
        if "reasoning" in action.metadata:
            self.logger.log(EventType.AGENT_REASONING, {
                "player_id": player_id,
                "agent_name": action.metadata.get("agent_name", f"Player_{player_id}"),
                "reasoning": action.metadata["reasoning"]
            }, player_id=player_id, is_private=True)
    
    def _advance_president(self):
        """Advance to next president, skipping dead players."""
        # Check if this was a special election
        if self.state.special_election_return_to is not None:
            # Return to the president after the one who called special election
            self.state.president_idx = self.state.special_election_return_to
            self.state.special_election_return_to = None
        
        # Move to next alive player
        next_idx = (self.state.president_idx + 1) % self.num_players
        while next_idx not in self.state.alive_players:
            next_idx = (next_idx + 1) % self.num_players
        
        self.state.president_idx = next_idx
    
    def _failed_election(self):
        """Handle failed election."""
        self.state.election_tracker += 1
        
        # Warn about impending chaos
        if self.state.election_tracker == 1:
            print(f"      ⚠️  Election Tracker: 1/3 (2 more failures = chaos)")
        elif self.state.election_tracker == 2:
            print(f"      🚨 Election Tracker: 2/3 (1 more failure = CHAOS!)")
        
        if self.state.election_tracker >= 3:
            # Chaos - enact top policy
            print(f"      💥 3 CONSECUTIVE FAILURES! Chaos policy will be enacted...")
            self._enact_chaos_policy()
    
    def _check_game_over(self) -> bool:
        """Check if game is over.
        
        Returns:
            True if game should end
        """
        # Check if already terminal
        if self.state.is_terminal:
            return True
        
        # Liberal victory - 5 liberal policies
        if self.state.liberal_policies >= 5:
            self._win(Party.LIBERAL, "5 Liberal policies enacted")
            return True
        
        # Liberal victory - Hitler assassinated
        hitler_dead = any(not p.is_alive and p.is_hitler for p in self.state.players)
        if hitler_dead:
            self._win(Party.LIBERAL, "Hitler was assassinated")
            return True
        
        # Fascist victory - 6 fascist policies
        if self.state.fascist_policies >= 6:
            self._win(Party.FASCIST, "6 Fascist policies enacted")
            return True
        
        return False
    
    def _win(self, party: Party, reason: str):
        """Set game winner."""
        self.state.is_terminal = True
        self.state.metadata["winner"] = party.name
        self.state.metadata["win_reason"] = reason
        
        # Log game end
        self.logger.log(EventType.GAME_END, {
            "winner": party.name,
            "reason": reason,
            "liberal_policies": self.state.liberal_policies,
            "fascist_policies": self.state.fascist_policies,
            "rounds": self.state.round_number
        })
        
        # Log to terminal
        print()
        print(f"🏁 GAME OVER - {party.name} WINS!")
        print(f"   {reason}")
    
    def _get_legal_chancellor_candidates(self) -> List[int]:
        """Get legal chancellor candidates."""
        candidates = [p for p in self.state.alive_players if p != self.state.president_idx]
        
        # Apply term limits
        if self.state.last_government:
            last_p = self.state.last_government.president
            last_c = self.state.last_government.chancellor
            
            if len(self.state.alive_players) > 5:
                candidates = [c for c in candidates if c not in [last_p, last_c]]
            else:
                candidates = [c for c in candidates if c != last_c]
        
        return candidates
    
    def _get_current_player(self) -> int:
        """Get current player ID."""
        if self.state.phase == Phase.ELECTION_NOMINATION:
            return self.state.president_idx
        return self.state.alive_players[0]  # Simplified
    
    def _validate_num_players(self):
        """Validate player count."""
        if not (5 <= self.num_players <= 10):
            raise EnvironmentError(f"Secret Hitler requires 5-10 players, got {self.num_players}")
    
    def get_winner(self):
        """Get game winner."""
        return self.state.metadata.get("winner")
    
    def get_win_reason(self):
        """Get win reason."""
        return self.state.metadata.get("win_reason")
    
    def _build_game_result(self):
        """Build GameResult from current game state."""
        from sdb.core.types import GameResult
        
        winner = self.get_winner() or "timeout"
        win_reason = self.get_win_reason() or "Game ended"
        
        # Calculate player stats
        player_stats = {}
        for player_info in self.state.players:
            pid = player_info.player_id
            party = player_info.party
            is_hitler = player_info.role == Role.HITLER
            role = "hitler" if is_hitler else party.value
            
            # Winner is based on party
            is_winner = (winner == "liberals" and party.value == "liberal") or \
                       (winner == "fascists" and party.value == "fascist")
            
            player_stats[pid] = {
                "score": 1.0 if is_winner else 0.0,
                "party": party.value,
                "role": role,
                "survived": pid in self.state.alive_players,
            }
        
        return GameResult(
            game_id=self.game_id,
            winner=winner,
            win_reason=win_reason,
            num_rounds=self.state.round_number,
            duration_seconds=0.0,
            player_stats=player_stats,
            metadata={
                "liberal_policies": self.state.liberal_policies,
                "fascist_policies": self.state.fascist_policies,
                "election_tracker": self.state.election_tracker,
            }
        )


# Import for logging
from sdb.logging.formats import EventType

