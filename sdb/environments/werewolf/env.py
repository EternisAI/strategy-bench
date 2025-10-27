"""Werewolf game environment."""

import random
from typing import Dict, List, Optional, Tuple

from sdb.core.base_env import BaseEnvironment
from sdb.core.types import Action, GameResult, Observation
from sdb.environments.werewolf.config import WerewolfConfig
from sdb.environments.werewolf.rules import (
    assign_roles,
    check_win_condition,
    get_max_bidders,
    get_team_for_role,
    get_vote_result,
    validate_bid,
    validate_night_action,
    validate_vote,
)
from sdb.environments.werewolf.state import WerewolfState
from sdb.environments.werewolf.types import (
    DayResult,
    NightResult,
    Phase,
    PlayerState,
    Role,
    Team,
)
from sdb.logging.formats import EventType


class WerewolfEnv(BaseEnvironment):
    """Werewolf game environment with night/day cycles."""
    
    def __init__(self, agents, config=None, game_id=None, logger=None, role_assignment=None):
        """Initialize Werewolf environment.
        
        Args:
            agents: List of agents (must match config.n_players)
            config: WerewolfConfig instance
            game_id: Optional game ID
            logger: Optional GameLogger instance
            role_assignment: Optional dict with 'villagers' and 'werewolves' player indices
        """
        config = config or WerewolfConfig()
        self.game_config = config
        self.logger = logger
        self.role_assignment = role_assignment  # Store for use in reset()
        super().__init__(agents, config=config.__dict__, game_id=game_id, seed=getattr(config, 'seed', None))
    
    def _validate_num_players(self):
        """Validate number of players."""
        if not (5 <= self.num_players <= 20):
            raise EnvironmentError(f"Werewolf requires 5-20 players, got {self.num_players}")
    
    def _get_current_player(self) -> Optional[int]:
        """Get the current player (for observation generation)."""
        # In night phases, return the specific role's player
        if self.state.phase == Phase.NIGHT_WEREWOLF:
            werewolves = self.state.get_alive_werewolves()
            return werewolves[0] if werewolves else None
        elif self.state.phase == Phase.NIGHT_DOCTOR:
            return self.state.get_player_by_role(Role.DOCTOR)
        elif self.state.phase == Phase.NIGHT_SEER:
            return self.state.get_player_by_role(Role.SEER)
        elif self.state.phase == Phase.DAY_DEBATE:
            return self.state.current_speaker
        # For bidding and voting, all players act
        return None
    
    def get_winner(self) -> Optional[str]:
        """Get the winning team."""
        if self.state.winner:
            return self.state.winner.value
        return None
    
    def get_win_reason(self) -> str:
        """Get the reason for winning."""
        return self.state.win_reason
    
    def reset(self) -> Dict[int, Observation]:
        """Reset the game state."""
        # Initialize state
        self.state = WerewolfState()
        
        # Assign roles
        if self.role_assignment:
            # Use fixed role assignment from tournament schedule
            villager_indices = self.role_assignment.get('villagers', [])
            werewolf_indices = self.role_assignment.get('werewolves', [])
            
            # Create roles array with fixed assignments
            roles = [None] * self.num_players
            
            # Assign werewolves first
            for idx in werewolf_indices:
                roles[idx] = Role.WEREWOLF
            
            # Assign special roles (Seer, Doctor) to villager slots
            villager_roles = []
            if self.game_config.include_seer:
                villager_roles.append(Role.SEER)
            if self.game_config.include_doctor:
                villager_roles.append(Role.DOCTOR)
            
            # Fill remaining villager slots with regular villagers
            n_regular_villagers = len(villager_indices) - len(villager_roles)
            villager_roles.extend([Role.VILLAGER] * n_regular_villagers)
            
            # Assign villager roles to villager indices
            for i, idx in enumerate(villager_indices):
                roles[idx] = villager_roles[i]
        else:
            # Default random assignment
            roles = assign_roles(self.game_config)
        
        # Create player states
        self.state.players = {}
        for i, role in enumerate(roles):
            self.state.players[i] = PlayerState(
                player_id=i,
                name=self.agents[i].name,
                role=role,
                team=get_team_for_role(role),
                is_alive=True
            )
        
        # Initialize state
        self.state.phase = Phase.NIGHT_WEREWOLF
        self.state.round_number = 1
        self.state.current_speaker = None
        self.state.night_results = []
        self.state.day_results = []
        self.state.current_night = NightResult()
        self.state.current_day = DayResult()
        self.state.winner = None
        self.state.win_reason = ""
        
        # Log game start
        if self.logger:
            self.logger.log(
                event_type=EventType.GAME_START,
                data={
                    "n_players": self.game_config.n_players,
                    "n_werewolves": self.game_config.n_werewolves,
                }
            )
            
            # Log role assignments (private)
            self.logger.log(
                event_type=EventType.PLAYER_ACTION,
                data={
                    "action": "role_assignment",
                    "roles": [r.value for r in roles],
                },
                is_private=True
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
            
            self.logger.log(
                event_type=EventType.GAME_START,
                data={
                    "action": "agent_metadata",
                    "agents": agent_metadata
                },
                is_private=True
            )
        
        # Give initial observations
        for pid in self.state.players:
            self._give_role_information(pid)
        
        return self._get_observations()
    
    def _give_role_information(self, player_id: int):
        """Give a player their role information."""
        player = self.state.players[player_id]
        
        # Basic role info
        self.state.add_observation(
            player_id,
            f"You are a {player.role.value.upper()}."
        )
        
        # Werewolves learn about each other
        if player.role == Role.WEREWOLF:
            other_wolves = [
                self.state.players[pid].name
                for pid in self.state.get_alive_werewolves()
                if pid != player_id
            ]
            if other_wolves:
                self.state.add_observation(
                    player_id,
                    f"Your fellow werewolf(ves): {', '.join(other_wolves)}"
                )
    
    def step(self, actions: Dict[int, Action]) -> Tuple[Dict[int, Observation], Dict[int, float], bool, Dict]:
        """Execute one step of the game.
        
        Args:
            actions: Dictionary mapping player_id to Action
        """
        # Process actions based on phase
        # For night phases, typically only one actor; for day phases, multiple actors
        for player_id, action in actions.items():
            if self.state.phase == Phase.NIGHT_WEREWOLF:
                self._handle_werewolf_action(action)
            elif self.state.phase == Phase.NIGHT_DOCTOR:
                self._handle_doctor_action(action)
            elif self.state.phase == Phase.NIGHT_SEER:
                self._handle_seer_action(action)
            elif self.state.phase == Phase.DAY_BIDDING:
                self._handle_bidding_action(action)
            elif self.state.phase == Phase.DAY_DEBATE:
                self._handle_debate_action(action)
            elif self.state.phase == Phase.DAY_VOTING:
                self._handle_voting_action(action)
        
        # Get observations
        observations = self._get_observations()
        
        # Check if game is over
        done = self.state.phase == Phase.GAME_END
        
        # Calculate rewards (only at game end)
        rewards = {}
        if done:
            for pid, player in self.state.players.items():
                rewards[pid] = 1.0 if player.team == self.state.winner else 0.0
        else:
            rewards = {pid: 0.0 for pid in self.state.players}
        
        info = {"phase": self.state.phase.value}
        
        return observations, rewards, done, info
    
    def _handle_werewolf_action(self, action: Action):
        """Handle werewolf elimination choice."""
        target = action.data.get("target")
        
        if target is None:
            self.logger.log(
                EventType.ERROR,
                {"player_id": action.player_id, "error": "No target specified"}
            )
            # Move to next phase anyway
            self.state.phase = Phase.NIGHT_DOCTOR if self.game_config.include_doctor else (
                Phase.NIGHT_SEER if self.game_config.include_seer else Phase.DAY_BIDDING
            )
            return
        
        # Validate action
        alive = self.state.get_alive_players()
        valid, error = validate_night_action("eliminate", target, alive, action.player_id)
        
        if not valid:
            self.logger.log(
                EventType.ERROR,
                {"player_id": action.player_id, "error": error}
            )
            return
        
        # Record target
        self.state.current_night.werewolf_target = target
        
        # Log action
        if self.logger:
            self.logger.log(
                EventType.PLAYER_ACTION,
                {
                    "round": self.state.round_number,
                    "actor": action.player_id,
                    "action": "eliminate",
                    "target": target,
                },
                is_private=True
            )
        
        # Add to werewolves' observations
        for wolf_id in self.state.get_alive_werewolves():
            self.state.add_observation(
                wolf_id,
                f"We chose to eliminate {self.state.players[target].name}."
            )
        
        # Move to next phase
        if self.game_config.include_doctor and self.state.is_player_alive(
            self.state.get_player_by_role(Role.DOCTOR)
        ):
            self.state.phase = Phase.NIGHT_DOCTOR
        elif self.game_config.include_seer and self.state.is_player_alive(
            self.state.get_player_by_role(Role.SEER)
        ):
            self.state.phase = Phase.NIGHT_SEER
        else:
            self._resolve_night()
    
    def _handle_doctor_action(self, action: Action):
        """Handle doctor protection choice."""
        target = action.data.get("target")
        
        if target is None:
            # Move to next phase
            self.state.phase = Phase.NIGHT_SEER if self.game_config.include_seer else Phase.DAY_BIDDING
            self._resolve_night()
            return
        
        # Validate action
        alive = self.state.get_alive_players()
        valid, error = validate_night_action("protect", target, alive, action.player_id)
        
        if not valid:
            self.logger.log(
                EventType.ERROR,
                {"player_id": action.player_id, "error": error}
            )
            return
        
        # Record protection
        self.state.current_night.doctor_target = target
        
        # Log action
        if self.logger:
            self.logger.log(
                EventType.PLAYER_ACTION,
                {
                    "round": self.state.round_number,
                    "actor": action.player_id,
                    "action": "protect",
                    "target": target,
                },
                is_private=True
            )
        
        # Add to doctor's observations
        self.state.add_observation(
            action.player_id,
            f"I protected {self.state.players[target].name}."
        )
        
        # Move to next phase
        if self.game_config.include_seer and self.state.is_player_alive(
            self.state.get_player_by_role(Role.SEER)
        ):
            self.state.phase = Phase.NIGHT_SEER
        else:
            self._resolve_night()
    
    def _handle_seer_action(self, action: Action):
        """Handle seer investigation."""
        target = action.data.get("target")
        
        if target is None:
            # Move to day
            self._resolve_night()
            return
        
        # Validate action
        alive = self.state.get_alive_players()
        valid, error = validate_night_action("investigate", target, alive, action.player_id)
        
        if not valid:
            self.logger.log(
                EventType.ERROR,
                {"player_id": action.player_id, "error": error}
            )
            return
        
        # Record investigation
        self.state.current_night.seer_target = target
        revealed_role = self.state.players[target].role
        self.state.current_night.seer_result = revealed_role
        
        # Log action
        if self.logger:
            self.logger.log(
                EventType.PLAYER_ACTION,
                {
                    "round": self.state.round_number,
                    "actor": action.player_id,
                    "action": "investigate",
                    "target": target,
                    "result": revealed_role.value,
                },
                is_private=True
            )
        
        # Add to seer's observations
        self.state.add_observation(
            action.player_id,
            f"I investigated {self.state.players[target].name} and learned they are a {revealed_role.value.upper()}."
        )
        
        # Resolve night and move to day
        self._resolve_night()
    
    def _resolve_night(self):
        """Resolve night actions and move to day."""
        # Determine if anyone dies
        target = self.state.current_night.werewolf_target
        protected = self.state.current_night.doctor_target
        investigated = self.state.current_night.seer_target
        
        # Log comprehensive night results
        if self.logger:
            night_log = {
                "round": self.state.round_number,
                "werewolf_target": target,
                "werewolf_target_name": self.state.players[target].name if target is not None else None,
                "doctor_target": protected,
                "doctor_target_name": self.state.players[protected].name if protected is not None else None,
                "seer_target": investigated,
                "seer_target_name": self.state.players[investigated].name if investigated is not None else None,
                "kill_successful": target is not None and target != protected,
                "kill_blocked": target is not None and target == protected,
            }
            self.logger.log(
                EventType.PHASE_CHANGE,
                night_log,
                is_private=False
            )
        
        if target is not None and target != protected:
            # Player dies
            self.state.players[target].is_alive = False
            self.state.current_night.eliminated_player = target
            
            # Broadcast death
            self.state.broadcast_observation(
                f"Moderator Announcement: {self.state.players[target].name} was eliminated during the night!",
                include_dead=False
            )
            
            # Log death
            if self.logger:
                self.logger.log(
                    EventType.PLAYER_ELIMINATED,
                    {
                        "round": self.state.round_number,
                        "player_id": target,
                        "player_name": self.state.players[target].name,
                        "role": self.state.players[target].role.value,
                        "by": "werewolves",
                    }
                )
        elif target is not None and target == protected:
            # Protection saved the target
            self.state.broadcast_observation(
                "Moderator Announcement: No one was eliminated during the night. The Doctor's protection was successful!",
                include_dead=False
            )
        else:
            # No target (werewolves didn't act or skipped)
            self.state.broadcast_observation(
                "Moderator Announcement: No one was eliminated during the night.",
                include_dead=False
            )
        
        # Save night results
        self.state.night_results.append(self.state.current_night)
        self.state.current_night = NightResult()
        
        # Check win condition
        if self._check_game_over():
            return
        
        # Move to day phase
        self.state.phase = Phase.DAY_BIDDING
        self.state.current_day = DayResult()
    
    def _handle_bidding_action(self, action: Action):
        """Handle bidding to speak."""
        bid = action.data.get("bid", 0)
        
        # Validate bid
        valid, error = validate_bid(bid)
        if not valid:
            self.logger.log(
                EventType.ERROR,
                {"player_id": action.player_id, "error": error}
            )
            bid = 0
        
        # Store bid
        if not self.state.current_day.bids:
            self.state.current_day.bids = [{}]
        
        self.state.current_day.bids[-1][action.player_id] = bid
        
        # Check if all alive players have bid
        alive = self.state.get_alive_players()
        last_bids = self.state.current_day.bids[-1]
        
        # Exclude previous speaker from bidding
        previous_speaker = self.state.current_day.debate[-1][0] if self.state.current_day.debate else None
        expected_bidders = [p for p in alive if p != previous_speaker]
        
        if len(last_bids) >= len(expected_bidders):
            # All players have bid, determine speaker
            max_bidders = get_max_bidders(last_bids)
            
            if len(max_bidders) == 1:
                self.state.current_speaker = max_bidders[0]
                self.state.phase = Phase.DAY_DEBATE
            else:
                # Tie, pick randomly
                self.state.current_speaker = random.choice(max_bidders)
                self.state.phase = Phase.DAY_DEBATE
    
    def _handle_debate_action(self, action: Action):
        """Handle debate statement."""
        statement = action.data.get("statement", "")
        
        if not statement:
            self.logger.log(
                EventType.ERROR,
                {"player_id": action.player_id, "error": "Empty statement"}
            )
            # Skip to next turn
            self._next_debate_turn()
            return
        
        # Add statement to debate
        self.state.current_day.debate.append((action.player_id, statement))
        
        # Broadcast statement
        speaker_name = self.state.players[action.player_id].name
        self.state.broadcast_observation(
            f"{speaker_name}: {statement}",
            include_dead=False
        )
        
        # Log debate
        if self.logger:
            self.logger.log(
                EventType.DISCUSSION,
                {
                    "round": self.state.round_number,
                    "player_id": action.player_id,
                    "player_name": speaker_name,
                    "statement": statement,
                },
                is_private=False
            )
        
        # Check if debate is complete
        if len(self.state.current_day.debate) >= self.game_config.max_debate_turns:
            # Move to voting
            self.state.phase = Phase.DAY_VOTING
            self.state.current_speaker = None
        else:
            # Next bidding round
            self._next_debate_turn()
    
    def _next_debate_turn(self):
        """Move to next debate turn."""
        self.state.phase = Phase.DAY_BIDDING
        self.state.current_speaker = None
        self.state.current_day.bids.append({})
    
    def _handle_voting_action(self, action: Action):
        """Handle voting to eliminate."""
        target = action.data.get("target")
        
        if target is None:
            self.logger.log(
                EventType.ERROR,
                {"player_id": action.player_id, "error": "No target specified"}
            )
            return
        
        # Validate vote
        alive = self.state.get_alive_players()
        valid, error = validate_vote(action.player_id, target, alive)
        
        if not valid:
            self.logger.log(
                EventType.ERROR,
                {"player_id": action.player_id, "error": error}
            )
            return
        
        # Record vote
        self.state.current_day.votes[action.player_id] = target
        
        # Log vote
        if self.logger:
            self.logger.log(
                EventType.VOTE_CAST,
                {
                    "round": self.state.round_number,
                    "voter": action.player_id,
                    "target": target,
                },
                is_private=True
            )
        
        # Check if all alive players have voted
        if len(self.state.current_day.votes) >= len(alive):
            self._resolve_day_vote()
    
    def _resolve_day_vote(self):
        """Resolve the day vote."""
        alive = self.state.get_alive_players()
        eliminated, vote_count = get_vote_result(
            self.state.current_day.votes,
            n_alive=len(alive),
            require_majority=self.game_config.vote_requires_majority
        )
        
        # Log complete vote results
        if self.logger:
            # Count votes per target
            vote_counts = {}
            for voter, target in self.state.current_day.votes.items():
                vote_counts[target] = vote_counts.get(target, 0) + 1
            
            self.logger.log(
                EventType.ELECTION_RESULT,
                {
                    "round": self.state.round_number,
                    "votes": self.state.current_day.votes,  # voter_id -> target_id
                    "vote_counts": vote_counts,  # target_id -> count
                    "eliminated": eliminated,
                    "vote_count": vote_count,
                    "tied": eliminated is None,
                },
                is_private=False
            )
        
        if eliminated is not None:
            # Someone was eliminated
            self.state.players[eliminated].is_alive = False
            self.state.current_day.eliminated_player = eliminated
            
            # Broadcast result
            eliminated_player = self.state.players[eliminated]
            self.state.broadcast_observation(
                f"Moderator Announcement: {eliminated_player.name} was voted out! "
                f"They were a {eliminated_player.role.value.upper()}.",
                include_dead=True
            )
            
            # Log elimination
            if self.logger:
                self.logger.log(
                    EventType.PLAYER_ELIMINATED,
                    {
                        "round": self.state.round_number,
                        "player_id": eliminated,
                        "player_name": eliminated_player.name,
                        "role": eliminated_player.role.value,
                        "by": "vote",
                        "vote_count": vote_count,
                    }
                )
        else:
            # No elimination (tie or no majority)
            # Determine reason from vote counts
            vote_counts = {}
            for voter, target in self.state.current_day.votes.items():
                vote_counts[target] = vote_counts.get(target, 0) + 1
            
            if len(vote_counts) == 0:
                reason = "No votes were cast."
            elif len([k for k, v in vote_counts.items() if v == max(vote_counts.values())]) > 1:
                reason = "The vote was tied."
            elif self.game_config.vote_requires_majority:
                reason = "No player received a majority of votes."
            else:
                reason = "The vote did not result in an elimination."
            
            self.state.broadcast_observation(
                f"Moderator Announcement: {reason} No one was eliminated.",
                include_dead=False
            )
        
        # Save day results
        self.state.day_results.append(self.state.current_day)
        self.state.current_day = DayResult()
        
        # Check win condition
        if self._check_game_over():
            return
        
        # Move to next round
        self.state.round_number += 1
        self.state.phase = Phase.NIGHT_WEREWOLF
        
        # Check round limit
        if self.state.round_number > self.game_config.max_rounds:
            self.state.winner = None
            self.state.win_reason = "Maximum rounds reached (draw)"
            self.state.phase = Phase.GAME_END
    
    def _check_game_over(self) -> bool:
        """Check if game is over."""
        alive_werewolves = len(self.state.get_alive_werewolves())
        alive_villagers = len(self.state.get_alive_villagers())
        
        game_over, winner, reason = check_win_condition(alive_werewolves, alive_villagers)
        
        if game_over:
            self.state.winner = winner
            self.state.win_reason = reason
            self.state.phase = Phase.GAME_END
            
            if self.logger:
                self.logger.log(
                    EventType.GAME_END,
                    {
                        "winner": winner.value if winner else "draw",
                        "reason": reason,
                        "rounds": self.state.round_number,
                    }
                )
            
            return True
        
        return False
    
    def _format_player_observations(self, player_id: int) -> str:
        """Format all private observations for a player.
        
        Args:
            player_id: ID of the player
            
        Returns:
            Formatted string of all observations
        """
        player = self.state.players[player_id]
        if not player.observations:
            return "None yet."
        
        formatted = []
        for i, obs in enumerate(player.observations, 1):
            formatted.append(f"   {i}. {obs}")
        return "\n".join(formatted)
    
    def _format_debate_history(self) -> str:
        """Format the complete debate history for the current day.
        
        Returns:
            Formatted string of debate statements
        """
        if not self.state.current_day.debate:
            return "   (Debate has not started)"
        
        formatted = []
        for speaker_id, statement in self.state.current_day.debate:
            speaker_name = self.state.players[speaker_id].name
            formatted.append(f"   • {speaker_name}: \"{statement}\"")
        return "\n".join(formatted)
    
    def _format_previous_rounds_summary(self) -> str:
        """Format summary of previous rounds (who was eliminated).
        
        Returns:
            Formatted string of round results
        """
        if not self.state.day_results:
            return "   (First round)"
        
        formatted = []
        for round_idx, day_result in enumerate(self.state.day_results, start=1):
            if day_result.eliminated_player is not None:
                elim_name = self.state.players[day_result.eliminated_player].name
                elim_role = self.state.players[day_result.eliminated_player].role.value
                formatted.append(f"   • Round {round_idx}: {elim_name} ({elim_role}) was eliminated")
            else:
                formatted.append(f"   • Round {round_idx}: No elimination")
        return "\n".join(formatted) if formatted else "   (First round)"
    
    def _get_observations(self) -> Dict[int, Observation]:
        """Generate observations for all players."""
        observations = {}
        alive = self.state.get_alive_players()
        
        for pid in self.state.players:
            player = self.state.players[pid]
            
            # Base observation data
            obs_data = {
                "round": self.state.round_number,
                "alive_players": [self.state.players[p].name for p in alive],
                "n_alive": len(alive),
                "observations": player.observations,
                "is_alive": player.is_alive,
            }
            
            # Add phase-specific instructions
            instruction = ""
            obs_type = "observe"  # Default to observe
            
            if not player.is_alive:
                instruction = "You are dead. You can only observe."
                obs_type = "observe"
            
            elif self.state.phase == Phase.NIGHT_WEREWOLF:
                if player.role == Role.WEREWOLF:
                    werewolf_ids = self.state.get_alive_werewolves()
                    teammates = [self.state.players[wid].name for wid in werewolf_ids if wid != pid]
                    teammates_str = ", ".join(teammates) if teammates else "none (you are alone)"
                    
                    targets_list = [f"Player {p} ({self.state.players[p].name})" for p in alive if p not in werewolf_ids]
                    
                    instruction = f"""=== NIGHT {self.state.round_number} - WEREWOLF ACTION ===

YOUR ROLE: Werewolf (Evil Team)
YOUR TEAMMATES: {teammates_str}

🔍 PREVIOUS ELIMINATIONS:
{self._format_previous_rounds_summary()}

⚡ ACTION REQUIRED:
Choose a player to eliminate tonight.

Available targets: {targets_list}

Respond with JSON using the numeric player_id: {{"type": "eliminate", "target": <player_id>}}
Example: {{"type": "eliminate", "target": 0}}"""
                    obs_type = "act"
                else:
                    instruction = "NIGHT PHASE: You are asleep."
                    obs_type = "observe"
            
            elif self.state.phase == Phase.NIGHT_DOCTOR:
                if player.role == Role.DOCTOR:
                    targets_list = [f"Player {p} ({self.state.players[p].name})" for p in alive]
                    
                    instruction = f"""=== NIGHT {self.state.round_number} - DOCTOR ACTION ===

YOUR ROLE: Doctor (Good Team)

🔍 PREVIOUS ELIMINATIONS:
{self._format_previous_rounds_summary()}

⚡ ACTION REQUIRED:
Choose a player to protect from Werewolf elimination tonight.

Available targets: {targets_list}

Strategy: Protect suspected power roles (Seer) or vulnerable villagers.

Respond with JSON using the numeric player_id: {{"type": "protect", "target": <player_id>}}
Example: {{"type": "protect", "target": 2}}"""
                    obs_type = "act"
                else:
                    instruction = "NIGHT PHASE: You are asleep."
                    obs_type = "observe"
            
            elif self.state.phase == Phase.NIGHT_SEER:
                if player.role == Role.SEER:
                    targets_list = [f"Player {p} ({self.state.players[p].name})" for p in alive if p != pid]
                    
                    instruction = f"""=== NIGHT {self.state.round_number} - SEER ACTION ===

YOUR ROLE: Seer (Good Team) - You can investigate players to learn their role!

🔮 YOUR PRIVATE OBSERVATIONS (Keep secret!):
{self._format_player_observations(pid)}

🔍 PREVIOUS ELIMINATIONS:
{self._format_previous_rounds_summary()}

⚡ ACTION REQUIRED:
Choose a player to investigate and learn if they are Werewolf or not.

Available targets: {targets_list}

Strategy: Investigate suspicious players or confirm trusted allies.

Respond with JSON using the numeric player_id: {{"type": "investigate", "target": <player_id>}}
Example: {{"type": "investigate", "target": 3}}"""
                    obs_type = "act"
                else:
                    instruction = "NIGHT PHASE: You are asleep."
                    obs_type = "observe"
            
            elif self.state.phase == Phase.DAY_BIDDING:
                # Check if this player should bid
                previous_speaker = self.state.current_day.debate[-1][0] if self.state.current_day.debate else None
                if pid == previous_speaker:
                    instruction = "Waiting for others to bid."
                    obs_type = "observe"
                else:
                    turns_left = self.game_config.max_debate_turns - len(self.state.current_day.debate)
                    
                    # Include recent observations (like night kill announcements)
                    recent_obs = ""
                    if player.observations:
                        last_3_obs = player.observations[-3:]  # Show last 3 observations
                        recent_obs = "\n📰 RECENT EVENTS:\n" + "\n".join([f"   • {obs}" for obs in last_3_obs]) + "\n"
                    
                    instruction = f"""=== DAY {self.state.round_number} - BIDDING TO SPEAK ===

YOUR ROLE: {player.role.value} ({player.team.value} Team)
{recent_obs}
🔍 PREVIOUS ELIMINATIONS:
{self._format_previous_rounds_summary()}

💬 DEBATE SO FAR THIS ROUND:
{self._format_debate_history()}

⚡ ACTION REQUIRED:
Bid 0-4 to speak next. Higher bids speak first. {turns_left} debate turns remaining.

BID OPTIONS:
  0: Observe and listen for now
  1: General thoughts to share
  2: Something critical to contribute
  3: Absolutely urgent to speak
  4: Must respond directly to someone

Strategy: {"Consider revealing your Seer information or defending yourself!" if player.role == Role.SEER and player.observations else "Listen carefully and identify suspicious behavior."}

Respond with JSON: {{"type": "bid", "bid": <0-4>}}"""
                    obs_type = "act"
                    obs_data["current_debate"] = [
                        f"{self.state.players[speaker].name}: {stmt}"
                        for speaker, stmt in self.state.current_day.debate
                    ]
            
            elif self.state.phase == Phase.DAY_DEBATE:
                if pid == self.state.current_speaker:
                    role_strategy = ""
                    if player.role == Role.WEREWOLF:
                        role_strategy = "\n🐺 WEREWOLF STRATEGY: Sow chaos, cast suspicion on Villagers, deflect from yourself and teammates. Deception is your weapon!"
                    elif player.role == Role.SEER and player.observations:
                        role_strategy = f"\n🔮 SEER STRATEGY: You have private information! Consider revealing it strategically, but beware - revealing makes you a Werewolf target!"
                    elif player.role in [Role.DOCTOR, Role.VILLAGER]:
                        role_strategy = "\n👥 VILLAGER STRATEGY: Scrutinize accusations, expose inconsistencies, call out suspicious behavior. Work together to find Werewolves!"
                    
                    # Include recent observations (like night kill announcements)
                    recent_obs = ""
                    if player.observations:
                        last_3_obs = player.observations[-3:]  # Show last 3 observations
                        recent_obs = "\n📰 RECENT EVENTS:\n" + "\n".join([f"   • {obs}" for obs in last_3_obs]) + "\n"
                    
                    instruction = f"""=== DAY {self.state.round_number} - YOUR TURN TO SPEAK ===

YOUR ROLE: {player.role.value} ({player.team.value} Team){role_strategy}
{recent_obs}
🔍 PREVIOUS ELIMINATIONS:
{self._format_previous_rounds_summary()}

💬 DEBATE SO FAR THIS ROUND:
{self._format_debate_history()}

⚡ ACTION REQUIRED:
You won the bid! Make a public statement. Be strategic and persuasive!

Respond with JSON: {{"type": "debate", "statement": "<your statement>"}}"""
                    obs_type = "act"
                else:
                    instruction = f"Waiting for {self.state.players[self.state.current_speaker].name} to speak."
                    obs_type = "observe"
                
                obs_data["current_debate"] = [
                    f"{self.state.players[speaker].name}: {stmt}"
                    for speaker, stmt in self.state.current_day.debate
                ]
            
            elif self.state.phase == Phase.DAY_VOTING:
                role_strategy = ""
                if player.role == Role.WEREWOLF:
                    werewolf_ids = self.state.get_alive_werewolves()
                    role_strategy = "\n🐺 WEREWOLF STRATEGY: Target influential Villagers. If Villagers suspect one of their own, join the vote against them!"
                elif player.role == Role.SEER and player.observations:
                    role_strategy = "\n🔮 SEER STRATEGY: Vote based on your investigations! Look for Werewolves or inconsistencies."
                else:
                    role_strategy = "\n👥 VILLAGER STRATEGY: Look for inconsistencies, deflection, discord-sowing, or unusually quiet players."
                
                targets_list = [f"Player {p} ({self.state.players[p].name})" for p in alive if p != pid]
                
                # Add majority requirement notice
                majority_text = ""
                if self.game_config.vote_requires_majority:
                    from math import floor
                    required = floor(len(alive) / 2) + 1
                    majority_text = f"\n⚖️  MAJORITY REQUIRED: {required}/{len(alive)} votes needed to eliminate. If no majority, no elimination occurs."
                
                # Include recent observations (like night kill announcements)
                recent_obs = ""
                if player.observations:
                    last_3_obs = player.observations[-3:]  # Show last 3 observations
                    recent_obs = "\n📰 RECENT EVENTS:\n" + "\n".join([f"   • {obs}" for obs in last_3_obs]) + "\n"
                
                instruction = f"""=== DAY {self.state.round_number} - VOTING ===

YOUR ROLE: {player.role.value} ({player.team.value} Team){role_strategy}
{recent_obs}
🔍 PREVIOUS ELIMINATIONS:
{self._format_previous_rounds_summary()}

💬 COMPLETE DEBATE THIS ROUND:
{self._format_debate_history()}

⚡ ACTION REQUIRED:
Vote to eliminate a player. Your vote is PRIVATE.{majority_text}

Available targets: {targets_list}

Respond with JSON using the numeric player_id: {{"type": "vote", "target": <player_id>}}
Example: {{"type": "vote", "target": 1}}"""
                obs_type = "act"
                obs_data["current_debate"] = [
                    f"{self.state.players[speaker].name}: {stmt}"
                    for speaker, stmt in self.state.current_day.debate
                ]
            
            elif self.state.phase == Phase.GAME_END:
                instruction = f"Game over! {self.state.winner.value if self.state.winner else 'Draw'}"
                obs_type = "observe"
            
            else:
                instruction = "Waiting..."
                obs_type = "observe"
            
            obs_data["instruction"] = instruction
            obs_data["type"] = obs_type  # Add obs_type to data for script filtering
            
            observations[pid] = Observation(
                player_id=pid,
                obs_type=obs_type,
                phase=self.state.phase.value,
                data=obs_data
            )
        
        return observations
    
    async def play_game(self):
        """Play a complete Werewolf game with configured agents.
        
        Returns:
            GameResult with winner, scores, and stats
        """
        import asyncio
        from sdb.core.types import GameResult
        
        if not self.agents:
            raise RuntimeError("No agents configured")
        
        # Environment already initialized in __init__ (reset was called there)
        round_count = 0
        max_rounds = 100  # Safety limit
        
        while not self.state.winner and round_count < max_rounds:
            round_count += 1
            
            # Get observations
            obs = self._get_observations()
            
            # Collect players who need to act
            act_players = [
                (player_id, observation)
                for player_id, observation in obs.items()
                if observation.data.get("type") == "act" and observation.data.get("instruction")
            ]
            
            # Call agents in parallel
            actions = {}
            if act_players:
                tasks = []
                for pid, observation in act_players:
                    agent = self.agents[pid]
                    # Check if agent has async method
                    if hasattr(agent, 'act_async'):
                        tasks.append(agent.act_async(observation))
                    else:
                        # Wrap sync call in coroutine
                        async def sync_act(a, o):
                            return a.act(o)
                        tasks.append(sync_act(agent, observation))
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Collect successful actions
                for (pid, _), result in zip(act_players, results):
                    if not isinstance(result, Exception):
                        actions[pid] = result
            
            # Step environment
            if actions:
                obs, rewards, done, info = self.step(actions)
            
            if self.state.winner:
                break
        
        # Build result
        winner = self.get_winner() or "timeout"
        win_reason = self.get_win_reason() or "Game reached maximum rounds"
        
        # Calculate player stats
        player_stats = {}
        for pid, player in self.state.players.items():
            if self.state.winner:
                if self.state.winner.value == "werewolves":
                    score = 1.0 if player.role.value == "werewolf" else 0.0
                else:  # villagers
                    score = 1.0 if player.role.value != "werewolf" else 0.0
            else:  # timeout
                score = 0.0
            
            player_stats[pid] = {
                "score": score,
                "role": player.role.value,
                "survived": player.is_alive,
            }
        
        alive_players = self.state.get_alive_players()
        
        return GameResult(
            game_id=self.game_id,
            winner=winner,
            win_reason=win_reason,
            num_rounds=round_count,
            duration_seconds=0.0,
            player_stats=player_stats,
            metadata={
                "players_remaining": len(alive_players),
                "night_phases": len(self.state.night_results),
                "day_phases": len(self.state.day_results),
            }
        )

