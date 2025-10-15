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
    
    def __init__(self, agents, config=None, game_id=None, logger=None):
        """Initialize Werewolf environment.
        
        Args:
            agents: List of agents (must match config.n_players)
            config: WerewolfConfig instance
            game_id: Optional game ID
            logger: Optional GameLogger instance
        """
        self.config = config or WerewolfConfig()
        super().__init__(agents, game_id, logger)
        self.state: WerewolfState = WerewolfState()
    
    def _validate_num_players(self, num_players: int) -> bool:
        """Validate number of players."""
        return 5 <= num_players <= 20
    
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
        # Assign roles
        roles = assign_roles(self.config)
        
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
            self.logger.log_event(
                event_type=EventType.GAME_START,
                data={
                    "n_players": self.config.n_players,
                    "n_werewolves": self.config.n_werewolves,
                    "roles": [r.value for r in roles],
                }
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
    
    def step(self, action: Action) -> Tuple[Dict[int, Observation], Dict[int, float], bool, Dict]:
        """Execute one step of the game."""
        # Handle action based on phase
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
            self.logger.log_event(
                EventType.ERROR,
                {"player_id": action.player_id, "error": "No target specified"}
            )
            # Move to next phase anyway
            self.state.phase = Phase.NIGHT_DOCTOR if self.config.include_doctor else (
                Phase.NIGHT_SEER if self.config.include_seer else Phase.DAY_BIDDING
            )
            return
        
        # Validate action
        alive = self.state.get_alive_players()
        valid, error = validate_night_action("eliminate", target, alive, action.player_id)
        
        if not valid:
            self.logger.log_event(
                EventType.ERROR,
                {"player_id": action.player_id, "error": error}
            )
            return
        
        # Record target
        self.state.current_night.werewolf_target = target
        
        # Log action
        if self.logger:
            self.logger.log_event(
                EventType.NIGHT_ACTION,
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
        if self.config.include_doctor and self.state.is_player_alive(
            self.state.get_player_by_role(Role.DOCTOR)
        ):
            self.state.phase = Phase.NIGHT_DOCTOR
        elif self.config.include_seer and self.state.is_player_alive(
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
            self.state.phase = Phase.NIGHT_SEER if self.config.include_seer else Phase.DAY_BIDDING
            self._resolve_night()
            return
        
        # Validate action
        alive = self.state.get_alive_players()
        valid, error = validate_night_action("protect", target, alive, action.player_id)
        
        if not valid:
            self.logger.log_event(
                EventType.ERROR,
                {"player_id": action.player_id, "error": error}
            )
            return
        
        # Record protection
        self.state.current_night.doctor_target = target
        
        # Log action
        if self.logger:
            self.logger.log_event(
                EventType.NIGHT_ACTION,
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
        if self.config.include_seer and self.state.is_player_alive(
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
            self.logger.log_event(
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
            self.logger.log_event(
                EventType.NIGHT_ACTION,
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
                self.logger.log_event(
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
            self.logger.log_event(
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
            self.logger.log_event(
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
            self.logger.log_event(
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
        if len(self.state.current_day.debate) >= self.config.max_debate_turns:
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
            self.logger.log_event(
                EventType.ERROR,
                {"player_id": action.player_id, "error": "No target specified"}
            )
            return
        
        # Validate vote
        alive = self.state.get_alive_players()
        valid, error = validate_vote(action.player_id, target, alive)
        
        if not valid:
            self.logger.log_event(
                EventType.ERROR,
                {"player_id": action.player_id, "error": error}
            )
            return
        
        # Record vote
        self.state.current_day.votes[action.player_id] = target
        
        # Log vote
        if self.logger:
            self.logger.log_event(
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
        eliminated, vote_count = get_vote_result(self.state.current_day.votes)
        
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
                self.logger.log_event(
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
            # Tie - no elimination
            self.state.broadcast_observation(
                "Moderator Announcement: The vote was tied. No one was eliminated.",
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
        if self.state.round_number > self.config.max_rounds:
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
                self.logger.log_event(
                    EventType.GAME_END,
                    {
                        "winner": winner.value if winner else "draw",
                        "reason": reason,
                        "rounds": self.state.round_number,
                    }
                )
            
            return True
        
        return False
    
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
            
            if not player.is_alive:
                instruction = "You are dead. You can only observe."
                obs_type = "observe"
            
            elif self.state.phase == Phase.NIGHT_WEREWOLF:
                if player.role == Role.WEREWOLF:
                    instruction = (
                        "NIGHT PHASE: As a Werewolf, choose a player to eliminate. "
                        "Respond with JSON: {\"type\": \"eliminate\", \"target\": <player_id>}"
                    )
                    obs_type = "act"
                else:
                    instruction = "NIGHT PHASE: You are asleep."
                    obs_type = "observe"
            
            elif self.state.phase == Phase.NIGHT_DOCTOR:
                if player.role == Role.DOCTOR:
                    instruction = (
                        "NIGHT PHASE: As the Doctor, choose a player to protect. "
                        "Respond with JSON: {\"type\": \"protect\", \"target\": <player_id>}"
                    )
                    obs_type = "act"
                else:
                    instruction = "NIGHT PHASE: You are asleep."
                    obs_type = "observe"
            
            elif self.state.phase == Phase.NIGHT_SEER:
                if player.role == Role.SEER:
                    instruction = (
                        "NIGHT PHASE: As the Seer, choose a player to investigate. "
                        "Respond with JSON: {\"type\": \"investigate\", \"target\": <player_id>}"
                    )
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
                    instruction = (
                        "DAY PHASE - BIDDING: Bid to speak (0-4). Higher bids speak first. "
                        "Respond with JSON: {\"type\": \"bid\", \"bid\": <0-4>}"
                    )
                    obs_type = "act"
                    obs_data["current_debate"] = [
                        f"{self.state.players[speaker].name}: {stmt}"
                        for speaker, stmt in self.state.current_day.debate
                    ]
            
            elif self.state.phase == Phase.DAY_DEBATE:
                if pid == self.state.current_speaker:
                    instruction = (
                        "DAY PHASE - DEBATE: You won the bid! Make a statement. "
                        "Respond with JSON: {\"type\": \"debate\", \"statement\": \"<your statement>\"}"
                    )
                    obs_type = "act"
                else:
                    instruction = f"Waiting for {self.state.players[self.state.current_speaker].name} to speak."
                    obs_type = "observe"
                
                obs_data["current_debate"] = [
                    f"{self.state.players[speaker].name}: {stmt}"
                    for speaker, stmt in self.state.current_day.debate
                ]
            
            elif self.state.phase == Phase.DAY_VOTING:
                instruction = (
                    "DAY PHASE - VOTING: Vote to eliminate a player. "
                    "Respond with JSON: {\"type\": \"vote\", \"target\": <player_id>}"
                )
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
            
            observations[pid] = Observation(
                player_id=pid,
                obs_type=obs_type,
                phase=self.state.phase.value,
                data=obs_data
            )
        
        return observations

