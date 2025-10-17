"""Among Us game environment with spatial mechanics.

Features:
- Spatial movement system (Skeld ship with 14 rooms)
- Corridor-based movement for all players
- Vent system for impostor fast-travel
- Task locations tied to specific rooms
- Visibility limited to same room
- Task rounds where crewmates complete tasks and impostors can kill
- Meetings triggered by body reports or emergency buttons
- Discussion and voting phases
- Win conditions based on tasks, eliminations, or impostor majority
"""

import random
from typing import Dict, List, Optional, Tuple

from sdb.core.base_env import BaseEnvironment
from sdb.core.types import Action, GameResult, Observation
from sdb.environments.among_us.config import AmongUsConfig
from sdb.environments.among_us.map import SpaceshipMap
from sdb.environments.among_us.rules import (
    assign_roles,
    check_win_condition,
    get_vote_result,
    validate_emergency_call,
    validate_kill,
    validate_vote,
)
from sdb.environments.among_us.state import AmongUsState
from sdb.environments.among_us.types import (
    MeetingResult,
    Phase,
    PlayerRole,
    PlayerState,
    TaskRoundResult,
)
from sdb.logging.formats import EventType


class AmongUsEnv(BaseEnvironment):
    """Among Us environment with spatial mechanics.
    
    Features spatial movement, room-based tasks, and visibility constraints.
    """
    
    def __init__(self, agents, config=None, game_id=None, logger=None):
        """Initialize Among Us environment.
        
        Args:
            agents: List of agents (must match config.n_players)
            config: AmongUsConfig instance
            game_id: Optional game ID
            logger: Optional GameLogger instance
        """
        config = config or AmongUsConfig()
        self.game_config = config
        self.rng = random.Random()
        self.ship_map = SpaceshipMap()
        self.logger = logger
        super().__init__(agents, config=config.__dict__, game_id=game_id, seed=getattr(config, 'seed', None))
        
        # Track actions for current task round
        self.pending_task_actions: Dict[int, Action] = {}
    
    def _assign_tasks_to_rooms(self, player: PlayerState) -> None:
        """Assign tasks to specific rooms for a crewmate.
        
        Args:
            player: PlayerState to assign tasks to
        """
        if player.role != PlayerRole.CREWMATE:
            return
        
        # Get all possible task locations
        all_tasks = []
        for room_name in self.ship_map.get_all_room_names():
            tasks_in_room = self.ship_map.get_tasks_in_room(room_name)
            for task_name in tasks_in_room:
                all_tasks.append((task_name, room_name))
        
        # Randomly assign tasks to this player
        num_tasks = min(player.total_tasks, len(all_tasks))
        assigned = self.rng.sample(all_tasks, num_tasks)
        player.assigned_tasks = assigned
        player.total_tasks = len(assigned)
    
    def _validate_num_players(self):
        """Validate number of players."""
        if not (4 <= self.num_players <= 15):
            raise EnvironmentError(f"Among Us requires 4-15 players, got {self.num_players}")
    
    def _get_current_player(self) -> Optional[int]:
        """Get the current acting player."""
        # In this simplified version, most phases have all players act simultaneously
        # Return first player who hasn't acted yet
        alive = self.state.get_alive_players()
        
        if self.state.phase == Phase.TASK:
            # Return first player who hasn't submitted action
            for pid in alive:
                if pid not in self.pending_task_actions:
                    return pid
        elif self.state.phase == Phase.DISCUSSION:
            # Everyone can discuss
            return alive[0] if alive else None
        elif self.state.phase == Phase.VOTING:
            # Return first player who hasn't voted
            for pid in alive:
                if pid not in self.state.current_votes:
                    return pid
        
        return None
    
    def get_winner(self) -> Optional[str]:
        """Get the winner."""
        return self.state.winner
    
    def get_win_reason(self) -> str:
        """Get the reason for winning."""
        return self.state.win_reason
    
    def reset(self) -> Dict[int, Observation]:
        """Reset the game state."""
        # Initialize state
        self.state = AmongUsState()
        self.state.ship_map = self.ship_map
        
        # Reset spatial map
        self.ship_map.reset()
        
        # Assign roles
        roles = assign_roles(self.game_config, self.rng)
        
        # Initialize player states
        self.state.players = {}
        for i, role in enumerate(roles):
            player = PlayerState(
                player_id=i,
                name=self.agents[i].name,
                role=role,
                is_alive=True,
                tasks_completed=0,
                total_tasks=self.game_config.tasks_per_player if role == PlayerRole.CREWMATE else 0,
                has_called_emergency=False,
                location="Cafeteria",  # Everyone starts in Cafeteria
            )
            
            # Assign tasks to specific rooms for crewmates
            if role == PlayerRole.CREWMATE:
                self._assign_tasks_to_rooms(player)
            
            self.state.players[i] = player
            
            # Add player to map
            self.ship_map.move_player(i, "", "Cafeteria")
        
        # Initialize state
        self.state.phase = Phase.TASK
        self.state.round_number = 0
        self.state.task_round_results = []
        self.state.meeting_results = []
        self.state.impostor_kill_cooldowns = {
            pid: 0 for pid, p in self.state.players.items() if p.role == PlayerRole.IMPOSTOR
        }
        self.state.current_meeting_statements = []
        self.state.current_votes = {}
        self.state.winner = None
        self.state.win_reason = ""
        
        self.pending_task_actions = {}
        
        # Log game start
        if self.logger:
            self.logger.log(
                event_type=EventType.GAME_START,
                data={
                    "n_players": self.game_config.n_players,
                    "n_impostors": self.game_config.n_impostors,
                }
            )
            
            # Log role assignments (private)
            self.logger.log(
                event_type=EventType.PLAYER_ACTION,
                data={
                    "action": "role_assignment",
                    "roles": [r.value for r in roles],
                    "role_map": {str(i): r.value for i, r in enumerate(roles)}
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
        
        return self._get_observations()
    
    def step(self, actions: Dict[int, Action]) -> Tuple[Dict[int, Observation], Dict[int, float], bool, Dict]:
        """Execute one step of the game."""
        # Handle actions based on phase
        for player_id, action in actions.items():
            if self.state.phase == Phase.TASK:
                # Store action without triggering resolution yet
                self.pending_task_actions[action.player_id] = action
            elif self.state.phase == Phase.DISCUSSION:
                self._handle_discussion_action(action)
            elif self.state.phase == Phase.VOTING:
                self._handle_voting_action(action)
        
        # After all actions are stored, check if we should resolve
        if self.state.phase == Phase.TASK:
            alive = self.state.get_alive_players()
            if len(self.pending_task_actions) >= len(alive):
                self._resolve_task_round()
        
        # Get observations
        observations = self._get_observations()
        
        # Check if game is over
        done = self.state.phase == Phase.GAME_END
        
        # Calculate rewards (only at game end)
        rewards = {}
        if done:
            for pid, player in self.state.players.items():
                if self.state.winner == "crewmates":
                    rewards[pid] = 1.0 if player.role == PlayerRole.CREWMATE else 0.0
                elif self.state.winner == "impostors":
                    rewards[pid] = 1.0 if player.role == PlayerRole.IMPOSTOR else 0.0
                else:
                    rewards[pid] = 0.5  # Draw
        else:
            rewards = {pid: 0.0 for pid in self.state.players}
        
        info = {
            "phase": self.state.phase.value,
            "round": self.state.round_number,
            "task_completion": self.state.get_total_task_completion()
        }
        
        return observations, rewards, done, info
    
    def _handle_task_action(self, action: Action):
        """Handle task phase actions."""
        # Store action
        self.pending_task_actions[action.player_id] = action
        
        # Check if all alive players have acted
        alive = self.state.get_alive_players()
        if len(self.pending_task_actions) >= len(alive):
            self._resolve_task_round()
    
    def _resolve_task_round(self):
        """Resolve all task actions for this round using TWO-PHASE resolution.
        
        Phase 1: Resolve kills based on pre-move positions
        Phase 2: Apply movements, tasks, and reports
        
        This prevents order-dependent kill outcomes.
        """
        result = TaskRoundResult()
        
        # 0) Snapshot start-of-round state
        pre_loc = {pid: p.location for pid, p in self.state.players.items()}
        alive_start = set(self.state.get_alive_players())
        
        # 1) Partition actions by type (preserve arrival order)
        moves, vents, kills, tasks, reports, emergencies = [], [], [], [], [], []
        for pid, action in self.pending_task_actions.items():
            t = action.data.get("type", "")
            if t == "move":
                moves.append((pid, action))
            elif t == "vent":
                vents.append((pid, action))
            elif t == "kill":
                kills.append((pid, action))
            elif t == "complete_task":
                tasks.append((pid, action))
            elif t == "report_body":
                reports.append((pid, action))
            elif t == "call_emergency":
                emergencies.append((pid, action))
        
        # 2) Resolve KILLS using pre-move locations
        newly_dead = set()
        for pid, action in kills:
            player = self.state.players[pid]
            if not player.is_alive:
                continue
            
            target = action.data.get("target")
            # Coerce string numbers to int
            if isinstance(target, str) and target.isdigit():
                target = int(target)
            
            can_kill = self.state.can_impostor_kill(pid)
            valid, error = validate_kill(pid, target, list(alive_start), can_kill)
            
            if not valid:
                self.state.players[pid].last_error = error
                if self.logger:
                    self.logger.log(
                        EventType.ERROR,
                        {
                            "player_id": pid,
                            "error": error,
                            "error_code": error.replace(" ", "_").upper() if error else "UNKNOWN",
                            "target_provided": target,
                            "target_type": type(target).__name__
                        }
                    )
                continue
            
            # Same-room check vs pre-round positions
            if pre_loc.get(pid) == pre_loc.get(target):
                self.state.players[target].is_alive = False
                # Keep body where kill occurred
                self.state.players[target].location = pre_loc[pid]
                self.state.impostor_kill_cooldowns[pid] = self.game_config.kill_cooldown
                newly_dead.add(target)
                result.kills.append((pid, target))
                
                if self.logger:
                    self.logger.log(
                        EventType.PLAYER_ACTION,
                        {
                            "round": self.state.round_number,
                            "killer": pid,
                            "victim": target,
                            "action": "kill",
                            "location": pre_loc[pid],
                            "pre_move_kill": True
                        },
                        is_private=True
                    )
            else:
                killer_room = pre_loc.get(pid, "unknown")
                targ_room = pre_loc.get(target, "unknown")
                error_msg = f"Target not in same room (killer in {killer_room}, target in {targ_room})"
                self.state.players[pid].last_error = error_msg
                if self.logger:
                    self.logger.log(
                        EventType.ERROR,
                        {
                            "player_id": pid,
                            "error": error_msg,
                            "error_code": "TARGET_DIFFERENT_ROOM",
                            "killer_location": killer_room,
                            "target_location": targ_room
                        }
                    )
        
        # 3) Resolve EMERGENCY (first valid one wins)
        for pid, action in emergencies:
            if pid in newly_dead or not self.state.players[pid].is_alive:
                continue
            player = self.state.players[pid]
            ok, msg = validate_emergency_call(pid, player.has_called_emergency)
            if ok:
                player.has_called_emergency = True
                result.emergency_called = pid
                if self.logger:
                    self.logger.log(
                        EventType.PLAYER_ACTION,
                        {
                            "round": self.state.round_number,
                            "player_id": pid,
                            "action": "call_emergency"
                        }
                    )
                break
            else:
                self.state.players[pid].last_error = msg
                if self.logger:
                    self.logger.log(EventType.ERROR, {"player_id": pid, "error": msg, "error_code": "EMERGENCY_UNAVAILABLE"})
        
        # Helper: Check if player is alive and not killed this round
        def _alive(pid):
            return self.state.players[pid].is_alive and pid not in newly_dead
        
        # 4) Resolve MOVES for survivors only
        for pid, action in moves:
            if not _alive(pid):
                continue
            player = self.state.players[pid]
            dest = action.data.get("room", "")
            if self.state.can_player_move_to(pid, dest):
                old = player.location
                self.ship_map.move_player(pid, old, dest)
                player.location = dest
                if self.logger:
                    self.logger.log(
                        EventType.PLAYER_ACTION,
                        {
                            "round": self.state.round_number,
                            "player_id": pid,
                            "action": "move",
                            "from": old,
                            "to": dest
                        }
                    )
            else:
                error_msg = f"Invalid move to {dest}"
                self.state.players[pid].last_error = error_msg
                if self.logger:
                    self.logger.log(EventType.ERROR, {"player_id": pid, "error": error_msg, "error_code": "INVALID_MOVE"})
        
        # 5) Resolve VENTS for survivors only
        for pid, action in vents:
            if not _alive(pid):
                continue
            player = self.state.players[pid]
            if player.role != PlayerRole.IMPOSTOR:
                continue
            dest = action.data.get("room", "")
            vent_dests = self.ship_map.get_vent_destinations(player.location)
            if dest in vent_dests:
                old = player.location
                self.ship_map.move_player(pid, old, dest)
                player.location = dest
                if self.logger:
                    self.logger.log(
                        EventType.PLAYER_ACTION,
                        {
                            "round": self.state.round_number,
                            "player_id": pid,
                            "action": "vent",
                            "from": old,
                            "to": dest
                        },
                        is_private=True
                    )
            else:
                error_msg = f"Invalid vent to {dest}"
                self.state.players[pid].last_error = error_msg
                if self.logger:
                    self.logger.log(EventType.ERROR, {"player_id": pid, "error": error_msg, "error_code": "INVALID_VENT"})
        
        # 6) Resolve TASKS for survivors only
        for pid, action in tasks:
            if not _alive(pid):
                continue
            player = self.state.players[pid]
            if player.role != PlayerRole.CREWMATE:
                continue
            if player.tasks_completed < player.total_tasks:
                task_name, task_room = player.assigned_tasks[player.tasks_completed]
                if player.location == task_room:
                    player.tasks_completed += 1
                    result.tasks_completed.append(pid)
                    if self.logger:
                        self.logger.log(
                            EventType.PLAYER_ACTION,
                            {
                                "round": self.state.round_number,
                                "player_id": pid,
                                "action": "complete_task",
                                "task": task_name,
                                "room": task_room,
                                "progress": f"{player.tasks_completed}/{player.total_tasks}"
                            }
                        )
                else:
                    error_msg = f"Cannot complete {task_name} - not in {task_room} (currently in {player.location})"
                    self.state.players[pid].last_error = error_msg
                    if self.logger:
                        self.logger.log(
                            EventType.ERROR,
                            {
                                "player_id": pid,
                                "error": error_msg,
                                "error_code": "WRONG_ROOM_FOR_TASK"
                            }
                        )
        
        # 7) Body reports (first valid reporter wins; don't overwrite)
        for pid, action in reports:
            if not _alive(pid):
                continue
            player = self.state.players[pid]
            victim = action.data.get("victim")
            if victim is not None and victim in self.state.players:
                victim_player = self.state.players[victim]
                # Only report bodies from kills, not ejections
                if (not victim_player.is_alive 
                    and victim_player.location != "EJECTED" 
                    and victim_player.location == player.location):
                    if result.body_reported is None:  # First reporter wins
                        result.body_reported = (pid, victim)
                        if self.logger:
                            self.logger.log(
                                EventType.PLAYER_ACTION,
                                {
                                    "round": self.state.round_number,
                                    "player_id": pid,
                                    "action": "report_body",
                                    "victim": victim,
                                    "location": player.location
                                }
                            )
                    break
        
        # Store result
        self.state.task_round_results.append(result)
        self.pending_task_actions = {}
        
        # Decrease kill cooldowns
        self.state.decrease_kill_cooldowns()
        
        # Increment round
        self.state.round_number += 1
        
        # Check for meeting trigger
        if result.body_reported or result.emergency_called:
            self._start_meeting(result)
        else:
            # Check win condition
            if self._check_game_over():
                return
            
            # Check round limit
            if self.state.round_number >= self.game_config.max_task_rounds:
                # Game ends in impostor victory (ran out of time)
                self.state.winner = "impostors"
                self.state.win_reason = "Time limit reached"
                self.state.phase = Phase.GAME_END
                
                if self.logger:
                    self.logger.log(
                        EventType.GAME_END,
                        {
                            "winner": self.state.winner,
                            "reason": self.state.win_reason
                        }
                    )
    
    def _start_meeting(self, trigger: TaskRoundResult):
        """Start a meeting phase."""
        # Log meeting start
        if self.logger:
            if trigger.body_reported:
                reporter, victim = trigger.body_reported
                self.logger.log(
                    EventType.PHASE_CHANGE,
                    {
                        "from_phase": "task",
                        "to_phase": "discussion",
                        "trigger": "body_report",
                        "reporter": reporter,
                        "victim": victim
                    }
                )
            elif trigger.emergency_called:
                self.logger.log(
                    EventType.PHASE_CHANGE,
                    {
                        "from_phase": "task",
                        "to_phase": "discussion",
                        "trigger": "emergency",
                        "caller": trigger.emergency_called
                    }
                )
        
        self.state.phase = Phase.DISCUSSION
        self.state.current_meeting_statements = []
        self.state.current_votes = {}
        self.state.discussion_round = 0
        self.state.players_spoken_this_round = set()
    
    def _handle_discussion_action(self, action: Action):
        """Handle discussion phase actions."""
        action_type = action.data.get("type", "")
        
        if action_type == "discuss":
            statement = action.data.get("statement", "")
            if statement:
                self.state.current_meeting_statements.append((action.player_id, statement))
                self.state.players_spoken_this_round.add(action.player_id)
                
                # Broadcast statement
                player_name = self.state.players[action.player_id].name
                if self.logger:
                    self.logger.log(
                        EventType.DISCUSSION,
                        {
                            "round": self.state.round_number,
                            "player_id": action.player_id,
                            "player_name": player_name,
                            "statement": statement,
                            "discussion_round": self.state.discussion_round
                        },
                        is_private=False
                    )
                
                # Check if all alive players have spoken this round
                alive_players = set(self.state.get_alive_players())
                if self.state.players_spoken_this_round >= alive_players:
                    # Round complete, advance to next round
                    self.state.discussion_round += 1
                    self.state.players_spoken_this_round = set()
                    
                    # Check if we've completed all discussion rounds
                    if self.state.discussion_round >= self.game_config.discussion_rounds:
                        self.state.phase = Phase.VOTING
        
        elif action_type == "vote_now":
            # Move to voting
            self.state.phase = Phase.VOTING
    
    def _handle_voting_action(self, action: Action):
        """Handle voting phase actions."""
        target = action.data.get("target")  # None means skip
        
        # Validate vote
        alive = self.state.get_alive_players()
        valid, error = validate_vote(action.player_id, target, alive)
        
        if not valid:
            if self.logger:
                self.logger.log(
                    EventType.ERROR,
                    {"player_id": action.player_id, "error": error}
                )
            return
        
        # Record vote
        self.state.current_votes[action.player_id] = target
        
        # Log vote
        if self.logger:
            self.logger.log(
                EventType.VOTE_CAST,
                {
                    "round": self.state.round_number,
                    "voter": action.player_id,
                    "target": target if target is not None else "skip"
                },
                is_private=True
            )
        
        # Check if all alive players have voted
        if len(self.state.current_votes) >= len(alive):
            self._resolve_voting()
    
    def _resolve_voting(self):
        """Resolve the voting phase."""
        ejected, is_skip = get_vote_result(self.state.current_votes)
        
        # Create meeting result
        result = MeetingResult(
            discussion_statements=self.state.current_meeting_statements,
            votes=self.state.current_votes,
            ejected=ejected,
            skipped=is_skip
        )
        self.state.meeting_results.append(result)
        
        # Log result
        if self.logger:
            self.logger.log(
                EventType.ELECTION_RESULT,
                {
                    "ejected": ejected,
                    "skipped": is_skip,
                    "votes": self.state.current_votes
                }
            )
        
        # Eject player if not skipped/tied
        if ejected is not None:
            self.state.players[ejected].is_alive = False
            # Remove ejected player from map (no body to report)
            old_location = self.state.players[ejected].location
            # Remove from the room's player set
            if old_location in self.ship_map.rooms:
                self.ship_map.rooms[old_location].remove_player(ejected)
            self.state.players[ejected].location = "EJECTED"  # Mark as ejected, not killed
            
            if self.logger:
                self.logger.log(
                    EventType.PLAYER_ELIMINATED,
                    {
                        "round": self.state.round_number,
                        "player_id": ejected,
                        "player_name": self.state.players[ejected].name,
                        "role": self.state.players[ejected].role.value,
                        "by": "vote",
                        "ejected_from": old_location
                    }
                )
        
        # Clear meeting state
        self.state.current_meeting_statements = []
        self.state.current_votes = {}
        
        # Check win condition
        if self._check_game_over():
            return
        
        # Return to task phase
        self.state.phase = Phase.TASK
    
    def _check_game_over(self) -> bool:
        """Check if game is over."""
        alive_crewmates = len(self.state.get_alive_crewmates())
        alive_impostors = len(self.state.get_alive_impostors())
        task_completion = self.state.get_total_task_completion()
        
        game_over, winner, reason = check_win_condition(
            alive_crewmates,
            alive_impostors,
            task_completion
        )
        
        if game_over:
            self.state.winner = winner
            self.state.win_reason = reason
            self.state.phase = Phase.GAME_END
            
            if self.logger:
                self.logger.log(
                    EventType.GAME_END,
                    {
                        "winner": winner,
                        "reason": reason,
                        "task_completion": task_completion,
                        "alive_crewmates": alive_crewmates,
                        "alive_impostors": alive_impostors
                    }
                )
            
            return True
        
        return False
    
    def _format_elimination_history(self) -> str:
        """Format history of all eliminations.
        
        Returns:
            Formatted string of eliminations
        """
        eliminated = [pid for pid, p in self.state.players.items() if not p.is_alive]
        
        if not eliminated:
            return "   (No eliminations yet)"
        
        formatted = []
        for pid in eliminated:
            player = self.state.players[pid]
            formatted.append(
                f"   • Player {pid} ({player.role.value}) - eliminated"
            )
        return "\n".join(formatted)
    
    def _format_meeting_history(self) -> str:
        """Format complete meeting history.
        
        Returns:
            Formatted string of all meetings and votes
        """
        if not hasattr(self.state, 'meeting_history') or not self.state.meeting_history:
            return "   (No meetings yet)"
        
        formatted = []
        for i, meeting in enumerate(self.state.meeting_history, 1):
            caller = meeting.get('caller', '?')
            reason = meeting.get('reason', 'emergency')
            voted_out = meeting.get('voted_out', None)
            formatted.append(
                f"   Meeting {i}: Called by Player {caller} ({reason})"
            )
            if voted_out is not None:
                formatted.append(f"      → Player {voted_out} was voted out")
            else:
                formatted.append(f"      → No one voted out (skip/tie)")
        return "\n".join(formatted) if formatted else "   (No meetings yet)"
    
    def _get_observations(self) -> Dict[int, Observation]:
        """Generate observations for all players."""
        observations = {}
        alive = self.state.get_alive_players()
        
        for pid in self.state.players:
            player = self.state.players[pid]
            
            # Base observation data
            obs_data = {
                "round": self.state.round_number,
                "is_alive": player.is_alive,
                "role": player.role.value,
                "alive_count": len(alive),
                "task_completion": self.state.get_total_task_completion(),
            }
            
            # Add player directory (lossless ID ↔ name mapping)
            obs_data["player_directory"] = [
                {
                    "id": p_id,
                    "name": p.name,
                    "alive": p.is_alive
                }
                for p_id, p in self.state.players.items()
            ]
            
            # Add error feedback
            obs_data["last_error"] = player.last_error
            
            # Add round progress
            obs_data["rounds_left"] = self.game_config.max_task_rounds - self.state.round_number
            
            # Add spatial information
            current_location = player.location
            obs_data["location"] = current_location
            obs_data["adjacent_rooms"] = self.ship_map.get_adjacent_rooms(current_location)
            
            # Visible players (same room, alive)
            visible_players = self.state.get_visible_players(pid)
            obs_data["visible_players"] = [self.state.players[vpid].name for vpid in visible_players]
            
            # Tasks in current room
            tasks_here = self.ship_map.get_tasks_in_room(current_location)
            obs_data["tasks_in_room"] = tasks_here
            
            # Add role-specific info
            if player.role == PlayerRole.CREWMATE:
                obs_data["my_tasks"] = f"{player.tasks_completed}/{player.total_tasks}"
                obs_data["can_call_emergency"] = not player.has_called_emergency
                # Show assigned task locations
                remaining_tasks = player.assigned_tasks[player.tasks_completed:]
                obs_data["my_remaining_tasks"] = [
                    f"{task} (in {room})" for task, room in remaining_tasks
                ]
            elif player.role == PlayerRole.IMPOSTOR:
                obs_data["fellow_impostors"] = [
                    self.state.players[imp_id].name
                    for imp_id in self.state.get_alive_impostors()
                    if imp_id != pid
                ]
                obs_data["can_kill"] = self.state.can_impostor_kill(pid)
                obs_data["kill_cooldown"] = self.state.impostor_kill_cooldowns.get(pid, 0)
                # Impostors can see vent connections
                obs_data["vent_destinations"] = self.ship_map.get_vent_destinations(current_location)
            
            # Add formatted full context
            obs_data["formatted_eliminations"] = self._format_elimination_history()
            obs_data["formatted_meetings"] = self._format_meeting_history()
            
            # Add phase-specific instructions
            instruction = ""
            obs_type = "observe"
            
            if not player.is_alive:
                instruction = "You are dead. You can only observe."
                obs_type = "observe"
            
            elif self.state.phase == Phase.TASK:
                if player.role == PlayerRole.CREWMATE:
                    actions = []
                    action_choices = []
                    
                    # Movement
                    adj_rooms = obs_data["adjacent_rooms"]
                    if adj_rooms:
                        actions.append(f"{{\"type\": \"move\", \"room\": \"<room_name>\"}} (adjacent: {', '.join(adj_rooms)})")
                        for room in adj_rooms:
                            action_choices.append({
                                "id": f"move:{room}",
                                "type": "move",
                                "payload": {"type": "move", "room": room}
                            })
                    
                    # Task completion (only if in correct room)
                    if player.tasks_completed < player.total_tasks:
                        next_task_name, next_task_room = player.assigned_tasks[player.tasks_completed]
                        if next_task_room == current_location:
                            actions.append(f"{{\"type\": \"complete_task\"}} (complete {next_task_name} here)")
                            action_choices.append({
                                "id": "complete_task",
                                "type": "complete_task",
                                "payload": {"type": "complete_task"}
                            })
                        else:
                            actions.append(f"(Next task: {next_task_name} in {next_task_room})")
                    
                    # Emergency button (only in Cafeteria)
                    if not player.has_called_emergency and current_location == "Cafeteria":
                        actions.append("{\"type\": \"call_emergency\"}")
                        action_choices.append({
                            "id": "call_emergency",
                            "type": "call_emergency",
                            "payload": {"type": "call_emergency"}
                        })
                    
                    # Body reporting (only if body in same room, exclude ejected players)
                    for body_pid, body_player in self.state.players.items():
                        if (not body_player.is_alive 
                            and body_player.location == current_location 
                            and body_player.location != "EJECTED"):
                            actions.append(f"{{\"type\": \"report_body\", \"victim\": {body_pid}}} (report {body_player.name}'s body)")
                            action_choices.append({
                                "id": f"report_body:{body_pid}",
                                "type": "report_body",
                                "payload": {"type": "report_body", "victim": body_pid}
                            })
                    
                    obs_data["action_choices"] = action_choices
                    instruction = f"TASK PHASE (Location: {current_location}): You MUST choose one from action_choices. Options:\n" + "\n".join(f"- {a}" for a in actions)
                    obs_type = "act"
                else:  # Impostor
                    actions = []
                    action_choices = []
                    
                    # Movement (corridors)
                    adj_rooms = obs_data["adjacent_rooms"]
                    if adj_rooms:
                        actions.append(f"{{\"type\": \"move\", \"room\": \"<room_name>\"}} (adjacent: {', '.join(adj_rooms)})")
                        for room in adj_rooms:
                            action_choices.append({
                                "id": f"move:{room}",
                                "type": "move",
                                "payload": {"type": "move", "room": room}
                            })
                    
                    # Vent travel (impostors only)
                    vent_dests = obs_data.get("vent_destinations", [])
                    if vent_dests:
                        actions.append(f"{{\"type\": \"vent\", \"room\": \"<room_name>\"}} (vent to: {', '.join(vent_dests)})")
                        for room in vent_dests:
                            action_choices.append({
                                "id": f"vent:{room}",
                                "type": "vent",
                                "payload": {"type": "vent", "room": room}
                            })
                    
                    # Kill (only if can kill and targets in same room)
                    if self.state.can_impostor_kill(pid):
                        targets_here = [vpid for vpid in visible_players if vpid != pid]
                        if targets_here:
                            target_list = [f"Player {t} ({self.state.players[t].name})" for t in targets_here]
                            actions.append(f"{{\"type\": \"kill\", \"target\": <player_id>}} (targets: {', '.join(target_list)})")
                            for t in targets_here:
                                action_choices.append({
                                    "id": f"kill:{t}",
                                    "type": "kill",
                                    "payload": {"type": "kill", "target": t}
                                })
                        else:
                            actions.append("(No targets in this room)")
                    else:
                        actions.append(f"(Kill on cooldown: {self.state.impostor_kill_cooldowns[pid]} rounds)")
                    
                    # Body reporting (exclude ejected players)
                    for body_pid, body_player in self.state.players.items():
                        if (not body_player.is_alive 
                            and body_player.location == current_location 
                            and body_player.location != "EJECTED" 
                            and body_pid != pid):
                            actions.append(f"{{\"type\": \"report_body\", \"victim\": {body_pid}}} (report {body_player.name}'s body)")
                            action_choices.append({
                                "id": f"report_body:{body_pid}",
                                "type": "report_body",
                                "payload": {"type": "report_body", "victim": body_pid}
                            })
                    
                    obs_data["action_choices"] = action_choices
                    instruction = f"TASK PHASE (Location: {current_location}): You MUST choose one from action_choices. Options:\n" + "\n".join(f"- {a}" for a in actions)
                    obs_type = "act"
            
            elif self.state.phase == Phase.DISCUSSION:
                # Check if player has spoken this round
                has_spoken_this_round = pid in self.state.players_spoken_this_round
                
                if has_spoken_this_round:
                    instruction = f"DISCUSSION PHASE: Waiting for other players (Discussion Round {self.state.discussion_round + 1}/{self.game_config.discussion_rounds})"
                    obs_type = "observe"
                else:
                    instruction = (
                        f"DISCUSSION PHASE (Round {self.state.discussion_round + 1}/{self.game_config.discussion_rounds}): "
                        f"Make ONE statement about what you saw, who you suspect, or your alibi. "
                        f"Respond with JSON: {{\"type\": \"discuss\", \"statement\": \"<your statement>\"}}"
                    )
                    obs_type = "act"
                
                obs_data["discussion"] = [
                    f"{self.state.players[speaker].name}: {stmt}"
                    for speaker, stmt in self.state.current_meeting_statements
                ]
                obs_data["discussion_round"] = self.state.discussion_round
                obs_data["has_spoken_this_round"] = has_spoken_this_round
            
            elif self.state.phase == Phase.VOTING:
                if pid not in self.state.current_votes:
                    # Build list of alive players for voting
                    alive_players = [
                        p_id for p_id in self.state.players
                        if self.state.players[p_id].is_alive and p_id != pid
                    ]
                    player_list = ", ".join([
                        f"Player {p_id} ({self.state.players[p_id].name})"
                        for p_id in alive_players
                    ])
                    
                    instruction = (
                        f"VOTING PHASE: Vote to eject someone or skip.\n"
                        f"Alive players: {player_list}\n"
                        f"Based on the discussion, who do you think is the impostor?\n"
                        f"Respond with JSON:\n"
                        f"  {{\"type\": \"vote\", \"target\": <player_id>}} to vote for that player\n"
                        f"  {{\"type\": \"vote\", \"target\": null}} to skip/abstain\n"
                        f"Use numeric player IDs (e.g., 0, 2, 3), NOT names."
                    )
                    obs_type = "act"
                else:
                    instruction = "Waiting for other votes."
                    obs_type = "observe"
                
                obs_data["discussion"] = [
                    f"{self.state.players[speaker].name}: {stmt}"
                    for speaker, stmt in self.state.current_meeting_statements
                ]
            
            elif self.state.phase == Phase.GAME_END:
                instruction = f"Game over! Winner: {self.state.winner}"
                obs_type = "observe"
            
            obs_data["instruction"] = instruction
            obs_data["type"] = obs_type  # Add obs_type to data for filtering in play_game()
            
            observations[pid] = Observation(
                player_id=pid,
                obs_type=obs_type,
                phase=self.state.phase.value,
                data=obs_data
            )
            
            # Log private observations for each player
            if self.logger:
                self.logger.log(
                    event_type=EventType.INFO,
                    data={
                        "round": self.state.round_number,
                        "player_id": pid,
                        "observation": obs_data,
                    },
                    player_id=pid,
                    is_private=True
                )
        
        return observations
    
    async def play_game(self):
        """Play a complete Among Us game with configured agents.
        
        Returns:
            GameResult with winner, scores, and stats
        """
        import asyncio
        from sdb.core.types import GameResult
        
        if not self.agents:
            raise RuntimeError("No agents configured")
        
        # Environment already initialized in __init__ (reset was called there)
        round_count = 0
        max_rounds = 200  # Safety limit
        
        # Ensure state is initialized
        if not hasattr(self, 'state') or self.state is None:
            raise RuntimeError("State not initialized - reset() should have been called in __init__")
        
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
        winner = self.state.winner or "timeout"
        win_reason = self.get_win_reason() or "Game reached maximum rounds"
        
        # Calculate player stats (scores for winners)
        player_stats = {}
        for player in self.state.players.values():
            if self.state.winner == "crewmates":
                score = 1.0 if player.role == "crewmate" else 0.0
            elif self.state.winner == "impostors":
                score = 1.0 if player.role == "impostor" else 0.0
            else:  # timeout
                score = 0.0
            
            player_stats[player.player_id] = {
                "score": score,
                "role": player.role,
                "survived": player.is_alive,
            }
        
        return GameResult(
            game_id=self.game_id,
            winner=winner,
            win_reason=win_reason,
            num_rounds=round_count,
            duration_seconds=0.0,  # Could track this if needed
            player_stats=player_stats,
            metadata={
                "kills": len([p for p in self.state.players.values() if not p.is_alive and p.location == "DEAD"]),
                "meetings": len(self.state.meeting_results),
                "tasks_completed": sum(p.tasks_completed for p in self.state.players.values()),
            }
        )

