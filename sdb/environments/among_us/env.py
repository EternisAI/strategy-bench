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
        self.config = config or AmongUsConfig()
        super().__init__(agents, game_id, logger)
        self.state: AmongUsState = AmongUsState()
        self.rng = random.Random()
        
        # Initialize spatial map
        self.ship_map = SpaceshipMap()
        self.state.ship_map = self.ship_map
        
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
    
    def _validate_num_players(self, num_players: int) -> bool:
        """Validate number of players."""
        return 4 <= num_players <= 15
    
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
        # Reset spatial map
        self.ship_map.reset()
        
        # Assign roles
        roles = assign_roles(self.config, self.rng)
        
        # Initialize player states
        self.state.players = {}
        for i, role in enumerate(roles):
            player = PlayerState(
                player_id=i,
                name=self.agents[i].name,
                role=role,
                is_alive=True,
                tasks_completed=0,
                total_tasks=self.config.tasks_per_player if role == PlayerRole.CREWMATE else 0,
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
            self.logger.log_event(
                event_type=EventType.GAME_START,
                data={
                    "n_players": self.config.n_players,
                    "n_impostors": self.config.n_impostors,
                    "roles": [r.value for r in roles],
                }
            )
        
        return self._get_observations()
    
    def step(self, action: Action) -> Tuple[Dict[int, Observation], Dict[int, float], bool, Dict]:
        """Execute one step of the game."""
        # Handle action based on phase
        if self.state.phase == Phase.TASK:
            self._handle_task_action(action)
        elif self.state.phase == Phase.DISCUSSION:
            self._handle_discussion_action(action)
        elif self.state.phase == Phase.VOTING:
            self._handle_voting_action(action)
        
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
        """Resolve all task actions for this round."""
        result = TaskRoundResult()
        
        # Process each action
        for pid, action in self.pending_task_actions.items():
            player = self.state.players[pid]
            action_type = action.data.get("type", "")
            
            # Handle movement (both roles)
            if action_type == "move":
                target_room = action.data.get("room", "")
                if self.state.can_player_move_to(pid, target_room):
                    old_location = player.location
                    self.ship_map.move_player(pid, old_location, target_room)
                    player.location = target_room
                    
                    if self.logger:
                        self.logger.log_event(
                            EventType.PLAYER_ACTION,
                            {
                                "round": self.state.round_number,
                                "player_id": pid,
                                "action": "move",
                                "from": old_location,
                                "to": target_room
                            }
                        )
                elif self.logger:
                    self.logger.log_event(
                        EventType.ERROR,
                        {
                            "player_id": pid,
                            "error": f"Invalid move to {target_room}"
                        }
                    )
            
            # Handle vent (impostors only)
            elif action_type == "vent" and player.role == PlayerRole.IMPOSTOR:
                target_room = action.data.get("room", "")
                vent_dests = self.ship_map.get_vent_destinations(player.location)
                if target_room in vent_dests:
                    old_location = player.location
                    self.ship_map.move_player(pid, old_location, target_room)
                    player.location = target_room
                    
                    if self.logger:
                        self.logger.log_event(
                            EventType.PLAYER_ACTION,
                            {
                                "round": self.state.round_number,
                                "player_id": pid,
                                "action": "vent",
                                "from": old_location,
                                "to": target_room
                            },
                            is_private=True  # Venting is secret
                        )
                elif self.logger:
                    self.logger.log_event(
                        EventType.ERROR,
                        {
                            "player_id": pid,
                            "error": f"Invalid vent to {target_room}"
                        }
                    )
            
            if player.role == PlayerRole.CREWMATE:
                if action_type == "complete_task":
                    # Complete a task (must be in correct room)
                    if player.tasks_completed < player.total_tasks:
                        task_name, task_room = player.assigned_tasks[player.tasks_completed]
                        
                        # Check if player is in the correct room
                        if player.location == task_room:
                            player.tasks_completed += 1
                            result.tasks_completed.append(pid)
                            
                            if self.logger:
                                self.logger.log_event(
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
                        elif self.logger:
                            self.logger.log_event(
                                EventType.ERROR,
                                {
                                    "player_id": pid,
                                    "error": f"Cannot complete {task_name} - not in {task_room} (currently in {player.location})"
                                }
                            )
                elif action_type == "call_emergency":
                    # Call emergency meeting
                    valid, error = validate_emergency_call(pid, player.has_called_emergency)
                    if valid:
                        player.has_called_emergency = True
                        result.emergency_called = pid
                        
                        if self.logger:
                            self.logger.log_event(
                                EventType.PLAYER_ACTION,
                                {
                                    "round": self.state.round_number,
                                    "player_id": pid,
                                    "action": "call_emergency"
                                }
                            )
            
            elif player.role == PlayerRole.IMPOSTOR:
                if action_type == "kill":
                    # Attempt kill (must be in same room)
                    target = action.data.get("target")
                    can_kill = self.state.can_impostor_kill(pid)
                    
                    # Check if target is in the same room
                    target_player = self.state.players.get(target)
                    same_room = target_player and target_player.location == player.location
                    
                    valid, error = validate_kill(pid, target, alive := self.state.get_alive_players(), can_kill)
                    
                    if valid and same_room:
                        # Kill successful
                        self.state.players[target].is_alive = False
                        # Keep body location for reporting
                        self.state.players[target].location = player.location
                        result.kills.append((pid, target))
                        self.state.impostor_kill_cooldowns[pid] = self.config.kill_cooldown
                        
                        if self.logger:
                            self.logger.log_event(
                                EventType.PLAYER_ACTION,
                                {
                                    "round": self.state.round_number,
                                    "killer": pid,
                                    "victim": target,
                                    "action": "kill",
                                    "location": player.location
                                },
                                is_private=True
                            )
                    elif self.logger:
                        if not same_room:
                            error = f"Target not in same room (killer in {player.location}, target in {target_player.location if target_player else 'unknown'})"
                        self.logger.log_event(
                            EventType.ERROR,
                            {"player_id": pid, "error": error}
                        )
                elif action_type == "report_body":
                    # Impostor can also report bodies
                    victim = action.data.get("victim")
                    if victim is not None and not self.state.players[victim].is_alive:
                        result.body_reported = (pid, victim)
        
        # Check for body reports from crewmates
        for pid, action in self.pending_task_actions.items():
            if action.data.get("type") == "report_body":
                victim = action.data.get("victim")
                if victim is not None and not self.state.players[victim].is_alive:
                    result.body_reported = (pid, victim)
                    break  # Only one report needed
        
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
            if self.state.round_number >= self.config.max_task_rounds:
                # Game ends in impostor victory (ran out of time)
                self.state.winner = "impostors"
                self.state.win_reason = "Time limit reached"
                self.state.phase = Phase.GAME_END
                
                if self.logger:
                    self.logger.log_event(
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
                self.logger.log_event(
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
                self.logger.log_event(
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
    
    def _handle_discussion_action(self, action: Action):
        """Handle discussion phase actions."""
        action_type = action.data.get("type", "")
        
        if action_type == "discuss":
            statement = action.data.get("statement", "")
            if statement:
                self.state.current_meeting_statements.append((action.player_id, statement))
                
                # Broadcast statement
                player_name = self.state.players[action.player_id].name
                if self.logger:
                    self.logger.log_event(
                        EventType.DISCUSSION,
                        {
                            "round": self.state.round_number,
                            "player_id": action.player_id,
                            "player_name": player_name,
                            "statement": statement
                        },
                        is_private=False
                    )
        
        elif action_type == "vote_now":
            # Move to voting
            self.state.phase = Phase.VOTING
        
        # Check discussion limit
        if len(self.state.current_meeting_statements) >= self.config.discussion_rounds:
            self.state.phase = Phase.VOTING
    
    def _handle_voting_action(self, action: Action):
        """Handle voting phase actions."""
        target = action.data.get("target")  # None means skip
        
        # Validate vote
        alive = self.state.get_alive_players()
        valid, error = validate_vote(action.player_id, target, alive)
        
        if not valid:
            if self.logger:
                self.logger.log_event(
                    EventType.ERROR,
                    {"player_id": action.player_id, "error": error}
                )
            return
        
        # Record vote
        self.state.current_votes[action.player_id] = target
        
        # Log vote
        if self.logger:
            self.logger.log_event(
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
            self.logger.log_event(
                EventType.VOTE_RESULT,
                {
                    "ejected": ejected,
                    "skipped": is_skip,
                    "votes": self.state.current_votes
                }
            )
        
        # Eject player if not skipped/tied
        if ejected is not None:
            self.state.players[ejected].is_alive = False
            
            if self.logger:
                self.logger.log_event(
                    EventType.PLAYER_ELIMINATED,
                    {
                        "round": self.state.round_number,
                        "player_id": ejected,
                        "player_name": self.state.players[ejected].name,
                        "role": self.state.players[ejected].role.value,
                        "by": "vote"
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
                self.logger.log_event(
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
        if not self.state.eliminated_players:
            return "   (No eliminations yet)"
        
        formatted = []
        for pid, info in self.state.eliminated_players.items():
            reason = info.get('reason', 'unknown')
            round_num = info.get('round', '?')
            formatted.append(
                f"   • Round {round_num}: Player {pid} eliminated ({reason})"
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
                    
                    # Movement
                    adj_rooms = obs_data["adjacent_rooms"]
                    if adj_rooms:
                        actions.append(f"{{\"type\": \"move\", \"room\": \"<room_name>\"}} (adjacent: {', '.join(adj_rooms)})")
                    
                    # Task completion (only if in correct room)
                    if player.tasks_completed < player.total_tasks:
                        next_task_name, next_task_room = player.assigned_tasks[player.tasks_completed]
                        if next_task_room == current_location:
                            actions.append(f"{{\"type\": \"complete_task\"}} (complete {next_task_name} here)")
                        else:
                            actions.append(f"(Next task: {next_task_name} in {next_task_room})")
                    
                    # Emergency button (only in Cafeteria)
                    if not player.has_called_emergency and current_location == "Cafeteria":
                        actions.append("{\"type\": \"call_emergency\"}")
                    
                    # Body reporting (only if body in same room)
                    for body_pid, body_player in self.state.players.items():
                        if not body_player.is_alive and body_player.location == current_location:
                            actions.append(f"{{\"type\": \"report_body\", \"victim\": {body_pid}}} (report {body_player.name}'s body)")
                    
                    instruction = f"TASK PHASE (Location: {current_location}): Choose an action. Options:\n" + "\n".join(f"- {a}" for a in actions)
                    obs_type = "act"
                else:  # Impostor
                    actions = []
                    
                    # Movement (corridors)
                    adj_rooms = obs_data["adjacent_rooms"]
                    if adj_rooms:
                        actions.append(f"{{\"type\": \"move\", \"room\": \"<room_name>\"}} (adjacent: {', '.join(adj_rooms)})")
                    
                    # Vent travel (impostors only)
                    vent_dests = obs_data.get("vent_destinations", [])
                    if vent_dests:
                        actions.append(f"{{\"type\": \"vent\", \"room\": \"<room_name>\"}} (vent to: {', '.join(vent_dests)})")
                    
                    # Kill (only if can kill and targets in same room)
                    if self.state.can_impostor_kill(pid):
                        targets_here = [vpid for vpid in visible_players if vpid != pid]
                        if targets_here:
                            target_names = [self.state.players[t].name for t in targets_here]
                            actions.append(f"{{\"type\": \"kill\", \"target\": <player_id>}} (targets here: {', '.join(target_names)})")
                        else:
                            actions.append("(No targets in this room)")
                    else:
                        actions.append(f"(Kill on cooldown: {self.state.impostor_kill_cooldowns[pid]} rounds)")
                    
                    # Body reporting
                    for body_pid, body_player in self.state.players.items():
                        if not body_player.is_alive and body_player.location == current_location and body_pid != pid:
                            actions.append(f"{{\"type\": \"report_body\", \"victim\": {body_pid}}} (report {body_player.name}'s body)")
                    
                    instruction = f"TASK PHASE (Location: {current_location}): Choose an action. Options:\n" + "\n".join(f"- {a}" for a in actions)
                    obs_type = "act"
            
            elif self.state.phase == Phase.DISCUSSION:
                instruction = (
                    "DISCUSSION PHASE: Make a statement OR move to voting. "
                    "Respond with JSON: {\"type\": \"discuss\", \"statement\": \"<your statement>\"} "
                    "OR {\"type\": \"vote_now\"}"
                )
                obs_type = "act"
                obs_data["discussion"] = [
                    f"{self.state.players[speaker].name}: {stmt}"
                    for speaker, stmt in self.state.current_meeting_statements
                ]
            
            elif self.state.phase == Phase.VOTING:
                if pid not in self.state.current_votes:
                    instruction = (
                        "VOTING PHASE: Vote to eject a player or skip. "
                        "Respond with JSON: {\"type\": \"vote\", \"target\": <player_id>} "
                        "OR {\"type\": \"vote\", \"target\": null} to skip"
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
            
            observations[pid] = Observation(
                player_id=pid,
                obs_type=obs_type,
                phase=self.state.phase.value,
                data=obs_data
            )
        
        return observations

