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
from sdb.logging.formats import EventType

from .config import AmongUsConfig
from .map import SpaceshipMap
from .rules import (
    assign_roles,
    check_win_condition,
    get_vote_result,
    handle_move,
    handle_vent,
    validate_emergency_call,
    validate_kill,
    validate_vote,
)
from .state import (
    AmongUsState,
    MeetingState,
    cast_vote,
    close_meeting,
    normalize_target,
    SKIP_TOKEN,
)
from .types import (
    MeetingResult,
    Phase,
    PlayerRole,
    PlayerState,
    TaskRoundResult,
)


SKIP_TOKENS = {"skip", "none", ""}

def normalize_agent_field(val):
    """Normalize agent field values to int or None.
    
    Handles:
    - integers: pass through
    - None: pass through as None
    - "skip"/"none"/"": convert to None
    - "Agent_2"/"Player 3"/"3": extract integer 2 or 3
    
    Examples:
        normalize_agent_field(2) -> 2
        normalize_agent_field("Agent_2") -> 2
        normalize_agent_field("Player 3") -> 3
        normalize_agent_field("skip") -> None
        normalize_agent_field("") -> None
        normalize_agent_field(None) -> None
    
    Returns:
        int or None
    """
    if val is None:
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        s = val.strip().lower()
        if s in SKIP_TOKENS:
            return None
        # Extract digits from string like "Agent_3", "Player 3", "3"
        digits = "".join(ch for ch in s if ch.isdigit())
        if digits.isdigit():
            return int(digits)
        return None
    return None

def normalize_target_id(target):
    """Normalize target to player ID: 'Agent_2', 'Player 2', '2', 2 -> 2.
    
    Deprecated: Use normalize_agent_field() instead.
    This is kept for backward compatibility but delegates to normalize_agent_field().
    """
    return normalize_agent_field(target)

def normalize_payload(d: dict):
    """Apply normalization to known fields that expect IDs."""
    # apply to known fields that expect IDs
    for k in ("target", "victim", "voter", "reporter"):
        if k in d: 
            d[k] = normalize_agent_field(d[k])
    return d


# Error throttling
from time import time
ERROR_COOLDOWN_SEC = 3.0
_last_error_at: dict[tuple[int,str,str], float] = {}

def emit_error_throttled(player_id: int, code: str, detail: str, **payload):
    """Emit error with throttling to prevent spam."""
    key = (player_id, code, detail)
    now = time()
    last = _last_error_at.get(key, 0)
    if now - last < ERROR_COOLDOWN_SEC:
        return  # suppress duplicate
    _last_error_at[key] = now
    return {"error": code, "detail": detail, **payload}


class AmongUsEnv(BaseEnvironment):
    """Among Us environment with spatial mechanics.
    
    Features spatial movement, room-based tasks, and visibility constraints.
    """
    
    def __init__(self, agents, config=None, game_id=None, logger=None, role_assignment=None):
        """Initialize Among Us environment.
        
        Args:
            agents: List of agents (must match config.n_players)
            config: AmongUsConfig instance
            game_id: Optional game ID
            logger: Optional GameLogger instance
            role_assignment: Optional dict with 'impostors' and 'crewmates' player indices
        """
        config = config or AmongUsConfig()
        self.game_config = config
        self.rng = random.Random()
        self.ship_map = SpaceshipMap()
        self.map = self.ship_map  # Alias for rules.py compatibility
        self.logger = logger
        self.role_assignment = role_assignment  # Store for use in reset()
        super().__init__(agents, config=config.__dict__, game_id=game_id, seed=getattr(config, 'seed', None))
        
        # Track actions for current task round
        self.pending_task_actions: Dict[int, Action] = {}
    
    def error(self, error_type: str, **kwargs) -> dict:
        """Helper method for rules.py to return error results.
        
        Args:
            error_type: Type of error (e.g., "INVALID_MOVE")
            **kwargs: Additional error context
            
        Returns:
            Dict with "error" key and additional context
        """
        result = {"error": error_type}
        result.update(kwargs)
        return result
    
    def ok(self, event_type: str, **kwargs) -> dict:
        """Helper method for rules.py to return success results.
        
        Args:
            event_type: Type of event (e.g., "PLAYER_ACTION")
            **kwargs: Event data
            
        Returns:
            Dict with event data (no "error" key)
        """
        result = {"event_type": event_type}
        result.update(kwargs)
        return result
    
    def is_impostor(self, player_id: int) -> bool:
        """Check if a player is an impostor.
        
        Args:
            player_id: Player ID to check
            
        Returns:
            True if player is an impostor, False otherwise
        """
        if player_id not in self.state.players:
            return False
        return self.state.players[player_id].role == PlayerRole.IMPOSTOR
    
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
        if self.role_assignment:
            # Use fixed role assignment from tournament schedule
            impostor_indices = self.role_assignment.get('impostors', [])
            crewmate_indices = self.role_assignment.get('crewmates', [])
            
            # Create roles array with fixed assignments
            roles = [None] * self.num_players
            
            # Assign impostors
            for idx in impostor_indices:
                roles[idx] = PlayerRole.IMPOSTOR
            
            # Assign crewmates
            for idx in crewmate_indices:
                roles[idx] = PlayerRole.CREWMATE
        else:
            # Default random assignment
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
            
            # Add player to map at starting location
            # Use add_player directly since there's no "from" location yet
            if "Cafeteria" in self.ship_map.rooms:
                self.ship_map.rooms["Cafeteria"].add_player(i)
        
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
        self.state.meeting_history = []
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
            # CRITICAL: Block dead players from taking any actions
            if player_id not in self.state.players or not self.state.players[player_id].is_alive:
                error_msg = "Dead players cannot take actions"
                if player_id in self.state.players:
                    self.state.players[player_id].last_error = error_msg
                if self.logger:
                    self.logger.log(
                        EventType.ERROR,
                        {
                            "player_id": player_id,
                            "error": error_msg,
                            "error_code": "DEAD_PLAYER_ACTION_BLOCKED",
                            "attempted_action": action.data.get("type", "unknown")
                        }
                    )
                continue  # Skip this action entirely
            
            # Normalize action payload
            action.data = normalize_payload(action.data)
            
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
        """Resolve all task actions for this round.
        
        Action Resolution Order (for fairness and consistency):
        1. Moves & Vents (all movement resolves first)
        2. Kills (using POST-move locations - escapees get away)
        3. Body Reports & Emergency Calls (reactions to kills)
        4. Tasks (task completion)
        5. Win Checks (after all actions resolved)
        
        This order gives crewmates a fair chance to escape and rewards spatial play.
        """
        result = TaskRoundResult()
        
        # 0) Snapshot start-of-round state
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
        
        # Helper: Check if player is alive
        def _alive(pid):
            return self.state.players[pid].is_alive
        
        # 2) Resolve MOVES first (all alive players)
        for pid, action in moves:
            if not _alive(pid):
                continue
            dest = action.data.get("room", "")
            move_result = handle_move(self, pid, dest)
            if "error" in move_result:
                self.state.players[pid].last_error = move_result["error"]
                throttled_error = emit_error_throttled(pid, "INVALID_MOVE", move_result["error"], allowed=move_result.get("allowed", []))
                if throttled_error and self.logger:
                    self.logger.log(EventType.ERROR, {
                        "player_id": pid, 
                        "error": move_result["error"], 
                        "error_code": "INVALID_MOVE",
                        "from_room": move_result.get("from_room"),
                        "attempted_room": move_result.get("to_room"),
                        "allowed": move_result.get("allowed", [])
                    })
            else:
                # Update player location
                from_room = self.state.players[pid].location
                self.state.players[pid].location = dest
                if self.logger:
                    self.logger.log(
                        EventType.PLAYER_ACTION,
                        {
                            "round": self.state.round_number,
                            "player_id": pid,
                            "action": "move",
                            "from_room": from_room,
                            "to_room": dest
                        }
                    )
        
        # 3) Resolve VENTS (all alive impostors)
        for pid, action in vents:
            if not _alive(pid):
                continue
            dest = action.data.get("room", "")
            vent_result = handle_vent(self, pid, dest)
            if "error" in vent_result:
                self.state.players[pid].last_error = vent_result["error"]
                throttled_error = emit_error_throttled(pid, "INVALID_VENT", vent_result["error"], allowed=vent_result.get("allowed", []))
                if throttled_error and self.logger:
                    self.logger.log(EventType.ERROR, {
                        "player_id": pid, 
                        "error": vent_result["error"], 
                        "error_code": "INVALID_VENT",
                        "from_room": vent_result.get("from_room"),
                        "attempted_room": vent_result.get("to_room"),
                        "allowed": vent_result.get("allowed", [])
                    })
            else:
                # Update player location
                from_room = self.state.players[pid].location
                self.state.players[pid].location = dest
                if self.logger:
                    self.logger.log(
                        EventType.PLAYER_ACTION,
                        {
                            "round": self.state.round_number,
                            "player_id": pid,
                            "action": "vent",
                            "from_room": from_room,
                            "to_room": dest
                        }
                    )
        
        # 4) Resolve KILLS using POST-MOVE locations (escapees get away!)
        newly_dead = set()
        for pid, action in kills:
            player = self.state.players[pid]
            if not player.is_alive:
                continue
            
            # Normalize target (handles "Agent_2" -> 2)
            target_raw = action.data.get("target")
            target = normalize_target_id(target_raw)
            
            can_kill = self.state.can_impostor_kill(pid)
            valid, error = validate_kill(pid, target, list(alive_start), can_kill)
            
            if not valid:
                # Improve error message to show both ID and name
                if target is not None and isinstance(target, int) and target in self.state.players:
                    target_name = self.state.players[target].name
                    error = f"{error} (target: {target} = {target_name})"
                self.state.players[pid].last_error = error
                if self.logger:
                    self.logger.log(
                        EventType.ERROR,
                        {
                            "player_id": pid,
                            "error": error,
                            "error_code": error.replace(" ", "_").upper() if error else "UNKNOWN",
                            "target_provided": target_raw,
                            "target_normalized": target,
                            "target_type": type(target_raw).__name__
                        }
                    )
                continue
            
            # Check if target exists and is alive
            if target not in self.state.players:
                error_msg = f"Target player {target} ({self.state.players.get(target, 'unknown').name if target in self.state.players else 'unknown'}) does not exist"
                self.state.players[pid].last_error = error_msg
                if self.logger:
                    self.logger.log(
                        EventType.ERROR,
                        {
                            "player_id": pid,
                            "error": error_msg,
                            "error_code": "TARGET_NOT_FOUND",
                            "target": target,
                            "target_raw": target_raw
                        }
                    )
                continue
            
            if not self.state.players[target].is_alive:
                target_name = self.state.players[target].name
                error_msg = f"Target player {target} ({target_name}) is already dead"
                self.state.players[pid].last_error = error_msg
                if self.logger:
                    self.logger.log(
                        EventType.ERROR,
                        {
                            "player_id": pid,
                            "error": error_msg,
                            "error_code": "TARGET_ALREADY_DEAD",
                            "target": target,
                            "target_name": target_name
                        }
                    )
                continue
            
            # Same-room check using CURRENT (post-move) positions
            killer_room = self.state.players[pid].location
            target_room = self.state.players[target].location
            
            if killer_room == target_room:
                self.state.players[target].is_alive = False
                # Keep body where kill occurred
                self.state.players[target].location = killer_room
                self.state.impostor_kill_cooldowns[pid] = self.game_config.kill_cooldown
                newly_dead.add(target)
                result.kills.append((pid, target))
                
                target_name = self.state.players[target].name
                if self.logger:
                    self.logger.log(
                        EventType.PLAYER_ACTION,
                        {
                            "round": self.state.round_number,
                            "killer": pid,
                            "killer_name": self.state.players[pid].name,
                            "victim": target,
                            "victim_name": target_name,
                            "action": "kill",
                            "location": killer_room,
                            "post_move_kill": True  # Flag that this is post-move
                        },
                        is_private=True
                    )
            else:
                target_name = self.state.players[target].name
                error_msg = f"Target {target} ({target_name}) not in same room (killer in {killer_room}, target in {target_room})"
                self.state.players[pid].last_error = error_msg
                if self.logger:
                    self.logger.log(
                        EventType.ERROR,
                        {
                            "player_id": pid,
                            "error": error_msg,
                            "error_code": "TARGET_DIFFERENT_ROOM",
                            "killer_location": killer_room,
                            "target_location": target_room,
                            "target": target,
                            "target_name": target_name
                        }
                    )
        
        # 5) CRITICAL WIN CHECK: Check immediately after kills resolve
        # If impostors >= crewmates or all impostors dead, game ends NOW (no meeting)
        if result.kills:  # Only check if kills occurred this round
            if self._check_game_over():
                # Game ended due to kills - store result and exit
                self.state.task_round_results.append(result)
                self.pending_task_actions = {}
                return
        
        # Helper: Check if player is alive and not killed this round
        def _alive_survivor(pid):
            return self.state.players[pid].is_alive and pid not in newly_dead
        
        # 6) Resolve BODY REPORTS (first valid reporter wins)
        for pid, action in reports:
            if not _alive_survivor(pid):
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
        
        # 7) Resolve EMERGENCY (first valid one wins, but only if no body report)
        if result.body_reported:
            # Block all emergencies when a body report was successful
            for pid, action in emergencies:
                if not _alive_survivor(pid):
                    continue
                error_msg = "Cannot call emergency meeting - body report in progress"
                self.state.players[pid].last_error = error_msg
                if self.logger:
                    self.logger.log(EventType.ERROR, {
                        "player_id": pid, 
                        "error": error_msg, 
                        "error_code": "BODY_REPORT_TAKES_PRECEDENCE"
                    })
        else:
            # No body report - process emergency calls
            for pid, action in emergencies:
                if not _alive_survivor(pid):
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
        
        # 8) Resolve TASKS for survivors only
        for pid, action in tasks:
            if not _alive_survivor(pid):
                continue
            player = self.state.players[pid]
            
            # Impostors cannot complete real tasks (they can only fake them for cover)
            if player.role != PlayerRole.CREWMATE:
                error_msg = "Impostors cannot complete tasks"
                self.state.players[pid].last_error = error_msg
                if self.logger:
                    self.logger.log(
                        EventType.ERROR,
                        {
                            "player_id": pid,
                            "error": error_msg,
                            "error_code": "IMPOSTOR_CANNOT_COMPLETE_TASKS"
                        }
                    )
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
        
        # 9) Store result and finalize round
        self.state.task_round_results.append(result)
        self.pending_task_actions = {}
        
        # Decrease kill cooldowns
        self.state.decrease_kill_cooldowns()
        
        # Increment round
        self.state.round_number += 1
        
        # CRITICAL WIN CHECK: Check win condition after tasks (before meeting logic)
        # This catches task completion wins even if a meeting was also triggered
        if self._check_game_over():
            return
        
        # Check for meeting trigger
        if result.body_reported or result.emergency_called:
            self._start_meeting(result)
            
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
        # Add meeting to history
        meeting_info = {
            "caller": None,
            "reason": "unknown",
            "voted_out": None
        }
        
        if trigger.body_reported:
            reporter, victim = trigger.body_reported
            meeting_info["caller"] = reporter
            meeting_info["reason"] = "body_report"
            meeting_info["victim"] = victim
        elif trigger.emergency_called is not None:
            meeting_info["caller"] = trigger.emergency_called
            meeting_info["reason"] = "emergency"
        
        self.state.meeting_history.append(meeting_info)
        
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
        self.state.discussion_started_round = self.state.round_number  # Track when discussion started
        self.state.discussion_attempts = 0  # Reset attempt counter
    
    def _handle_discussion_action(self, action: Action):
        """Handle discussion phase actions with timeout mechanism."""
        action_type = action.data.get("type", "")
        
        # Check for discussion timeout (prevent deadlock if players don't respond)
        discussion_duration = self.state.round_number - self.state.discussion_started_round
        max_discussion_attempts = self.game_config.discussion_rounds * len(self.state.get_alive_players()) * 2  # 2x buffer
        
        if discussion_duration > 10 or self.state.discussion_attempts > max_discussion_attempts:
            # Force move to voting after timeout
            if self.logger:
                self.logger.log(
                    EventType.INFO,
                    {
                        "message": "Discussion timeout - moving to voting phase",
                        "discussion_duration": discussion_duration,
                        "discussion_attempts": self.state.discussion_attempts
                    }
                )
            self.state.phase = Phase.VOTING
            return
        
        self.state.discussion_attempts += 1
        
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
        target_raw = action.data.get("target")  # None means skip
        target = normalize_target_id(target_raw)
        
        # Use new voting system
        if not self.state.meeting:
            alive = self.state.get_alive_players()
            self.state.meeting = MeetingState(alive)
            self.state.voting_started_round = self.state.round_number  # Track when voting started
        
        result = cast_vote(self.state.meeting, action.player_id, target)
        
        if "error" in result:
            # Improve error message to show both ID and name
            error_msg = result["error"]
            if "allowed" in result:
                allowed_names = [f"{pid} ({self.state.players[pid].name})" for pid in result["allowed"] if isinstance(pid, int)]
                error_msg += f" - Allowed targets: {', '.join(allowed_names) if allowed_names else 'skip only'}"
            
            throttled_error = emit_error_throttled(action.player_id, "VOTE_ERROR", error_msg, allowed=result.get("allowed", []))
            if throttled_error and self.logger:
                self.logger.log(
                    EventType.ERROR,
                    {"player_id": action.player_id, "error": error_msg, 
                     "target_provided": target_raw,
                     "target_normalized": target,
                     "allowed": result.get("allowed", [])}
                )
            return
        
        # Log vote
        if self.logger:
            target_name = self.state.players[target].name if target is not None and isinstance(target, int) else "skip"
            self.logger.log(
                EventType.VOTE_CAST,
                {
                    "round": self.state.round_number,
                    "voter": action.player_id,
                    "voter_name": self.state.players[action.player_id].name,
                    "target": target if target is not None else "skip",
                    "target_name": target_name
                },
                is_private=True
            )
        
        # Check if all alive players have voted OR voting timeout reached
        voting_duration = self.state.round_number - self.state.voting_started_round
        max_voting_rounds = 5  # Allow 5 rounds for all players to vote
        
        if len(self.state.meeting.ballots) >= len(self.state.meeting.alive):
            self._resolve_voting()
        elif voting_duration > max_voting_rounds:
            # Force abstain votes for missing players after timeout
            if self.logger:
                self.logger.log(
                    EventType.INFO,
                    {
                        "message": "Voting timeout - forcing skip votes for non-voters",
                        "voting_duration": voting_duration,
                        "votes_cast": len(self.state.meeting.ballots),
                        "votes_needed": len(self.state.meeting.alive)
                    }
                )
            # Auto-skip for players who haven't voted
            for pid in self.state.meeting.alive:
                if pid not in self.state.meeting.ballots:
                    cast_vote(self.state.meeting, pid, SKIP_TOKEN)
            self._resolve_voting()
    
    def _resolve_voting(self):
        """Resolve the voting phase."""
        if not self.state.meeting:
            return
            
        result = close_meeting(self.state.meeting)
        
        # Create meeting result
        meeting_result = MeetingResult(
            discussion_statements=self.state.current_meeting_statements,
            votes=result["votes"],
            ejected=result.get("ejected"),
            skipped=result["skipped"]
        )
        self.state.meeting_results.append(meeting_result)
        
        # Update meeting history with voting result
        if self.state.meeting_history:
            self.state.meeting_history[-1]["voted_out"] = result.get("ejected")
        
        # Log result
        ejected = result.get("ejected")
        is_skip = result["skipped"]
        if self.logger:
            self.logger.log(
                EventType.ELECTION_RESULT,
                {
                    "ejected": ejected,
                    "skipped": is_skip,
                    "votes": result["votes"]
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
        self.state.meeting = None  # Clear meeting state
        
        # MEETING END RESET (critical for fairness):
        # 1. Teleport all alive players to Cafeteria (standard Among Us rules)
        # 2. Reset kill cooldowns for all impostors to full
        for pid, player in self.state.players.items():
            if player.is_alive:
                # Remove from current room
                old_loc = player.location
                if old_loc in self.ship_map.rooms:
                    self.ship_map.rooms[old_loc].remove_player(pid)
                # Teleport to Cafeteria
                player.location = "Cafeteria"
                self.ship_map.rooms["Cafeteria"].add_player(pid)
                
            # Reset impostor kill cooldowns to full
            if player.role == PlayerRole.IMPOSTOR:
                self.state.impostor_kill_cooldowns[pid] = self.game_config.kill_cooldown
        
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
        if not self.state.meeting_history:
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
            
            # Visible players (same room, alive) - show both ID and name
            visible_players = self.state.get_visible_players(pid)
            obs_data["visible_players"] = [
                {"id": vpid, "name": self.state.players[vpid].name}
                for vpid in visible_players
            ]
            
            # Tasks in current room
            tasks_here = self.ship_map.get_tasks_in_room(current_location)
            obs_data["tasks_in_room"] = tasks_here
            
            # Add role-specific info (only for alive players)
            if player.is_alive:
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
                        {"id": imp_id, "name": self.state.players[imp_id].name}
                        for imp_id in self.state.get_alive_impostors()
                        if imp_id != pid
                    ]
                    obs_data["can_kill"] = self.state.can_impostor_kill(pid)
                    obs_data["kill_cooldown"] = self.state.impostor_kill_cooldowns.get(pid, 0)
                    # Impostors can see vent connections
                    obs_data["vent_destinations"] = self.ship_map.get_vent_destinations(current_location)
            else:
                # Dead players have no capabilities
                if player.role == PlayerRole.CREWMATE:
                    obs_data["my_tasks"] = f"{player.tasks_completed}/{player.total_tasks}"
                    obs_data["can_call_emergency"] = False
                    obs_data["my_remaining_tasks"] = []
                elif player.role == PlayerRole.IMPOSTOR:
                    obs_data["can_kill"] = False
                    obs_data["kill_cooldown"] = None
                    obs_data["vent_destinations"] = []
                    # Don't include fellow_impostors for dead players
            
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
                        # visible_players is now a list of dicts with id and name
                        targets_here = [vp["id"] if isinstance(vp, dict) else vp for vp in visible_players if (vp["id"] if isinstance(vp, dict) else vp) != pid]
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
                
                # Collect successful actions and log failures
                for (pid, _), result in zip(act_players, results):
                    if isinstance(result, Exception):
                        # Log the exception with traceback
                        import traceback
                        agent_name = self.agents[pid].name if pid in self.agents else f"Agent_{pid}"
                        error_msg = f"Agent {agent_name} (ID {pid}) failed to act: {type(result).__name__}: {str(result)}"
                        traceback_str = ''.join(traceback.format_exception(type(result), result, result.__traceback__))
                        print(f"❌ {error_msg}")
                        print(f"   Traceback:\n{traceback_str}")
                        if self.logger:
                            self.logger.log(
                                EventType.ERROR,
                                {
                                    "player_id": pid,
                                    "agent_name": agent_name,
                                    "error_type": type(result).__name__,
                                    "error_message": str(result),
                                    "traceback": traceback_str,
                                    "round": round_count
                                }
                            )
                    else:
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

