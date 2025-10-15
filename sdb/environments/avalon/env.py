"""Avalon environment implementation."""

import random
from typing import Dict, List, Optional, Any, Tuple

from sdb.core.base_env import BaseEnvironment
from sdb.core.base_agent import BaseAgent
from sdb.core.types import Action, Observation, GameResult, GamePhase, ObservationType
from sdb.logging.game_logger import GameLogger
from sdb.logging.formats import EventType

from .config import AvalonConfig
from .state import AvalonState
from .types import (
    Role, Team, Phase, VoteChoice, QuestChoice, TeamProposal, QuestResult,
    MAX_REJECTIONS,
)
from .rules import (
    assign_roles,
    check_quest_result,
    check_game_end,
    find_assassin,
    find_merlin,
    validate_team_proposal,
    get_role_info_for_player,
)


class AvalonEnv(BaseEnvironment):
    """Avalon (The Resistance: Avalon) environment.
    
    A team-based social deduction game where Good tries to complete 3 quests
    and Evil tries to sabotage them. Features asymmetric information through
    special roles like Merlin who knows all evil players.
    
    Game Flow:
    1. Team Selection: Leader proposes a team for the quest
    2. Team Voting: All players vote to approve/reject
    3. Quest Voting: Team members secretly vote success/fail
    4. Repeat until 3 quests succeed or fail
    5. Assassination: If Good wins, Assassin tries to kill Merlin
    """

    def __init__(
        self,
        agents: List[BaseAgent],
        config: Optional[AvalonConfig] = None,
        game_id: Optional[str] = None,
        logger: Optional[GameLogger] = None,
    ):
        """Initialize Avalon environment.
        
        Args:
            agents: List of agents (must match config.n_players)
            config: Game configuration
            game_id: Unique game identifier
            logger: Game logger instance
        """
        config = config or AvalonConfig(n_players=len(agents))
        
        # Set game config before calling super().__init__()
        self.game_config = config
        self.logger = logger
        self.rng = random.Random(config.seed)
        
        super().__init__(agents=agents, config=config.__dict__, game_id=game_id, seed=config.seed)

    def reset(self) -> Dict[int, Observation]:
        """Reset the game to initial state."""
        # Assign roles
        players = assign_roles(self.game_config, self.rng)
        
        # Initialize state
        self.state = AvalonState(
            config=self.game_config,
            rng=self.rng,
            players=players,
            quest_leader=self.rng.randint(0, self.game_config.n_players - 1),
            current_phase=Phase.TEAM_SELECTION,
            current_quest=0,
            current_round=0,
        )
        
        # Log game start
        if self.logger:
            self.logger.log_event(
                EventType.GAME_START,
                {
                    "n_players": self.game_config.n_players,
                    "quest_leader": self.state.quest_leader,
                    "roles": [p.role.value for p in players],
                }
            )
            
            # Log role information for each player (private)
            for pid, player in enumerate(players):
                role_info = get_role_info_for_player(pid, players)
                self.logger.log_event(
                    EventType.ROLE_REVEAL,
                    {
                        "player_id": pid,
                        "role": player.role.value,
                        "team": player.team.value,
                        "info": role_info,
                    },
                    is_private=True,
                )
        
        return self._get_observations()

    def get_state(self) -> AvalonState:
        """Get current game state."""
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self.state

    def _get_observations(self) -> Dict[int, Observation]:
        """Generate observations for all players."""
        st = self.state
        obs = {}
        
        # Map phase to GamePhase enum
        phase_map = {
            Phase.TEAM_SELECTION: GamePhase.TEAM_SELECTION,
            Phase.TEAM_VOTING: GamePhase.VOTING,
            Phase.QUEST_VOTING: GamePhase.QUEST,
            Phase.ASSASSINATION: GamePhase.ASSASSINATION,
            Phase.GAME_END: GamePhase.TERMINAL,
        }
        game_phase = phase_map.get(st.current_phase, GamePhase.SETUP)
        
        for player in st.players:
            # Get role visibility
            visibility = st.get_role_visibility(player.pid)
            
            # Build instruction based on phase
            instruction = ""
            if st.current_phase == Phase.TEAM_SELECTION:
                if player.pid == st.quest_leader:
                    instruction = f"YOU ARE THE QUEST LEADER. Select {st.get_team_size()} players for the quest team. Provide your action as: {{\"type\": \"propose_team\", \"team\": [player_ids]}}"
                else:
                    instruction = f"Waiting for quest leader (Player {st.quest_leader}) to propose a team."
            elif st.current_phase == Phase.TEAM_VOTING:
                instruction = f"Vote on the proposed team: {st.current_proposal.team}. Provide: {{\"type\": \"vote\", \"vote\": \"approve\"}} or {{\"type\": \"vote\", \"vote\": \"reject\"}}"
            elif st.current_phase == Phase.QUEST_VOTING:
                if player.pid in st.current_proposal.team:
                    if player.team == Team.GOOD:
                        instruction = "You are on the quest team. Good players can only vote SUCCESS. Provide: {\"type\": \"quest_vote\", \"quest_vote\": \"success\"}"
                    else:
                        instruction = "You are on the quest team. You can vote SUCCESS or FAIL. Provide: {\"type\": \"quest_vote\", \"quest_vote\": \"success\"} or {\"type\": \"quest_vote\", \"quest_vote\": \"fail\"}"
                else:
                    instruction = f"Waiting for quest team {st.current_proposal.team} to vote."
            elif st.current_phase == Phase.ASSASSINATION:
                assassin_pid = find_assassin(st.players)
                if player.pid == assassin_pid:
                    instruction = "YOU ARE THE ASSASSIN. Choose a player to assassinate (try to kill Merlin). Provide: {\"type\": \"assassinate\", \"target\": player_id}"
                else:
                    instruction = f"Waiting for Assassin (Player {assassin_pid}) to choose target."
            
            # Build data dictionary
            data = {
                # Public information
                "phase": st.current_phase.value,
                "quest_number": st.current_quest + 1,  # 1-indexed for display
                "quest_leader": st.quest_leader,
                "is_quest_leader": player.pid == st.quest_leader,
                "team_size_needed": st.get_team_size(),
                "fails_needed": st.get_fails_needed(),
                "team_rejections": st.team_rejections,
                "quests_succeeded": st.quests_succeeded,
                "quests_failed": st.quests_failed,
                "instruction": instruction,
                
                # Private information
                "player_id": player.pid,
                "role": player.role.value,
                "team": player.team.value,
                "role_info": get_role_info_for_player(player.pid, st.players),
                "visibility": {
                    pid: team.value if team else "unknown"
                    for pid, team in visibility.items()
                },
                
                # Current proposal (if exists)
                "current_proposal": None,
                "proposed_team": None,
                "on_proposed_team": False,
                
                # Quest history
                "quest_history": [
                    {
                        "quest_num": qr.quest_num + 1,
                        "team": qr.team_members,
                        "success_votes": qr.success_votes,
                        "fail_votes": qr.fail_votes,
                        "succeeded": qr.succeeded,
                    }
                    for qr in st.quest_results
                ],
            }
            
            # Add current proposal info if exists
            if st.current_proposal:
                data["current_proposal"] = {
                    "leader": st.current_proposal.leader,
                    "team": st.current_proposal.team,
                }
                data["proposed_team"] = st.current_proposal.team
                data["on_proposed_team"] = player.pid in st.current_proposal.team
            
            obs[player.pid] = Observation(
                player_id=player.pid,
                obs_type=ObservationType.ROLE_SPECIFIC,
                phase=game_phase,
                data=data,
            )
        
        return obs

    def step(self, actions: Dict[int, Action]) -> Tuple[
        Dict[int, Observation],
        Dict[int, float],
        bool,
        Dict[str, Any],
    ]:
        """Execute actions and advance game state."""
        st = self.state
        
        if st.current_phase == Phase.TEAM_SELECTION:
            self._handle_team_selection(actions)
        elif st.current_phase == Phase.TEAM_VOTING:
            self._handle_team_voting(actions)
        elif st.current_phase == Phase.QUEST_VOTING:
            self._handle_quest_voting(actions)
        elif st.current_phase == Phase.ASSASSINATION:
            self._handle_assassination(actions)
        
        # Get observations
        obs = self._get_observations()
        rewards = {p.pid: 0.0 for p in st.players}
        done = st.game_over
        
        return obs, rewards, done, {}

    def _handle_team_selection(self, actions: Dict[int, Action]):
        """Handle team selection phase."""
        st = self.state
        
        # Only quest leader acts
        if st.quest_leader not in actions:
            return
        
        action = actions[st.quest_leader]
        proposed_team = action.data.get("team", [])
        
        # Debug: print what we got
        if self.logger:
            self.logger.log_event(
                EventType.ACTION,
                {
                    "debug": "team_selection_attempt",
                    "leader": st.quest_leader,
                    "action_data": action.data,
                    "proposed_team": proposed_team,
                }
            )
        
        # Validate team
        required_size = st.get_team_size()
        if not validate_team_proposal(proposed_team, required_size, self.game_config.n_players):
            # Invalid proposal, stay in same phase
            if self.logger:
                self.logger.log_event(
                    EventType.ACTION,
                    {
                        "error": "invalid_team_proposal",
                        "proposed_team": proposed_team,
                        "required_size": required_size,
                        "reason": f"Team size={len(proposed_team)}, required={required_size}"
                    }
                )
            return
        
        # Create proposal
        st.current_proposal = TeamProposal(
            leader=st.quest_leader,
            team=proposed_team,
        )
        
        # Log
        if self.logger:
            self.logger.log_event(
                EventType.ACTION,
                {
                    "phase": "team_selection",
                    "leader": st.quest_leader,
                    "proposed_team": proposed_team,
                    "quest": st.current_quest + 1,
                }
            )
        
        # Move to team voting
        st.current_phase = Phase.TEAM_VOTING
        
        if self.logger:
            self.logger.log_event(EventType.PHASE_CHANGE, {"new_phase": "team_voting"})

    def _handle_team_voting(self, actions: Dict[int, Action]):
        """Handle team voting phase."""
        st = self.state
        
        # Collect votes from all players
        votes = {}
        for pid in range(self.game_config.n_players):
            if pid in actions:
                vote = actions[pid].data.get("vote")
                if vote in ("approve", "reject"):
                    votes[pid] = vote
        
        # Need all votes
        if len(votes) < self.game_config.n_players:
            return
        
        # Count votes
        approves = sum(1 for v in votes.values() if v == "approve")
        rejects = sum(1 for v in votes.values() if v == "reject")
        
        st.current_proposal.approve_votes = approves
        st.current_proposal.reject_votes = rejects
        st.current_proposal.approved = approves > rejects
        
        # Log
        if self.logger:
            self.logger.log_event(
                EventType.VOTING,
                {
                    "phase": "team_voting",
                    "proposal": st.current_proposal.team,
                    "approves": approves,
                    "rejects": rejects,
                    "approved": st.current_proposal.approved,
                    "votes": votes,
                }
            )
        
        # Add to history
        st.proposal_history.append(st.current_proposal)
        
        if st.current_proposal.approved:
            # Team approved, move to quest
            st.current_phase = Phase.QUEST_VOTING
            st.team_rejections = 0
            
            if self.logger:
                self.logger.log_event(EventType.PHASE_CHANGE, {"new_phase": "quest_voting"})
        else:
            # Team rejected
            st.team_rejections += 1
            
            if st.team_rejections >= MAX_REJECTIONS:
                # Force quest failure after 5 rejections
                self._force_quest_failure()
            else:
                # Advance quest leader and try again
                st.advance_quest_leader()
                st.current_proposal = None
                st.current_phase = Phase.TEAM_SELECTION
                st.current_round += 1
                
                if self.logger:
                    self.logger.log_event(
                        EventType.PHASE_CHANGE,
                        {
                            "new_phase": "team_selection",
                            "new_leader": st.quest_leader,
                            "rejections": st.team_rejections,
                        }
                    )

    def _handle_quest_voting(self, actions: Dict[int, Action]):
        """Handle quest voting phase."""
        st = self.state
        team = st.current_proposal.team
        
        # Collect votes from team members only
        votes = {}
        for pid in team:
            if pid in actions:
                vote = actions[pid].data.get("quest_vote")
                if vote in ("success", "fail"):
                    votes[pid] = vote
        
        # Need all team member votes
        if len(votes) < len(team):
            return
        
        # Count votes
        success_votes = sum(1 for v in votes.values() if v == "success")
        fail_votes = sum(1 for v in votes.values() if v == "fail")
        
        fails_needed = st.get_fails_needed()
        succeeded = check_quest_result(fail_votes, fails_needed)
        
        # Create quest result
        quest_result = QuestResult(
            quest_num=st.current_quest,
            team_members=team,
            success_votes=success_votes,
            fail_votes=fail_votes,
            succeeded=succeeded,
        )
        st.quest_results.append(quest_result)
        
        if succeeded:
            st.quests_succeeded += 1
        else:
            st.quests_failed += 1
        
        # Log
        if self.logger:
            self.logger.log_event(
                EventType.QUEST,
                {
                    "quest": st.current_quest + 1,
                    "team": team,
                    "success_votes": success_votes,
                    "fail_votes": fail_votes,
                    "succeeded": succeeded,
                    "fails_needed": fails_needed,
                }
            )
        
        # Check if game ended
        game_over, winner = check_game_end(st.quests_succeeded, st.quests_failed)
        
        if game_over:
            if winner == Team.GOOD:
                # Good wins, but Assassin gets to try to kill Merlin
                merlin_pid = find_merlin(st.players)
                if merlin_pid >= 0:
                    st.current_phase = Phase.ASSASSINATION
                    if self.logger:
                        self.logger.log_event(EventType.PHASE_CHANGE, {"new_phase": "assassination"})
                else:
                    # No Merlin, Good wins outright
                    self._end_game(Team.GOOD)
            else:
                # Evil wins
                self._end_game(Team.EVIL)
        else:
            # Continue to next quest
            st.current_quest += 1
            st.current_round = 0
            st.current_proposal = None
            st.proposal_history = []
            st.advance_quest_leader()
            st.current_phase = Phase.TEAM_SELECTION
            
            if self.logger:
                self.logger.log_event(
                    EventType.PHASE_CHANGE,
                    {
                        "new_phase": "team_selection",
                        "next_quest": st.current_quest + 1,
                        "new_leader": st.quest_leader,
                    }
                )

    def _handle_assassination(self, actions: Dict[int, Action]):
        """Handle assassination phase."""
        st = self.state
        
        # Find assassin
        assassin_pid = find_assassin(st.players)
        
        # Get assassin's target
        if assassin_pid not in actions:
            return
        
        target = actions[assassin_pid].data.get("target")
        if target is None or target < 0 or target >= self.game_config.n_players:
            return
        
        st.assassin_target = target
        target_player = st.get_player(target)
        
        # Log
        if self.logger:
            self.logger.log_event(
                EventType.ACTION,
                {
                    "phase": "assassination",
                    "assassin": assassin_pid,
                    "target": target,
                    "target_role": target_player.role.value,
                }
            )
        
        # Check if assassin killed Merlin
        if target_player.role == Role.MERLIN:
            # Evil wins by killing Merlin
            self._end_game(Team.EVIL, f"Assassin killed Merlin (Player {target})")
        else:
            # Good wins, assassin missed
            self._end_game(Team.GOOD, f"Assassin failed to kill Merlin (targeted Player {target})")

    def _force_quest_failure(self):
        """Force quest failure after 5 team rejections."""
        st = self.state
        
        # Quest automatically fails
        quest_result = QuestResult(
            quest_num=st.current_quest,
            team_members=[],
            success_votes=0,
            fail_votes=1,
            succeeded=False,
        )
        st.quest_results.append(quest_result)
        st.quests_failed += 1
        
        if self.logger:
            self.logger.log_event(
                EventType.QUEST,
                {
                    "quest": st.current_quest + 1,
                    "forced_failure": True,
                    "reason": "5 consecutive team rejections",
                }
            )
        
        # Check game end
        game_over, winner = check_game_end(st.quests_succeeded, st.quests_failed)
        
        if game_over:
            self._end_game(winner)
        else:
            # Continue to next quest
            st.current_quest += 1
            st.current_round = 0
            st.current_proposal = None
            st.proposal_history = []
            st.team_rejections = 0
            st.advance_quest_leader()
            st.current_phase = Phase.TEAM_SELECTION

    def _end_game(self, winner: Team, reason: str = ""):
        """End the game."""
        st = self.state
        st.game_over = True
        st.winner = winner
        st.current_phase = Phase.GAME_END
        
        if self.logger:
            self.logger.log_event(
                EventType.GAME_END,
                {
                    "winner": winner.value,
                    "reason": reason,
                    "quests_succeeded": st.quests_succeeded,
                    "quests_failed": st.quests_failed,
                }
            )

    def _validate_num_players(self):
        """Validate player count."""
        if len(self.agents) < 5 or len(self.agents) > 10:
            from sdb.core.exceptions import EnvironmentError
            raise EnvironmentError(f"Avalon requires 5-10 players, got {len(self.agents)}")
    
    def _get_current_player(self) -> int:
        """Get the ID of the current active player."""
        if self.state is None:
            return 0
        
        if self.state.current_phase == Phase.TEAM_SELECTION:
            return self.state.quest_leader
        elif self.state.current_phase == Phase.ASSASSINATION:
            return find_assassin(self.state.players)
        else:
            # Voting phases - all players act
            return 0
    
    def get_winner(self):
        """Get game winner."""
        if self.state is None or not self.state.game_over:
            return None
        return self.state.winner.value if self.state.winner else None
    
    def get_win_reason(self):
        """Get the reason for the win."""
        if self.state is None or not self.state.game_over:
            return None
        
        st = self.state
        if st.winner == Team.GOOD:
            if st.assassin_target is not None:
                target_role = st.get_player(st.assassin_target).role.value
                return f"Good won: 3 quests succeeded, Assassin missed Merlin (targeted {target_role})"
            return f"Good won: {st.quests_succeeded} quests succeeded"
        else:
            if st.assassin_target is not None and st.get_player(st.assassin_target).role == Role.MERLIN:
                return "Evil won: Assassin killed Merlin"
            return f"Evil won: {st.quests_failed} quests failed"

    def play_game(self) -> GameResult:
        """Play a complete game with the configured agents."""
        if not self.agents:
            raise RuntimeError("No agents configured")
        
        obs = self.reset()
        done = False
        num_rounds = 0
        
        while not done and num_rounds < 1000:  # Safety limit
            st = self.state
            
            # Collect actions based on phase
            actions = {}
            
            if st.current_phase == Phase.TEAM_SELECTION:
                # Only quest leader acts
                agent = self.agents[st.quest_leader]
                actions[st.quest_leader] = agent.act(obs[st.quest_leader])
            
            elif st.current_phase == Phase.TEAM_VOTING:
                # All players vote
                for pid in range(self.game_config.n_players):
                    agent = self.agents[pid]
                    actions[pid] = agent.act(obs[pid])
            
            elif st.current_phase == Phase.QUEST_VOTING:
                # Only team members vote
                if st.current_proposal:
                    for pid in st.current_proposal.team:
                        agent = self.agents[pid]
                        actions[pid] = agent.act(obs[pid])
            
            elif st.current_phase == Phase.ASSASSINATION:
                # Only assassin acts
                assassin_pid = find_assassin(st.players)
                agent = self.agents[assassin_pid]
                actions[assassin_pid] = agent.act(obs[assassin_pid])
            
            # Execute actions
            obs, rewards, done, info = self.step(actions)
            num_rounds += 1
        
        # Create result
        st = self.state
        winner = st.winner.value if st.winner else None
        
        # Calculate scores (1.0 for winners, 0.0 for losers)
        scores = {}
        for player in st.players:
            scores[player.pid] = 1.0 if player.team == st.winner else 0.0
        
        return GameResult(
            winner=winner,
            num_rounds=num_rounds,
            final_scores=scores,
            players=[i for i in range(self.game_config.n_players)],
        )

