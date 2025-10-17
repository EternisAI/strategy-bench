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
        
        # Log game start (PUBLIC - no sensitive info)
        if self.logger:
            # Initialize round counter to track quest progress
            self.logger.current_round = 0
            
            self.logger.log(
                EventType.GAME_START,
                {
                    "n_players": self.game_config.n_players,
                    "quest_leader": self.state.quest_leader,
                    "special_roles_enabled": {
                        "merlin": self.game_config.include_merlin,
                        "percival": self.game_config.include_percival,
                        "morgana": self.game_config.include_morgana,
                        "mordred": self.game_config.include_mordred,
                        "oberon": self.game_config.include_oberon,
                    },
                    "seed": self.game_config.seed,
                },
                is_private=False,
            )
            
            # Log role information for each player (PRIVATE)
            for pid, player in enumerate(players):
                role_info = get_role_info_for_player(pid, players)
                self.logger.log(
                    EventType.INFO,
                    {
                        "event": "role_assignment",
                        "player_id": pid,
                        "role": player.role.value,
                        "team": player.team.value,
                        "visible_info": role_info,
                    },
                    player_id=pid,
                    is_private=True,
                )
        
        return self._get_observations()

    def get_state(self) -> AvalonState:
        """Get current game state."""
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self.state

    def _format_quest_history(self) -> str:
        """Format complete quest history for display.
        
        Returns:
            Formatted string of all quest results
        """
        if not self.state.quest_results:
            return "   (No quests completed yet)"
        
        formatted = []
        for qr in self.state.quest_results:
            result_str = "✅ SUCCEEDED" if qr.succeeded else "❌ FAILED"
            formatted.append(
                f"   • Quest {qr.quest_num + 1}: Team {qr.team_members} → "
                f"{qr.success_votes} Success, {qr.fail_votes} Fail → {result_str}"
            )
        return "\n".join(formatted)
    
    def _format_proposal_history(self) -> str:
        """Format team proposal history with individual votes.
        
        Returns:
            Formatted string of all proposals and individual votes
        """
        if not self.state.proposal_history:
            return "   (No proposals yet)"
        
        formatted = []
        for prop in self.state.proposal_history:
            # Show proposal ID, quest, leader, team, and result
            result = "✅ APPROVED" if prop.approved else "❌ REJECTED"
            quest_label = f"Q{prop.quest_num + 1}" if hasattr(prop, 'quest_num') else f"Q?"
            proposal_id = f"#{prop.proposal_idx}" if hasattr(prop, 'proposal_idx') and prop.proposal_idx > 0 else f"#{len(formatted)+1}"
            round_label = f"R{prop.round_idx}" if hasattr(prop, 'round_idx') and prop.round_idx > 0 else ""
            formatted.append(
                f"   {proposal_id} ({quest_label}{round_label}) - Leader {prop.leader} proposed {prop.team} → {result}"
            )
            
            # Show individual votes if available
            if prop.votes:
                vote_strs = []
                for pid in sorted(prop.votes.keys()):
                    vote = prop.votes[pid]
                    emoji = "✅" if vote == "approve" else "❌"
                    vote_strs.append(f"P{pid}:{emoji}")
                formatted.append(f"      Votes: {' '.join(vote_strs)}")
            else:
                # Fallback to tallies if individual votes not available
                formatted.append(
                    f"      Votes: {prop.approve_votes} Approve, {prop.reject_votes} Reject"
                )
        
        return "\n".join(formatted)
    
    def _format_game_state_summary(self) -> str:
        """Format current game state summary.
        
        Returns:
            Formatted string of key game state info
        """
        st = self.state
        
        # Build quest results summary
        quest_summary = []
        for i, qr in enumerate(st.quest_results):
            status = "✅ SUCCESS" if qr.succeeded else "❌ FAILED"
            quest_summary.append(f"Quest {qr.quest_num + 1}: {status}")
        quest_line = " | ".join(quest_summary) if quest_summary else "(No quests completed)"
        
        return f"""📊 GAME STATE:
   • Quest Results: {quest_line}
   • Score: Good {st.quests_succeeded} - {st.quests_failed} Evil (first to 3 wins)
   • Current Quest: {st.current_quest + 1}/5
   • Team Rejections: {st.team_rejections}/{MAX_REJECTIONS}
   • Quest Leader: Player {st.quest_leader}"""
    
    def _get_observations(self) -> Dict[int, Observation]:
        """Generate observations for all players."""
        st = self.state
        obs = {}
        
        # Map phase to GamePhase enum
        phase_map = {
            Phase.TEAM_SELECTION: GamePhase.TEAM_SELECTION,
            Phase.TEAM_DISCUSSION: GamePhase.DISCUSSION,
            Phase.TEAM_VOTING: GamePhase.VOTING,
            Phase.QUEST_VOTING: GamePhase.QUEST,
            Phase.ASSASSINATION: GamePhase.ASSASSINATION,
            Phase.GAME_END: GamePhase.TERMINAL,
        }
        game_phase = phase_map.get(st.current_phase, GamePhase.SETUP)
        
        for player in st.players:
            # Get role visibility
            visibility = st.get_role_visibility(player.pid)
            
            # Build instruction based on phase with full context
            instruction = ""
            if st.current_phase == Phase.TEAM_SELECTION:
                if player.pid == st.quest_leader:
                    available_players = list(range(self.game_config.n_players))
                    instruction = f"""=== QUEST {st.current_quest + 1} - TEAM SELECTION ===

{self._format_game_state_summary()}

📜 QUEST HISTORY:
{self._format_quest_history()}

📋 ALL PROPOSAL HISTORY (All Quests):
{self._format_proposal_history()}

⚡ YOUR ACTION:
YOU ARE THE QUEST LEADER. Select {st.get_team_size()} players for Quest {st.current_quest + 1}.

Available players: {available_players}
Team size needed: {st.get_team_size()}
Fails needed to sabotage: {st.get_fails_needed()}

Strategy: {"Choose players you trust to be Good." if player.team == Team.GOOD else "Include Evil players to sabotage, or build trust by succeeding."}

Respond with JSON:
{{"type": "propose_team", "team": [list of {st.get_team_size()} player IDs]}}

Example: {{"type": "propose_team", "team": {available_players[:st.get_team_size()]}}}"""
                else:
                    instruction = f"Waiting for quest leader (Player {st.quest_leader}) to propose a team."
            
            elif st.current_phase == Phase.TEAM_DISCUSSION:
                # Build dialogue history from current_discussion
                dialogue_history = [
                    (stmt.speaker_id, stmt.statement)
                    for stmt in st.current_discussion
                ]
                
                # Determine if this player is current speaker
                current_speaker = st.discussion_order[st.next_speaker_index] if st.next_speaker_index < len(st.discussion_order) else None
                
                if player.pid == current_speaker:
                    # This player's turn to speak
                    from .prompts import get_team_discussion_instruction
                    is_leader = (player.pid == st.quest_leader)
                    instruction = get_team_discussion_instruction(
                        quest_number=st.current_quest + 1,
                        quest_leader=st.quest_leader,
                        is_leader=is_leader,
                        dialogue_history=dialogue_history,
                        team_size=st.get_team_size(),
                    )
                    # Add context about the proposed team
                    if st.current_proposal:
                        instruction += f"\n\n💡 Proposed Team: {st.current_proposal.team}"
                    
                    # Add full game context
                    instruction += f"\n\n{self._format_game_state_summary()}\n\n📜 QUEST HISTORY:\n{self._format_quest_history()}"
                else:
                    # Waiting for another player
                    instruction = f"DISCUSSION PHASE: Waiting for Player {current_speaker} to speak."
                    if dialogue_history:
                        dialogue_text = "\n".join([
                            f"  - Player {speaker}: \"{stmt}\""
                            for speaker, stmt in dialogue_history
                        ])
                        instruction += f"\n\n📜 Dialogue so far:\n{dialogue_text}"
            
            elif st.current_phase == Phase.TEAM_VOTING:
                instruction = f"""=== QUEST {st.current_quest + 1} - TEAM VOTING ===

{self._format_game_state_summary()}

📜 QUEST HISTORY:
{self._format_quest_history()}

📋 ALL PROPOSAL HISTORY (All Quests):
{self._format_proposal_history()}

⚠️  CURRENT PROPOSAL:
   Leader: Player {st.current_proposal.leader}
   Proposed Team: {st.current_proposal.team}

⚡ YOUR VOTE:
Vote to APPROVE or REJECT this team.

Consider:
- Do you trust these players based on past quests?
- What patterns do you see in voting/quest results?
- What happens if this proposal fails? ({MAX_REJECTIONS - st.team_rejections} rejections left before auto-pass!)

Respond with JSON:
{{"type": "vote", "vote": "approve"}}  to APPROVE
{{"type": "vote", "vote": "reject"}}   to REJECT"""
            elif st.current_phase == Phase.QUEST_VOTING:
                if player.pid in st.current_proposal.team:
                    if player.team == Team.GOOD:
                        instruction = f"""=== QUEST {st.current_quest + 1} - QUEST VOTING ===

{self._format_game_state_summary()}

📜 QUEST HISTORY:
{self._format_quest_history()}

👥 YOUR QUEST TEAM: {st.current_proposal.team}

⚡ YOUR VOTE:
You are GOOD. You can ONLY vote SUCCESS.

Respond with JSON:
{{"type": "quest_vote", "quest_vote": "success"}}"""
                    else:
                        instruction = f"""=== QUEST {st.current_quest + 1} - QUEST VOTING ===

{self._format_game_state_summary()}

📜 QUEST HISTORY:
{self._format_quest_history()}

👥 YOUR QUEST TEAM: {st.current_proposal.team}

⚡ YOUR VOTE:
You are EVIL. You can vote SUCCESS or FAIL.

Strategy:
- Sabotaging advances Evil toward victory
- But succeeding can build trust for future quests
- Consider how many fails will be visible ({st.get_fails_needed()} needed to fail quest)

Respond with JSON:
{{"type": "quest_vote", "quest_vote": "success"}}  to help quest succeed
{{"type": "quest_vote", "quest_vote": "fail"}}     to sabotage quest"""
                else:
                    instruction = f"Waiting for quest team {st.current_proposal.team} to vote."
            elif st.current_phase == Phase.ASSASSINATION:
                assassin_pid = find_assassin(st.players)
                if player.pid == assassin_pid:
                    good_players = [p.pid for p in st.players if p.team == Team.GOOD]
                    instruction = f"""=== ASSASSINATION PHASE ===

🎯 YOU ARE THE ASSASSIN!

Good completed 3 quests, but you have ONE LAST CHANCE!
If you correctly assassinate MERLIN, Evil wins!

{self._format_game_state_summary()}

📜 COMPLETE QUEST HISTORY:
{self._format_quest_history()}

📋 ALL PROPOSALS:
{self._format_proposal_history()}

⚡ YOUR ACTION:
Analyze the game and identify who behaved like Merlin.

Who seemed to have perfect information?
Who was trusted but not too obvious?
Who guided Good's decisions?

Available targets (Good players): {good_players}

Respond with JSON:
{{"type": "assassinate", "target": <player_id>}}

Example: {{"type": "assassinate", "target": {good_players[0] if good_players else 0}}}"""
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
                
                # Quest history with clear outcomes
                "quest_history": [
                    {
                        "quest_num": qr.quest_num + 1,
                        "team": qr.team_members,
                        "team_size": len(qr.team_members),
                        "success_votes": qr.success_votes,
                        "fail_votes": qr.fail_votes,
                        "succeeded": qr.succeeded,
                        "result": "SUCCESS" if qr.succeeded else "FAILED",
                    }
                    for qr in st.quest_results
                ],
                
                # Summary statistics
                "quests_succeeded": st.quests_succeeded,
                "quests_failed": st.quests_failed,
                "quests_remaining": 5 - (st.quests_succeeded + st.quests_failed),
                
                # Formatted full context for better readability
                "formatted_quest_history": self._format_quest_history(),
                "formatted_proposal_history": self._format_proposal_history(),
                "formatted_game_state": self._format_game_state_summary(),
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

    def _validate_action(self, player_id: int, action: Action, phase: Phase) -> tuple[bool, Optional[str]]:
        """Validate that an action is allowed in the current phase.
        
        Args:
            player_id: ID of acting player
            action: Action being attempted
            phase: Current game phase
            
        Returns:
            (is_valid, error_message)
        """
        st = self.state
        action_type = action.data.get("type", "")
        
        # Phase-specific action validation
        ALLOWED_ACTIONS = {
            Phase.TEAM_SELECTION: {"propose_team"},
            Phase.TEAM_DISCUSSION: {"discuss_team", "wait"},
            Phase.TEAM_VOTING: {"team_vote", "vote"},  # accept both for compatibility
            Phase.QUEST_VOTING: {"quest_vote"},
            Phase.ASSASSINATION: {"assassinate"},
        }
        
        if action_type not in ALLOWED_ACTIONS.get(phase, set()):
            return False, f"Action '{action_type}' not allowed in phase {phase.value}"
        
        # Team selection: only leader can propose
        if phase == Phase.TEAM_SELECTION and player_id != st.quest_leader:
            return False, f"Only quest leader (Player {st.quest_leader}) can propose team"
        
        # Quest voting: only team members can vote
        if phase == Phase.QUEST_VOTING:
            if st.current_proposal is None:
                return False, "No team proposal exists for quest voting"
            if player_id not in st.current_proposal.team:
                return False, f"Player {player_id} not on quest team {st.current_proposal.team}"
            # Validate quest vote is success/fail (check both 'vote' and 'quest_vote' keys)
            vote = action.data.get("quest_vote") or action.data.get("vote", "")
            if vote not in ["success", "fail"]:
                return False, f"Quest vote must be 'success' or 'fail', got '{vote}'"
            # Prevent double voting
            if player_id in st.quest_voters_done:
                return False, f"Player {player_id} already voted in this quest voting phase"
        
        # Team voting: validate approve/reject
        if phase == Phase.TEAM_VOTING:
            vote = action.data.get("vote", "")
            if vote not in ["approve", "reject"]:
                return False, f"Team vote must be 'approve' or 'reject', got '{vote}'"
            # Prevent double voting
            if player_id in st.team_votes_cast:
                return False, f"Player {player_id} already voted in this team voting phase"
        
        # Assassination: only assassin can act
        if phase == Phase.ASSASSINATION:
            assassin_pid = find_assassin(st.players)
            if player_id != assassin_pid:
                return False, f"Only assassin (Player {assassin_pid}) can assassinate"
        
        return True, None
    
    def step(self, actions: Dict[int, Action]) -> Tuple[
        Dict[int, Observation],
        Dict[int, float],
        bool,
        Dict[str, Any],
    ]:
        """Execute actions and advance game state."""
        st = self.state
        
        # Validate and filter actions
        validated_actions = {}
        for player_id, action in actions.items():
            is_valid, error = self._validate_action(player_id, action, st.current_phase)
            if is_valid:
                validated_actions[player_id] = action
            elif error:
                # Log validation error privately
                if self.logger:
                    self.logger.log(
                        EventType.INFO,
                        {
                            "event": "action_rejected",
                            "player_id": player_id,
                            "action_type": action.data.get("type"),
                            "phase": st.current_phase.value,
                            "reason": error,
                        },
                        is_private=True
                    )
        
        # Execute validated actions
        if st.current_phase == Phase.TEAM_SELECTION:
            self._handle_team_selection(validated_actions)
        elif st.current_phase == Phase.TEAM_DISCUSSION:
            self._handle_team_discussion(validated_actions)
        elif st.current_phase == Phase.TEAM_VOTING:
            self._handle_team_voting(validated_actions)
        elif st.current_phase == Phase.QUEST_VOTING:
            self._handle_quest_voting(validated_actions)
        elif st.current_phase == Phase.ASSASSINATION:
            self._handle_assassination(validated_actions)
        
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
            self.logger.log(
                EventType.PLAYER_ACTION,
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
                self.logger.log(
                    EventType.PLAYER_ACTION,
                    {
                        "error": "invalid_team_proposal",
                        "proposed_team": proposed_team,
                        "required_size": required_size,
                        "reason": f"Team size={len(proposed_team)}, required={required_size}"
                    }
                )
            return
        
        # Increment proposal counters
        st.total_proposals += 1
        st.current_round += 1
        
        # Create proposal with tracking indices
        st.current_proposal = TeamProposal(
            leader=st.quest_leader,
            team=proposed_team,
            quest_num=st.current_quest,
            proposal_idx=st.total_proposals,
            round_idx=st.current_round,
        )
        
        # Log
        if self.logger:
            self.logger.log(
                EventType.PLAYER_ACTION,
                {
                    "phase": "team_selection",
                    "leader": st.quest_leader,
                    "proposed_team": proposed_team,
                    "quest": st.current_quest + 1,
                }
            )
        
        # Initialize discussion for this proposal
        st.current_discussion = []
        st.next_speaker_index = 0
        # Discussion order: leader first, then all others in round-robin
        st.discussion_order = [st.quest_leader] + [
            pid for pid in range(self.game_config.n_players) if pid != st.quest_leader
        ]
        
        # Clear vote tracking for new proposal
        st.team_votes_cast = set()
        st.quest_votes_by_player = {}
        st.quest_voters_done = set()
        
        # Clear discussion tracking for new discussion round
        st.spoken_this_round = set()
        
        # Move to team discussion
        st.current_phase = Phase.TEAM_DISCUSSION
        
        if self.logger:
            self.logger.log(EventType.PHASE_CHANGE, {
                "new_phase": "team_discussion",
                "discussion_order": st.discussion_order
            })

    def _handle_team_discussion(self, actions: Dict[int, Action]):
        """Handle team discussion phase - sequential dialogue."""
        st = self.state
        
        # Determine current speaker
        if st.next_speaker_index >= len(st.discussion_order):
            # All players have spoken, move to voting
            st.current_phase = Phase.TEAM_VOTING
            if self.logger:
                self.logger.log(EventType.PHASE_CHANGE, {"new_phase": "team_voting"})
            return
        
        current_speaker = st.discussion_order[st.next_speaker_index]
        
        # Skip if already spoken this round (duplicate protection)
        if current_speaker in st.spoken_this_round:
            st.next_speaker_index += 1
            return
        
        # Wait for current speaker's action
        if current_speaker not in actions:
            return
        
        action = actions[current_speaker]
        statement = action.data.get("statement", "")
        
        if not statement or not statement.strip():
            # Empty statement, just skip
            statement = "(no comment)"
        
        # Normalize statement for duplicate detection
        normalized_stmt = " ".join(statement.strip().split()).lower()
        
        # Check for duplicate (same player, same quest/proposal, similar text)
        is_duplicate = False
        for prev_stmt in st.current_discussion:
            if prev_stmt.speaker_id == current_speaker:
                prev_normalized = " ".join(prev_stmt.statement.split()).lower()
                if prev_normalized == normalized_stmt:
                    is_duplicate = True
                    break
        
        if is_duplicate:
            # Skip duplicate, don't record
            st.next_speaker_index += 1
            st.spoken_this_round.add(current_speaker)
            return
        
        # Record statement
        from .types import DiscussionStatement
        discussion_stmt = DiscussionStatement(
            speaker_id=current_speaker,
            statement=statement.strip(),
            quest_num=st.current_quest,
            round_num=st.current_round
        )
        st.current_discussion.append(discussion_stmt)
        st.spoken_this_round.add(current_speaker)  # Mark as spoken
        
        # Log discussion
        if self.logger:
            self.logger.log(
                EventType.DISCUSSION,
                {
                    "quest": st.current_quest + 1,
                    "round": st.current_round,
                    "speaker": current_speaker,
                    "statement": statement.strip(),
                    "discussion_length": len(st.current_discussion),
                },
                is_private=False  # Public discussion
            )
        
        # Notify all agents of the statement
        for pid in range(self.game_config.n_players):
            agent = self.agents[pid]
            if hasattr(agent, 'add_memory'):
                agent.add_memory(f"Player {current_speaker} said: \"{statement.strip()}\"")
        
        # Move to next speaker
        st.next_speaker_index += 1
    
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
                    st.team_votes_cast.add(pid)  # Mark as voted
        
        # Need all votes
        if len(votes) < self.game_config.n_players:
            return
        
        # Count votes
        approves = sum(1 for v in votes.values() if v == "approve")
        rejects = sum(1 for v in votes.values() if v == "reject")
        
        # Store individual votes and tallies in proposal
        st.current_proposal.votes = votes
        st.current_proposal.approve_votes = approves
        st.current_proposal.reject_votes = rejects
        st.current_proposal.approved = approves > rejects
        
        # Log
        if self.logger:
            self.logger.log(
                EventType.PLAYER_VOTE,
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
            # Clear quest voting tracking
            st.quest_votes_by_player = {}
            st.quest_voters_done = set()
            
            if self.logger:
                self.logger.log(EventType.PHASE_CHANGE, {"new_phase": "quest_voting"})
        else:
            # Team rejected
            st.team_rejections += 1
            
            if st.team_rejections >= MAX_REJECTIONS:
                # Evil wins immediately after 5 rejections (standard Avalon rule)
                self._force_evil_win_by_rejections()
            else:
                # Advance quest leader and try again
                st.advance_quest_leader()
                st.current_proposal = None
                st.current_phase = Phase.TEAM_SELECTION
                st.current_round += 1
                
                if self.logger:
                    self.logger.log(
                        EventType.PHASE_CHANGE,
                        {
                            "new_phase": "team_selection",
                            "new_leader": st.quest_leader,
                            "rejections": st.team_rejections,
                        }
                    )

    def _handle_quest_voting(self, actions: Dict[int, Action]):
        """Handle quest voting phase - only team members can vote, exactly once."""
        st = self.state
        team = st.current_proposal.team
        
        # Collect votes from team members only (idempotent - duplicates ignored)
        for pid in team:
            if pid in actions and pid not in st.quest_voters_done:
                # Accept both 'quest_vote' and 'vote' keys for compatibility
                vote = actions[pid].data.get("quest_vote") or actions[pid].data.get("vote")
                if vote in ("success", "fail"):
                    st.quest_votes_by_player[pid] = vote
                    st.quest_voters_done.add(pid)
                    
                    # Log privately (ballots are anonymous)
                    if self.logger:
                        self.logger.log(
                            EventType.INFO,
                            {
                                "event": "quest_ballot_recorded",
                                "player": pid,
                                "ballot": vote,
                                "progress": f"{len(st.quest_voters_done)}/{len(team)}",
                            },
                            player_id=pid,
                            is_private=True
                        )
        
        # Check if all team members have voted
        if len(st.quest_voters_done) < len(team):
            # Still waiting for votes
            return
        
        # All votes collected - close quest voting
        success_votes = sum(1 for v in st.quest_votes_by_player.values() if v == "success")
        fail_votes = len(team) - success_votes
        
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
        
        # Log quest result (PUBLIC - only show anonymous counts, never individual ballots)
        if self.logger:
            self.logger.log(
                EventType.QUEST_RESULT,
                {
                    "quest": st.current_quest + 1,
                    "team": team,
                    "success_votes": success_votes,
                    "fail_votes": fail_votes,
                    "succeeded": succeeded,
                    "fails_needed": fails_needed,
                },
                is_private=False  # Public: only counts, not individual ballots
            )
        
        # Clear quest voting tracking after processing
        st.quest_votes_by_player = {}
        st.quest_voters_done = set()
        
        # Check if game ended
        game_over, winner = check_game_end(st.quests_succeeded, st.quests_failed)
        
        if game_over:
            if winner == Team.GOOD:
                # Good wins, but Assassin gets to try to kill Merlin
                merlin_pid = find_merlin(st.players)
                if merlin_pid >= 0:
                    st.current_phase = Phase.ASSASSINATION
                    if self.logger:
                        self.logger.log(EventType.PHASE_CHANGE, {"new_phase": "assassination"})
                else:
                    # No Merlin, Good wins outright
                    reason = f"Good won: {st.quests_succeeded} quests succeeded (no Merlin to assassinate)"
                    self._end_game(Team.GOOD, reason)
            else:
                # Evil wins
                reason = f"Evil won: {st.quests_failed} quests failed"
                self._end_game(Team.EVIL, reason)
        else:
            # Continue to next quest
            st.current_quest += 1
            st.current_round = 0
            st.current_proposal = None
            # DON'T clear proposal_history - keep it for strategic analysis
            st.advance_quest_leader()
            st.current_phase = Phase.TEAM_SELECTION
            
            if self.logger:
                # Update logger's round counter to match current quest
                self.logger.current_round = st.current_quest
                
                self.logger.log(
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
        
        # Log assassination (PRIVATE - contains sensitive role information)
        if self.logger:
            self.logger.log(
                EventType.PLAYER_ACTION,
                {
                    "phase": "assassination",
                    "assassin": assassin_pid,
                    "target": target,
                    "target_role": target_player.role.value,
                    "is_merlin": target_player.role == Role.MERLIN,
                },
                is_private=True  # Keep role information private
            )
            
            # Also log a public version without sensitive info
            self.logger.log(
                EventType.INFO,
                {
                    "event": "assassination_occurred",
                    "target": target,
                    "result": "success" if target_player.role == Role.MERLIN else "failure"
                },
                is_private=False
            )
        
        # Check if assassin killed Merlin
        if target_player.role == Role.MERLIN:
            # Evil wins by killing Merlin
            self._end_game(Team.EVIL, f"Assassin killed Merlin (Player {target})")
        else:
            # Good wins, assassin missed
            self._end_game(Team.GOOD, f"Assassin failed to kill Merlin (targeted Player {target})")

    def _force_evil_win_by_rejections(self):
        """Evil wins immediately after 5 consecutive team rejections.
        
        This is the standard Avalon rule: 5 rejections → Evil wins instantly,
        not through quest failure mechanics.
        """
        st = self.state
        
        # Log the rejection-triggered win
        if self.logger:
            self.logger.log(
                EventType.INFO,
                {
                    "event": "five_rejections_triggered",
                    "quest": st.current_quest + 1,
                    "rejections": st.team_rejections,
                    "reason": "5 consecutive team proposal rejections",
                }
            )
        
        # Evil wins immediately (standard Avalon rule)
        reason = f"Evil wins: 5 consecutive team rejections on Quest {st.current_quest + 1}"
        self._end_game(Team.EVIL, reason)

    def _end_game(self, winner: Team, reason: str = ""):
        """End the game with comprehensive summary."""
        st = self.state
        st.game_over = True
        st.winner = winner
        st.current_phase = Phase.GAME_END
        
        if self.logger:
            # Build comprehensive game summary
            summary_data = {
                # Core outcome
                "winner": winner.value,
                "win_reason": reason or "Unknown",
                
                # Quest ledger
                "quests_succeeded": st.quests_succeeded,
                "quests_failed": st.quests_failed,
                "total_quests_played": len(st.quest_results),
                
                # Quest details
                "quest_results": [
                    {
                        "quest_num": qr.quest_num + 1,
                        "team": qr.team_members,
                        "success_votes": qr.success_votes,
                        "fail_votes": qr.fail_votes,
                        "succeeded": qr.succeeded,
                    }
                    for qr in st.quest_results
                ],
                
                # Proposal statistics
                "total_proposals": st.total_proposals,
                "total_rejections": sum(1 for p in st.proposal_history if not p.approved),
                "total_approvals": sum(1 for p in st.proposal_history if p.approved),
                "max_consecutive_rejections": st.team_rejections,
                
                # Assassination details (if occurred)
                "assassination_occurred": st.assassin_target is not None,
                "assassin_target": st.assassin_target,
                "assassin_correct": None,
                
                # Configuration snapshot
                "config": {
                    "n_players": self.game_config.n_players,
                    "seed": self.game_config.seed,
                    "special_roles": {
                        "merlin": self.game_config.include_merlin,
                        "percival": self.game_config.include_percival,
                        "morgana": self.game_config.include_morgana,
                        "mordred": self.game_config.include_mordred,
                        "oberon": self.game_config.include_oberon,
                    },
                },
                
                # Role assignments (private)
                "roles": [p.role.value for p in st.players],
                "teams": [p.team.value for p in st.players],
            }
            
            # Add assassination correctness if it occurred
            if st.assassin_target is not None:
                target_player = st.get_player(st.assassin_target)
                summary_data["assassin_correct"] = (target_player.role == Role.MERLIN)
                summary_data["target_role"] = target_player.role.value
            
            # Log comprehensive summary
            self.logger.log(
                EventType.GAME_END,
                summary_data,
                is_private=False  # Summary is public except roles/teams
            )
            
            # Also log private version with full details
            self.logger.log(
                EventType.INFO,
                {
                    "event": "game_summary_private",
                    **summary_data,
                },
                is_private=True
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
        
        # Get initial observations (reset was already called in __init__)
        obs = self._get_observations()
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
            
            elif st.current_phase == Phase.TEAM_DISCUSSION:
                # All players can discuss
                for pid in range(self.game_config.n_players):
                    # Check if this player has spoken this round
                    if pid not in st.spoken_this_round:
                        agent = self.agents[pid]
                        actions[pid] = agent.act(obs[pid])
            
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
        win_reason = self.get_win_reason() if st.winner else "Game reached maximum rounds"
        
        # Calculate player stats
        player_stats = {}
        for player in st.players:
            score = 1.0 if player.team == st.winner else 0.0
            player_stats[player.pid] = {
                "score": score,
                "team": player.team.value if player.team else "unknown",
                "role": player.role.value if player.role else "unknown",
            }
        
        return GameResult(
            game_id=self.game_id,
            winner=winner,
            win_reason=win_reason,
            num_rounds=num_rounds,
            duration_seconds=0.0,
            player_stats=player_stats,
            metadata={
                "quests_succeeded": st.quests_succeeded,
                "quests_failed": st.quests_failed,
                "total_proposals": st.total_proposals if hasattr(st, 'total_proposals') else 0,
            }
        )

