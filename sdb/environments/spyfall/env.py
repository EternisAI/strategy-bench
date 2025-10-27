"""Spyfall game environment."""

import random
from typing import Dict, List, Optional, Tuple

from sdb.core.base_env import BaseEnvironment
from sdb.core.types import Action, GameResult, Observation
from sdb.environments.spyfall.config import SpyfallConfig
from sdb.environments.spyfall.rules import (
    assign_roles,
    calculate_scores,
    get_voting_result,
    validate_accusation_target,
    validate_question_target,
    validate_spy_guess,
)
from sdb.environments.spyfall.state import SpyfallState
from sdb.environments.spyfall.types import (
    AccusationState,
    FinalVoteState,
    Phase,
)
from sdb.logging.formats import EventType


class SpyfallEnv(BaseEnvironment):
    """Spyfall game environment with Q&A and deduction mechanics."""
    
    def __init__(self, agents, config=None, game_id=None, logger=None, role_assignment=None):
        """Initialize Spyfall environment.
        
        Args:
            agents: List of agents (must match config.n_players)
            config: SpyfallConfig instance
            game_id: Optional game ID
            logger: Optional GameLogger instance
            role_assignment: Optional dict with 'spy' key specifying spy index
        """
        config = config or SpyfallConfig()
        self.game_config = config
        self.rng = random.Random()
        self.logger = logger
        self.role_assignment = role_assignment  # Store for use in reset()
        super().__init__(agents, config=config.__dict__, game_id=game_id, seed=getattr(config, 'seed', None))
    
    def _validate_num_players(self):
        """Validate number of players."""
        if not (3 <= self.num_players <= 12):
            raise EnvironmentError(f"Spyfall requires 3-12 players, got {self.num_players}")
    
    def _get_current_player(self) -> Optional[int]:
        """Get the current acting player."""
        if self.state.phase == Phase.QANDA:
            if self.state.awaiting_answer_from is not None:
                return self.state.awaiting_answer_from
            return self.state.current_asker
        elif self.state.phase == Phase.ACCUSATION_VOTE:
            # Return first player who hasn't voted yet
            if self.state.accusation:
                for voter in self.state.accusation.voters:
                    if voter not in self.state.accusation.votes:
                        return voter
        elif self.state.phase == Phase.FINAL_VOTE:
            if self.state.final_vote:
                if self.state.final_vote.current_suspect is None:
                    return self.state.final_vote.current_nominator
                # Return first player who hasn't voted
                for pid in range(self.game_config.n_players):
                    if pid != self.state.final_vote.current_suspect and pid not in self.state.final_vote.votes:
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
        self.state = SpyfallState()
        
        # Assign roles and location
        # If role_assignment provided (from tournament), use it; otherwise random
        if self.role_assignment and 'spy' in self.role_assignment:
            spy_index = self.role_assignment['spy']
            location, _, cards = assign_roles(self.game_config, self.rng, fixed_spy_index=spy_index)
        else:
            location, spy_index, cards = assign_roles(self.game_config, self.rng)
        
        # Initialize state
        self.state.phase = Phase.QANDA
        self.state.turn = 0
        self.state.location = location
        self.state.spy_index = spy_index
        self.state.cards = cards
        self.state.qa_history = []
        self.state.current_asker = self.game_config.dealer_index
        self.state.awaiting_answer_from = None
        self.state.cannot_ask_back = None
        self.state.stops_used = {i: False for i in range(self.game_config.n_players)}
        self.state.spy_guess_allowed = True
        self.state.accusation = None
        self.state.final_vote = None
        self.state.winner = None
        self.state.scores = {i: 0 for i in range(self.game_config.n_players)}
        self.state.win_reason = ""
        
        # Log game start
        if self.logger:
            self.logger.log(
                event_type=EventType.GAME_START,
                data={
                    "n_players": self.game_config.n_players,
                    "dealer": self.game_config.dealer_index,
                }
            )
            
            # Log role assignments (private) - location and spy
            self.logger.log(
                event_type=EventType.PLAYER_ACTION,
                data={
                    "action": "role_assignment",
                    "location": location,
                    "spy_index": spy_index,
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
        # Handle actions from relevant players based on phase
        for player_id, action in actions.items():
            if self.state.phase == Phase.QANDA:
                self._handle_qanda_action(action)
            elif self.state.phase == Phase.ACCUSATION_VOTE:
                self._handle_accusation_vote(action)
            elif self.state.phase == Phase.FINAL_VOTE:
                # Only process actions from players who should act
                if self.state.final_vote:
                    if self.state.final_vote.current_suspect is None:
                        # Only nominator can act
                        if player_id == self.state.final_vote.current_nominator:
                            self._handle_final_vote_action(action)
                    else:
                        # Everyone except suspect can vote
                        if player_id != self.state.final_vote.current_suspect:
                            self._handle_final_vote_action(action)
            elif self.state.phase == Phase.SPY_GUESS:
                # Only process action from the spy
                if player_id == self.state.spy_index:
                    self._handle_spy_guess_action(action)
        
        # Get observations
        observations = self._get_observations()
        
        # Check if game is over
        done = self.state.phase == Phase.GAME_END
        
        # Calculate rewards
        rewards = {pid: 0.0 for pid in range(self.game_config.n_players)}
        if done:
            for pid, score in self.state.scores.items():
                rewards[pid] = float(score)
        
        info = {
            "phase": self.state.phase.value,
            "turn": self.state.turn
        }
        
        return observations, rewards, done, info
    
    def _handle_qanda_action(self, action: Action):
        """Handle Q&A phase actions."""
        action_type = action.data.get("type", "")
        
        if action_type == "ask":
            self._handle_ask(action)
        elif action_type == "answer":
            self._handle_answer(action)
        elif action_type == "accuse":
            self._handle_accuse(action)
        elif action_type == "spy_guess":
            self._handle_spy_guess_initiation(action)
        elif action_type == "wait":
            pass  # No action needed
        else:
            if self.logger:
                self.logger.log(
                    EventType.ERROR,
                    {"player_id": action.player_id, "error": f"Unknown action type: {action_type}"}
                )
    
    def _handle_ask(self, action: Action):
        """Handle asking a question."""
        # Disallow new question if we are waiting for an answer
        if self.state.awaiting_answer_from is not None:
            if self.logger:
                self.logger.log(
                    EventType.ERROR,
                    {
                        "player_id": action.player_id,
                        "error": "Cannot ask while awaiting an answer"
                    }
                )
            return
        
        target = action.data.get("target")
        question = action.data.get("question", "")
        
        if not question:
            if self.logger:
                self.logger.log(
                    EventType.ERROR,
                    {"player_id": action.player_id, "error": "Empty question"}
                )
            return
        
        # Check if target is provided
        if target is None:
            if self.logger:
                self.logger.log(
                    EventType.ERROR,
                    {"player_id": action.player_id, "error": "No target provided for question"}
                )
            return
        
        # Validate target
        valid, error = validate_question_target(
            action.player_id,
            target,
            self.game_config.n_players,
            self.state.cannot_ask_back
        )
        
        if not valid:
            if self.logger:
                self.logger.log(
                    EventType.ERROR,
                    {"player_id": action.player_id, "error": error}
                )
            return
        
        # Log question
        if self.logger:
            self.logger.log(
                EventType.DISCUSSION,
                {
                    "turn": self.state.turn,
                    "asker": action.player_id,
                    "target": target,
                    "question": question,
                },
                is_private=False
            )
        
        # Update state
        self.state.awaiting_answer_from = target
        self.state.turn += 1
    
    def _handle_answer(self, action: Action):
        """Handle answering a question."""
        answer = action.data.get("answer", "")
        
        if not answer:
            if self.logger:
                self.logger.log(
                    EventType.ERROR,
                    {"player_id": action.player_id, "error": "Empty answer"}
                )
            return
        
        asker = self.state.current_asker
        answerer = action.player_id
        
        # Get the last question (should be the one being answered)
        question = "unknown"
        for event in reversed(self.state.qa_history):
            if event.answerer == answerer:
                question = event.question
                break
        
        # Add to QA history
        self.state.add_qa(asker, answerer, question, answer)
        
        # Log answer
        if self.logger:
            self.logger.log(
                EventType.DISCUSSION,
                {
                    "turn": self.state.turn,
                    "answerer": answerer,
                    "answer": answer,
                },
                is_private=False
            )
        
        # Answerer becomes the new asker
        self.state.current_asker = answerer
        self.state.awaiting_answer_from = None
        self.state.cannot_ask_back = asker  # Can't ask back to previous asker
        
        # Check if turn limit reached
        if self.state.turn >= self.game_config.max_turns:
            self._start_final_voting()
    
    def _handle_accuse(self, action: Action):
        """Handle stopping the clock to accuse someone."""
        suspect = action.data.get("suspect")
        
        # Check if suspect is provided
        if suspect is None:
            if self.logger:
                self.logger.log(
                    EventType.ERROR,
                    {"player_id": action.player_id, "error": "No suspect provided in accusation"}
                )
            return
        
        # Validate accusation
        valid, error = validate_accusation_target(
            action.player_id,
            suspect,
            self.game_config.n_players
        )
        
        if not valid:
            if self.logger:
                self.logger.log(
                    EventType.ERROR,
                    {"player_id": action.player_id, "error": error}
                )
            return
        
        # Mark stop used
        self.state.stops_used[action.player_id] = True
        self.state.spy_guess_allowed = False  # Spy can't guess after someone stops
        
        # Log accusation
        if self.logger:
            self.logger.log(
                EventType.PLAYER_ACTION,
                {
                    "turn": self.state.turn,
                    "accuser": action.player_id,
                    "suspect": suspect,
                    "action": "accuse",
                },
                is_private=False
            )
        
        # Start accusation vote
        voters = [i for i in range(self.game_config.n_players) if i != suspect]
        self.state.accusation = AccusationState(
            accuser=action.player_id,
            suspect=suspect,
            voters=voters,
            votes={}
        )
        self.state.phase = Phase.ACCUSATION_VOTE
    
    def _handle_spy_guess_initiation(self, action: Action):
        """Handle spy attempting to guess location."""
        if action.player_id != self.state.spy_index:
            if self.logger:
                self.logger.log(
                    EventType.ERROR,
                    {"player_id": action.player_id, "error": "Only spy can guess location"}
                )
            return
        
        if not self.state.can_spy_guess():
            if self.logger:
                self.logger.log(
                    EventType.ERROR,
                    {"player_id": action.player_id, "error": "Spy cannot guess location now"}
                )
            return
        
        # Log spy guess attempt
        if self.logger:
            self.logger.log(
                EventType.PLAYER_ACTION,
                {
                    "turn": self.state.turn,
                    "player_id": action.player_id,
                    "action": "spy_guess_initiation",
                },
                is_private=False
            )
        
        self.state.phase = Phase.SPY_GUESS
    
    def _handle_spy_guess_action(self, action: Action):
        """Handle spy's location guess."""
        guess = action.data.get("guess")
        
        # Check if guess is provided
        if guess is None:
            if self.logger:
                self.logger.log(
                    EventType.ERROR,
                    {"player_id": action.player_id, "error": "No location guess provided"}
                )
            return
        
        # Validate guess
        valid, is_correct, error = validate_spy_guess(
            guess,
            self.state.location,
            self.game_config.locations
        )
        
        if not valid:
            if self.logger:
                self.logger.log(
                    EventType.ERROR,
                    {"player_id": action.player_id, "error": error}
                )
            return
        
        # Log guess
        if self.logger:
            self.logger.log(
                EventType.PLAYER_ACTION,
                {
                    "player_id": action.player_id,
                    "action": "spy_guess",
                    "guess": guess,
                    "correct": is_correct,
                },
                is_private=False
            )
        
        # End game
        if is_correct:
            self.state.winner = "spy"
            self.state.win_reason = "Spy guessed location correctly"
        else:
            self.state.winner = "non_spy"
            self.state.win_reason = "Spy guessed location incorrectly"
        
        self.state.scores = calculate_scores(
            self.game_config.n_players,
            self.state.spy_index,
            self.state.winner,
            is_correct
        )
        self.state.phase = Phase.GAME_END
        
        if self.logger:
            self.logger.log(
                EventType.GAME_END,
                {
                    "winner": self.state.winner,
                    "reason": self.state.win_reason,
                    "scores": self.state.scores,
                }
            )
    
    def _handle_accusation_vote(self, action: Action):
        """Handle voting on an accusation."""
        vote = action.data.get("vote")
        
        if vote is None:
            if self.logger:
                self.logger.log(
                    EventType.ERROR,
                    {"player_id": action.player_id, "error": "No vote provided"}
                )
            return
        
        # Check for duplicate vote
        if action.player_id in self.state.accusation.votes:
            # Already voted - ignore duplicate
            return
        
        # Record vote
        self.state.accusation.votes[action.player_id] = bool(vote)
        
        # Log vote
        if self.logger:
            self.logger.log(
                EventType.VOTE_CAST,
                {
                    "voter": action.player_id,
                    "vote": vote,
                    "suspect": self.state.accusation.suspect,
                },
                is_private=True
            )
        
        # Check if voting complete
        if self.state.accusation.is_complete():
            self._resolve_accusation()
    
    def _resolve_accusation(self):
        """Resolve the accusation vote."""
        passed = self.state.accusation.is_successful()
        suspect = self.state.accusation.suspect
        
        # Log result
        if self.logger:
            self.logger.log(
                EventType.ELECTION_RESULT,
                {
                    "passed": passed,
                    "suspect": suspect,
                    "votes": self.state.accusation.votes,
                }
            )
        
        if passed:
            # Accusation passed - game ends
            is_spy = (suspect == self.state.spy_index)
            
            if is_spy:
                self.state.winner = "non_spy"
                self.state.win_reason = "Spy was correctly identified"
            else:
                self.state.winner = "spy"
                self.state.win_reason = "Wrong player was accused"
            
            self.state.scores = calculate_scores(
                self.game_config.n_players,
                self.state.spy_index,
                self.state.winner,
                False
            )
            self.state.phase = Phase.GAME_END
            
            if self.logger:
                self.logger.log(
                    EventType.GAME_END,
                    {
                        "winner": self.state.winner,
                        "reason": self.state.win_reason,
                        "scores": self.state.scores,
                    }
                )
        else:
            # Accusation failed - return to Q&A or end game
            if self.state.turn >= self.game_config.max_turns:
                self._start_final_voting()
            else:
                self.state.phase = Phase.QANDA
                self.state.accusation = None
    
    def _start_final_voting(self):
        """Start final voting phase."""
        # Log start of final voting
        if self.logger:
            self.logger.log(
                EventType.PHASE_CHANGE,
                {
                    "from_phase": self.state.phase.value,
                    "to_phase": "final_vote",
                    "reason": "Turn limit reached",
                }
            )
        
        self.state.phase = Phase.FINAL_VOTE
        self.state.final_vote = FinalVoteState(
            current_nominator=0,  # Start with player 0
            current_suspect=None,
            votes={},
            nominators_tried=set()
        )
    
    def _handle_final_vote_action(self, action: Action):
        """Handle final voting phase actions."""
        action_type = action.data.get("type", "")
        
        if action_type == "nominate":
            self._handle_final_nominate(action)
        elif action_type == "vote":
            self._handle_final_vote(action)
        else:
            if self.logger:
                self.logger.log(
                    EventType.ERROR,
                    {"player_id": action.player_id, "error": f"Unknown action type: {action_type}"}
                )
    
    def _handle_final_nominate(self, action: Action):
        """Handle nominating a suspect in final voting."""
        suspect = action.data.get("suspect")
        
        # Check if suspect is provided
        if suspect is None:
            if self.logger:
                self.logger.log(
                    EventType.ERROR,
                    {"player_id": action.player_id, "error": "No suspect provided in nomination"}
                )
            return
        
        # Validate
        valid, error = validate_accusation_target(
            action.player_id,
            suspect,
            self.game_config.n_players
        )
        
        if not valid:
            if self.logger:
                self.logger.log(
                    EventType.ERROR,
                    {"player_id": action.player_id, "error": error}
                )
            return
        
        # Log nomination
        if self.logger:
            self.logger.log(
                EventType.PLAYER_ACTION,
                {
                    "nominator": action.player_id,
                    "suspect": suspect,
                    "action": "final_nominate",
                },
                is_private=False
            )
        
        self.state.final_vote.current_suspect = suspect
        self.state.final_vote.total_voters = self.game_config.n_players - 1  # Everyone except suspect
        self.state.final_vote.votes = {}
    
    def _handle_final_vote(self, action: Action):
        """Handle voting in final voting phase."""
        vote = action.data.get("vote")
        
        if vote is None:
            if self.logger:
                self.logger.log(
                    EventType.ERROR,
                    {"player_id": action.player_id, "error": "No vote provided"}
                )
            return
        
        # Check for duplicate vote
        if action.player_id in self.state.final_vote.votes:
            # Already voted - ignore duplicate
            return
        
        # Record vote
        self.state.final_vote.votes[action.player_id] = bool(vote)
        
        # Log vote
        if self.logger:
            self.logger.log(
                EventType.VOTE_CAST,
                {
                    "voter": action.player_id,
                    "vote": vote,
                    "suspect": self.state.final_vote.current_suspect,
                },
                is_private=True
            )
        
        # Check if voting complete (all players except suspect have voted)
        expected_votes = self.game_config.n_players - 1  # Everyone except suspect
        if len(self.state.final_vote.votes) >= expected_votes:
            self._resolve_final_vote()
    
    def _resolve_final_vote(self):
        """Resolve final vote."""
        # Check if vote passed (majority, not unanimous)
        yes_votes = sum(1 for v in self.state.final_vote.votes.values() if v)
        total_votes = len(self.state.final_vote.votes)
        passed = yes_votes > (total_votes / 2)  # Strict majority
        suspect = self.state.final_vote.current_suspect
        
        # Log result
        if self.logger:
            self.logger.log(
                EventType.ELECTION_RESULT,
                {
                    "passed": passed,
                    "suspect": suspect,
                    "votes": self.state.final_vote.votes,
                }
            )
        
        if passed:
            # Vote passed
            is_spy = (suspect == self.state.spy_index)
            
            if is_spy:
                # Spy was correctly identified - give spy chance to guess location!
                if self.logger:
                    self.logger.log(
                        EventType.INFO,
                        {
                            "message": "Spy correctly identified! Spy gets one guess to steal the win.",
                            "spy": suspect,
                        }
                    )
                # Transition to spy guess phase
                self.state.phase = Phase.SPY_GUESS
                self.state.spy_guess_allowed = True
            else:
                # Wrong player accused - spy wins
                self.state.winner = "spy"
                self.state.win_reason = "Wrong player was accused in final vote"
                
                self.state.scores = calculate_scores(
                    self.game_config.n_players,
                    self.state.spy_index,
                    self.state.winner,
                    False
                )
                self.state.phase = Phase.GAME_END
                
                if self.logger:
                    self.logger.log(
                        EventType.GAME_END,
                        {
                            "winner": self.state.winner,
                            "reason": self.state.win_reason,
                            "scores": self.state.scores,
                        }
                    )
        else:
            # Vote failed - try next nominator or end game
            self.state.final_vote.nominators_tried.add(self.state.final_vote.current_nominator)
            
            if len(self.state.final_vote.nominators_tried) >= self.game_config.n_players:
                # All players tried - spy wins
                self.state.winner = "spy"
                self.state.win_reason = "No consensus reached in final voting"
                self.state.scores = calculate_scores(
                    self.game_config.n_players,
                    self.state.spy_index,
                    self.state.winner,
                    False
                )
                self.state.phase = Phase.GAME_END
                
                if self.logger:
                    self.logger.log(
                        EventType.GAME_END,
                        {
                            "winner": self.state.winner,
                            "reason": self.state.win_reason,
                            "scores": self.state.scores,
                        }
                    )
            else:
                # Reset for next nominator
                self.state.final_vote.current_suspect = None
                self.state.final_vote.votes = {}
                # Move to next nominator (will be handled in next iteration)
                remaining = [i for i in range(self.game_config.n_players) if i not in self.state.final_vote.nominators_tried]
                if remaining:
                    self.state.final_vote.current_nominator = remaining[0]
    
    def _format_qa_history(self) -> str:
        """Format complete Q&A history for display.
        
        Returns:
            Formatted string of all Q&A exchanges
        """
        if not self.state.qa_history:
            return "   (No Q&A yet)"
        
        formatted = []
        for qa in self.state.qa_history:
            formatted.append(
                f"   Turn {qa.turn}: Player {qa.asker} asked Player {qa.answerer}:\n"
                f"      Q: \"{qa.question}\"\n"
                f"      A: \"{qa.answer}\""
            )
        return "\n".join(formatted)
    
    def _format_game_state(self) -> str:
        """Format current game state summary.
        
        Returns:
            Formatted string of key game info
        """
        return f"""📊 GAME STATE:
   • Turn: {self.state.turn}/{self.game_config.max_turns}
   • Phase: {self.state.phase.value}
   • Current Asker: Player {self.state.current_asker if self.state.current_asker is not None else "N/A"}"""
    
    def _get_observations(self) -> Dict[int, Observation]:
        """Generate observations for all players."""
        observations = {}
        
        for pid in range(self.game_config.n_players):
            # Base observation data
            obs_data = {
                "turn": self.state.turn,
                "max_turns": self.game_config.max_turns,
                "is_spy": self.state.is_spy(pid),
                "location": self.state.get_player_location(pid),
                "role": self.state.get_player_role(pid),
                "qa_history": [
                    {
                        "turn": qa.turn,
                        "asker": qa.asker,
                        "answerer": qa.answerer,
                        "question": qa.question,
                        "answer": qa.answer,
                    }
                    for qa in self.state.qa_history
                ],
                "can_stop_clock": self.state.can_stop_clock(pid),
                
                # Formatted full context
                "formatted_qa_history": self._format_qa_history(),
                "formatted_game_state": self._format_game_state(),
            }
            
            # Add phase-specific info
            instruction = ""
            obs_type = "observe"
            
            if self.state.phase == Phase.QANDA:
                # Exactly one actor at a time in Q&A:
                # If someone is awaited to answer, only they act.
                if self.state.awaiting_answer_from is not None:
                    if self.state.awaiting_answer_from == pid:
                        role_hint = f" (Your Role: {self.state.get_player_role(pid)})" if not self.state.is_spy(pid) else " (You're the SPY!)"
                        location_hint = f" Location: {self.state.get_player_location(pid)}" if not self.state.is_spy(pid) else " You DON'T know the location!"
                        
                        instruction = f"""=== Q&A PHASE - YOUR TURN TO ANSWER ===

{self._format_game_state()}

YOUR ROLE:{role_hint}
{location_hint}

📜 COMPLETE Q&A HISTORY:
{self._format_qa_history()}

⚡ YOUR ACTION:
Answer the question. {"Be vague to blend in!" if self.state.is_spy(pid) else "Be specific to prove you know the location!"}

⚠️  CRITICAL SPYFALL RULES:
• NEVER directly name or state the location in your answer
• {"Avoid being too specific or you'll reveal yourself as the spy" if self.state.is_spy(pid) else "Include 1-2 concrete details that prove you know the location"}
• {"Be plausible but generic enough to blend in" if self.state.is_spy(pid) else "Reference role-specific activities, objects, or sensory details"}

Respond with JSON:
{{"type": "answer", "answer": "<your answer>"}}"""
                        
                        if self.state.can_stop_clock(pid):
                            instruction += "\n\nOR accuse: {\"type\": \"accuse\", \"suspect\": <player_id>}"
                        if self.state.is_spy(pid) and self.state.can_spy_guess():
                            instruction += f"\n\nOR guess location: {{\"type\": \"spy_guess\", \"guess\": \"<location>\"}} from {self.game_config.locations}"
                        obs_type = "act"
                    else:
                        instruction = "Waiting for the answer."
                        obs_type = "observe"
                elif self.state.current_asker == pid:
                    targets = [i for i in range(self.game_config.n_players) if i != pid and i != self.state.cannot_ask_back]
                    role_hint = f" (Your Role: {self.state.get_player_role(pid)})" if not self.state.is_spy(pid) else " (You're the SPY!)"
                    location_hint = f" Location: {self.state.get_player_location(pid)}" if not self.state.is_spy(pid) else " You DON'T know the location!"
                    
                    instruction = f"""=== Q&A PHASE - YOUR TURN TO ASK ===

{self._format_game_state()}

YOUR ROLE:{role_hint}
{location_hint}

📜 COMPLETE Q&A HISTORY:
{self._format_qa_history()}

⚡ YOUR ACTION:
Ask a question to another player. {"Ask vague questions to blend in!" if self.state.is_spy(pid) else "Ask questions that test if they know the location!"}

⚠️  CRITICAL SPYFALL RULE: You MUST NOT directly ask "where are we?", "what is this place?", "what location?", or similar direct location questions.
Instead, ask indirect questions about:
• Sensory details (sights, sounds, smells)
• Activities or tasks performed here
• Tools, equipment, or objects present
• Clothing or dress code
• Time-related constraints
• Physical environment or dangers

Available targets: {targets}
Cannot ask back to: {self.state.cannot_ask_back if self.state.cannot_ask_back is not None else "None"}

Respond with JSON:
{{"type": "ask", "target": <player_id>, "question": "<your question>"}}"""
                    
                    if self.state.can_stop_clock(pid):
                        instruction += "\n\nOR accuse: {\"type\": \"accuse\", \"suspect\": <player_id>}"
                    if self.state.is_spy(pid) and self.state.can_spy_guess():
                        instruction += f"\n\nOR guess location: {{\"type\": \"spy_guess\", \"guess\": \"<location>\"}} from {self.game_config.locations}"
                    obs_type = "act"
                else:
                    instruction = "Waiting for the question."
                    obs_type = "observe"
            
            elif self.state.phase == Phase.ACCUSATION_VOTE:
                if pid == self.state.accusation.suspect:
                    instruction = f"You are accused of being the spy! Waiting for vote result."
                    obs_type = "observe"
                elif pid in self.state.accusation.voters and pid not in self.state.accusation.votes:
                    instruction = (
                        f"Player {self.state.accusation.suspect} is accused of being the spy. "
                        "Vote yes/no (unanimous yes needed). "
                        "Respond with JSON: {\"type\": \"vote\", \"vote\": true/false}"
                    )
                    obs_type = "act"
                else:
                    instruction = "Waiting for votes."
                    obs_type = "observe"
                
                obs_data["accusation"] = {
                    "accuser": self.state.accusation.accuser,
                    "suspect": self.state.accusation.suspect,
                }
            
            elif self.state.phase == Phase.FINAL_VOTE:
                if self.state.final_vote.current_suspect is None:
                    if pid == self.state.final_vote.current_nominator:
                        instruction = (
                            "Final voting: Nominate a suspect. "
                            "Respond with JSON: {\"type\": \"nominate\", \"suspect\": <player_id>}"
                        )
                        obs_type = "act"
                    else:
                        instruction = "Waiting for nomination."
                        obs_type = "observe"
                else:
                    if pid == self.state.final_vote.current_suspect:
                        instruction = "You are nominated! Waiting for vote result."
                        obs_type = "observe"
                    elif pid not in self.state.final_vote.votes:
                        instruction = (
                            f"Player {self.state.final_vote.current_suspect} is nominated. "
                            "Vote yes/no (unanimous yes needed). "
                            "Respond with JSON: {\"type\": \"vote\", \"vote\": true/false}"
                        )
                        obs_type = "act"
                    else:
                        instruction = "Waiting for votes."
                        obs_type = "observe"
            
            elif self.state.phase == Phase.SPY_GUESS:
                if pid == self.state.spy_index:
                    instruction = (
                        f"Guess the location from: {self.game_config.locations}. "
                        "Respond with JSON: {\"type\": \"guess\", \"guess\": \"<location>\"}"
                    )
                    obs_type = "act"
                else:
                    instruction = "The spy is guessing the location!"
                    obs_type = "observe"
            
            elif self.state.phase == Phase.GAME_END:
                instruction = f"Game over! Winner: {self.state.winner}"
                obs_type = "observe"
            
            obs_data["instruction"] = instruction
            obs_data["type"] = obs_type  # Add type to data dict for easy access
            
            observations[pid] = Observation(
                player_id=pid,
                obs_type=obs_type,
                phase=self.state.phase.value,
                data=obs_data
            )
        
        return observations
    
    async def play_game(self):
        """Play a complete Spyfall game with configured agents.
        
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
        winner = self.state.winner or "timeout"
        win_reason = self.get_win_reason() or "Game reached maximum rounds"
        
        # Calculate player stats
        player_stats = {}
        spy_idx = self.state.spy_index
        
        for pid in range(self.game_config.n_players):
            if self.state.winner == "spy":
                score = 1.0 if pid == spy_idx else 0.0
            elif self.state.winner == "non-spies":
                score = 1.0 if pid != spy_idx else 0.0
            else:  # timeout
                score = 0.0
            
            player_stats[pid] = {
                "score": score,
                "is_spy": pid == spy_idx,
            }
        
        return GameResult(
            game_id=self.game_id,
            winner=winner,
            win_reason=win_reason,
            num_rounds=round_count,
            duration_seconds=0.0,
            player_stats=player_stats,
            metadata={
                "spy_index": spy_idx,
                "location": self.state.location,
                "qa_exchanges": len(self.state.qa_history),
                "accusations": len([e for e in self.state.qa_history if hasattr(e, 'type') and e.type == 'accusation']),
            }
        )

