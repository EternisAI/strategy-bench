"""LLM agent using OpenRouter API."""

import asyncio
from typing import Any, Dict, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from sdb.core.base_agent import BaseAgent
from sdb.core.types import Action, Observation, ActionType
from sdb.core.exceptions import AgentError
from sdb.llm_interface.openrouter import OpenRouterClient
from sdb.memory.short_term import ShortTermMemory
from sdb.memory.belief_tracker import BeliefTracker


class OpenRouterAgent(BaseAgent):
    """Agent that uses OpenRouter API for decision making.
    
    This agent maintains short-term memory and belief tracking,
    and uses LLM reasoning to choose actions.
    """
    
    def __init__(
        self,
        player_id: int,
        name: Optional[str] = None,
        model: str = "openai/gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        memory_capacity: int = 100,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize OpenRouter agent.
        
        Args:
            player_id: Unique identifier for this agent
            name: Human-readable name
            model: Model identifier for OpenRouter
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: System prompt for the agent
            memory_capacity: Size of short-term memory
            config: Additional configuration
        """
        super().__init__(player_id=player_id, name=name, config=config)
        
        # LLM client
        self.llm_client = OpenRouterClient(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Memory systems
        self.memory = ShortTermMemory(capacity=memory_capacity)
        self.beliefs = BeliefTracker()
        
        # Prompts
        self.system_prompt = system_prompt or self._default_system_prompt()
        
        # Track conversation history
        self.conversation_history: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]
    
    def _default_system_prompt(self) -> str:
        """Get default system prompt."""
        return f"""You are Player {self.player_id} ({self.name}) in a social deduction game.

Your goal is to achieve your win condition through strategic reasoning, observation, and communication.

When making decisions:
1. Analyze the current game state carefully
2. Consider what other players' actions reveal about their roles
3. Plan your actions strategically
4. Adapt your strategy based on new information

Respond in the following format:
[Reasoning]: <your thought process>
[Action]: <the action you will take>
"""
    
    def act(self, observation: Observation) -> Action:
        """Choose an action based on observation (synchronous wrapper).
        
        Args:
            observation: Current observation
            
        Returns:
            Action to take
        """
        # Run async version in event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.act_async(observation))
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def _call_llm_with_retry(self, messages: List[Dict[str, str]]) -> Any:
        """Call LLM with automatic retry logic.
        
        Args:
            messages: Conversation messages
            
        Returns:
            LLM response
            
        Raises:
            Exception: If all retries fail
        """
        response = await self.llm_client.chat_completion(messages=messages)
        return response
    
    async def act_async(self, observation: Observation) -> Action:
        """Choose an action based on observation using LLM (async version).
        
        Args:
            observation: Current observation
            
        Returns:
            Action to take
            
        Raises:
            AgentError: If LLM fails after all retries
        """
        # Store observation in memory with context
        event_summary = self._summarize_observation(observation)
        self.memory.add(
            content=event_summary,
            importance=0.8
        )
        
        # Update beliefs based on observation
        self._update_beliefs_from_observation(observation)
        
        # Build prompt from observation (includes memories and beliefs)
        prompt = self._build_action_prompt(observation)
        
        # Add to conversation
        self.conversation_history.append({
            "role": "user",
            "content": prompt
        })
        
        try:
            # Get LLM response with automatic retry
            response = await self._call_llm_with_retry(
                messages=self.conversation_history[-15:]  # Keep context manageable
            )
            
            # Log to terminal [[memory:7216156]]
            print(f"      🤖 {self.name} thinks: {response.content[:80]}...")
            
            # Parse action from response
            action = self._parse_action_from_llm(response.content, observation)
            
            # Store full reasoning in action metadata for logging
            action.metadata["reasoning"] = response.content
            action.metadata["agent_name"] = self.name
            
            # Add assistant response to history (truncated)
            self.conversation_history.append({
                "role": "assistant",
                "content": response.content[:500]  # Keep history manageable
            })
            
            # Update stats
            if hasattr(self.llm_client, 'total_tokens'):
                self.metadata["total_tokens"] = self.llm_client.total_tokens
            if hasattr(self.llm_client, 'total_cost'):
                self.metadata["total_cost"] = self.llm_client.total_cost
            
            return action
            
        except Exception as e:
            # All retries exhausted - raise error
            error_msg = f"LLM failed after all retries for {self.name}: {str(e)}"
            print(f"      ❌ {error_msg}")
            raise AgentError(
                error_msg,
                details={"player_id": self.player_id, "observation": str(observation.data)[:200]}
            )
    
    def _observation_to_text(self, observation: Observation) -> str:
        """Convert observation to text prompt.
        
        Args:
            observation: Observation to convert
            
        Returns:
            Text representation
        """
        text_parts = [
            f"Phase: {observation.phase.name}",
            f"Your ID: {observation.player_id}",
        ]
        
        # Add observation data
        for key, value in observation.data.items():
            text_parts.append(f"{key}: {value}")
        
        # Add recent memories
        recent_memories = self.memory.get_recent(n=5)
        if recent_memories:
            text_parts.append("\nRecent events:")
            for mem in recent_memories:
                text_parts.append(f"  - {mem.content}")
        
        # Add high-confidence beliefs
        high_conf_beliefs = self.beliefs.get_high_confidence_beliefs(threshold=0.7)
        if high_conf_beliefs:
            text_parts.append("\nYour beliefs:")
            for belief in high_conf_beliefs:
                text_parts.append(
                    f"  - Player {belief.subject} {belief.predicate} "
                    f"(confidence: {belief.confidence:.2f})"
                )
        
        return "\n".join(text_parts)
    
    def _summarize_observation(self, observation: Observation) -> str:
        """Create human-readable summary of observation for memory.
        
        Args:
            observation: Current observation
            
        Returns:
            Summary string
        """
        phase = observation.phase.name if hasattr(observation, 'phase') else "UNKNOWN"
        action = observation.data.get("action_required", "observe")
        
        if action == "nominate_chancellor":
            return f"[{phase}] I am President, must nominate a Chancellor"
        elif action == "vote":
            pres = observation.data.get('president', '?')
            nom = observation.data.get('nominee', '?')
            return f"[{phase}] President {pres} nominated Player {nom} for Chancellor"
        else:
            return f"[{phase}] Observing game state"
    
    def _update_beliefs_from_observation(self, observation: Observation) -> None:
        """Update beliefs about other players based on observations.
        
        Args:
            observation: Current observation
        """
        # Update beliefs based on known fascists (from role info)
        if 'fascist_team' in observation.data:
            for player_id in observation.data['fascist_team']:
                if player_id != self.player_id:
                    self.beliefs.add_belief(
                        subject=player_id,
                        predicate="is_fascist",
                        confidence=1.0,
                        evidence=["Revealed at game start"]
                    )
        
        if 'hitler_id' in observation.data:
            hitler_id = observation.data['hitler_id']
            if hitler_id != self.player_id:
                self.beliefs.add_belief(
                    subject=hitler_id,
                    predicate="is_hitler",
                    confidence=1.0,
                    evidence=["Revealed at game start"]
                )
        
        # Track nominations
        if 'nominee' in observation.data and 'president' in observation.data:
            pres = observation.data['president']
            nom = observation.data['nominee']
            # Presidents who nominate the same people repeatedly might trust them
            existing = self.beliefs.get_beliefs_about(pres)
            if existing:
                # Increment trust
                self.beliefs.add_belief(
                    subject=pres,
                    predicate=f"trusts_player_{nom}",
                    confidence=0.6,
                    evidence=[f"Nominated player {nom}"]
                )
    
    def _build_action_prompt(self, observation: Observation) -> str:
        """Build detailed prompt for action decision with memory and beliefs.
        
        Args:
            observation: Current observation
            
        Returns:
            Formatted prompt string
        """
        action_type = observation.data.get("action_required", "observe")
        phase = observation.phase.name if hasattr(observation, 'phase') else "UNKNOWN"
        
        prompt_parts = [
            f"=== SECRET HITLER - {phase} ===",
            f"\nYour Role: {observation.data.get('your_role', 'UNKNOWN')}",
            f"Your Party: {observation.data.get('your_party', 'UNKNOWN')}",
        ]
        
        # Add known team info
        if 'fascist_team' in observation.data:
            team = observation.data['fascist_team']
            prompt_parts.append(f"Your Fascist Team: {team}")
        if 'hitler_id' in observation.data:
            prompt_parts.append(f"Hitler is Player {observation.data['hitler_id']}")
        
        # Add game state
        if 'liberal_policies' in observation.data:
            prompt_parts.append(f"\nBoard: 🔵 {observation.data['liberal_policies']}/5 Liberal, 🔴 {observation.data['fascist_policies']}/6 Fascist")
        
        # Add recent memories
        recent_memories = self.memory.get_recent(n=5)
        if recent_memories:
            prompt_parts.append("\n📝 Recent Events:")
            for mem in recent_memories[-3:]:  # Last 3 events
                prompt_parts.append(f"   - {mem.content}")
        
        # Add beliefs about other players
        high_conf_beliefs = self.beliefs.get_high_confidence_beliefs(threshold=0.7)
        if high_conf_beliefs:
            prompt_parts.append("\n🤔 Your Beliefs:")
            for belief in high_conf_beliefs[:5]:  # Top 5 beliefs
                prompt_parts.append(f"   - Player {belief.subject} {belief.predicate} ({belief.confidence:.0%} sure)")
        
        # Add phase-specific information
        if action_type == "nominate_chancellor":
            legal = observation.data.get("legal_candidates", [])
            prompt_parts.append(f"\n📋 You are PRESIDENT. Nominate a Chancellor from: {legal}")
            prompt_parts.append("\n💡 Think about:")
            prompt_parts.append("   - Who can you trust?")
            prompt_parts.append("   - Who might be Fascist/Hitler?")
            prompt_parts.append("\nRespond with ONLY the player number you nominate (e.g., '2')")
            
        elif action_type == "vote":
            president = observation.data.get('president', '?')
            nominee = observation.data.get('nominee', '?')
            prompt_parts.append(f"\n🗳️  VOTE: President {president} nominated Player {nominee} for Chancellor")
            prompt_parts.append("\n💡 Consider:")
            prompt_parts.append("   - Do you trust this government?")
            prompt_parts.append("   - What policies might they enact?")
            prompt_parts.append("\nRespond with ONLY 'Ja' (yes) or 'Nein' (no)")
            
        elif action_type == "discard_policy":
            policies = observation.data.get('policies', [])
            prompt_parts.append(f"\n📜 You drew policies: {policies}")
            prompt_parts.append("\n💡 As President, discard ONE policy:")
            prompt_parts.append("\nRespond with ONLY '0', '1', or '2' for the policy index to discard")
            
        elif action_type == "enact_policy":
            policies = observation.data.get('policies', [])
            prompt_parts.append(f"\n📜 Chancellor choice: {policies}")
            prompt_parts.append("\n💡 Which policy do you enact?")
            prompt_parts.append("\nRespond with ONLY '0' or '1' for the policy index to enact")
        
        elif action_type == "discuss_nomination":
            president = observation.data.get('president', '?')
            nominee = observation.data.get('chancellor_nominee', '?')
            previous = observation.data.get('previous_statements', [])
            
            prompt_parts.append(f"\n💬 PUBLIC DISCUSSION PHASE")
            prompt_parts.append(f"President {president} has nominated Player {nominee} for Chancellor.")
            prompt_parts.append(f"\n⚠️  IMPORTANT: This is PUBLIC - ALL PLAYERS will see your statement!")
            
            if previous:
                prompt_parts.append(f"\nPrevious statements in this discussion:")
                for stmt in previous:
                    prompt_parts.append(f"   Player {stmt['speaker']}: \"{stmt['statement']}\"")
            
            prompt_parts.append("\n💡 You can:")
            prompt_parts.append("   - Express your opinion about the nomination")
            prompt_parts.append("   - Share information (or misinformation)")
            prompt_parts.append("   - Stay silent (respond with empty string)")
            prompt_parts.append("\n🎭 Remember: Use deception strategically if you're Fascist/Hitler")
            prompt_parts.append("\nRespond with your PUBLIC statement, or empty string to stay silent")
        
        elif action_type == "discuss_veto":
            president = observation.data.get('president', '?')
            chancellor = observation.data.get('chancellor', '?')
            previous = observation.data.get('previous_statements', [])
            
            prompt_parts.append(f"\n🚨 VETO DISCUSSION PHASE (PUBLIC)")
            prompt_parts.append(f"Chancellor {chancellor} has proposed a VETO!")
            prompt_parts.append(f"President {president} must decide whether to accept.")
            prompt_parts.append(f"\n⚠️  IMPORTANT: This is PUBLIC - ALL PLAYERS will see your statement!")
            
            if previous:
                prompt_parts.append(f"\nPrevious statements in this veto discussion:")
                for stmt in previous:
                    prompt_parts.append(f"   Player {stmt['speaker']}: \"{stmt['statement']}\"")
            
            prompt_parts.append("\n💡 You can:")
            prompt_parts.append("   - Argue for/against accepting the veto")
            prompt_parts.append("   - Share your reasoning (or mislead)")
            prompt_parts.append("   - Stay silent (respond with empty string)")
            prompt_parts.append("\n🎭 Remember: This is a strategic moment - use it wisely!")
            prompt_parts.append("\nRespond with your PUBLIC statement, or empty string to stay silent")
        
        elif action_type == "veto_response":
            chancellor = observation.data.get('chancellor', '?')
            prompt_parts.append(f"\n🚫 VETO RESPONSE")
            prompt_parts.append(f"Chancellor {chancellor} has proposed a veto.")
            prompt_parts.append("\n💡 As President, do you accept the veto?")
            prompt_parts.append("   - Accept: Both policies discarded, election tracker increases")
            prompt_parts.append("   - Reject: Chancellor must enact one of the two policies")
            prompt_parts.append("\nRespond with 'accept' or 'reject'")
        
        else:
            prompt_parts.append(f"\n⏸️  Observing the game...")
        
        return "\n".join(prompt_parts)
    
    def _parse_action_from_llm(self, llm_response: str, observation: Observation) -> Action:
        """Parse action from LLM response for Secret Hitler.
        
        Args:
            llm_response: Response from LLM
            observation: Current observation
            
        Returns:
            Parsed action
        """
        action_type = observation.data.get("action_required", "speak")
        response = llm_response.strip().lower()
        
        # Extract just the answer if there's reasoning
        if '\n' in response:
            # Take last line as the answer
            response = response.split('\n')[-1].strip()
        
        if action_type == "discuss_nomination" or action_type == "discuss_veto":
            # Extract statement from response
            statement = llm_response.strip()
            
            # Try to extract just the action/statement part (skip reasoning if present)
            # Look for common markers like [Action]:, [Statement]:, or after reasoning
            statement_lower = statement.lower()
            
            if "[action]:" in statement_lower:
                # Split case-insensitively
                idx = statement_lower.find("[action]:")
                statement = statement[idx + len("[action]:"):].strip()
            elif "[statement]:" in statement_lower:
                idx = statement_lower.find("[statement]:")
                statement = statement[idx + len("[statement]:"):].strip()
            elif "[reasoning]:" in statement_lower and "\n\n" in statement:
                # If there's reasoning followed by blank line, take everything after
                parts = statement.split("\n\n", 1)
                if len(parts) > 1:
                    statement = parts[1].strip()
            
            # Remove quotes if present
            statement = statement.strip('"\'')
            
            return Action(
                player_id=self.player_id,
                action_type=ActionType.SPEAK,
                data={"statement": statement}
            )
        
        elif action_type == "veto_response":
            # Check for accept/reject
            if "accept" in response:
                return Action(
                    player_id=self.player_id,
                    action_type=ActionType.VOTE,
                    data={"accept_veto": True}
                )
            else:
                return Action(
                    player_id=self.player_id,
                    action_type=ActionType.VOTE,
                    data={"accept_veto": False}
                )
        
        elif action_type == "nominate_chancellor":
            # Extract number from response
            legal = observation.data.get("legal_candidates", [])
            for word in response.split():
                if word.isdigit():
                    nominee = int(word)
                    if nominee in legal:
                        return Action(
                            player_id=self.player_id,
                            action_type=ActionType.NOMINATE,
                            target=nominee,
                            data={}
                        )
            # Fallback: pick first legal
            return Action(
                player_id=self.player_id,
                action_type=ActionType.NOMINATE,
                target=legal[0] if legal else 0,
                data={}
            )
            
        elif action_type == "vote":
            # Parse Ja/Nein
            vote = "ja" in response or "yes" in response
            return Action(
                player_id=self.player_id,
                action_type=ActionType.VOTE,
                data={"vote": vote}
            )
            
        elif action_type in ["discard_policy", "enact_policy"]:
            # Extract index
            for char in response:
                if char.isdigit():
                    idx = int(char)
                    if idx in [0, 1, 2]:
                        return Action(
                            player_id=self.player_id,
                            action_type=ActionType.POLICY_ACTION,
                            data={"index": idx}
                        )
            # Fallback
            return Action(
                player_id=self.player_id,
                action_type=ActionType.POLICY_ACTION,
                data={"index": 0}
            )
        
        # Default
        return Action(
            player_id=self.player_id,
            action_type=ActionType.SPEAK,
            data={"message": "..."}
        )
    
    
    def notify(self, event_type: str, data: Dict[str, Any]) -> None:
        """Process game events and update beliefs.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        # Create readable memory entry
        memory_text = self._format_event_for_memory(event_type, data)
        
        # Store event in memory
        self.memory.add(
            content=memory_text,
            importance=self._get_event_importance(event_type),
            source=event_type
        )
        
        # Update beliefs based on events
        self._update_beliefs_from_event(event_type, data)
    
    def _format_event_for_memory(self, event_type: str, data: Dict[str, Any]) -> str:
        """Format event for memory storage.
        
        Args:
            event_type: Type of event
            data: Event data
            
        Returns:
            Formatted string
        """
        if event_type == "game_start":
            role = data.get("role", "UNKNOWN")
            return f"Game started. I am {role}."
        elif event_type == "player_eliminated":
            player_id = data.get("player_id", "?")
            was_hitler = data.get("was_hitler", False)
            if was_hitler:
                return f"Player {player_id} was executed and revealed as HITLER!"
            return f"Player {player_id} was eliminated."
        elif event_type == "policy_enacted":
            policy = data.get("policy", "?")
            return f"{policy} policy enacted."
        elif event_type == "investigation_result":
            target = data.get("target", "?")
            party = data.get("party", "?")
            return f"Investigated Player {target}: they are {party}."
        elif event_type == "player_executed":
            player_id = data.get("player_id", "?")
            return f"Player {player_id} was executed."
        elif event_type == "discussion_statement":
            speaker = data.get("speaker", "?")
            statement = data.get("statement", "")
            context = data.get("context", "discussion")
            
            if context == "nomination_discussion":
                pres = data.get("president", "?")
                nominee = data.get("chancellor_nominee", "?")
                return f"[DISCUSSION] Player {speaker} said: \"{statement}\" (about President {pres} nominating Player {nominee})"
            elif context == "veto_discussion":
                pres = data.get("president", "?")
                chanc = data.get("chancellor", "?")
                return f"[VETO DISCUSSION] Player {speaker} said: \"{statement}\" (Chancellor {chanc} proposed veto to President {pres})"
            else:
                return f"[DISCUSSION] Player {speaker} said: \"{statement}\""
        else:
            return f"{event_type}: {str(data)[:100]}"
    
    def _get_event_importance(self, event_type: str) -> float:
        """Determine importance of event for memory.
        
        Args:
            event_type: Type of event
            
        Returns:
            Importance score (0-1)
        """
        importance_map = {
            "game_start": 1.0,
            "player_eliminated": 0.9,
            "investigation_result": 0.9,
            "policy_enacted": 0.7,
            "player_executed": 0.9,
            "discussion_statement": 0.8,  # PUBLIC discussions are important
            "election_result": 0.6,
        }
        return importance_map.get(event_type, 0.5)
    
    def _update_beliefs_from_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Update beliefs based on game events.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        if event_type == "player_eliminated":
            eliminated_id = data.get("player_id")
            if eliminated_id is not None:
                self.beliefs.add_belief(
                    subject=eliminated_id,
                    predicate="is_eliminated",
                    confidence=1.0,
                    evidence=["Player was eliminated"]
                )
                
                # If Hitler was eliminated, update belief
                if data.get("was_hitler", False):
                    self.beliefs.add_belief(
                        subject=eliminated_id,
                        predicate="was_hitler",
                        confidence=1.0,
                        evidence=["Revealed when executed"]
                    )
        
        elif event_type == "investigation_result":
            # This is private info only the investigator gets
            target = data.get("target")
            party = data.get("party")
            if target is not None and party:
                predicate = "is_fascist" if party == "FASCIST" else "is_liberal"
                self.beliefs.add_belief(
                    subject=target,
                    predicate=predicate,
                    confidence=1.0,
                    evidence=["Investigated by President"]
                )
        
        elif event_type == "policy_enacted":
            # Track who was in government when policies were enacted
            pres = data.get("president")
            chanc = data.get("chancellor")
            policy = data.get("policy")
            
            if pres is not None and chanc is not None and policy:
                # If Fascist policy, slightly increase suspicion
                if policy == "FASCIST":
                    for player_id in [pres, chanc]:
                        if player_id != self.player_id:
                            existing = self.beliefs.get_belief(player_id, "might_be_fascist")
                            current_conf = existing.confidence if existing else 0.3
                            self.beliefs.add_belief(
                                subject=player_id,
                                predicate="might_be_fascist",
                                confidence=min(current_conf + 0.15, 0.9),
                                evidence=[f"Enacted {policy} policy"]
                            )
        
        elif event_type == "discussion_statement":
            # Track who speaks and what they say during public discussions
            speaker = data.get("speaker")
            statement = data.get("statement", "").lower()
            
            if speaker is not None and speaker != self.player_id:
                # Basic belief updates based on discussion content
                # This is intentionally simple - more sophisticated analysis could be added
                
                # Track that this player participated in discussion
                self.beliefs.add_belief(
                    subject=speaker,
                    predicate="participates_in_discussion",
                    confidence=0.9,
                    evidence=["Made public statement"]
                )
                
                # If statement contains keywords, update beliefs accordingly
                # Note: In a real game, agents should analyze statements more carefully
                if any(word in statement for word in ["trust", "support", "agree"]):
                    self.beliefs.add_belief(
                        subject=speaker,
                        predicate="expresses_trust",
                        confidence=0.6,
                        evidence=[f"Said: {statement[:50]}"]
                    )
                elif any(word in statement for word in ["suspicious", "don't trust", "disagree", "concerned"]):
                    self.beliefs.add_belief(
                        subject=speaker,
                        predicate="expresses_suspicion",
                        confidence=0.6,
                        evidence=[f"Said: {statement[:50]}"]
                    )
    
    def reset(self) -> None:
        """Reset agent for new game."""
        super().reset()
        self.memory.clear()
        self.beliefs.clear()
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.llm_client.reset_stats()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        stats = super().get_stats()
        stats.update({
            "llm_stats": self.llm_client.get_stats(),
            "memory_size": self.memory.size(),
            "beliefs_count": len(self.beliefs.beliefs),
        })
        return stats

