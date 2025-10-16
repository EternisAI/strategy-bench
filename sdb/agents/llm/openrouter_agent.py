"""Generic LLM agent using OpenRouter API - works with all game environments."""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from sdb.core.base_agent import BaseAgent
from sdb.core.types import Action, Observation, ActionType
from sdb.core.exceptions import AgentError
from sdb.llm_interface.openrouter import OpenRouterClient
from sdb.memory.short_term import ShortTermMemory
from sdb.memory.belief_tracker import BeliefTracker


class OpenRouterAgent(BaseAgent):
    """Generic agent that uses OpenRouter API for decision making.
    
    This agent maintains short-term memory and belief tracking,
    and uses LLM reasoning to choose actions. It works with any game
    environment that provides clear instructions in observations.
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
5. Always respond with valid JSON as specified in the instructions

Response Format:
- Read the instruction carefully to understand the required JSON format
- Provide your reasoning first (optional)
- Then provide the JSON action exactly as specified
- Example: {{"type": "action_name", "parameter": value}}
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
        # Store observation in memory
        event_summary = self._summarize_observation(observation)
        self.memory.add(
            content=event_summary,
            importance=0.8
        )
        
        # Update beliefs based on observation
        self._update_beliefs_from_observation(observation)
        
        # Build prompt from observation
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
            
            # Log thinking to private only (not public console)
            # print(f"      🤖 {self.name} thinks: {response.content[:80]}...")
            
            # Parse action from response
            action = self._parse_action_from_llm(response.content, observation)
            
            # Check if JSON parsing failed (indicated by wait action)
            if action.data.get("type") == "wait" and "raw_response" in action.data:
                # Log to private only
                # print(f"      ⚠️  JSON parse failed, retrying with 2x max_tokens...")
                
                # Save original max_tokens
                original_max_tokens = self.llm_client.max_tokens
                
                try:
                    # Double max_tokens for retry
                    self.llm_client.max_tokens = original_max_tokens * 2
                    
                    # Add a clarification message
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": action.data.get("raw_response", "")[:200]
                    })
                    self.conversation_history.append({
                        "role": "user",
                        "content": "⚠️ Your response did not contain valid JSON. Please respond again with ONLY valid JSON in the exact format specified. No extra text, just the JSON object."
                    })
                    
                    # Retry LLM call
                    retry_response = await self._call_llm_with_retry(
                        messages=self.conversation_history[-15:]
                    )
                    
                    # Log to private only
                    # print(f"      🔄 Retry response: {retry_response.content[:80]}...")
                    
                    # Parse retry response
                    action = self._parse_action_from_llm(retry_response.content, observation)
                    
                    # Store retry reasoning
                    action.metadata["reasoning"] = retry_response.content
                    action.metadata["retry_attempt"] = True
                    
                finally:
                    # Restore original max_tokens
                    self.llm_client.max_tokens = original_max_tokens
            
            # Store full reasoning in action metadata for logging
            if "reasoning" not in action.metadata:
                action.metadata["reasoning"] = response.content
            action.metadata["agent_name"] = self.name
            
            # Add assistant response to history (truncated)
            self.conversation_history.append({
                "role": "assistant",
                "content": action.metadata.get("reasoning", response.content)[:500]  # Keep history manageable
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
    
    def _summarize_observation(self, observation: Observation) -> str:
        """Create human-readable summary of observation for memory.
        
        Args:
            observation: Current observation
            
        Returns:
            Summary string
        """
        phase = observation.phase if isinstance(observation.phase, str) else observation.phase.name
        obs_type = observation.obs_type
        
        # Try to extract key info for summary
        summary_parts = [f"[{phase}]"]
        
        if observation.data.get("instruction"):
            # Extract first sentence of instruction for summary
            instruction = observation.data["instruction"]
            first_sentence = instruction.split('.')[0] if '.' in instruction else instruction[:50]
            summary_parts.append(first_sentence)
        elif obs_type == "observe":
            summary_parts.append("Observing game state")
        else:
            summary_parts.append("Action required")
        
        return " ".join(summary_parts)
    
    def _update_beliefs_from_observation(self, observation: Observation) -> None:
        """Update beliefs about other players based on observations.
        
        This is a generic belief update that looks for common patterns
        across different games (team info, role reveals, etc.).
        
        Args:
            observation: Current observation
        """
        # Update beliefs based on team information
        team_keys = ['team', 'fascist_team', 'evil_team', 'werewolves', 'fellow_impostors']
        for key in team_keys:
            if key in observation.data:
                team_members = observation.data[key]
                if isinstance(team_members, list):
                    for member_id in team_members:
                        if member_id != self.player_id:
                            self.beliefs.add_belief(
                                subject=member_id,
                                predicate=f"is_on_my_team",
                                confidence=1.0,
                                evidence=[f"Revealed at game start via {key}"]
                            )
        
        # Update beliefs based on role reveals
        role_keys = ['hitler_id', 'spy_index', 'impostor_ids', 'merlin']
        for key in role_keys:
            if key in observation.data:
                player_id = observation.data[key]
                if isinstance(player_id, int) and player_id != self.player_id:
                    self.beliefs.add_belief(
                        subject=player_id,
                        predicate=f"has_role_{key}",
                        confidence=1.0,
                        evidence=[f"Revealed via {key}"]
                    )
    
    def _build_action_prompt(self, observation: Observation) -> str:
        """Build action prompt from observation.
        
        This is a completely generic prompt builder that relies on the
        game environment to provide clear instructions.
        
        Args:
            observation: Current observation
            
        Returns:
            Formatted prompt string
        """
        phase = observation.phase if isinstance(observation.phase, str) else observation.phase.name
        
        prompt_parts = [
            f"=== GAME STATE - {phase.upper()} ===\n",
        ]
        
        # Add role/team info if available (common across games)
        if "role" in observation.data:
            role = observation.data["role"]
            role_str = role.value if hasattr(role, 'value') else str(role)
            prompt_parts.append(f"Your Role: {role_str.upper()}")
        
        if "team" in observation.data:
            team = observation.data["team"]
            team_str = team.value if hasattr(team, 'value') else str(team)
            prompt_parts.append(f"Your Team: {team_str.upper()}")
        
        # Add recent memories
        recent_memories = self.memory.get_recent(n=15)
        if recent_memories:
            prompt_parts.append("\n📝 Recent Events:")
            for mem in recent_memories[-8:]:
                prompt_parts.append(f"   • {mem.content}")
        
        # Add high-confidence beliefs (increased to show more)
        high_conf_beliefs = self.beliefs.get_high_confidence_beliefs(threshold=0.6)
        if high_conf_beliefs and len(high_conf_beliefs) > 0:
            prompt_parts.append("\n🤔 Your Beliefs:")
            # Show up to 10 beliefs for better strategic reasoning
            for belief in high_conf_beliefs[:10]:
                prompt_parts.append(
                    f"   • Player {belief.subject}: {belief.predicate} "
                    f"({belief.confidence:.0%} confident)"
                )
        
        # Add the instruction (MOST IMPORTANT - from game environment)
        instruction = observation.data.get("instruction", "")
        if instruction:
            prompt_parts.append(f"\n⚡ INSTRUCTION:")
            prompt_parts.append(f"{instruction}")
        
        # Add relevant observation data (filtered)
        relevant_keys = [
            'alive_players', 'n_alive', 'current_debate', 'discussion',
            'qa_history', 'tasks_completed', 'quest_number', 'proposals_rejected',
            'round', 'turn', 'max_turns', 'phase_info', 'available_actions'
        ]
        
        relevant_data = {}
        for key in relevant_keys:
            if key in observation.data:
                relevant_data[key] = observation.data[key]
        
        if relevant_data:
            prompt_parts.append(f"\n📊 Game Info:")
            prompt_parts.append(json.dumps(relevant_data, indent=2, default=str))
        
        # Final reminder about JSON format
        prompt_parts.append("\n🎯 RESPOND WITH VALID JSON:")
        prompt_parts.append("   1. Think through your strategy")
        prompt_parts.append("   2. Provide the JSON action exactly as specified above")
        prompt_parts.append("   3. Example: {\"type\": \"action_name\", \"parameter\": value}")
        
        return "\n".join(prompt_parts)
    
    def _parse_action_from_llm(self, llm_response: str, observation: Observation) -> Action:
        """Parse action from LLM response.
        
        This parser expects JSON format and extracts it from the LLM response.
        The game environment is responsible for specifying the exact format needed.
        
        Args:
            llm_response: Response from LLM
            observation: Current observation
            
        Returns:
            Parsed action
        """
        # Try to extract and parse JSON from response
        if '{' in llm_response and '}' in llm_response:
            # Find the JSON object (handles nested braces)
            start = llm_response.find('{')
            if start >= 0:
                brace_count = 0
                for i in range(start, len(llm_response)):
                    if llm_response[i] == '{':
                        brace_count += 1
                    elif llm_response[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # Found matching closing brace
                            json_str = llm_response[start:i+1]
                            try:
                                action_data = json.loads(json_str)
                                
                                # Validate that we have a type field
                                if "type" not in action_data:
                                    raise ValueError("JSON missing 'type' field")
                                
                                return Action(
                                    player_id=self.player_id,
                                    action_type=ActionType.SPEAK,  # Generic type
                                    data=action_data
                                )
                            except (json.JSONDecodeError, ValueError) as e:
                                # Try to continue searching for another JSON object
                                continue
                            break
        
        # If we couldn't parse JSON, try to extract a simple response
        # This is a fallback for edge cases
        response_clean = llm_response.strip()
        
        # Remove common markdown/formatting
        response_clean = re.sub(r'```json\s*', '', response_clean)
        response_clean = re.sub(r'```\s*', '', response_clean)
        response_clean = response_clean.strip()
        
        # Try parsing again after cleanup
        if response_clean.startswith('{') and response_clean.endswith('}'):
            try:
                action_data = json.loads(response_clean)
                if "type" in action_data:
                    return Action(
                        player_id=self.player_id,
                        action_type=ActionType.SPEAK,
                        data=action_data
                    )
            except json.JSONDecodeError:
                pass
        
        # Last resort: create a wait action with the raw response
        print(f"⚠️  Could not parse valid JSON from LLM response. Using wait action.")
        return Action(
            player_id=self.player_id,
            action_type=ActionType.SPEAK,
            data={
                "type": "wait",
                "raw_response": llm_response[:200]
            }
        )
    
    def observe(self, observation: Observation) -> None:
        """Process observation without taking action.
        
        Args:
            observation: Observation to process
        """
        # Store in memory
        event_summary = self._summarize_observation(observation)
        self.memory.add(
            content=event_summary,
            importance=0.5  # Observations are less important than actions
        )
        
        # Update beliefs
        self._update_beliefs_from_observation(observation)
    
    def reset(self) -> None:
        """Reset agent state for a new game."""
        super().reset()
        self.memory.clear()
        self.beliefs.clear()
        
        # Reset conversation history but keep system prompt
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of agent's memory and beliefs.
        
        Returns:
            Dictionary with memory and belief information
        """
        return {
            "memory_size": len(self.memory.memories),
            "recent_memories": [m.content for m in self.memory.get_recent(n=5)],
            "high_confidence_beliefs": [
                {
                    "subject": b.subject,
                    "predicate": b.predicate,
                    "confidence": b.confidence
                }
                for b in self.beliefs.get_high_confidence_beliefs(threshold=0.7)
            ]
        }
