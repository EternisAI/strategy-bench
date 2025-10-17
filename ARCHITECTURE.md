# Strategy Bench - Architecture

## Overview

Strategy Bench is a modular, extensible framework for benchmarking Large Language Models in strategic multi-party games. This document describes the technical architecture, design patterns, and implementation details.

## Design Principles

### 1. **Separation of Concerns**
```
Core Framework (base classes, types, utils)
    ↓
Infrastructure (LLM interface, memory, logging)
    ↓
Agents (uses infrastructure to make decisions)
    ↓
Environments (orchestrates agents and game rules)
    ↓
Tournament/Evaluation (runs multiple games, analyzes results)
```

### 2. **Consistent Interfaces**
- All games implement `BaseEnvironment`
- All agents implement `BaseAgent`
- All game states extend `BaseState`

### 3. **Public/Private Information Separation**
- States clearly separate public vs private info
- Observations are filtered per player
- Logging respects privacy settings

### 4. **Async-First Design**
- LLM calls are asynchronous
- Parallel game execution supported
- Built-in rate limiting

### 5. **Comprehensive Logging**
- Every event tracked with structured data
- Replay capability from logs
- Privacy-aware (can filter private info)

---

## Core Components

### 1. Core Framework (`sdb/core/`)

#### `types.py` - Type System
```python
# Enums for game states
class GamePhase(Enum):
    SETUP, ONGOING, ENDED

class ActionType(Enum):
    SPEAK, VOTE, NOMINATE, INVESTIGATE, etc.

# Core data structures
@dataclass
class Action:
    player_id: int
    action_type: ActionType
    target: Optional[int]
    data: Dict[str, Any]
    metadata: Dict[str, Any]  # For agent reasoning, etc.

@dataclass
class Observation:
    player_id: int
    phase: GamePhase
    data: Dict[str, Any]  # Game-specific info

@dataclass
class GameResult:
    game_id: str
    winner: Union[str, List[int]]
    win_reason: str
    num_rounds: int
    duration_seconds: float
    player_stats: Dict[int, Dict[str, Any]]
```

#### `base_state.py` - Abstract State Class
```python
class BaseState(ABC):
    """Base class for game state management."""
    
    # Core attributes
    game_id: str
    num_players: int
    current_phase: GamePhase
    alive_players: List[int]
    
    # Abstract methods (must implement)
    @abstractmethod
    def get_observation(self, player_id: int) -> Observation:
        """Get player-specific observation (filtered for private info)."""
        
    @abstractmethod
    def is_action_legal(self, action: Action) -> bool:
        """Check if action is legal in current state."""
        
    @abstractmethod
    def get_legal_actions(self, player_id: int) -> List[ActionType]:
        """Get list of legal actions for player."""
```

#### `base_agent.py` - Abstract Agent Interface
```python
class BaseAgent(ABC):
    """Base class for all agents."""
    
    def __init__(self, player_id: int, name: str):
        self.player_id = player_id
        self.name = name
        self.observation_history: List[Observation] = []
        self.action_history: List[Action] = []
    
    @abstractmethod
    def act(self, observation: Observation) -> Action:
        """Choose an action based on observation."""
        
    def notify(self, event_type: str, data: Dict[str, Any]) -> None:
        """Receive game events (for memory/belief updates)."""
        
    def reset(self) -> None:
        """Reset agent for new game."""
```

#### `base_env.py` - Abstract Environment Class
```python
class BaseEnvironment(ABC):
    """Base class for all game environments."""
    
    def __init__(self, agents: List[BaseAgent], config: Any, logger: Optional[GameLogger] = None):
        self.agents = agents
        self.config = config
        self.logger = logger
        self.state: BaseState = None
    
    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """Initialize new game."""
        
    @abstractmethod
    def step(self, action: Action) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
        """Execute one game step. Returns (observations, done, info)."""
        
    @abstractmethod
    def get_winner(self) -> Union[str, List[int]]:
        """Determine winner(s)."""
        
    def play_game(self) -> GameResult:
        """Run complete game loop."""
        self.reset()
        while not self._is_game_over():
            self._run_round()
        return self._build_game_result()
```

---

### 2. LLM Interface (`sdb/llm_interface/`)

#### `openrouter.py` - OpenRouter Client
```python
class OpenRouterClient(BaseLLMClient):
    """Client for OpenRouter API (supports multiple providers)."""
    
    def __init__(self, model="openai/gpt-4o-mini", temperature=0.7, max_tokens=4096):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Statistics tracking
        self.total_calls = 0
        self.total_tokens = 0
        self.total_cost = 0.0
    
    @retry_with_backoff(max_retries=3)
    async def chat_completion(self, messages: List[Dict]) -> LLMResponse:
        """Make async API call with automatic retry."""
        # Validate and send request
        # Track tokens and cost
        # Return structured response
```

**Key Features:**
- Async API calls with `aiohttp`
- Automatic retry with exponential backoff
- Token usage and cost tracking
- Support for all OpenRouter models
- Error handling with `LLMError`

#### `utils.py` - LLM Utilities
```python
@retry_with_backoff(max_retries=3, base_delay=0.1)
async def api_call_with_retry(func):
    """Decorator for robust API calls."""

class RateLimiter:
    """Rate limiting for API calls (calls per minute)."""
    
def estimate_tokens(text: str) -> int:
    """Estimate token count (~4 chars per token)."""
    
def calculate_cost(prompt_tokens, completion_tokens, model) -> float:
    """Calculate API call cost based on model pricing."""
```

---

### 3. Memory System (`sdb/memory/`)

#### `short_term.py` - Short-Term Memory
```python
class ShortTermMemory(BaseMemory):
    """FIFO memory with importance-based retrieval."""
    
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.entries: deque[MemoryEntry] = deque(maxlen=capacity)
    
    def add(self, content: str, importance: float, source: str):
        """Add entry (auto-evicts oldest if at capacity)."""
        
    def get_recent(self, n: int) -> List[MemoryEntry]:
        """Get n most recent entries."""
        
    def get_important(self, threshold: float) -> List[MemoryEntry]:
        """Get entries above importance threshold."""
```

#### `belief_tracker.py` - Belief Tracking
```python
@dataclass
class Belief:
    subject: int          # Player ID
    predicate: str        # e.g., "is_fascist", "is_trustworthy"
    confidence: float     # 0.0 to 1.0
    evidence: List[str]   # Supporting evidence
    updated_at: datetime

class BeliefTracker:
    """Track beliefs about other players."""
    
    def add_belief(self, subject, predicate, confidence, evidence):
        """Add or update belief."""
        
    def get_belief(self, subject, predicate) -> Optional[Belief]:
        """Get specific belief."""
        
    def get_high_confidence_beliefs(self, threshold=0.7) -> List[Belief]:
        """Get beliefs above confidence threshold."""
```

---

### 4. Agents (`sdb/agents/`)

#### `llm/openrouter_agent.py` - LLM Agent
```python
class OpenRouterAgent(BaseAgent):
    """Agent powered by LLM via OpenRouter."""
    
    def __init__(self, player_id, name, model="openai/gpt-4o-mini", 
                 temperature=0.7, memory_capacity=50):
        super().__init__(player_id, name)
        self.llm_client = OpenRouterClient(model=model, temperature=temperature)
        self.memory = ShortTermMemory(capacity=memory_capacity)
        self.beliefs = BeliefTracker()
        self.conversation_history = []
    
    async def act_async(self, observation: Observation) -> Action:
        """Generate action using LLM."""
        # 1. Build prompt from observation + memory + beliefs
        prompt = self._build_action_prompt(observation)
        
        # 2. Get LLM response
        response = await self.llm_client.chat_completion(
            messages=self.conversation_history + [{"role": "user", "content": prompt}]
        )
        
        # 3. Parse action from response
        action = self._parse_action_from_llm(response.content, observation)
        
        # 4. Store reasoning in action metadata
        action.metadata["reasoning"] = response.content
        action.metadata["agent_name"] = self.name
        
        return action
    
    def _build_action_prompt(self, observation: Observation) -> str:
        """Build context-aware prompt including:
        - Current game state
        - Recent memories
        - High-confidence beliefs
        - Phase-specific instructions
        """
        
    def notify(self, event_type: str, data: Dict):
        """Receive and process game events."""
        # Format for memory
        memory_text = self._format_event_for_memory(event_type, data)
        
        # Store in memory
        self.memory.add(
            content=memory_text,
            importance=self._get_event_importance(event_type),
            source=event_type
        )
        
        # Update beliefs
        self._update_beliefs_from_event(event_type, data)
```

**Key Features:**
- Context-aware prompts with memory and beliefs
- Automatic belief updates from events
- Stores full reasoning in action metadata
- Conversation history management
- Token and cost tracking

---

### 5. Logging System (`sdb/logging/`)

#### `game_logger.py` - Game Logger
```python
class GameLogger:
    """Comprehensive game event logging."""
    
    def __init__(self, game_id: str, output_dir: str, log_private_info=False):
        self.game_id = game_id
        self.output_dir = Path(output_dir)
        self.log_private_info = log_private_info
        self.entries: List[LogEntry] = []
    
    def log(self, event_type: EventType, data: Dict, 
            player_id: Optional[int] = None, is_private: bool = False):
        """Log an event."""
        if is_private and not self.log_private_info:
            return  # Skip private events if not logging them
            
        entry = LogEntry(
            timestamp=datetime.now(),
            event_type=event_type,
            player_id=player_id,
            data=data,
            is_private=is_private
        )
        self.entries.append(entry)
    
    def save(self):
        """Save logs to JSONL file."""
        filepath = self.output_dir / f"{self.game_id}.jsonl"
        with open(filepath, 'w') as f:
            for entry in self.entries:
                f.write(entry.to_json() + '\n')
```

**Event Types:**
- Game: `GAME_START`, `ROUND_START`, `GAME_END`
- Actions: `PLAYER_NOMINATE`, `VOTE_CAST`, `POLICY_ENACTED`
- Discussions: `DISCUSSION`, `AGENT_REASONING`
- Powers: `INVESTIGATION`, `EXECUTION`, `SPECIAL_ELECTION`
- Veto: `VETO_PROPOSED`, `VETO_RESPONSE`

---

## Game Implementation: Secret Hitler

### Architecture

```
SecretHitlerEnv (extends BaseEnvironment)
    ├── SecretHitlerState (extends BaseState)
    ├── SecretHitlerConfig (game configuration)
    ├── GameRules (win conditions, validation)
    └── PolicyDeck (deck management)
```

### Key Classes

#### `state.py` - Game State
```python
@dataclass
class SecretHitlerState(BaseState):
    """Manages all game state."""
    
    # Public state
    liberal_policies: int = 0
    fascist_policies: int = 0
    election_tracker: int = 0
    veto_unlocked: bool = False
    president_idx: int = 0
    chancellor_nominee: Optional[int] = None
    
    # Private state
    player_roles: Dict[int, Role] = field(default_factory=dict)
    player_parties: Dict[int, Party] = field(default_factory=dict)
    
    # Discussion tracking
    current_discussion: List[Dict] = field(default_factory=list)
    
    def get_observation(self, player_id: int) -> Observation:
        """Return player-specific view (filters private info)."""
        public_data = {...}  # Visible to all
        
        private_data = {}
        if self.player_parties[player_id] == Party.FASCIST:
            private_data["fascist_team"] = [...]  # Fascists know each other
            private_data["hitler_id"] = ...
        
        return Observation(player_id=player_id, data={**public_data, **private_data})
```

#### `env.py` - Game Flow
```python
class SecretHitlerEnv(BaseEnvironment):
    """Secret Hitler game implementation."""
    
    def _run_round(self):
        """Execute one round of the game."""
        # 1. Nomination phase
        self._nomination_phase()
        
        # 2. Discussion phase (PUBLIC - all players can speak)
        self._discussion_phase()
        
        # 3. Voting phase
        votes_passed = self._voting_phase()
        
        if not votes_passed:
            self._failed_election()
            return
        
        # 4. Legislative session
        self._legislative_session()
        
        # 5. Presidential power (if applicable)
        self._execute_presidential_power()
        
        # 6. Advance to next president
        self._advance_president()
    
    def _discussion_phase(self):
        """Allow agents to make PUBLIC statements."""
        self.state.current_discussion = []
        
        # Randomize speaking order
        discussion_order = self.state.alive_players.copy()
        random.shuffle(discussion_order)
        
        for player_id in discussion_order:
            # Agent receives observation with action_required="discuss_nomination"
            obs = self.state.get_observation(player_id)
            obs.data["action_required"] = "discuss_nomination"
            obs.data["discussion_is_public"] = True
            obs.data["previous_statements"] = self.state.current_discussion.copy()
            
            # Agent decides what to say (or stay silent)
            action = self.agents[player_id].act(obs)
            statement = action.data.get("statement", "")
            
            if statement:
                # Log publicly
                self.logger.log(EventType.DISCUSSION, {
                    "speaker": player_id,
                    "statement": statement,
                    "context": "nomination_discussion"
                }, is_private=False)
                
                # Notify ALL agents about this PUBLIC statement
                for agent_id in self.state.alive_players:
                    if hasattr(self.agents[agent_id], 'notify'):
                        self.agents[agent_id].notify("discussion_statement", {
                            "speaker": player_id,
                            "statement": statement,
                            "president": self.state.president_idx,
                            "chancellor_nominee": self.state.chancellor_nominee
                        })
```

### Discussion Memory Integration

**Flow:**
1. Agent makes statement during discussion phase
2. Statement logged publicly (`is_private=False`)
3. ALL agents notified via `notify("discussion_statement", ...)`
4. Each agent's memory system stores the statement
5. Agents' belief trackers update based on statement content

**Result:** Agents build shared knowledge of who said what, enabling more sophisticated social deduction strategies.

---

## Data Flow

### 1. Game Initialization
```
BaseEnvironment.reset()
    → State initialized with roles/parties
    → Agents receive initial observations
    → Logger records game start
```

### 2. Action Loop
```
Environment requests action from agent
    → Agent.act(observation)
        → Build prompt with memory + beliefs
        → Call LLM
        → Parse response to Action
        → Store reasoning in metadata
    ← Return Action
    
Environment validates and executes action
    → State.is_action_legal()
    → Update state
    → Logger records event
    → Notify all agents of public events
```

### 3. Agent Memory Update
```
Environment.notify_agents(event_type, data)
    → For each agent:
        → Agent.notify(event_type, data)
            → Format event for memory
            → Store in short-term memory
            → Update beliefs based on event
```

---

## Testing Strategy

### Unit Tests
- Core types and utilities
- Base classes
- Memory systems
- Belief tracking
- LLM interface (mocked)
- Logging system

### Integration Tests
- Full game simulation
- Multi-agent interaction
- Discussion phases
- Presidential powers
- Win condition validation

### Property Tests
- State consistency
- Action legality
- Observation filtering
- Memory eviction

---

## Performance Considerations

### LLM Calls
- **Async**: All LLM calls are async for parallel execution
- **Retry Logic**: Automatic retry with exponential backoff
- **Rate Limiting**: Built-in rate limiter (calls per minute)
- **Cost Tracking**: Token usage and cost per game

### Memory Management
- **Fixed Capacity**: Short-term memory has max capacity (FIFO eviction)
- **Efficient Retrieval**: Recent and important memories retrieved quickly
- **Belief Pruning**: Old beliefs can be pruned (configurable)

### Logging
- **Deferred Writes**: Events buffered in memory, written at end
- **JSONL Format**: One event per line for streaming
- **Privacy Filtering**: Private events skipped if `log_private_info=False`

---

## Design Patterns

### 1. Strategy Pattern
- `BaseAgent` with multiple implementations (LLM, Random, Rule-based)
- Pluggable LLM clients

### 2. Observer Pattern
- Agents receive notifications via `notify()`
- Event-driven architecture

### 3. Factory Pattern
- Environment creation from configs
- Agent instantiation

### 4. Template Method
- `BaseEnvironment.play_game()` defines flow
- Subclasses override specific methods (`_run_round`, `_voting_phase`)

### 5. Decorator Pattern
- `@retry_with_backoff` for error handling
- `@rate_limit` for API calls

---

## Extension Points

### Adding a New Game

1. **Create directory**: `sdb/environments/new_game/`
2. **Implement classes**:
   - `types.py` - Game-specific enums and dataclasses
   - `config.py` - Configuration
   - `rules.py` - Game rules and validation
   - `state.py` - Extends `BaseState`
   - `env.py` - Extends `BaseEnvironment`

3. **Implement required methods**:
   ```python
   class NewGameEnv(BaseEnvironment):
       def reset(self): ...
       def _run_round(self): ...
       def get_winner(self): ...
       def get_win_reason(self): ...
   ```

4. **Register environment**:
   ```python
   from sdb.environments.registry import registry
   registry.register("new_game", NewGameEnv, ...)
   ```

### Adding a New LLM Provider

1. **Create client**:
   ```python
   class NewProviderClient(BaseLLMClient):
       async def chat_completion(self, messages, **kwargs):
           # Implementation
   ```

2. **Add pricing** to `llm_interface/utils.py`

3. **Register** in `__init__.py`

---

## Metrics & Evaluation

### Standard Metrics
- Win rate by role/party
- Average game length
- Token usage and cost
- Action distribution

### Deception Metrics
- Lying frequency (claims vs actual roles)
- Awareness score (correct beliefs about others)
- Planning depth (mention of future strategies)

### LLM Judge Evaluation
- Strategic quality (GPT-5 rates decision quality)
- Deception effectiveness
- Communication clarity

---

## File Statistics

| Module | Files | Lines | Purpose |
|--------|-------|-------|---------|
| `core/` | 6 | ~800 | Base framework |
| `llm_interface/` | 3 | ~300 | LLM clients |
| `memory/` | 3 | ~400 | Memory systems |
| `agents/` | 2 | ~750 | Agent implementations |
| `logging/` | 2 | ~500 | Logging system |
| `environments/secret_hitler/` | 6 | ~1500 | Secret Hitler game |
| **Total** | ~25 | ~4250 | Production code |

---

## Key Implementation Details

### Secret Hitler Specifics

**Veto Power:**
- Unlocks after 5 Fascist policies enacted
- Chancellor proposes veto
- Public discussion phase
- President accepts/rejects
- If accepted, both policies discarded, election tracker increments

**Discussion Phases:**
- Before voting on nomination
- During veto proposals
- All players can make PUBLIC statements
- Statements stored in agent memories
- Enables deception and information sharing

**Presidential Powers** (by player count):
- 5-6 players: Investigate → Investigate → Special Election → Execution → Execution
- 7-8 players: Investigate → Investigate → Special Election → Execution → Execution
- 9-10 players: Investigate → Investigate → Investigate → Execution → Execution

**Win Conditions:**
- Liberals: 5 Liberal policies OR assassinate Hitler
- Fascists: 6 Fascist policies OR elect Hitler as Chancellor (after 3 Fascist policies)

---

**Last Updated**: Secret Hitler complete, framework at ~80% completion
**Next Milestone**: Complete remaining game environments
