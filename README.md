# Strategy Bench

**A benchmark framework for evaluating Large Language Models in multi-agent social deduction games**

## What is Strategy Bench?

Strategy Bench is designed to test the exact capabilities personal AI will need when representing us in messy, multi-party settings and making decisions based on incomplete information. 

The benchmark uses strategic games where agents take actions, communicate with each other, and make decisions based on their world model. We focus on games that combine asymmetric/incomplete information, multi-party interactions, and possible deceptive behavior that agents must account for before making decisions.

This provides a crisp, high-signal way to evaluate how well AI agents can handle the complex social and strategic reasoning required in real-world scenarios when they represent human interests in the near future.

## Supported Games

- **Secret Hitler** - Political strategy with hidden roles, policies, and presidential powers
- **Among Us** - Spatial deduction with tasks, emergency meetings, and impostor kills
- **Avalon** - Quest-based team building with hidden roles and assassination
- **Spyfall** - Question-and-answer based location deduction
- **Werewolf** - Classic social deduction with night/day phases and special roles
- **Sheriff of Nottingham** - Bluffing and negotiation with goods inspection

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/social-deduction-bench.git
cd social-deduction-bench

# Install package
pip install -e .
```

### 2. Set up API Keys

```bash
# Option 1: Environment variable
export OPENROUTER_API_KEY="your_key_here"

# Option 2: Create .env file
echo "OPENROUTER_API_KEY=your_key_here" > .env
```

### 3. Run Your First Game

#### Using the CLI (Recommended)

```bash
# List all available games
python scripts/sdb_cli.py list

# Play any game with default settings
python scripts/sdb_cli.py play secret_hitler
python scripts/sdb_cli.py play among_us
python scripts/sdb_cli.py play avalon

# Customize game settings
python scripts/sdb_cli.py play secret_hitler --num-players 7 --model anthropic/claude-3.5-sonnet
python scripts/sdb_cli.py play werewolf --num-players 8 --temperature 0.9

# Use random agents for testing
python scripts/sdb_cli.py play spyfall --agent-type random --num-players 6
```

#### Using Example Scripts

```bash
# Run Secret Hitler with 7 players
python examples/run_secret_hitler.py

# Run Among Us with 6 players
python examples/run_among_us.py

# Run Avalon with 5 players
python examples/run_avalon.py

# Run Spyfall
python examples/run_spyfall.py

# Run Werewolf
python examples/run_werewolf.py

# Run Sheriff of Nottingham
python examples/run_sheriff.py
```

#### Using Python API

```python
from sdb.environments.secret_hitler import SecretHitlerEnv, SecretHitlerConfig
from sdb.agents.llm.openrouter_agent import OpenRouterAgent
from sdb.logging.game_logger import GameLogger

# Create agents
agents = [
    OpenRouterAgent(
        player_id=i,
        model="openai/gpt-4o-mini",
        temperature=0.7,
        memory_capacity=50
    )
    for i in range(5)
]

# Configure game
config = SecretHitlerConfig(n_players=5, log_private_info=True)
logger = GameLogger(output_dir="experiments/my_games")

# Run game
env = SecretHitlerEnv(agents=agents, config=config, logger=logger)
result = env.run()

print(f"Winner: {result.winner}")
print(f"Reason: {result.win_reason}")
print(f"Rounds: {result.num_rounds}")
```

## Configuration

All games can be configured via YAML files in the `configs/` directory:

- `configs/secret_hitler.yaml` - Secret Hitler settings
- `configs/among_us.yaml` - Among Us settings
- `configs/avalon.yaml` - Avalon settings
- `configs/spyfall.yaml` - Spyfall settings
- `configs/werewolf.yaml` - Werewolf settings
- `configs/sheriff.yaml` - Sheriff of Nottingham settings

Example configuration:

```yaml
# Game configuration
n_players: 7
n_impostors: 2
max_task_rounds: 50
discussion_rounds: 2

# LLM Agent Settings
agent:
  model: "anthropic/claude-3.5-sonnet"
  temperature: 0.8
  memory_capacity: 35

# Logging
logging:
  enabled: true
  output_dir: "experiments/my_game"
  log_private: false
```

## Game-Specific Features

### Secret Hitler

Political strategy game where Liberals try to enact 5 Liberal policies or assassinate Hitler, while Fascists try to enact 6 Fascist policies or elect Hitler as Chancellor.

**Features:**
- Full game mechanics (nomination, voting, legislative session, presidential powers)
- Veto power (unlocks after 5 Fascist policies)
- Discussion phases (public deliberation before votes)
- Memory integration (agents remember all discussions and events)
- Belief tracking (agents build models of other players)

### Among Us

Spatial deduction game with impostors and crewmates on a spaceship.

**Features:**
- Two-phase round resolution (deterministic, order-independent)
- Spatial map system (14 rooms with corridor connections)
- Movement and vent systems
- Task completion and progress tracking
- Emergency meetings and body reporting
- Discussion and voting mechanics
- Kill cooldowns and proper ejection handling

### Avalon

Quest-based team building with hidden roles.

**Features:**
- Team proposal and voting system
- Quest success/fail mechanics
- Assassination phase for Evil team
- Special roles (Merlin, Assassin)
- Pre-proposal discussion
- Proper round and proposal tracking

### Spyfall

Question-and-answer based location deduction.

**Features:**
- Turn-based Q&A system
- Location-based questioning
- Spy final guess mechanic
- Accusation and voting system
- Time limits and turn tracking

### Werewolf

Classic social deduction with night/day phases.

**Features:**
- Night phase (Werewolf kills, Doctor saves, Seer investigates)
- Day phase (Discussion and voting)
- Majority voting system
- Special role powers
- Proper phase transitions

### Sheriff of Nottingham

Bluffing and negotiation game with goods inspection.

**Features:**
- Market phase (card drawing)
- Loading phase (bag preparation)
- Declaration phase
- Negotiation phase (multi-round)
- Inspection phase with penalties
- Royal goods and contraband

## Supported Models

Strategy Bench uses **OpenRouter** to access multiple LLM providers:

### Recommended Models

```bash
# OpenAI
openai/gpt-4o              # Best performance
openai/gpt-4o-mini         # Fast & cheap (default)
openai/gpt-4-turbo         # Good balance

# Anthropic
anthropic/claude-opus-4.1      # High reasoning
anthropic/claude-3.5-sonnet    # Balanced (recommended)
anthropic/claude-haiku         # Fast

# Google
google/gemini-2.5-pro      # Best Gemini
google/gemini-1.5-flash    # Fast Gemini

# Meta
meta/llama-3.1-405b        # Open source
```

### Cost Optimization

```python
# Use cheaper models for bulk experiments
agent = OpenRouterAgent(
    player_id=0,
    model="openai/gpt-4o-mini",  # ~$0.15 per 1M tokens
    temperature=0.7,
    max_tokens=1024  # Limit response length
)
```

## Project Structure

```
social-deduction-bench/
├── sdb/                        # Main package
│   ├── core/                   # Base classes & interfaces
│   │   ├── base_env.py        # BaseEnvironment class
│   │   └── types.py           # Core types (Action, Observation, etc.)
│   ├── agents/                 # Agent implementations
│   │   └── llm/               # LLM agents (OpenRouter)
│   │       └── openrouter_agent.py
│   ├── environments/           # Game implementations
│   │   ├── secret_hitler/     # Secret Hitler game
│   │   ├── among_us/          # Among Us game
│   │   ├── avalon/            # Avalon game
│   │   ├── spyfall/           # Spyfall game
│   │   ├── werewolf/          # Werewolf game
│   │   └── sheriff/           # Sheriff of Nottingham
│   ├── memory/                # Memory & belief tracking
│   ├── logging/               # Logging system
│   └── llm_interface/         # LLM API clients
├── examples/                   # Example scripts
│   ├── run_secret_hitler.py
│   ├── run_among_us.py
│   ├── run_avalon.py
│   ├── run_spyfall.py
│   ├── run_werewolf.py
│   └── run_sheriff.py
├── configs/                    # YAML configuration files
├── experiments/                # Experiment outputs
└── tests/                      # Unit tests
```

## Analyzing Results

### View Game Logs

All games generate JSONL logs with detailed event information:

```python
import json

# Load game log
with open("experiments/my_game/game_xyz.jsonl") as f:
    events = [json.loads(line) for line in f]

# Filter specific events
discussions = [e for e in events if e["event_type"] == "DISCUSSION"]
votes = [e for e in events if e["event_type"] == "VOTE_CAST"]
actions = [e for e in events if e["event_type"] == "PLAYER_ACTION"]

# Analyze game flow
for event in events:
    print(f"{event['timestamp']}: {event['event_type']}")
```

### Log Event Types

Common event types across all games:
- `GAME_START` - Game initialization
- `PHASE_CHANGE` - Phase transitions
- `PLAYER_ACTION` - Player actions
- `DISCUSSION` - Discussion statements
- `VOTE_CAST` - Vote events
- `ELECTION_RESULT` - Voting outcomes
- `PLAYER_ELIMINATED` - Player eliminations
- `GAME_END` - Game conclusion
- `ERROR` - Error events with error codes

## Key Features

### 1. Memory-Aware Agents

Agents maintain:
- **Short-term memory**: Recent events and observations
- **Belief tracking**: Probabilistic models of other players
- **Discussion memory**: All public statements from all players

### 2. Comprehensive Logging

Every game event is logged:
- Player actions and reasoning
- Public discussions
- Private information (role assignments, investigations)
- Game state transitions
- Error events with detailed error codes

### 3. Generic Agent Design

- LLM agents are completely game-agnostic
- All game-specific prompts and actions are in environment folders
- Observations include `instruction` field with formatted context
- Hybrid memory: agents maintain short-term memory, games provide full history

### 4. Robust Error Handling

- Structured error codes (e.g., `INVALID_TARGET_ID`, `KILL_ON_COOLDOWN`)
- Last error tracking for agent self-correction
- JSON parse retries with increased token limits
- Fallback to safe actions on failures

### 5. Action Validation

- Phase-based action gating
- Explicit action choices provided to agents
- Player directory with ID-to-name mapping
- Prevention of duplicate votes and invalid actions

## Advanced Features

### Two-Phase Resolution (Among Us)

Among Us implements deterministic, order-independent action resolution:
1. Snapshot all positions at round start
2. Resolve kills based on pre-move positions
3. Apply movements for survivors
4. Process body reports and meetings

This prevents order-dependent kill failures and ensures fair gameplay.

### Discussion Rounds (Multiple Games)

Games track discussion rounds properly:
- Each player speaks once per round
- Rounds advance when all alive players have spoken
- Duplicate detection prevents repetitive statements
- Phase automatically advances after configured rounds

### Voting Systems

Different voting mechanics per game:
- **Secret Hitler**: Simple majority for governments
- **Werewolf**: Majority required for elimination
- **Avalon**: Team approval requires majority, quest voting is anonymous
- **Among Us**: Plurality voting for ejections
- **Spyfall**: Majority required for accusations

## Testing Games

Each game includes an example script in the `examples/` directory. To test:

```bash
# Test Secret Hitler
python examples/run_secret_hitler.py

# Test Among Us
python examples/run_among_us.py

# Test Avalon
python examples/run_avalon.py

# Test Spyfall
python examples/run_spyfall.py

# Test Werewolf
python examples/run_werewolf.py

# Test Sheriff of Nottingham
python examples/run_sheriff.py
```

Logs will be saved to `experiments/<game_name>/` by default.

## Documentation

- **Architecture**: See `ARCHITECTURE.md` for technical details
- **API Reference**: See inline docstrings in source code
- **Game Rules**: Each game environment has its own `rules.py` file
- **Fix Logs**: See `AMONG_US_FIXES.md` and `SHERIFF_ALL_FIXES.md` for detailed implementation notes

## Contributing

Contributions are welcome! To add a new game:

1. Create game directory in `sdb/environments/your_game/`
2. Implement required files:
   - `env.py` - Main environment (inherit from `BaseEnvironment`)
   - `state.py` - Game state management
   - `config.py` - Configuration dataclass
   - `types.py` - Game-specific types
   - `rules.py` - Rule validation functions
3. Add configuration YAML in `configs/your_game.yaml`
4. Create example script in `examples/run_your_game.py`
5. Write tests
6. Submit PR

See `ARCHITECTURE.md` for detailed implementation guide.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

This framework builds upon:
- **Secret Hitler** - Original game by Mike Boxleiter, Tommy Maranges, Max Temkin
- **AmongAgents** - Among Us implementation
- **AvalonBench & Strategist** (ICLR 2025) - Avalon with search agents
- **Spyfall** - Question-based deduction mechanics
- **Werewolf Arena** (Google) - Werewolf implementation
- **Sheriff of Nottingham** - Bluffing and negotiation mechanics



---

For technical architecture details, see `ARCHITECTURE.md`
