# 🎮 Social Deduction Bench (SDB)

**Unified benchmark framework for evaluating Large Language Models in social deduction games**

## 🎯 What is SDB?

Social Deduction Bench provides a production-ready platform for evaluating AI social intelligence through interactive games. Test LLMs on deception, strategic reasoning, and multi-agent coordination across multiple social deduction games with consistent interfaces.

## 🎮 Supported Games

- **Secret Hitler** ✅ - Political strategy with hidden roles, policies, and presidential powers
- **Among Us** 🔄 - Sandbox for studying agentic deception
- **Avalon** 🔄 - Multi-agent strategic reasoning with hidden roles
- **Spyfall** 🔄 - Question-and-answer based social deduction
- **Werewolf** 🔄 - Classic social deduction with night/day phases

**Legend**: ✅ Complete | 🔄 In Progress

## 🚀 Quick Start

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

#### Using the CLI

```bash
# List available games
python scripts/sdb_cli.py list

# Play Secret Hitler with 5 players
python scripts/sdb_cli.py play secret_hitler --num-players 5

# Use a specific model
python scripts/sdb_cli.py play secret_hitler --num-players 5 --model openai/gpt-4o

# Play with random agents (baseline)
python scripts/sdb_cli.py play secret_hitler --num-players 7 --agent-type random
```

#### Using Python API

```python
from sdb.environments.secret_hitler import SecretHitlerEnv, SecretHitlerConfig
from sdb.agents.llm import OpenRouterAgent
from sdb.logging import GameLogger

# Create agents
agents = [
    OpenRouterAgent(
        player_id=i, 
        name=f"Agent_{i}",
        model="openai/gpt-4o-mini",
        temperature=0.7,
        memory_capacity=50
    )
    for i in range(5)
]

# Configure game
config = SecretHitlerConfig(n_players=5, log_private_info=True)
logger = GameLogger(game_id="my_game", output_dir="experiments/my_games")

# Run game
env = SecretHitlerEnv(agents=agents, config=config, logger=logger)
result = env.play_game()

print(f"🏆 Winner: {result.winner}")
print(f"📋 Reason: {result.win_reason}")
print(f"🔢 Rounds: {result.num_rounds}")
print(f"📊 Log: {logger.get_log_file()}")
```

## 📊 Running Tournaments

```bash
# Run a tournament with 10 games
python scripts/sdb_cli.py tournament secret_hitler \
  --name my_tournament \
  --num-games 10 \
  --num-players 5 \
  --output-dir tournaments/my_tournament

# Evaluate results
python scripts/sdb_cli.py evaluate \
  tournaments/my_tournament/result.json \
  --output tournaments/my_tournament/metrics.json

# Create visualizations
python scripts/sdb_cli.py visualize \
  --tournament tournaments/my_tournament/result.json \
  --metrics tournaments/my_tournament/metrics.json \
  --output-dir tournaments/my_tournament/viz
```

## 🎮 Game-Specific Examples

### Secret Hitler

Secret Hitler is a social deduction game where Liberals try to enact 5 Liberal policies or assassinate Hitler, while Fascists try to enact 6 Fascist policies or elect Hitler as Chancellor after 3 Fascist policies are enacted.

```python
from sdb.environments.secret_hitler import SecretHitlerEnv, SecretHitlerConfig
from sdb.agents.llm import OpenRouterAgent

# Create 5-10 players
agents = [OpenRouterAgent(player_id=i, name=f"Player_{i}") for i in range(7)]

# Configure with custom settings
config = SecretHitlerConfig(
    n_players=7,
    seed=42,  # For reproducibility
    log_private_info=True  # Log hidden information for analysis
)

env = SecretHitlerEnv(agents=agents, config=config)
result = env.play_game()
```

**Key Features:**
- ✅ Full game mechanics (nomination, voting, legislative session, presidential powers)
- ✅ Veto power (unlocks after 5 Fascist policies)
- ✅ Discussion phases (public deliberation before votes)
- ✅ Memory integration (agents remember all discussions and events)
- ✅ Belief tracking (agents build models of other players)

## 🤖 Supported Models

SDB uses **OpenRouter** to access multiple LLM providers:

### Recommended Models

```bash
# OpenAI
--model openai/gpt-4o          # Best performance
--model openai/gpt-4o-mini     # Fast & cheap (default)
--model openai/gpt-4-turbo     # Good balance

# Anthropic
--model anthropic/claude-opus-4.1     # High reasoning
--model anthropic/claude-sonnet-3.5   # Balanced
--model anthropic/claude-haiku        # Fast

# Google
--model google/gemini-2.5-pro    # Best Gemini
--model google/gemini-1.5-flash  # Fast Gemini

# Meta
--model meta/llama-3.1-405b      # Open source
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

# Monitor costs
stats = agent.get_stats()
print(f"Total cost: ${stats['llm_stats']['total_cost']:.2f}")
```

## 📁 Project Structure

```
social-deduction-bench/
├── sdb/                      # Main package
│   ├── core/                # Base classes & interfaces
│   ├── agents/              # Agent implementations
│   │   ├── llm/            # LLM agents (OpenRouter)
│   │   └── baselines/      # Random, rule-based agents
│   ├── environments/        # Game implementations
│   │   ├── secret_hitler/  # ✅ Complete
│   │   ├── among_us/       # 🔄 In progress
│   │   └── ...
│   ├── memory/             # Memory & belief tracking
│   ├── llm_interface/      # LLM API clients
│   ├── tournament/         # Tournament framework
│   ├── evaluation/         # Metrics & evaluation
│   ├── logging/            # Logging system
│   └── analysis/           # Analysis & visualization
├── scripts/                # CLI entry points
│   └── sdb_cli.py         # Main CLI
├── configs/                # Configuration files
├── experiments/            # Experiment outputs
└── tests/                  # Unit tests
```

## 🔧 Configuration

### Game Configuration

```python
from sdb.environments.secret_hitler import SecretHitlerConfig

config = SecretHitlerConfig(
    n_players=7,
    seed=42,                    # Reproducibility
    log_private_info=True,      # Log hidden info for analysis
)
```

### Agent Configuration

```python
from sdb.agents.llm import OpenRouterAgent

agent = OpenRouterAgent(
    player_id=0,
    name="Strategic_Player",
    model="openai/gpt-4o-mini",
    temperature=0.7,             # Randomness (0=deterministic, 1=creative)
    memory_capacity=50,          # Number of events to remember
    max_tokens=2048,            # Max response length
)
```

## 📊 Analyzing Results

### View Game Logs

```python
import json

# Load game log
with open("experiments/my_game/secret_hitler_cli.jsonl") as f:
    events = [json.loads(line) for line in f]

# Filter specific events
discussions = [e for e in events if e["event_type"] == "DISCUSSION"]
votes = [e for e in events if e["event_type"] == "VOTE_CAST"]

# Analyze agent reasoning
for event in events:
    if event["event_type"] == "AGENT_REASONING":
        print(f"Player {event['data']['player_id']}: {event['data']['reasoning']}")
```

### Extract Metrics

```bash
# Get win rates, deception metrics, etc.
python scripts/sdb_cli.py evaluate \
  experiments/my_tournament/result.json \
  --output experiments/my_tournament/metrics.json
```

## 🎓 Key Features

### 1. **Memory-Aware Agents**
Agents maintain:
- **Short-term memory**: Recent events and observations
- **Belief tracking**: Probabilistic models of other players
- **Discussion memory**: All public statements from all players

### 2. **Comprehensive Logging**
Every game event is logged:
- Player actions and reasoning
- Public discussions
- Private information (role assignments, investigations)
- Game state transitions

### 3. **Reproducibility**
- Seed-based randomness
- Deterministic game flow
- Complete event replay from logs

### 4. **Cost Tracking**
- Automatic token counting
- Cost estimation per game
- Provider-specific pricing

## 🛠️ Advanced Usage

### Custom Agent Behavior

```python
class MyCustomAgent(OpenRouterAgent):
    def _build_system_prompt(self):
        return """You are a highly strategic player.
        Always consider:
        1. Other players' past actions
        2. Probability of hidden roles
        3. Long-term consequences
        """
    
    def notify(self, event_type, data):
        super().notify(event_type, data)
        # Custom logic when receiving events
        if event_type == "discussion_statement":
            # Analyze what others say
            self._analyze_statement(data)
```

### Parallel Games

```python
import asyncio

async def run_multiple_games():
    games = [create_game() for _ in range(10)]
    results = await asyncio.gather(*[game.play_game_async() for game in games])
    return results

results = asyncio.run(run_multiple_games())
```

## 📖 Documentation

- **Architecture**: See `ARCHITECTURE.md` for technical details
- **API Reference**: See inline docstrings
- **Examples**: See `experiments/` directory

## 🤝 Contributing

Contributions are welcome! To add a new game:

1. Create game directory in `sdb/environments/your_game/`
2. Implement `BaseEnvironment` interface
3. Add configuration and rules
4. Write tests
5. Submit PR

See `ARCHITECTURE.md` for detailed implementation guide.

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

This framework builds upon:
- **AmongAgents** - Among Us implementation
- **AvalonBench & Strategist** (ICLR 2025) - Avalon with search agents
- **Secret Hitler** - Original game mechanics
- **Spyfall** - Tournament system
- **Werewolf Arena** (Google) - Werewolf implementation

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/social-deduction-bench/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/social-deduction-bench/discussions)

## 📚 Citation

```bibtex
@software{social_deduction_bench2025,
  title={Social Deduction Bench: Unified Framework for LLMs in Social Deduction Games},
  author={Social Deduction Bench Contributors},
  year={2025},
  url={https://github.com/yourusername/social-deduction-bench}
}
```

---

**Status**: Secret Hitler ✅ Complete | Other games 🔄 In Progress

For technical architecture details, see [`ARCHITECTURE.md`](ARCHITECTURE.md)
