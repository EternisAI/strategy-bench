# Social Deduction Bench - Examples

This directory contains example scripts for running each game with LLM agents.

## Quick Start

1. **Install the package** (first time only):
```bash
cd /path/to/social-deduction-bench
pip install -e .
```

2. **Set your API key**:
```bash
export OPENROUTER_API_KEY='your-key-here'
```

3. **Run a game** (from project root):
```bash
cd /path/to/social-deduction-bench
python examples/run_avalon.py
python examples/run_secret_hitler.py
python examples/run_werewolf.py
# ... etc
```

## Available Examples

- `run_avalon.py` - Avalon with pre-proposal discussion
- `run_secret_hitler.py` - Secret Hitler with veto power
- `run_sheriff.py` - Sheriff of Nottingham with negotiation
- `run_werewolf.py` - Werewolf with bidding and debate
- `run_spyfall.py` - Spyfall with Q&A gameplay
- `run_among_us.py` - Among Us with spatial map

## Customization

Each script uses a corresponding config file in `configs/`. You can:
- Edit configs to change game parameters
- Modify scripts to use different LLM models
- Adjust logging and output directories
- Change agent behavior and memory settings

## Configuration Files

All game configs are in `configs/`:
- `configs/avalon.yaml`
- `configs/secret_hitler.yaml`
- `configs/sheriff.yaml`
- `configs/werewolf.yaml`
- `configs/spyfall.yaml`
- `configs/among_us.yaml`

