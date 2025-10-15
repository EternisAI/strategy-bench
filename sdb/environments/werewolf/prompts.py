"""Game-specific prompts and instructions for Werewolf.

This module contains all the instructions that agents see for different actions.
The generic LLM agent reads these instructions and responds with JSON.
"""

from typing import Dict, List, Any


def get_werewolf_eliminate_instruction(available_targets: List[int]) -> str:
    """Get instruction for werewolf elimination.
    
    Args:
        available_targets: List of player IDs that can be eliminated
        
    Returns:
        Instruction string with JSON format
    """
    return f"""NIGHT PHASE - WEREWOLF ACTION

You are a WEREWOLF. Choose a player to eliminate tonight.

Available targets: {available_targets}

Strategy:
- Target suspected power roles (Seer, Doctor)
- Avoid patterns that reveal you
- Consider who might be protected by Doctor

Respond with JSON:
{{"type": "eliminate", "target": <player_id>}}

Example: {{"type": "eliminate", "target": {available_targets[0]}}}"""


def get_doctor_protect_instruction(available_targets: List[int]) -> str:
    """Get instruction for doctor protection.
    
    Args:
        available_targets: List of player IDs that can be protected
        
    Returns:
        Instruction string with JSON format
    """
    return f"""NIGHT PHASE - DOCTOR ACTION

You are the DOCTOR. Choose a player to protect tonight.

Available targets: {available_targets}

Strategy:
- Protect suspected power roles
- Protect players likely to be targeted
- Consider self-protection

Respond with JSON:
{{"type": "protect", "target": <player_id>}}

Example: {{"type": "protect", "target": {available_targets[0]}}}"""


def get_seer_investigate_instruction(available_targets: List[int]) -> str:
    """Get instruction for seer investigation.
    
    Args:
        available_targets: List of player IDs that can be investigated
        
    Returns:
        Instruction string with JSON format
    """
    return f"""NIGHT PHASE - SEER ACTION

You are the SEER. Investigate a player to learn their role.

Available targets: {available_targets}

Strategy:
- Investigate suspicious players
- Build a network of known roles
- Be subtle - revealing yourself helps Werewolves

Respond with JSON:
{{"type": "investigate", "target": <player_id>}}

Example: {{"type": "investigate", "target": {available_targets[0]}}}"""


def get_bid_instruction(current_debate: List[tuple], turns_left: int) -> str:
    """Get instruction for bidding to speak.
    
    Args:
        current_debate: List of (speaker_id, statement) tuples
        turns_left: Number of debate turns remaining
        
    Returns:
        Instruction string with JSON format
    """
    debate_str = ""
    if current_debate:
        debate_str = "\nCurrent debate:\n"
        for speaker, stmt in current_debate:
            debate_str += f"  - Player {speaker}: {stmt}\n"
    
    return f"""DAY PHASE - BIDDING TO SPEAK

Bid 0-4 to speak. Highest bid speaks next.
{turns_left} debate turns remaining.
{debate_str}
Strategy:
- Bid high if you have important information
- Bid low to conserve or observe
- Villagers: Share suspicions
- Werewolves: Deflect, mislead

Respond with JSON:
{{"type": "bid", "bid": <0-4>}}

Example: {{"type": "bid", "bid": 2}}"""


def get_debate_instruction(current_debate: List[tuple]) -> str:
    """Get instruction for making a debate statement.
    
    Args:
        current_debate: List of (speaker_id, statement) tuples
        
    Returns:
        Instruction string with JSON format
    """
    debate_str = ""
    if current_debate:
        debate_str = "\nPrevious statements:\n"
        for speaker, stmt in current_debate:
            debate_str += f"  - Player {speaker}: {stmt}\n"
    
    return f"""DAY PHASE - DEBATE

You won the bid! Make your PUBLIC statement.

⚠️  ALL PLAYERS WILL SEE THIS!
{debate_str}
Strategy:
- Villagers: Share information, build cases
- Seer: Drop hints without revealing yourself
- Werewolves: Deflect suspicion, sow confusion

Respond with JSON:
{{"type": "debate", "statement": "<your public statement>"}}

Example: {{"type": "debate", "statement": "I think Player 2 is suspicious"}}"""


def get_vote_instruction(available_targets: List[int], current_debate: List[tuple]) -> str:
    """Get instruction for voting to eliminate.
    
    Args:
        available_targets: List of player IDs that can be voted for
        current_debate: List of (speaker_id, statement) tuples
        
    Returns:
        Instruction string with JSON format
    """
    debate_summary = ""
    if current_debate:
        debate_summary = f"\n{len(current_debate)} statements were made during debate.\n"
    
    return f"""DAY PHASE - VOTING TO ELIMINATE

Vote to eliminate a player from the game.
{debate_summary}
Available targets: {available_targets}

Consider:
- Who seemed suspicious in the debate?
- Voting patterns from previous rounds
- Your role's win condition

Respond with JSON:
{{"type": "vote", "target": <player_id>}}

Example: {{"type": "vote", "target": {available_targets[0]}}}"""


# Mapping from action types to instruction functions
INSTRUCTION_BUILDERS = {
    "eliminate": get_werewolf_eliminate_instruction,
    "protect": get_doctor_protect_instruction,
    "investigate": get_seer_investigate_instruction,
    "bid": get_bid_instruction,
    "debate": get_debate_instruction,
    "vote": get_vote_instruction,
}

