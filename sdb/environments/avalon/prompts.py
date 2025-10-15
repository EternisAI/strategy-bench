"""Game-specific prompts and instructions for Avalon.

This module contains all the instructions that agents see for different actions.
The generic LLM agent reads these instructions and responds with JSON.
"""

from typing import Dict, List, Any


def get_team_selection_instruction(quest_number: int, team_size: int, available_players: List[int]) -> str:
    """Get instruction for team selection.
    
    Args:
        quest_number: Current quest number (1-5)
        team_size: Required team size
        available_players: List of player IDs to choose from
        
    Returns:
        Instruction string with JSON format
    """
    return f"""You are the QUEST LEADER for Quest {quest_number}!

Select a team of {team_size} players for this quest.

Available players: {available_players}

Strategy:
- Good: Choose players you trust
- Evil: Include Evil players to sabotage OR build trust by succeeding
- Consider previous quest results and voting patterns

Respond with JSON:
{{"type": "propose_team", "team": [<list of {team_size} player IDs>]}}

Example: {{"type": "propose_team", "team": {available_players[:team_size]}}}"""


def get_team_discussion_instruction(
    quest_number: int,
    quest_leader: int,
    is_leader: bool,
    dialogue_history: List[tuple[int, str]],
    team_size: int,
) -> str:
    """Get instruction for team discussion phase.
    
    Args:
        quest_number: Current quest number (1-5)
        quest_leader: Quest leader player ID
        is_leader: Whether this player is the leader
        dialogue_history: List of (speaker_id, statement) tuples from discussion
        team_size: Required team size
        
    Returns:
        Instruction string with JSON format
    """
    role_text = "QUEST LEADER" if is_leader else "PARTICIPANT"
    
    # Format previous dialogue
    if dialogue_history:
        dialogue_text = "\n".join([
            f"  - Player {speaker}: \"{statement}\""
            for speaker, statement in dialogue_history
        ])
        previous_dialogue = f"\n📜 Previous Dialogue:\n{dialogue_text}\n"
    else:
        previous_dialogue = "\n📜 No dialogue yet (discussion just started)\n"
    
    if is_leader:
        guidance = f"""As LEADER, make an opening statement:
- Explain your planned team composition
- Give reasons for your choices
- Build trust or create misdirection
- You'll propose the actual team after discussion"""
    else:
        guidance = f"""As PARTICIPANT, contribute to the discussion:
- Question the leader's logic
- Suggest alternative team compositions
- Share your suspicions or defenses
- Try to identify good vs evil players"""
    
    return f"""🗣️  TEAM DISCUSSION - Quest {quest_number} ({role_text})

Leader: Player {quest_leader}
Team size needed: {team_size} players
{previous_dialogue}
⚠️  THIS IS PUBLIC - ALL PLAYERS WILL SEE YOUR STATEMENT!

{guidance}

Strategy Tips:
- **Good players**: Share genuine reasoning, ask probing questions
- **Evil players**: Blend in, create doubt, deflect suspicion
- **Merlin**: Guide subtly without revealing yourself
- **Assassin**: Watch for who seems too knowledgeable

Respond with JSON:
{{"type": "discuss_team", "statement": "<your public statement>"}}

Examples:
{{"type": "discuss_team", "statement": "I propose taking myself, Player 0, and Player 3 because they voted correctly last time."}}
{{"type": "discuss_team", "statement": "Player {quest_leader}, why not include Player 1? They seem trustworthy to me."}}
{{"type": "discuss_team", "statement": "I'm willing to go on this quest if the team trusts me."}}"""


def get_team_vote_instruction(leader: int, proposed_team: List[int], proposals_rejected: int) -> str:
    """Get instruction for voting on proposed team.
    
    Args:
        leader: Quest leader player ID
        proposed_team: Proposed team member IDs
        proposals_rejected: Number of rejected proposals so far
        
    Returns:
        Instruction string with JSON format
    """
    warning = ""
    if proposals_rejected >= 4:
        warning = "\n⚠️  WARNING: This is the 5th proposal! If rejected, Evil wins automatically!\n"
    
    return f"""VOTE on the proposed quest team!

Leader {leader} proposed: {proposed_team}
Proposals rejected so far: {proposals_rejected}/5
{warning}
Consider:
- Do you trust these players?
- What would Evil players vote?
- How does the voting pattern reveal information?

Respond with JSON:
{{"type": "vote_team", "approve": true}}  to APPROVE
{{"type": "vote_team", "approve": false}} to REJECT"""


def get_quest_vote_instruction(quest_number: int, team: List[int], fails_needed: int) -> str:
    """Get instruction for quest voting.
    
    Args:
        quest_number: Current quest number
        team: Team members on this quest
        fails_needed: Number of fails needed to fail quest
        
    Returns:
        Instruction string with JSON format
    """
    return f"""You are on Quest {quest_number}!

Team: {team}
Fails needed to fail this quest: {fails_needed}

SECRET VOTE:
- Good players MUST vote Success
- Evil players can vote Success OR Fail

Strategy:
- Evil: Consider whether to sabotage or build trust
- Everyone: Think about what the result reveals

Respond with JSON:
{{"type": "quest_vote", "success": true}}  to vote SUCCESS
{{"type": "quest_vote", "success": false}} to vote FAIL (Evil only!)"""


def get_assassination_instruction(available_targets: List[int]) -> str:
    """Get instruction for Assassin's assassination attempt.
    
    Args:
        available_targets: List of player IDs that can be assassinated
        
    Returns:
        Instruction string with JSON format
    """
    return f"""YOU ARE THE ASSASSIN!

Good completed 3 quests, but you have one last chance!

If you correctly assassinate MERLIN, Evil wins!

Available targets: {available_targets}

Strategy:
- Who behaved like Merlin?
- Who seemed to have perfect information?
- Who was trusted but not too obvious?

Respond with JSON:
{{"type": "assassinate", "target": <player_id>}}

Example: {{"type": "assassinate", "target": {available_targets[0]}}}"""


# Mapping from action types to instruction functions
INSTRUCTION_BUILDERS = {
    "propose_team": get_team_selection_instruction,
    "vote_team": get_team_vote_instruction,
    "quest_vote": get_quest_vote_instruction,
    "assassinate": get_assassination_instruction,
}

