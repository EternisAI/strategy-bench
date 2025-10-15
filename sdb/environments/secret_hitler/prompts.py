"""Game-specific prompts and instructions for Secret Hitler.

This module contains all the instructions that agents see for different actions.
The generic LLM agent reads these instructions and responds with JSON.
"""

from typing import Dict, List, Any


def get_nomination_instruction(legal_candidates: List[int], context: Dict[str, Any] = None) -> str:
    """Get instruction for chancellor nomination.
    
    Args:
        legal_candidates: List of valid candidate IDs
        context: Dict with game state (policies, history, etc.)
        
    Returns:
        Instruction string with JSON format
    """
    context = context or {}
    lib_policies = context.get("liberal_policies", 0)
    fas_policies = context.get("fascist_policies", 0)
    election_tracker = context.get("election_tracker", 0)
    history = context.get("formatted_history", "(No history yet)")
    discussion = context.get("formatted_discussion", "(No discussion yet)")
    round_num = context.get("round", 1)
    
    return f"""=== ROUND {round_num} - PRESIDENTIAL NOMINATION ===

GAME STATE:
📊 Policies Enacted: {lib_policies} Liberal, {fas_policies} Fascist
⚠️  Election Tracker: {election_tracker}/3 failed elections

📜 COMPLETE GAME HISTORY:
{history}

💬 CURRENT DISCUSSION:
{discussion}

⚡ YOUR ACTION:
You are the PRESIDENT. Nominate a Chancellor from the legal candidates.

Legal Candidates: {legal_candidates}

Strategy:
- Who do you trust based on previous actions?
- Who might be Fascist/Hitler?
- What policies might they enact?
- Consider the game state and what's at stake

Respond with JSON:
{{"type": "nominate", "nominee": <player_id>}}

Example: {{"type": "nominate", "nominee": {legal_candidates[0]}}}"""


def get_vote_instruction(president: int, nominee: int, your_party: str, context: Dict[str, Any] = None) -> str:
    """Get instruction for voting on government.
    
    Args:
        president: President player ID
        nominee: Nominated chancellor ID
        your_party: Player's party affiliation
        context: Dict with game state
        
    Returns:
        Instruction string with JSON format
    """
    context = context or {}
    lib_policies = context.get("liberal_policies", 0)
    fas_policies = context.get("fascist_policies", 0)
    election_tracker = context.get("election_tracker", 0)
    history = context.get("formatted_history", "(No history yet)")
    discussion = context.get("formatted_discussion", "(No discussion yet)")
    
    return f"""=== VOTING ON GOVERNMENT ===

PROPOSED GOVERNMENT:
👤 President: Player {president}
👤 Chancellor: Player {nominee}

YOUR PARTY: {your_party}

GAME STATE:
📊 Policies Enacted: {lib_policies} Liberal, {fas_policies} Fascist
⚠️  Election Tracker: {election_tracker}/3 failed elections

📜 COMPLETE GAME HISTORY:
{history}

💬 RECENT DISCUSSION:
{discussion}

⚡ YOUR VOTE:
Vote YES (Ja) to approve this government, NO (Nein) to reject.

Consider:
- Do you trust these players based on their history?
- What policies might they enact?
- How does this affect your win condition?
- What happens if election fails again? (3 failures = chaos!)

Respond with JSON:
{{"type": "vote", "vote": true}}  for YES/Ja
{{"type": "vote", "vote": false}} for NO/Nein"""


def get_discard_policy_instruction(policies: List[str]) -> str:
    """Get instruction for president policy discard.
    
    Args:
        policies: List of 3 policy names
        
    Returns:
        Instruction string with JSON format
    """
    return f"""You are PRESIDENT. You drew 3 policies: {policies}

Discard ONE policy. The remaining two go to the Chancellor.

Strategy:
- Liberals want Liberal policies enacted
- Fascists want Fascist policies enacted
- Consider what signal you're sending

Respond with JSON:
{{"type": "discard_policy", "index": <0, 1, or 2>}}

Example: {{"type": "discard_policy", "index": 0}} to discard the first policy"""


def get_enact_policy_instruction(policies: List[str], veto_available: bool) -> str:
    """Get instruction for chancellor policy enactment.
    
    Args:
        policies: List of 2 policy names
        veto_available: Whether veto power is unlocked
        
    Returns:
        Instruction string with JSON format
    """
    base_instruction = f"""You are CHANCELLOR. The President gave you 2 policies: {policies}

Enact ONE policy. This will be revealed to all players.

Strategy:
- Consider your role and win condition
- Think about what this signals to other players
"""
    
    if veto_available:
        base_instruction += """
VETO POWER UNLOCKED: You can propose to veto both policies.
- If President accepts: Both discarded, election tracker increases
- If President rejects: You MUST enact one of the policies

"""
    
    base_instruction += """Respond with JSON:
{"type": "enact_policy", "index": <0 or 1>}"""
    
    if veto_available:
        base_instruction += """
OR {"type": "propose_veto"}"""
    
    base_instruction += f"""

Example: {{"type": "enact_policy", "index": 0}} to enact the first policy"""
    
    return base_instruction


def get_discussion_instruction(context: Dict[str, Any]) -> str:
    """Get instruction for discussion phase.
    
    Args:
        context: Discussion context (president, nominee, previous statements)
        
    Returns:
        Instruction string with JSON format
    """
    president = context.get('president', '?')
    nominee = context.get('nominee', '?')
    previous = context.get('previous_statements', [])
    
    instruction = f"""PUBLIC DISCUSSION: President {president} nominated Chancellor {nominee}

⚠️  THIS IS PUBLIC - ALL PLAYERS WILL SEE YOUR STATEMENT!

"""
    
    if previous:
        instruction += "Previous statements:\n"
        for stmt in previous:
            instruction += f"  - Player {stmt['speaker']}: \"{stmt['statement']}\"\n"
        instruction += "\n"
    
    instruction += """You can:
- Express your opinion about the nomination
- Share information (or misinformation if Fascist)
- Ask questions to other players
- Stay silent (empty string)

Strategy:
- Liberals: Help identify Fascists
- Fascists: Sow confusion, protect Hitler, discredit Liberals
- Hitler: Blend in, gain trust

Respond with JSON:
{"type": "discuss", "statement": "<your public statement>"}
OR {"type": "discuss", "statement": ""} to stay silent

Example: {"type": "discuss", "statement": "I trust this government, we should vote Ja"}"""
    
    return instruction


def get_veto_discussion_instruction(context: Dict[str, Any]) -> str:
    """Get instruction for veto discussion phase.
    
    Args:
        context: Veto context (president, chancellor, previous statements)
        
    Returns:
        Instruction string with JSON format
    """
    president = context.get('president', '?')
    chancellor = context.get('chancellor', '?')
    previous = context.get('previous_statements', [])
    
    instruction = f"""PUBLIC VETO DISCUSSION: Chancellor {chancellor} proposed VETO!

President {president} must decide whether to accept or reject the veto.

⚠️  THIS IS PUBLIC - ALL PLAYERS WILL SEE YOUR STATEMENT!

"""
    
    if previous:
        instruction += "Previous statements:\n"
        for stmt in previous:
            instruction += f"  - Player {stmt['speaker']}: \"{stmt['statement']}\"\n"
        instruction += "\n"
    
    instruction += """Strategic considerations:
- What does the veto proposal reveal?
- Are they both Fascists trying to manipulate the election tracker?
- Or are they both Liberals protecting against Fascist policies?

Respond with JSON:
{"type": "discuss_veto", "statement": "<your public statement>"}
OR {"type": "discuss_veto", "statement": ""} to stay silent

Example: {"type": "discuss_veto", "statement": "This seems suspicious, why veto?"}"""
    
    return instruction


def get_veto_response_instruction(chancellor: int) -> str:
    """Get instruction for president's veto response.
    
    Args:
        chancellor: Chancellor player ID
        
    Returns:
        Instruction string with JSON format
    """
    return f"""Chancellor {chancellor} proposed to VETO both policies!

As President, you must decide:

ACCEPT VETO:
- Both policies are discarded
- Election tracker increases by 1
- If tracker reaches 3: Chaos - top policy auto-enacted

REJECT VETO:
- Chancellor MUST enact one of the two policies
- Game continues normally

Consider:
- Why did the Chancellor propose veto?
- What are both policies likely to be?
- Is this strategic or suspicious?

Respond with JSON:
{"type": "veto_response", "accept": true}  to ACCEPT veto
{"type": "veto_response", "accept": false} to REJECT veto"""


def get_investigate_instruction(available_players: List[int]) -> str:
    """Get instruction for investigate presidential power.
    
    Args:
        available_players: List of player IDs that can be investigated
        
    Returns:
        Instruction string with JSON format
    """
    return f"""PRESIDENTIAL POWER: INVESTIGATE a player's party membership!

You will secretly learn if they are Liberal or Fascist.
(Note: Hitler appears as Fascist)

Available players: {available_players}

Strategy:
- Investigate someone you're unsure about
- Use this information in future discussions
- Be careful - Fascists may lie about investigation results

Respond with JSON:
{{"type": "investigate", "target": <player_id>}}

Example: {{"type": "investigate", "target": {available_players[0]}}}"""


def get_special_election_instruction(available_players: List[int]) -> str:
    """Get instruction for special election presidential power.
    
    Args:
        available_players: List of player IDs that can be chosen
        
    Returns:
        Instruction string with JSON format
    """
    return f"""PRESIDENTIAL POWER: SPECIAL ELECTION - Choose the next President!

This is a one-time power. After their term, presidency returns to normal rotation.

Available players: {available_players}

Strategy:
- Give power to someone you trust
- Or give it to someone suspicious to test them
- Consider the current board state

Respond with JSON:
{{"type": "special_election", "target": <player_id>}}

Example: {{"type": "special_election", "target": {available_players[0]}}}"""


def get_peek_instruction() -> str:
    """Get instruction for peek policy presidential power.
    
    Returns:
        Instruction string with JSON format
    """
    return """PRESIDENTIAL POWER: PEEK at the top 3 policies!

You will secretly see the next 3 policies in the deck.

This is an automatic action - you just need to confirm.

Strategy:
- Use this information to guide future governments
- Share (or don't share) this information strategically
- Fascists may lie about what they saw

Respond with JSON:
{"type": "peek_policies", "confirm": true}"""


def get_execution_instruction(available_players: List[int]) -> str:
    """Get instruction for execution presidential power.
    
    Args:
        available_players: List of player IDs that can be executed
        
    Returns:
        Instruction string with JSON format
    """
    return f"""PRESIDENTIAL POWER: EXECUTION - Eliminate a player from the game!

⚠️  THIS IS PERMANENT - Choose carefully!

Available players: {available_players}

Strategy:
- Execute someone you believe is Fascist/Hitler
- Be cautious - killing Hitler as Fascist loses the game immediately
- Consider all information gathered so far

Respond with JSON:
{{"type": "execute", "target": <player_id>}}

Example: {{"type": "execute", "target": {available_players[0]}}}"""


# Mapping from action types to instruction functions
INSTRUCTION_BUILDERS = {
    "nominate_chancellor": get_nomination_instruction,
    "vote": get_vote_instruction,
    "discard_policy": get_discard_policy_instruction,
    "enact_policy": get_enact_policy_instruction,
    "discuss_nomination": get_discussion_instruction,
    "discuss_veto": get_veto_discussion_instruction,
    "veto_response": get_veto_response_instruction,
    "investigate": get_investigate_instruction,
    "special_election": get_special_election_instruction,
    "peek_policies": get_peek_instruction,
    "execute": get_execution_instruction,
}

