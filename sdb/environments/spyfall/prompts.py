"""Game-specific prompts and instructions for Spyfall.

This module contains all the instructions that agents see for different actions.
The generic LLM agent reads these instructions and responds with JSON.
"""

from typing import Dict, List, Any


def get_ask_question_instruction(available_targets: List[int], cannot_ask_back: int = None) -> str:
    """Get instruction for asking a question.
    
    Args:
        available_targets: List of player IDs that can be asked
        cannot_ask_back: Player ID that cannot be asked (if any)
        
    Returns:
        Instruction string with JSON format
    """
    note = ""
    if cannot_ask_back is not None:
        note = f"\nNote: You cannot ask Player {cannot_ask_back} (they just asked you).\n"
    
    return f"""Q&A PHASE - YOUR TURN TO ASK

Ask another player a question about the location.

Available targets: {available_targets}
{note}
Strategy:
- If you know the location: Ask questions that test others
- If you're the SPY: Ask vague questions to blend in
- Avoid revealing you don't know the location!

Respond with JSON:
{{"type": "ask", "target": <player_id>, "question": "<your question>"}}

Example: {{"type": "ask", "target": {available_targets[0]}, "question": "What color are the walls?"}}"""


def get_answer_question_instruction(question: str, asker: int, is_spy: bool) -> str:
    """Get instruction for answering a question.
    
    Args:
        question: The question being asked
        asker: Player ID who asked the question
        is_spy: Whether this player is the spy
        
    Returns:
        Instruction string with JSON format
    """
    spy_note = ""
    if is_spy:
        spy_note = """
⚠️  YOU ARE THE SPY! You don't know the location!
- Answer vaguely enough to blend in
- Try to infer the location from others' answers
"""
    
    return f"""Q&A PHASE - ANSWER THE QUESTION

Player {asker} asked you: "{question}"
{spy_note}
Strategy:
- If you know location: Answer accurately but test for the spy
- If you're the spy: Be vague, bluff confidently
- Your answer reveals information to everyone

Respond with JSON:
{{"type": "answer", "answer": "<your answer>"}}

Example: {{"type": "answer", "answer": "They're mostly white with some blue accents"}}"""


def get_accuse_instruction(available_suspects: List[int]) -> str:
    """Get instruction for accusing someone of being the spy.
    
    Args:
        available_suspects: List of player IDs that can be accused
        
    Returns:
        Instruction string with JSON format
    """
    return f"""STOP THE CLOCK - ACCUSATION!

You're calling for a vote to identify the spy!

Available suspects: {available_suspects}

⚠️  This requires UNANIMOUS vote from all other players!
- If unanimous YES and correct: You win!
- If unanimous YES but wrong: Spy wins!
- If not unanimous: Game continues

Strategy:
- Only accuse when confident
- Consider all questions and answers
- Spy can also accuse to deflect suspicion

Respond with JSON:
{{"type": "accuse", "suspect": <player_id>}}

Example: {{"type": "accuse", "suspect": {available_suspects[0]}}}"""


def get_spy_guess_instruction(locations: List[str]) -> str:
    """Get instruction for spy's location guess.
    
    Args:
        locations: List of possible location names
        
    Returns:
        Instruction string with JSON format
    """
    return f"""SPY GUESS - FINAL CHANCE!

You're the spy! Guess the location correctly to win!

Possible locations: {locations}

Based on the questions and answers, what is the location?

⚠️  If correct: Spy wins!
⚠️  If incorrect: Non-spy players win!

Respond with JSON:
{{"type": "spy_guess", "guess": "<location_name>"}}

Example: {{"type": "spy_guess", "guess": "{locations[0]}"}}"""


def get_accusation_vote_instruction(suspect: int, accuser: int) -> str:
    """Get instruction for voting on an accusation.
    
    Args:
        suspect: Player ID being accused
        accuser: Player ID who made the accusation
        
    Returns:
        Instruction string with JSON format
    """
    return f"""VOTE ON ACCUSATION

Player {accuser} accused Player {suspect} of being the spy!

⚠️  Vote must be UNANIMOUS to eliminate!

Consider:
- Do you think they're the spy?
- What did they say that was suspicious?
- If wrong, spy wins immediately!

Respond with JSON:
{{"type": "vote", "vote": true}}  to vote YES (eliminate suspect)
{{"type": "vote", "vote": false}} to vote NO (continue game)"""


def get_final_nominate_instruction(available_suspects: List[int]) -> str:
    """Get instruction for final voting nomination.
    
    Args:
        available_suspects: List of player IDs that can be nominated
        
    Returns:
        Instruction string with JSON format
    """
    return f"""FINAL VOTING - TIME'S UP!

Nominate someone you suspect is the spy.

Available players: {available_suspects}

⚠️  Requires unanimous vote to eliminate!

Strategy:
- Who was most evasive?
- Who asked suspicious questions?
- Whose answers didn't fit?

Respond with JSON:
{{"type": "nominate", "suspect": <player_id>}}

Example: {{"type": "nominate", "suspect": {available_suspects[0]}}}"""


def get_final_vote_instruction(suspect: int, nominator: int) -> str:
    """Get instruction for final voting.
    
    Args:
        suspect: Player ID being nominated
        nominator: Player ID who nominated
        
    Returns:
        Instruction string with JSON format
    """
    return f"""FINAL VOTE

Player {nominator} nominated Player {suspect} as the spy!

⚠️  Vote must be UNANIMOUS!

This is your last chance to identify the spy!

Respond with JSON:
{{"type": "vote", "vote": true}}  to vote YES
{{"type": "vote", "vote": false}} to vote NO"""


# Mapping from action types to instruction functions
INSTRUCTION_BUILDERS = {
    "ask": get_ask_question_instruction,
    "answer": get_answer_question_instruction,
    "accuse": get_accuse_instruction,
    "spy_guess": get_spy_guess_instruction,
    "vote": get_accusation_vote_instruction,
    "nominate": get_final_nominate_instruction,
    "vote_final": get_final_vote_instruction,
}

