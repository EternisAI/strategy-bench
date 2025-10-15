"""Game-specific prompts and instructions for Sheriff of Nottingham.

This module contains all the instructions that agents see for different actions.
The generic LLM agent reads these instructions and responds with JSON.
"""

from typing import Dict, List, Any


def get_load_bag_instruction(hand: List[str], max_goods: int) -> str:
    """Get instruction for loading goods into bag.
    
    Args:
        hand: List of card names in player's hand
        max_goods: Maximum number of goods allowed
        
    Returns:
        Instruction string with JSON format
    """
    return f"""MARKET PHASE - LOAD YOUR BAG

Your hand: {hand}

Load up to {max_goods} cards into your bag.

Legal goods: Apples, Cheese, Bread, Chickens (4 types)
Contraband: Silk, Pepper, Mead, Crossbows, Royal Goods (higher value, illegal!)

Strategy:
- Honest: Load only legal goods
- Bluffing: Mix in contraband and lie in declaration
- High risk, high reward with contraband!

Respond with JSON:
{{"type": "load_bag", "cards": ["<card_name>", "<card_name>", ...]}}

Example: {{"type": "load_bag", "cards": ["Apples", "Apples", "Cheese"]}}"""


def get_declare_instruction(bag_contents: List[str]) -> str:
    """Get instruction for declaring bag contents.
    
    Args:
        bag_contents: Actual contents of the bag
        
    Returns:
        Instruction string with JSON format
    """
    return f"""DECLARATION PHASE

Your bag contains: {bag_contents}

Declare what you claim is in your bag.

Rules:
- You can only declare ONE type of good
- You can declare any quantity (but must match actual count if honest)
- You CAN LIE about contents!

Strategy:
- Tell truth: Safe but boring
- Lie: Can smuggle contraband for big profits
- Sheriff may inspect based on your behavior

Respond with JSON:
{{"type": "declare", "good": "<good_type>", "quantity": <number>}}

Example: {{"type": "declare", "good": "Apples", "quantity": 3}}"""


def get_sheriff_inspect_instruction(merchant: int, declared_good: str, declared_quantity: int, gold: int) -> str:
    """Get instruction for sheriff's inspection decision.
    
    Args:
        merchant: Player ID of current merchant
        declared_good: What merchant declared
        declared_quantity: How many they declared
        gold: Sheriff's current gold
        
    Returns:
        Instruction string with JSON format
    """
    return f"""YOU ARE THE SHERIFF!

Merchant {merchant} declares: {declared_quantity} {declared_good}

Your gold: {gold}

Options:
1. LET PASS: Merchant keeps goods, no penalty
2. INSPECT: 
   - If lying: Merchant pays penalty, confiscate contraband
   - If honest: YOU pay merchant for each good!

Strategy:
- Watch for suspicious behavior
- Consider merchant's past actions
- Risk vs reward of inspection

Respond with JSON:
{{"type": "inspect", "inspect": true}}  to INSPECT
{{"type": "inspect", "inspect": false}} to LET PASS"""


def get_offer_bribe_instruction(sheriff: int, your_gold: int, bag_contents: List[str], declared: Dict) -> str:
    """Get instruction for offering bribe to sheriff.
    
    Args:
        sheriff: Sheriff player ID
        your_gold: Merchant's gold
        bag_contents: Actual bag contents
        declared: What was declared
        
    Returns:
        Instruction string with JSON format
    """
    return f"""NEGOTIATION PHASE

You declared: {declared['quantity']} {declared['good']}
Your actual bag: {bag_contents}
Your gold: {your_gold}

Sheriff {sheriff} is deciding whether to inspect!

You can:
- OFFER BRIBE: Gold or cards to avoid inspection
- STAY SILENT: Hope they let you pass
- NEGOTIATE: Make a deal

Strategy:
- If honest: No need to bribe (but might anyway for King/Queen bonus)
- If lying: Bribe might be cheaper than penalty
- Be persuasive!

Respond with JSON:
{{"type": "offer_bribe", "gold": <amount>, "cards": ["<card>", ...], "message": "<persuasive message>"}}

OR {{"type": "offer_bribe", "gold": 0, "cards": [], "message": ""}} for no bribe"""


def get_bribe_response_instruction(merchant: int, bribe: Dict) -> str:
    """Get instruction for sheriff's response to bribe.
    
    Args:
        merchant: Merchant player ID
        bribe: Bribe offer details
        
    Returns:
        Instruction string with JSON format
    """
    gold = bribe.get('gold', 0)
    cards = bribe.get('cards', [])
    message = bribe.get('message', '')
    
    offer_str = f"{gold} gold" if gold > 0 else ""
    if cards:
        offer_str += f" + {len(cards)} cards" if offer_str else f"{len(cards)} cards"
    if not offer_str:
        offer_str = "nothing"
    
    return f"""BRIBE DECISION

Merchant {merchant} offers: {offer_str}
Their message: "{message}"

Do you accept the bribe and let them pass?

Consider:
- Is the bribe worth it?
- Are they likely lying?
- What if they're honest and you inspect?

Respond with JSON:
{{"type": "bribe_response", "accept": true}}  to ACCEPT bribe (let pass)
{{"type": "bribe_response", "accept": false}} to REJECT bribe (will inspect)"""


# Mapping from action types to instruction functions
INSTRUCTION_BUILDERS = {
    "load_bag": get_load_bag_instruction,
    "declare": get_declare_instruction,
    "inspect": get_sheriff_inspect_instruction,
    "offer_bribe": get_offer_bribe_instruction,
    "bribe_response": get_bribe_response_instruction,
}

