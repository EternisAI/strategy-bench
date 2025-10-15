"""Game-specific prompts and instructions for Among Us with spatial mechanics.

This module contains all the instructions that agents see for different actions.
The generic LLM agent reads these instructions and responds with JSON.

SPATIAL MECHANICS:
- The spaceship has 14 rooms (Skeld map)
- Players move via corridors (1 room per action)
- Impostors can use vents for fast travel
- Only players in the same room can see each other
- Tasks must be completed in specific rooms
- Kills only work if impostor and target are in the same room
"""

from typing import Dict, List, Any


SPATIAL_INTRO = """🚀 SPATIAL MECHANICS:
- You are on the Skeld spaceship with 14 rooms
- Move via corridors (1 room per turn) or stay in place
- You can only see players in your CURRENT room
- Tasks must be completed in their assigned rooms
"""


def get_crewmate_task_instruction(
    tasks_completed: int,
    total_tasks: int,
    can_call_emergency: bool,
    location: str = "unknown",
    adjacent_rooms: List[str] = None,
    remaining_tasks: List[str] = None
) -> str:
    """Get instruction for crewmate task phase with spatial info.
    
    Args:
        tasks_completed: Tasks completed by this player
        total_tasks: Total tasks assigned to this player
        can_call_emergency: Whether player can call emergency meeting
        location: Current room location
        adjacent_rooms: List of adjacent rooms
        remaining_tasks: List of remaining tasks with locations
        
    Returns:
        Instruction string with JSON format
    """
    adj_str = ", ".join(adjacent_rooms) if adjacent_rooms else "none"
    tasks_str = "\n".join(f"  • {t}" for t in (remaining_tasks or []))
    
    return f"""{SPATIAL_INTRO}
📍 Your Location: {location}
🚪 Adjacent Rooms: {adj_str}

TASK PHASE - CREWMATE
Your tasks: {tasks_completed}/{total_tasks} completed

Remaining tasks:
{tasks_str or "  (All tasks completed!)"}

Available actions:
  - {{"type": "move", "room": "<room_name>"}}  to move to adjacent room
  - {{"type": "complete_task"}}  if you're in the correct room
  - {{"type": "call_emergency"}}  (only in Cafeteria, once per game)
  - {{"type": "report_body", "victim": <player_id>}}  if you see a body

Strategy:
- Move to task locations and complete them
- Pay attention to who you see in each room
- Report bodies immediately
- Call emergency if you have critical info

Respond with JSON - choose ONE action."""


def get_impostor_task_instruction(
    can_kill: bool,
    kill_cooldown: int,
    alive_crewmates: int,
    location: str = "unknown",
    adjacent_rooms: List[str] = None,
    vent_destinations: List[str] = None,
    visible_players: List[str] = None
) -> str:
    """Get instruction for impostor task phase with spatial info.
    
    Args:
        can_kill: Whether impostor can currently kill
        kill_cooldown: Rounds until can kill (if on cooldown)
        alive_crewmates: Number of alive crewmates
        location: Current room location
        adjacent_rooms: List of adjacent rooms (corridors)
        vent_destinations: List of rooms reachable via vents
        visible_players: List of players in current room
        
    Returns:
        Instruction string with JSON format
    """
    adj_str = ", ".join(adjacent_rooms) if adjacent_rooms else "none"
    vent_str = ", ".join(vent_destinations) if vent_destinations else "none"
    visible_str = ", ".join(visible_players) if visible_players else "empty room"
    
    return f"""{SPATIAL_INTRO}
💀 IMPOSTOR ABILITY: You can use VENTS for fast travel!

📍 Your Location: {location}
🚪 Adjacent Rooms (corridor): {adj_str}
🕳️  Vent Destinations: {vent_str}
👁️  Visible Players: {visible_str}

TASK PHASE - IMPOSTOR
Alive crewmates: {alive_crewmates}
Kill cooldown: {"Ready!" if can_kill else f"{kill_cooldown} rounds"}

Available actions:
  - {{"type": "move", "room": "<room_name>"}}  move via corridor
  - {{"type": "vent", "room": "<room_name>"}}  FAST TRAVEL via vent!
  - {{"type": "kill", "target": <player_id>}}  eliminate (must be in same room{", READY!" if can_kill else f", cooldown: {kill_cooldown}"})
  - {{"type": "report_body", "victim": <player_id>}}  report body (deflect suspicion)
  - {{"type": "wait"}}  do nothing

Strategy:
- Use vents to move quickly and create alibis
- Only kill when no witnesses present
- Kill in isolated rooms, then vent away
- Report bodies to appear innocent
- Avoid being seen near bodies

⚠️  You can only kill players in your CURRENT room!

Respond with JSON - choose ONE action."""


def get_emergency_discussion_instruction(caller: int, context: str) -> str:
    """Get instruction for emergency meeting discussion.
    
    Args:
        caller: Player ID who called the emergency
        context: Context about why meeting was called
        
    Returns:
        Instruction string with JSON format
    """
    return f"""EMERGENCY MEETING DISCUSSION

Player {caller} called an emergency meeting!
{context}

⚠️  THIS IS PUBLIC - ALL PLAYERS WILL SEE YOUR STATEMENT!

Share information, accusations, or defend yourself.

💡 Spatial Info Matters:
- Mention which rooms you were in
- Say who you saw in each room
- Question others about their locations

Strategy:
- Crewmates: Share suspicions, alibi, locations
- Impostors: Create fake alibis, deflect, sow confusion

Respond with JSON:
{{"type": "discuss", "statement": "<your public statement>"}}
OR {{"type": "discuss", "statement": ""}} to stay silent

Example: {{"type": "discuss", "statement": "I was in Electrical, saw Player 3 venting!"}}"""


def get_body_discussion_instruction(reporter: int, victim: int, body_location: str = "unknown") -> str:
    """Get instruction for body report discussion.
    
    Args:
        reporter: Player ID who reported the body
        victim: Player ID of the victim
        body_location: Room where body was found
        
    Returns:
        Instruction string with JSON format
    """
    return f"""BODY REPORTED - DISCUSSION

Player {reporter} reported Player {victim}'s body in {body_location}!

⚠️  THIS IS PUBLIC - ALL PLAYERS WILL SEE YOUR STATEMENT!

Share what you know, ask questions, make accusations.

💡 Key Questions:
- Where were you when the kill happened?
- Who did you see in which rooms?
- Who was near {body_location}?
- Any suspicious movements or vent usage?

Strategy:
- Share your location trail and who you saw
- Question others about their movements
- Look for inconsistencies in alibis

Respond with JSON:
{{"type": "discuss", "statement": "<your public statement>"}}
OR {{"type": "discuss", "statement": ""}} to stay silent

Example: {{"type": "discuss", "statement": "I was in Weapons the whole time, Player 3 came from {body_location}!"}}"""


def get_vote_instruction(available_targets: List[int], discussion_summary: str) -> str:
    """Get instruction for voting phase.
    
    Args:
        available_targets: List of player IDs that can be voted for
        discussion_summary: Summary of what was discussed
        
    Returns:
        Instruction string with JSON format
    """
    return f"""VOTING PHASE - EJECT OR SKIP

Available players: {available_targets}

{discussion_summary}

Vote to eject a player OR skip.

⚠️  Ejected player is permanently removed!

Strategy:
- Vote based on evidence and suspicion
- Skip if uncertain (no consensus)
- Consider what others will vote

Respond with JSON:
{{"type": "vote", "target": <player_id>}}  to vote for someone
{{"type": "vote", "target": null}}  to SKIP/abstain

Example: {{"type": "vote", "target": {available_targets[0] if available_targets else 'null'}}}"""


# Mapping from action types to instruction functions
INSTRUCTION_BUILDERS = {
    "complete_task": get_crewmate_task_instruction,
    "kill": get_impostor_task_instruction,
    "call_emergency": get_crewmate_task_instruction,
    "report_body": get_crewmate_task_instruction,
    "discuss": get_emergency_discussion_instruction,
    "vote": get_vote_instruction,
}

