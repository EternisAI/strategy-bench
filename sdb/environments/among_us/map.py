"""Spatial map system for Among Us - Skeld ship layout.

This module provides the spatial graph representation of the Among Us spaceship,
including rooms, corridors, vents, and tasks locations.
"""

from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field

# --- Canonical Skeld data (right side has NO vents) ---
ROOM_DATA = {
    "Cafeteria": {"tasks": ["Download Data","Empty Garbage","Fix Wiring"],
                  "vent_connections": [], "special_actions": ["Emergency Button"]},
    "Weapons": {"tasks": ["Accept Diverted Power","Clear Asteroids","Download Data"],
                "vent_connections": [], "special_actions": []},
    "Navigation": {"tasks": ["Accept Diverted Power","Chart Course","Download Data","Fix Wiring","Stabilize Steering"],
                   "vent_connections": [], "special_actions": []},
    "O2": {"tasks": ["Clean O2 Filter","Empty Chute","Accept Diverted Power"],
           "vent_connections": [], "special_actions": ["Oxygen Depleted"]},
    "Shields": {"tasks": ["Accept Diverted Power","Prime Shields"],
                "vent_connections": [], "special_actions": []},
    "Communications": {"tasks": ["Accept Diverted Power","Download Data"],
                       "vent_connections": [], "special_actions": ["Comms Sabotaged"]},
    "Storage": {"tasks": ["Empty Garbage","Empty Chute"],
                "vent_connections": [], "special_actions": []},
    "Admin": {"tasks": ["Fix Wiring","Swipe Card","Upload Data"],
              "vent_connections": [], "special_actions": ["Admin Map"]},
    # Left-side vent triangles only:
    "Electrical": {"tasks": ["Calibrate Distributor","Divert Power","Download Data","Fix Wiring"],
                   "vent_connections": ["Medbay","Security"], "special_actions": ["Fix Lights"]},
    "Lower Engine": {"tasks": ["Accept Diverted Power","Align Engine Output","Fuel Engines"],
                     "vent_connections": ["Reactor","Upper Engine"], "special_actions": []},
    "Security": {"tasks": ["Accept Diverted Power","Fix Wiring"],
                 "vent_connections": ["Electrical","Medbay"], "special_actions": ["Security Cameras"]},
    "Reactor": {"tasks": ["Start Reactor","Unlock Manifolds"],
                "vent_connections": ["Upper Engine","Lower Engine"], "special_actions": ["Reactor Meltdown"]},
    "Upper Engine": {"tasks": ["Accept Diverted Power","Align Engine Output","Fuel Engines"],
                     "vent_connections": ["Reactor","Lower Engine"], "special_actions": []},
    "Medbay": {"tasks": ["Inspect Sample","Submit Scan"],
               "vent_connections": ["Electrical","Security"], "special_actions": ["Medbay Scan"]},
}

# --- Corridors (bi-directional). Remove invalid edges like O2↔Admin, Admin↔Electrical, Security↔Upper Engine ---
CORRIDOR_CONNECTIONS = [
    ("Cafeteria","Weapons"), ("Cafeteria","Admin"), ("Cafeteria","Upper Engine"), ("Cafeteria","Medbay"),
    ("Weapons","Navigation"), ("Weapons","O2"),
    ("Navigation","O2"),
    ("O2","Shields"),
    ("Shields","Communications"), ("Shields","Storage"),
    ("Communications","Storage"),
    ("Storage","Admin"), ("Storage","Electrical"), ("Storage","Lower Engine"),
    ("Electrical","Lower Engine"),
    ("Lower Engine","Security"), ("Lower Engine","Reactor"),
    ("Reactor","Security"), ("Reactor","Upper Engine"),
    ("Upper Engine","Medbay"),
]


@dataclass
class Room:
    """Represents a room in the spaceship."""
    
    name: str
    tasks: List[str] = field(default_factory=list)
    vent_connections: List[str] = field(default_factory=list)
    special_actions: List[str] = field(default_factory=list)
    players: Set[int] = field(default_factory=set)  # Player IDs in this room
    
    def add_player(self, player_id: int) -> None:
        """Add a player to this room."""
        self.players.add(player_id)
    
    def remove_player(self, player_id: int) -> None:
        """Remove a player from this room."""
        self.players.discard(player_id)
    
    def has_task(self, task_name: str) -> bool:
        """Check if a task can be done in this room."""
        return task_name in self.tasks
    
    def has_vent(self) -> bool:
        """Check if this room has a vent."""
        return len(self.vent_connections) > 0


class SpaceshipMap:
    """Manages the spatial graph of the Among Us spaceship.
    
    Features:
    - Room-based spatial graph
    - Corridor connections for movement
    - Vent connections for impostor fast-travel
    - Task locations
    - Player location tracking
    """
    
    def __init__(self):
        """Initialize the spaceship map."""
        # Build rooms
        self.rooms: Dict[str, Room] = {}
        for room_name, data in ROOM_DATA.items():
            self.rooms[room_name] = Room(
                name=room_name,
                tasks=data["tasks"],
                vent_connections=data["vent_connections"],
                special_actions=data["special_actions"],
            )
        
        # Build connectivity graph
        self.corridor_graph: Dict[str, Set[str]] = {room: set() for room in self.rooms}
        for room1, room2 in CORRIDOR_CONNECTIONS:
            self.corridor_graph[room1].add(room2)
            self.corridor_graph[room2].add(room1)
        
        # In __init__, force symmetric vent graph (prevents one-way typos)
        self.vent_graph = {room: set() for room in self.rooms}
        for room_name, room in self.rooms.items():
            for dst in room.vent_connections:
                if dst in self.rooms:
                    self.vent_graph[room_name].add(dst)
                    self.vent_graph[dst].add(room_name)
    
    def get_room(self, room_name: str) -> Optional[Room]:
        """Get a room by name."""
        return self.rooms.get(room_name)
    
    def get_adjacent_rooms(self, room_name: str) -> List[str]:
        """Get rooms connected via corridors.
        
        Args:
            room_name: Name of the room
            
        Returns:
            List of adjacent room names
        """
        if room_name not in self.corridor_graph:
            return []
        return list(self.corridor_graph[room_name])
    
    def get_vent_destinations(self, room_name: str) -> List[str]:
        """Get rooms connected via vents (impostor only).
        
        Args:
            room_name: Name of the room
            
        Returns:
            List of vent-connected room names
        """
        if room_name not in self.vent_graph:
            return []
        return list(self.vent_graph[room_name])
    
    
    def get_players_in_room(self, room_name: str, alive_only: bool = True) -> Set[int]:
        """Get all players in a room.
        
        Args:
            room_name: Name of the room
            alive_only: Only return alive players (ignored here, handled by env)
            
        Returns:
            Set of player IDs in the room
        """
        room = self.rooms.get(room_name)
        return room.players.copy() if room else set()
    
    def get_visible_players(self, room_name: str, player_id: int) -> Set[int]:
        """Get players visible from a room (same room).
        
        Args:
            room_name: Name of the room
            player_id: ID of the observing player
            
        Returns:
            Set of visible player IDs (excluding self)
        """
        room = self.rooms.get(room_name)
        if not room:
            return set()
        
        visible = room.players.copy()
        visible.discard(player_id)  # Don't see yourself
        return visible
    
    def can_move_via_corridor(self, from_room: str, to_room: str) -> bool:
        """Check if two rooms are connected via corridor.
        
        Args:
            from_room: Starting room
            to_room: Destination room
            
        Returns:
            True if connected
        """
        return to_room in self.corridor_graph.get(from_room, set())
    
    def can_move_via_vent(self, from_room: str, to_room: str) -> bool:
        """Check if two rooms are connected via vent.
        
        Args:
            from_room: Starting room (must have vent)
            to_room: Destination room
            
        Returns:
            True if connected via vent
        """
        return to_room in self.vent_graph.get(from_room, set())
    
    def get_all_room_names(self) -> List[str]:
        """Get list of all room names."""
        return list(self.rooms.keys())
    
    def get_tasks_in_room(self, room_name: str) -> List[str]:
        """Get all tasks available in a room.
        
        Args:
            room_name: Name of the room
            
        Returns:
            List of task names
        """
        room = self.rooms.get(room_name)
        return room.tasks.copy() if room else []
    
    def reset(self) -> None:
        """Reset the map (clear all player locations)."""
        for room in self.rooms.values():
            room.players.clear()
    
    def get_player_location(self, player_id: int) -> Optional[str]:
        """Find which room a player is in.
        
        Args:
            player_id: ID of the player
            
        Returns:
            Room name or None if not found
        """
        for room_name, room in self.rooms.items():
            if player_id in room.players:
                return room_name
        return None

    # --- Validated movement APIs inside SpaceshipMap ---
    def allowed_corridor_neighbors(self, room: str) -> List[str]:
        """Get allowed corridor neighbors for a room."""
        return sorted(self.corridor_graph.get(room, set()))
    
    def allowed_vent_neighbors(self, room: str) -> List[str]:
        """Get allowed vent neighbors for a room."""
        return sorted(self.vent_graph.get(room, set()))

    def can_move_via_corridor(self, frm: str, to: str) -> bool:
        """Check if movement via corridor is allowed."""
        return to in self.corridor_graph.get(frm, set())

    def can_move_via_vent(self, frm: str, to: str) -> bool:
        """Check if movement via vent is allowed."""
        return to in self.vent_graph.get(frm, set())

    def move_player(self, player_id: int, frm: str, to: str) -> bool:
        """Move a player via corridor with validation."""
        if not self.can_move_via_corridor(frm, to): 
            return False
        if frm in self.rooms: 
            self.rooms[frm].remove_player(player_id)
        self.rooms[to].add_player(player_id)
        return True

    def vent_player(self, player_id: int, frm: str, to: str) -> bool:
        """Move a player via vent with validation."""
        if not self.can_move_via_vent(frm, to): 
            return False
        if frm in self.rooms: 
            self.rooms[frm].remove_player(player_id)
        self.rooms[to].add_player(player_id)
        return True

