"""Configuration for Spyfall game."""

from dataclasses import dataclass, field
from typing import Dict, List


# Default locations and roles for Spyfall (from original game)
# Expanded location set for more challenging gameplay
DEFAULT_LOCATIONS = [
    # Sci-Fi & Future
    "Lunar Base", "Space Station", "Alien Laboratory", "Time Machine", "Robot Factory",
    # Historical & Cultural  
    "Leonardo's Studio", "The Shaolin Temple", "Roman Senate", "Medieval Castle", "Ancient Library",
    # Adventure & Exploration
    "Western Saloon", "Pirate Ship", "Desert Oasis", "Mountain Cabin", "Submarine",
    # Modern & Urban
    "Corporate Office", "Shopping Mall", "Airport", "Hospital", "University Campus",
    # Entertainment & Leisure
    "Movie Theater", "Casino", "Sports Stadium", "Beach Resort", "Art Gallery",
    # Fantasy & Magic
    "Wizard Tower", "Dragon's Lair", "Enchanted Forest", "Crystal Cave", "Magic Academy"
]

DEFAULT_ROLES_BY_LOCATION = {
    # Sci-Fi & Future
    "Lunar Base": ["Commander","Engineer","Pilot","Scientist","Medic","Navigator","Technician"],
    "Space Station": ["Captain","Engineer","Communications Officer","Biologist","Security Chief","Maintenance Worker","Diplomat"],
    "Alien Laboratory": ["Head Scientist","Research Assistant","Test Subject","Xenobiologist","Lab Technician","Security Guard","Janitor"],
    "Time Machine": ["Inventor","Time Traveler","Historian","Mechanic","Security Agent","Tourist","Accident Victim"],
    "Robot Factory": ["Lead Engineer","Assembly Worker","Quality Inspector","Designer","Maintenance Tech","Security Officer","Delivery Driver"],
    
    # Historical & Cultural
    "Leonardo's Studio": ["Master Artist","Apprentice","Wealthy Patron","Portrait Model","Inventor","Curious Visitor","Art Restorer"],
    "The Shaolin Temple": ["Grand Master","Young Monk","Temple Cook","Martial Arts Student","Herbalist","Temple Guardian","Pilgrim"],
    "Roman Senate": ["Senator","Emperor","Consul","Scribe","Guard","Citizen","Foreign Ambassador"],
    "Medieval Castle": ["King","Knight","Court Jester","Cook","Blacksmith","Servant","Visiting Noble"],
    "Ancient Library": ["Head Librarian","Scholar","Scribe","Book Collector","Student","Guard","Cleaning Staff"],
    
    # Adventure & Exploration  
    "Western Saloon": ["Bartender","Gambler","Piano Player","Sheriff","Outlaw","Gold Prospector","Saloon Girl"],
    "Pirate Ship": ["Captain","First Mate","Navigator","Ship's Cook","Gunner","Cabin Boy","Prisoner"],
    "Desert Oasis": ["Caravan Leader","Desert Guide","Merchant","Traveler","Water Bearer","Nomad","Lost Wanderer"],
    "Mountain Cabin": ["Hermit","Hunter","Park Ranger","Hiker","Survivalist","Wildlife Photographer","Lost Tourist"],
    "Submarine": ["Captain","Sonar Operator","Engineer","Cook","Torpedo Specialist","Communications Officer","Mechanic"],
    
    # Modern & Urban
    "Corporate Office": ["CEO","Manager","Accountant","Secretary","IT Specialist","Security Guard","Janitor"],
    "Shopping Mall": ["Store Manager","Cashier","Security Guard","Shopper","Food Court Worker","Maintenance Person","Lost Child"],
    "Airport": ["Pilot","Flight Attendant","Air Traffic Controller","Security Officer","Baggage Handler","Passenger","Customs Agent"],
    "Hospital": ["Doctor","Nurse","Surgeon","Patient","Receptionist","Ambulance Driver","Hospital Administrator"],
    "University Campus": ["Professor","Student","Dean","Librarian","Campus Security","Maintenance Worker","Visiting Lecturer"],
    
    # Entertainment & Leisure
    "Movie Theater": ["Projectionist","Ticket Taker","Concession Worker","Movie Director","Actor","Audience Member","Janitor"],
    "Casino": ["Dealer","Pit Boss","Security Guard","High Roller","Cocktail Waitress","Slot Machine Technician","Comp Host"],
    "Sports Stadium": ["Coach","Star Player","Referee","Sports Announcer","Concession Worker","Stadium Security","Enthusiastic Fan"],
    "Beach Resort": ["Resort Manager","Lifeguard","Beach Volleyball Player","Tourist","Bartender","Hotel Maid","Surf Instructor"],
    "Art Gallery": ["Gallery Owner","Curator","Famous Artist","Art Critic","Security Guard","Visitor","Art Student"],
    
    # Fantasy & Magic
    "Wizard Tower": ["Archmage","Apprentice Wizard","Familiar","Spellbook Collector","Tower Guardian","Magic Student","Enchanted Servant"],
    "Dragon's Lair": ["Ancient Dragon","Dragon Hunter","Treasure Seeker","Captured Knight","Dragon Keeper","Brave Rescuer","Greedy Thief"],
    "Enchanted Forest": ["Forest Guardian","Woodland Elf","Lost Traveler","Herbalist","Magical Creature","Fairy","Nature Spirit"],
    "Crystal Cave": ["Crystal Miner","Cave Explorer","Geologist","Crystal Collector","Cave Guide","Treasure Hunter","Lost Spelunker"],
    "Magic Academy": ["Headmaster","Magic Teacher","Talented Student","School Librarian","Groundskeeper","Visiting Dignitary","New Recruit"]
}


@dataclass
class SpyfallConfig:
    """Configuration for Spyfall game.
    
    Attributes:
        n_players: Number of players (3-8 recommended)
        locations: List of possible locations
        roles_by_location: Mapping of location to possible roles
        max_turns: Maximum number of Q&A turns
        dealer_index: Index of player who asks first
        allow_followups: Whether to allow follow-up questions (default: False per rulebook)
    """
    n_players: int = 5
    locations: List[str] = field(default_factory=lambda: DEFAULT_LOCATIONS.copy())
    roles_by_location: Dict[str, List[str]] = field(
        default_factory=lambda: DEFAULT_ROLES_BY_LOCATION.copy()
    )
    max_turns: int = 24  # ~12 Q&A pairs
    dealer_index: int = 0
    allow_followups: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        if self.n_players < 3:
            raise ValueError("Spyfall requires at least 3 players")
        if self.n_players > 12:
            raise ValueError("Spyfall supports max 12 players")
        
        if not self.locations:
            raise ValueError("Must provide at least one location")
        
        for loc in self.locations:
            if loc not in self.roles_by_location:
                raise ValueError(f"No roles defined for location: {loc}")
            if not self.roles_by_location[loc]:
                raise ValueError(f"Must provide at least one role for location: {loc}")
        
        if self.max_turns < 1:
            raise ValueError("max_turns must be positive")
        
        if not (0 <= self.dealer_index < self.n_players):
            raise ValueError("dealer_index must be valid player index")

