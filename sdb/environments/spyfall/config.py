"""Configuration for Spyfall game."""

from dataclasses import dataclass, field
from typing import Dict, List


# Default locations and roles for Spyfall
DEFAULT_LOCATIONS = [
    "airplane", "bank", "beach", "casino", "cathedral",
    "circus_tent", "corporate_party", "crusader_army", "day_spa",
    "embassy", "hospital", "hotel", "military_base", "movie_studio",
    "ocean_liner", "passenger_train", "pirate_ship", "polar_station",
    "police_station", "restaurant", "school", "service_station",
    "space_station", "submarine", "supermarket", "theater",
    "university", "world_war_ii_squad"
]

DEFAULT_ROLES_BY_LOCATION = {
    "airplane": ["First Class Passenger", "Air Marshal", "Mechanic", "Economy Class Passenger", "Stewardess", "Co-Pilot", "Captain"],
    "bank": ["Armored Car Driver", "Manager", "Consultant", "Customer", "Robber", "Security Guard", "Teller"],
    "beach": ["Beach Waitress", "Kite Surfer", "Lifeguard", "Thief", "Beach Goer", "Beach Photographer", "Ice Cream Man"],
    "casino": ["Bartender", "Head Security Guard", "Bouncer", "Manager", "Hustler", "Dealer", "Gambler"],
    "cathedral": ["Priest", "Beggar", "Sinner", "Parishioner", "Tourist", "Sponsor", "Choir Singer"],
    "circus_tent": ["Acrobat", "Animal Trainer", "Magician", "Visitor", "Fire Eater", "Clown", "Juggler"],
    "corporate_party": ["Entertainer", "Manager", "Unwanted Guest", "Owner", "Secretary", "Accountant", "Delivery Boy"],
    "crusader_army": ["Monk", "Imprisoned Arab", "Servant", "Bishop", "Squire", "Archer", "Knight"],
    "day_spa": ["Customer", "Stylist", "Masseuse", "Manicurist", "Makeup Artist", "Dermatologist", "Beautician"],
    "embassy": ["Security Guard", "Secretary", "Ambassador", "Government Official", "Tourist", "Refugee", "Diplomat"],
    "hospital": ["Nurse", "Doctor", "Anesthesiologist", "Intern", "Patient", "Therapist", "Surgeon"],
    "hotel": ["Doorman", "Security Guard", "Manager", "Housekeeper", "Customer", "Bartender", "Bellman"],
    "military_base": ["Deserter", "Colonel", "Medic", "Soldier", "Sniper", "Officer", "Tank Engineer"],
    "movie_studio": ["Stunt Man", "Sound Engineer", "Camera Man", "Director", "Costume Artist", "Actor", "Producer"],
    "ocean_liner": ["Rich Passenger", "Cook", "Captain", "Bartender", "Musician", "Waiter", "Mechanic"],
    "passenger_train": ["Mechanic", "Border Patrol", "Train Attendant", "Passenger", "Restaurant Chef", "Engineer", "Stoker"],
    "pirate_ship": ["Cook", "Sailor", "Slave", "Cannoneer", "Bound Prisoner", "Cabin Boy", "Brave Captain"],
    "polar_station": ["Medic", "Geologist", "Expedition Leader", "Biologist", "Radioman", "Hydrologist", "Meteorologist"],
    "police_station": ["Detective", "Lawyer", "Journalist", "Criminalist", "Archivist", "Patrol Officer", "Criminal"],
    "restaurant": ["Musician", "Customer", "Bouncer", "Hostess", "Head Chef", "Food Critic", "Waiter"],
    "school": ["Gym Teacher", "Student", "Principal", "Security Guard", "Janitor", "Lunch Lady", "Maintenance Man"],
    "service_station": ["Manager", "Tire Specialist", "Biker", "Car Owner", "Car Wash Operator", "Electrician", "Auto Mechanic"],
    "space_station": ["Engineer", "Alien", "Space Tourist", "Pilot", "Commander", "Scientist", "Doctor"],
    "submarine": ["Cook", "Commander", "Sonar Technician", "Electronics Technician", "Sailor", "Radioman", "Navigator"],
    "supermarket": ["Customer", "Cashier", "Butcher", "Janitor", "Security Guard", "Food Sample Demonstrator", "Shelf Stocker"],
    "theater": ["Coat Check Lady", "Prompter", "Cashier", "Director", "Actor", "Crew Man", "Audience Member"],
    "university": ["Graduate Student", "Professor", "Dean", "Psychologist", "Maintenance Man", "Janitor", "Student"],
    "world_war_ii_squad": ["Resistance Fighter", "Radioman", "Scout", "Medic", "Cook", "Imprisoned Soldier", "Soldier"]
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

