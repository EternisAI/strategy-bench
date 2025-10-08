"""Secret Hitler game configuration."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from sdb.environments.secret_hitler.types import PresidentialPower


@dataclass
class SecretHitlerConfig:
    """Configuration for Secret Hitler game.
    
    Based on official Secret Hitler rules for 5-10 players.
    """
    
    n_players: int
    seed: Optional[int] = None
    log_private_info: bool = False
    
    # Official game rules
    ROLE_DISTRIBUTION: Dict[int, Dict[str, int]] = None
    PRESIDENTIAL_POWERS: Dict[int, List[PresidentialPower]] = None
    
    def __post_init__(self):
        """Initialize rules based on player count."""
        if self.n_players < 5 or self.n_players > 10:
            raise ValueError(f"Secret Hitler requires 5-10 players, got {self.n_players}")
        
        # Official role distribution
        if self.ROLE_DISTRIBUTION is None:
            self.ROLE_DISTRIBUTION = {
                5: {"liberals": 3, "fascists": 1, "hitler": 1},
                6: {"liberals": 4, "fascists": 1, "hitler": 1},
                7: {"liberals": 4, "fascists": 2, "hitler": 1},
                8: {"liberals": 5, "fascists": 2, "hitler": 1},
                9: {"liberals": 5, "fascists": 3, "hitler": 1},
                10: {"liberals": 6, "fascists": 3, "hitler": 1},
            }
        
        # Presidential powers by number of players and fascist policy count
        # Format: {n_players: [power_at_1st, power_at_2nd, power_at_3rd, power_at_4th, power_at_5th, power_at_6th]}
        if self.PRESIDENTIAL_POWERS is None:
            self.PRESIDENTIAL_POWERS = {
                5: [
                    PresidentialPower.NONE,
                    PresidentialPower.NONE,
                    PresidentialPower.POLICY_PEEK,
                    PresidentialPower.EXECUTION,
                    PresidentialPower.EXECUTION,
                    PresidentialPower.NONE  # Game ends
                ],
                6: [
                    PresidentialPower.NONE,
                    PresidentialPower.NONE,
                    PresidentialPower.POLICY_PEEK,
                    PresidentialPower.EXECUTION,
                    PresidentialPower.EXECUTION,
                    PresidentialPower.NONE
                ],
                7: [
                    PresidentialPower.NONE,
                    PresidentialPower.INVESTIGATE_LOYALTY,
                    PresidentialPower.CALL_SPECIAL_ELECTION,
                    PresidentialPower.EXECUTION,
                    PresidentialPower.EXECUTION,
                    PresidentialPower.NONE
                ],
                8: [
                    PresidentialPower.NONE,
                    PresidentialPower.INVESTIGATE_LOYALTY,
                    PresidentialPower.CALL_SPECIAL_ELECTION,
                    PresidentialPower.EXECUTION,
                    PresidentialPower.EXECUTION,
                    PresidentialPower.NONE
                ],
                9: [
                    PresidentialPower.INVESTIGATE_LOYALTY,
                    PresidentialPower.INVESTIGATE_LOYALTY,
                    PresidentialPower.CALL_SPECIAL_ELECTION,
                    PresidentialPower.EXECUTION,
                    PresidentialPower.EXECUTION,
                    PresidentialPower.NONE
                ],
                10: [
                    PresidentialPower.INVESTIGATE_LOYALTY,
                    PresidentialPower.INVESTIGATE_LOYALTY,
                    PresidentialPower.CALL_SPECIAL_ELECTION,
                    PresidentialPower.EXECUTION,
                    PresidentialPower.EXECUTION,
                    PresidentialPower.NONE
                ],
            }
    
    def get_roles(self) -> Dict[str, int]:
        """Get role distribution for current player count."""
        return self.ROLE_DISTRIBUTION[self.n_players]
    
    def get_presidential_power(self, fascist_policy_count: int) -> PresidentialPower:
        """Get presidential power for current fascist policy count.
        
        Args:
            fascist_policy_count: Number of fascist policies enacted (0-6)
            
        Returns:
            Presidential power to execute
        """
        if fascist_policy_count < 0 or fascist_policy_count > 6:
            return PresidentialPower.NONE
        
        powers = self.PRESIDENTIAL_POWERS[self.n_players]
        return powers[fascist_policy_count] if fascist_policy_count < len(powers) else PresidentialPower.NONE

