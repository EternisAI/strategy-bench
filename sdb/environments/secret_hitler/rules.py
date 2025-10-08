"""Secret Hitler game rules and validation."""

from typing import List
from sdb.environments.secret_hitler.types import Policy, Party
import random


class PolicyDeck:
    """Manages the policy deck."""
    
    def __init__(self, seed: int = None):
        """Initialize deck with 6 Liberal and 11 Fascist policies."""
        self.rng = random.Random(seed)
        self.deck: List[Policy] = []
        self.discard: List[Policy] = []
        self.reset()
    
    def reset(self):
        """Reset deck to initial state."""
        self.deck = (
            [Policy.LIBERAL] * 6 +
            [Policy.FASCIST] * 11
        )
        self.rng.shuffle(self.deck)
        self.discard = []
    
    def draw(self, n: int = 3) -> List[Policy]:
        """Draw n policies from the deck.
        
        Args:
            n: Number of policies to draw
            
        Returns:
            List of drawn policies
        """
        # Reshuffle if needed
        if len(self.deck) < n:
            self.deck.extend(self.discard)
            self.discard = []
            self.rng.shuffle(self.deck)
        
        drawn = self.deck[:n]
        self.deck = self.deck[n:]
        return drawn
    
    def discard_policy(self, policy: Policy):
        """Discard a policy.
        
        Args:
            policy: Policy to discard
        """
        self.discard.append(policy)
    
    def peek_top(self, n: int = 3) -> List[Policy]:
        """Peek at top n policies without drawing.
        
        Args:
            n: Number of policies to peek at
            
        Returns:
            List of policies (does not modify deck)
        """
        return self.deck[:n]
    
    def policies_remaining(self) -> int:
        """Get number of policies remaining in deck."""
        return len(self.deck)


class GameRules:
    """Secret Hitler game rules and win conditions."""
    
    @staticmethod
    def check_liberal_victory(liberal_policies: int, hitler_eliminated: bool) -> bool:
        """Check if Liberals have won.
        
        Args:
            liberal_policies: Number of Liberal policies enacted
            hitler_eliminated: Whether Hitler was eliminated
            
        Returns:
            True if Liberals won
        """
        return liberal_policies >= 5 or hitler_eliminated
    
    @staticmethod
    def check_fascist_victory(fascist_policies: int, hitler_chancellor: bool) -> bool:
        """Check if Fascists have won.
        
        Args:
            fascist_policies: Number of Fascist policies enacted
            hitler_chancellor: Whether Hitler was elected Chancellor after 3 Fascist policies
            
        Returns:
            True if Fascists won
        """
        return fascist_policies >= 6 or hitler_chancellor
    
    @staticmethod
    def check_term_limits(
        president_id: int,
        chancellor_id: int,
        last_president: int,
        last_chancellor: int,
        alive_count: int
    ) -> bool:
        """Check if government respects term limits.
        
        Args:
            president_id: Proposed president
            chancellor_id: Proposed chancellor
            last_president: Previous president
            last_chancellor: Previous chancellor
            alive_count: Number of alive players
            
        Returns:
            True if term limits are respected
        """
        # President cannot nominate themselves
        if president_id == chancellor_id:
            return False
        
        # With >5 players, last president and chancellor are both ineligible
        if alive_count > 5:
            if chancellor_id == last_president or chancellor_id == last_chancellor:
                return False
        else:
            # With ≤5 players, only last chancellor is ineligible
            if chancellor_id == last_chancellor:
                return False
        
        return True
    
    @staticmethod
    def check_election_passed(ja_votes: int, total_votes: int) -> bool:
        """Check if election passed (majority Ja).
        
        Args:
            ja_votes: Number of Ja votes
            total_votes: Total number of votes
            
        Returns:
            True if election passed
        """
        return ja_votes > total_votes // 2
    
    @staticmethod
    def veto_available(fascist_policies: int) -> bool:
        """Check if veto power is available.
        
        Args:
            fascist_policies: Number of Fascist policies enacted
            
        Returns:
            True if veto is available (5+ Fascist policies)
        """
        return fascist_policies >= 5

