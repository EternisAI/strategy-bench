"""Belief tracking for reasoning about other players."""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Belief:
    """A belief about the game state or another player."""
    
    subject: Any  # What the belief is about (player ID, game state, etc.)
    predicate: str  # What is believed (e.g., "is_impostor", "is_trustworthy")
    confidence: float  # Confidence level (0.0-1.0)
    evidence: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_confidence(self, new_evidence: str, delta: float) -> None:
        """Update belief confidence based on new evidence.
        
        Args:
            new_evidence: Description of new evidence
            delta: Change in confidence (-1.0 to 1.0)
        """
        self.confidence = max(0.0, min(1.0, self.confidence + delta))
        self.evidence.append(new_evidence)
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class BeliefTracker:
    """Tracks beliefs about game state and other players."""
    
    def __init__(self):
        """Initialize belief tracker."""
        # Map from (subject, predicate) to Belief
        self.beliefs: Dict[tuple, Belief] = {}
        self.history: List[Belief] = []
    
    def add_belief(
        self,
        subject: Any,
        predicate: str,
        confidence: float,
        evidence: Optional[List[str]] = None,
        **metadata
    ) -> Belief:
        """Add or update a belief.
        
        Args:
            subject: What the belief is about
            predicate: What is believed
            confidence: Confidence level (0.0-1.0)
            evidence: List of evidence strings
            **metadata: Additional metadata
            
        Returns:
            The created or updated Belief
        """
        key = (subject, predicate)
        
        if key in self.beliefs:
            # Update existing belief
            belief = self.beliefs[key]
            belief.confidence = confidence
            if evidence:
                belief.evidence.extend(evidence)
            belief.timestamp = datetime.now()
            belief.metadata.update(metadata)
        else:
            # Create new belief
            belief = Belief(
                subject=subject,
                predicate=predicate,
                confidence=confidence,
                evidence=evidence or [],
                metadata=metadata
            )
            self.beliefs[key] = belief
        
        # Track in history
        self.history.append(belief)
        
        return belief
    
    def update_belief(
        self,
        subject: Any,
        predicate: str,
        new_evidence: str,
        delta: float
    ) -> Optional[Belief]:
        """Update an existing belief.
        
        Args:
            subject: What the belief is about
            predicate: What is believed
            new_evidence: Description of new evidence
            delta: Change in confidence
            
        Returns:
            Updated Belief or None if not found
        """
        key = (subject, predicate)
        if key in self.beliefs:
            belief = self.beliefs[key]
            belief.update_confidence(new_evidence, delta)
            self.history.append(belief)
            return belief
        return None
    
    def get_belief(self, subject: Any, predicate: str) -> Optional[Belief]:
        """Get a specific belief.
        
        Args:
            subject: What the belief is about
            predicate: What is believed
            
        Returns:
            Belief or None if not found
        """
        return self.beliefs.get((subject, predicate))
    
    def get_beliefs_about(self, subject: Any) -> List[Belief]:
        """Get all beliefs about a subject.
        
        Args:
            subject: What to get beliefs about
            
        Returns:
            List of beliefs about the subject
        """
        return [
            belief for (subj, _), belief in self.beliefs.items()
            if subj == subject
        ]
    
    def get_high_confidence_beliefs(self, threshold: float = 0.7) -> List[Belief]:
        """Get beliefs with high confidence.
        
        Args:
            threshold: Minimum confidence threshold
            
        Returns:
            List of high-confidence beliefs
        """
        return [
            belief for belief in self.beliefs.values()
            if belief.confidence >= threshold
        ]
    
    def clear(self) -> None:
        """Clear all beliefs."""
        self.beliefs.clear()
        self.history.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "beliefs": [belief.to_dict() for belief in self.beliefs.values()],
            "history_size": len(self.history),
        }

