"""Base memory interface for agents."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MemoryEntry:
    """Single memory entry."""
    
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
            "metadata": self.metadata,
        }


class BaseMemory(ABC):
    """Abstract base class for memory systems."""
    
    def __init__(self, capacity: Optional[int] = None):
        """Initialize memory.
        
        Args:
            capacity: Maximum number of entries (None for unlimited)
        """
        self.capacity = capacity
        self.entries: List[MemoryEntry] = []
    
    @abstractmethod
    def add(self, content: Any, importance: float = 1.0, **metadata) -> None:
        """Add a new memory entry.
        
        Args:
            content: Content to store
            importance: Importance score (0.0-1.0)
            **metadata: Additional metadata
        """
        pass
    
    @abstractmethod
    def retrieve(self, query: Optional[str] = None, k: int = 5) -> List[MemoryEntry]:
        """Retrieve relevant memories.
        
        Args:
            query: Query string for retrieval (None returns most recent)
            k: Number of memories to retrieve
            
        Returns:
            List of memory entries
        """
        pass
    
    def clear(self) -> None:
        """Clear all memories."""
        self.entries.clear()
    
    def size(self) -> int:
        """Get number of stored memories."""
        return len(self.entries)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary."""
        return {
            "capacity": self.capacity,
            "size": self.size(),
            "entries": [entry.to_dict() for entry in self.entries],
        }

