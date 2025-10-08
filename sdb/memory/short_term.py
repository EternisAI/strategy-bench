"""Short-term memory implementation with recency bias."""

from typing import Any, List, Optional
from collections import deque

from sdb.memory.base import BaseMemory, MemoryEntry


class ShortTermMemory(BaseMemory):
    """Short-term memory with FIFO eviction and recency bias.
    
    Suitable for storing recent observations and actions.
    """
    
    def __init__(self, capacity: int = 100):
        """Initialize short-term memory.
        
        Args:
            capacity: Maximum number of entries
        """
        super().__init__(capacity=capacity)
        self._deque = deque(maxlen=capacity)
    
    def add(self, content: Any, importance: float = 1.0, **metadata) -> None:
        """Add a new memory entry.
        
        Args:
            content: Content to store
            importance: Importance score (ignored in short-term memory)
            **metadata: Additional metadata
        """
        entry = MemoryEntry(
            content=content,
            importance=importance,
            metadata=metadata
        )
        self._deque.append(entry)
        self.entries = list(self._deque)
    
    def retrieve(self, query: Optional[str] = None, k: int = 5) -> List[MemoryEntry]:
        """Retrieve most recent memories.
        
        Args:
            query: Ignored in short-term memory (uses recency)
            k: Number of recent memories to retrieve
            
        Returns:
            List of k most recent memory entries
        """
        return list(self._deque)[-k:]
    
    def get_all(self) -> List[MemoryEntry]:
        """Get all memories in chronological order.
        
        Returns:
            List of all memory entries
        """
        return list(self._deque)
    
    def get_recent(self, n: int = 10) -> List[MemoryEntry]:
        """Get n most recent memories.
        
        Args:
            n: Number of recent memories
            
        Returns:
            List of n most recent entries
        """
        return self.retrieve(k=n)

