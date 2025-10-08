"""Base LLM client interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class LLMResponse:
    """Response from an LLM API call."""
    
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def cost(self) -> float:
        """Estimate cost in USD (if pricing info available in metadata)."""
        if "cost" in self.metadata:
            return self.metadata["cost"]
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "finish_reason": self.finish_reason,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class BaseLLMClient(ABC):
    """Abstract base class for LLM API clients."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 256,
        **kwargs
    ):
        """Initialize LLM client.
        
        Args:
            api_key: API key for the service
            model: Model identifier
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.config = kwargs
        
        # Statistics
        self.total_calls = 0
        self.total_tokens = 0
        self.total_cost = 0.0
    
    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Make a chat completion API call.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse object
        """
        pass
    
    def chat_completion_sync(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Synchronous chat completion (for backward compatibility).
        
        Default implementation raises NotImplementedError.
        Override in subclasses if sync support is needed.
        """
        raise NotImplementedError("Synchronous calls not supported. Use async version.")
    
    def _update_stats(self, response: LLMResponse) -> None:
        """Update client statistics."""
        self.total_calls += 1
        self.total_tokens += response.total_tokens
        self.total_cost += response.cost
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "model": self.model,
        }
    
    def reset_stats(self) -> None:
        """Reset client statistics."""
        self.total_calls = 0
        self.total_tokens = 0
        self.total_cost = 0.0

