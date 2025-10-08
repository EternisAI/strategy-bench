"""Utility functions for LLM interface."""

import asyncio
import functools
import time
from typing import Callable, Any, Optional
from collections import deque
from datetime import datetime, timedelta

from sdb.core.utils import exponential_backoff
from sdb.core.exceptions import LLMError


def retry_with_backoff(max_retries: int = 3, base_delay: float = 0.1):
    """Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay for exponential backoff
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    # Add small delay between all calls to avoid rate limiting
                    if attempt > 0:
                        await asyncio.sleep(0.5)
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = exponential_backoff(attempt, base_delay=base_delay)
                        print(f"      ⚠️  Retry {attempt + 1}/{max_retries} after {delay:.1f}s: {str(e)[:100]}")
                        await asyncio.sleep(delay)
                    else:
                        break
            
            # If all retries failed
            raise LLMError(
                f"Failed after {max_retries} retries",
                details={"last_error": str(last_exception)}
            )
        
        return wrapper
    return decorator


class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, calls_per_minute: int = 60):
        """Initialize rate limiter.
        
        Args:
            calls_per_minute: Maximum number of calls per minute
        """
        self.calls_per_minute = calls_per_minute
        self.call_times = deque()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait if necessary to respect rate limit."""
        async with self.lock:
            now = datetime.now()
            
            # Remove calls older than 1 minute
            while self.call_times and self.call_times[0] < now - timedelta(minutes=1):
                self.call_times.popleft()
            
            # If at limit, wait
            if len(self.call_times) >= self.calls_per_minute:
                oldest_call = self.call_times[0]
                wait_until = oldest_call + timedelta(minutes=1)
                wait_seconds = (wait_until - now).total_seconds()
                
                if wait_seconds > 0:
                    await asyncio.sleep(wait_seconds)
                
                # Clean up again
                now = datetime.now()
                while self.call_times and self.call_times[0] < now - timedelta(minutes=1):
                    self.call_times.popleft()
            
            # Record this call
            self.call_times.append(now)


def rate_limit(calls_per_minute: int = 60):
    """Decorator for rate limiting async functions.
    
    Args:
        calls_per_minute: Maximum calls per minute
    """
    limiter = RateLimiter(calls_per_minute)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            await limiter.acquire()
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def estimate_tokens(text: str) -> int:
    """Estimate number of tokens in text.
    
    Uses simple heuristic: ~4 characters per token.
    For more accurate estimates, use tiktoken library.
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    return len(text) // 4


def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str
) -> float:
    """Calculate estimated cost for API call.
    
    Pricing is approximate and may not reflect actual costs.
    
    Args:
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        model: Model identifier
        
    Returns:
        Estimated cost in USD
    """
    # Pricing per 1M tokens (approximate, as of 2025)
    pricing = {
        "gpt-5": {"prompt": 5.0, "completion": 15.0},
        "gpt-4": {"prompt": 30.0, "completion": 60.0},
        "gpt-4o": {"prompt": 5.0, "completion": 15.0},
        "gpt-4o-mini": {"prompt": 0.15, "completion": 0.6},
        "claude-opus-4.1": {"prompt": 15.0, "completion": 75.0},
        "claude-sonnet-3.5": {"prompt": 3.0, "completion": 15.0},
        "claude-haiku": {"prompt": 0.25, "completion": 1.25},
        "gemini-2.5-pro": {"prompt": 1.25, "completion": 5.0},
        "gemini-1.5-pro": {"prompt": 1.25, "completion": 5.0},
        "gemini-1.5-flash": {"prompt": 0.075, "completion": 0.3},
    }
    
    # Extract base model name
    for model_key in pricing:
        if model_key in model.lower():
            prices = pricing[model_key]
            prompt_cost = (prompt_tokens / 1_000_000) * prices["prompt"]
            completion_cost = (completion_tokens / 1_000_000) * prices["completion"]
            return prompt_cost + completion_cost
    
    # Default pricing if model not found
    return (prompt_tokens / 1_000_000) * 1.0 + (completion_tokens / 1_000_000) * 3.0

