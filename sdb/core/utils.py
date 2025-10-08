"""Shared utility functions for Social Deduction Bench."""

import random
import numpy as np
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
import json
import hashlib


def seed_everything(seed: int) -> None:
    """Set seed for reproducibility across all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    # If using torch in the future:
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


def get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()


def get_unix_timestamp() -> float:
    """Get current Unix timestamp."""
    return time.time()


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with update taking precedence."""
    result = base.copy()
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def generate_game_id(prefix: str = "game") -> str:
    """Generate a unique game ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    return f"{prefix}_{timestamp}_{random_suffix}"


def chunks(lst: List[Any], n: int) -> List[List[Any]]:
    """Split list into chunks of size n."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def flatten(nested_list: List[List[Any]]) -> List[Any]:
    """Flatten a nested list."""
    return [item for sublist in nested_list for item in sublist]


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """Safely dump object to JSON, handling datetime and other non-serializable types."""
    
    def default_handler(o):
        if isinstance(o, datetime):
            return o.isoformat()
        if hasattr(o, "__dict__"):
            return o.__dict__
        if hasattr(o, "to_dict"):
            return o.to_dict()
        return str(o)
    
    return json.dumps(obj, default=default_handler, **kwargs)


def safe_json_loads(s: str) -> Any:
    """Safely load JSON string."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def truncate_string(s: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate string to max length."""
    if len(s) <= max_length:
        return s
    return s[:max_length - len(suffix)] + suffix


def percentage(numerator: float, denominator: float, decimals: int = 2) -> float:
    """Calculate percentage with safe division."""
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, decimals)


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def exponential_backoff(
    attempt: int,
    base_delay: float = 0.1,
    max_delay: float = 60.0,
    exponential_base: float = 2.0
) -> float:
    """Calculate exponential backoff delay."""
    delay = min(base_delay * (exponential_base ** attempt), max_delay)
    # Add jitter
    jitter = random.uniform(0, delay * 0.1)
    return delay + jitter

