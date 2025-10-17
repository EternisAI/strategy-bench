"""Game logger for tracking events and actions."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from sdb.logging.formats import LogEntry, EventType
from sdb.core.utils import generate_game_id


class GameLogger:
    """Logger for game events and actions.
    
    Handles both in-memory and file-based logging with support for
    filtering private information.
    """
    
    def __init__(
        self,
        game_id: Optional[str] = None,
        output_dir: Optional[Path] = None,
        log_private: bool = True,
        enabled: bool = True,
    ):
        """Initialize game logger.
        
        Args:
            game_id: Unique game identifier
            output_dir: Directory to save logs (None for memory-only)
            log_private: Whether to log private information (default: True)
            enabled: Whether logging is enabled
        """
        self.game_id = game_id or generate_game_id()
        self.output_dir = Path(output_dir) if output_dir else None
        self.log_private = log_private
        self.enabled = enabled
        
        # In-memory log
        self.entries: List[LogEntry] = []
        
        # Current round number
        self.current_round = 0
        
        # Create output directory if needed
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = self.output_dir / f"{self.game_id}.jsonl"
        else:
            self.log_file = None
    
    def log(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        player_id: Optional[int] = None,
        is_private: bool = False,
        **metadata
    ) -> None:
        """Log an event.
        
        Args:
            event_type: Type of event
            data: Event data
            player_id: Player associated with event (if any)
            is_private: Whether this is private information
            **metadata: Additional metadata
        """
        if not self.enabled:
            return
        
        # Skip private events if not logging them
        if is_private and not self.log_private:
            return
        
        entry = LogEntry(
            timestamp=datetime.now(),
            event_type=event_type,
            game_id=self.game_id,
            round_number=self.current_round,
            data=data,
            player_id=player_id,
            is_private=is_private,
            metadata=metadata
        )
        
        # Store in memory
        self.entries.append(entry)
        
        # Write to file if configured
        if self.log_file:
            self._write_to_file(entry)
    
    def _write_to_file(self, entry: LogEntry) -> None:
        """Write entry to log file.
        
        Args:
            entry: Log entry to write
        """
        try:
            with open(self.log_file, 'a') as f:
                f.write(entry.to_json() + '\n')
        except Exception as e:
            print(f"Warning: Failed to write log entry: {e}")
    
    def log_game_start(self, config: Dict[str, Any]) -> None:
        """Log game start.
        
        Args:
            config: Game configuration
        """
        self.log(EventType.GAME_START, {"config": config})
    
    def log_game_end(self, winner: Any, reason: str, stats: Dict[str, Any]) -> None:
        """Log game end.
        
        Args:
            winner: Winner identifier
            reason: Win reason
            stats: Game statistics
        """
        self.log(
            EventType.GAME_END,
            {"winner": winner, "reason": reason, "stats": stats}
        )
    
    def log_phase_change(self, old_phase: str, new_phase: str) -> None:
        """Log phase change.
        
        Args:
            old_phase: Previous phase
            new_phase: New phase
        """
        self.log(
            EventType.PHASE_CHANGE,
            {"old_phase": old_phase, "new_phase": new_phase}
        )
    
    def log_round_start(self, round_number: int) -> None:
        """Log round start.
        
        Args:
            round_number: Round number
        """
        self.current_round = round_number
        self.log(EventType.ROUND_START, {"round": round_number})
    
    def log_action(
        self,
        player_id: int,
        action_type: str,
        target: Optional[int] = None,
        data: Optional[Dict[str, Any]] = None,
        is_private: bool = False
    ) -> None:
        """Log player action.
        
        Args:
            player_id: Player taking action
            action_type: Type of action
            target: Target of action (if any)
            data: Additional action data
            is_private: Whether action is private
        """
        self.log(
            EventType.PLAYER_ACTION,
            {
                "action_type": action_type,
                "target": target,
                "data": data or {}
            },
            player_id=player_id,
            is_private=is_private
        )
    
    def log_llm_call(
        self,
        player_id: int,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        response: Optional[str] = None
    ) -> None:
        """Log LLM API call.
        
        Args:
            player_id: Player making the call
            model: Model used
            prompt_tokens: Tokens in prompt
            completion_tokens: Tokens in completion
            response: Response text (optional for privacy)
        """
        self.log(
            EventType.LLM_CALL,
            {
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "response": response if self.log_private else None
            },
            player_id=player_id
        )
    
    def log_error(self, error_type: str, message: str, details: Optional[Dict] = None) -> None:
        """Log error.
        
        Args:
            error_type: Type of error
            message: Error message
            details: Additional details
        """
        self.log(
            EventType.ERROR,
            {
                "error_type": error_type,
                "message": message,
                "details": details or {}
            }
        )
    
    def get_entries(
        self,
        event_type: Optional[EventType] = None,
        player_id: Optional[int] = None,
        include_private: bool = False
    ) -> List[LogEntry]:
        """Get log entries with optional filtering.
        
        Args:
            event_type: Filter by event type
            player_id: Filter by player ID
            include_private: Include private entries
            
        Returns:
            Filtered list of log entries
        """
        entries = self.entries
        
        if event_type:
            entries = [e for e in entries if e.event_type == event_type]
        
        if player_id is not None:
            entries = [e for e in entries if e.player_id == player_id]
        
        if not include_private:
            entries = [e for e in entries if not e.is_private]
        
        return entries
    
    def export_to_json(self, filepath: Path, include_private: bool = False) -> None:
        """Export logs to JSON file.
        
        Args:
            filepath: Output file path
            include_private: Include private entries
        """
        entries = self.get_entries(include_private=include_private)
        data = [entry.to_dict() for entry in entries]
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "game_id": self.game_id,
            "total_entries": len(self.entries),
            "current_round": self.current_round,
            "private_entries": sum(1 for e in self.entries if e.is_private),
            "event_type_counts": self._count_event_types(),
        }
    
    def _count_event_types(self) -> Dict[str, int]:
        """Count entries by event type.
        
        Returns:
            Dictionary mapping event type to count
        """
        counts = {}
        for entry in self.entries:
            event_name = entry.event_type.name
            counts[event_name] = counts.get(event_name, 0) + 1
        return counts
    
    def clear(self) -> None:
        """Clear all log entries."""
        self.entries.clear()
        self.current_round = 0

