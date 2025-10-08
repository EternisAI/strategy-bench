"""Custom exceptions for Social Deduction Bench."""


class SDBException(Exception):
    """Base exception for all SDB errors."""
    
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class InvalidActionError(SDBException):
    """Raised when an invalid action is attempted."""
    
    pass


class InvalidStateError(SDBException):
    """Raised when the game state is invalid or inconsistent."""
    
    pass


class AgentError(SDBException):
    """Raised when an agent fails or behaves incorrectly."""
    
    pass


class EnvironmentError(SDBException):
    """Raised when an environment encounters an error."""
    
    pass


class ConfigurationError(SDBException):
    """Raised when configuration is invalid."""
    
    pass


class LLMError(SDBException):
    """Raised when LLM API calls fail."""
    
    pass


class TournamentError(SDBException):
    """Raised when tournament execution fails."""
    
    pass


class EvaluationError(SDBException):
    """Raised when evaluation fails."""
    
    pass

