class BookAssistantError(Exception):
    """Base exception for BookAssistant application"""
    pass

class FileNotFoundError(BookAssistantError):
    """Raised when a file cannot be found"""
    pass

class ProcessingError(BookAssistantError):
    """Raised when there's an error processing a book"""
    pass

class SearchError(BookAssistantError):
    """Raised when there's an error during search"""
    pass

class SummaryError(BookAssistantError):
    """Raised when there's an error generating summary"""
    pass