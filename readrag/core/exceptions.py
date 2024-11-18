class BookAssistantError(Exception):
    pass

class FileNotFoundError(BookAssistantError):
    pass

class ProcessingError(BookAssistantError):
    pass

class SearchError(BookAssistantError):
    pass

class SummaryError(BookAssistantError):
    pass