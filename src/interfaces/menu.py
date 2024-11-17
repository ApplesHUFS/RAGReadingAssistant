from typing import Dict, Any, Callable

class Menu:
    def __init__(self):
        self.options: Dict[str, Dict[str, Any]] = {}
        
    def add_option(self, key: str, description: str, handler: Callable):
        self.options[key] = {
            'description': description,
            'handler': handler
        }
        
    def display(self, title: str):
        print(f"\n=== {title} ===")
        for key, option in self.options.items():
            print(f"{key}. {option['description']}")
            
    def handle(self, choice: str) -> bool:
        if choice in self.options:
            self.options[choice]['handler']()
            return True
        return False