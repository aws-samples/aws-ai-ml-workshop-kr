"""
Custom state management system to replace LangGraph state functionality.
Provides thread-safe state management for multi-agent workflows.
"""

import copy
import threading
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class WorkflowState:
    """Represents the state of the workflow execution."""
    
    # Core workflow data
    messages: List[Dict[str, Any]] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    clues: str = ""
    full_plan: str = ""
    request: str = ""
    request_prompt: str = ""
    
    # Workflow control
    next_node: Optional[str] = None
    current_agent: str = ""
    messages_name: str = ""
    
    # Team and configuration
    team_members: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Custom fields for extensibility
    custom_data: Dict[str, Any] = field(default_factory=dict)
    
    def update(self, **kwargs) -> 'WorkflowState':
        """Update state fields and return updated state."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.custom_data[key] = value
        
        self.updated_at = datetime.now()
        return self
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get state value with fallback to custom_data."""
        if hasattr(self, key):
            return getattr(self, key)
        return self.custom_data.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        result = {}
        for key, value in self.__dict__.items():
            if key != 'custom_data':
                result[key] = value
        result.update(self.custom_data)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowState':
        """Create state from dictionary."""
        # Extract known fields
        known_fields = {
            field.name for field in cls.__dataclass_fields__.values()
        }
        
        state_data = {}
        custom_data = {}
        
        for key, value in data.items():
            if key in known_fields:
                state_data[key] = value
            else:
                custom_data[key] = value
        
        state = cls(**state_data)
        state.custom_data = custom_data
        return state


class StateManager:
    """Thread-safe state manager for workflow execution."""
    
    def __init__(self, initial_state: Optional[WorkflowState] = None):
        """Initialize state manager with optional initial state."""
        self._state = initial_state or WorkflowState()
        self._lock = threading.RLock()
        self._subscribers: List[Callable[[WorkflowState], None]] = []
    
    def get_state(self) -> WorkflowState:
        """Get current state (thread-safe copy)."""
        with self._lock:
            return copy.deepcopy(self._state)
    
    def update_state(self, **kwargs) -> WorkflowState:
        """Update state with given parameters (thread-safe)."""
        with self._lock:
            self._state.update(**kwargs)
            updated_state = copy.deepcopy(self._state)
            
            # Notify subscribers
            for subscriber in self._subscribers:
                try:
                    subscriber(updated_state)
                except Exception as e:
                    print(f"Error notifying subscriber: {e}")
            
            return updated_state
    
    def set_state(self, new_state: WorkflowState) -> None:
        """Replace entire state (thread-safe)."""
        with self._lock:
            self._state = copy.deepcopy(new_state)
    
    def get_field(self, key: str, default: Any = None) -> Any:
        """Get specific state field (thread-safe)."""
        with self._lock:
            return self._state.get(key, default)
    
    def append_to_list(self, field_name: str, item: Any) -> WorkflowState:
        """Append item to a list field (thread-safe)."""
        with self._lock:
            current_list = getattr(self._state, field_name, [])
            if not isinstance(current_list, list):
                current_list = []
                setattr(self._state, field_name, current_list)
            
            current_list.append(item)
            self._state.updated_at = datetime.now()
            
            return copy.deepcopy(self._state)
    
    def subscribe(self, callback: Callable[[WorkflowState], None]) -> None:
        """Subscribe to state changes."""
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable[[WorkflowState], None]) -> None:
        """Unsubscribe from state changes."""
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)
    
    def reset(self) -> None:
        """Reset state to initial values."""
        with self._lock:
            self._state = WorkflowState()
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize state to dictionary."""
        with self._lock:
            return self._state.to_dict()
    
    def deserialize(self, data: Dict[str, Any]) -> None:
        """Deserialize state from dictionary."""
        with self._lock:
            self._state = WorkflowState.from_dict(data)


# Global state manager instance (can be replaced with dependency injection)
_global_state_manager: Optional[StateManager] = None


def get_state_manager() -> StateManager:
    """Get global state manager instance."""
    global _global_state_manager
    if _global_state_manager is None:
        _global_state_manager = StateManager()
    return _global_state_manager


def set_state_manager(manager: StateManager) -> None:
    """Set global state manager instance."""
    global _global_state_manager
    _global_state_manager = manager


def reset_state_manager() -> None:
    """Reset global state manager."""
    global _global_state_manager
    _global_state_manager = StateManager()


# Context manager for scoped state management
class StateContext:
    """Context manager for scoped state management."""
    
    def __init__(self, initial_state: Optional[WorkflowState] = None):
        self.manager = StateManager(initial_state)
        self._previous_manager = None
    
    def __enter__(self) -> StateManager:
        global _global_state_manager
        self._previous_manager = _global_state_manager
        _global_state_manager = self.manager
        return self.manager
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        global _global_state_manager
        _global_state_manager = self._previous_manager