"""
Global event queue for streaming events across different components.
Allows coder_agent_tool and other tools to send streaming events to main.py
"""

import threading
from collections import deque
from typing import Dict, Any, Optional

# Global event queue
_global_event_queue = deque()
_queue_lock = threading.Lock()

def put_event(event: Dict[str, Any]) -> None:
    """Add an event to the global queue"""
    with _queue_lock:
        _global_event_queue.append(event)

def get_event() -> Optional[Dict[str, Any]]:
    """Get an event from the global queue (non-blocking)"""
    with _queue_lock:
        if _global_event_queue:
            return _global_event_queue.popleft()
        return None

def has_events() -> bool:
    """Check if there are events in the queue"""
    with _queue_lock:
        return len(_global_event_queue) > 0

def clear_queue() -> None:
    """Clear all events from the queue"""
    with _queue_lock:
        _global_event_queue.clear()