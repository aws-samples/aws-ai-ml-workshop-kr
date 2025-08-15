"""
Test script for the new state management system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.state import WorkflowState, StateManager, StateContext
from src.workflow_executor import WorkflowExecutor, build_workflow
import logging

# Configure logging for test
logging.basicConfig(level=logging.INFO)

def test_state_management():
    """Test basic state management functionality."""
    print("Testing state management...")
    
    # Test WorkflowState
    state = WorkflowState(
        request="Test request",
        request_prompt="Test prompt"
    )
    
    print(f"Initial state: {state.request}")
    
    # Test StateManager
    manager = StateManager(state)
    
    # Test state updates
    manager.update_state(current_agent="test_agent")
    updated_state = manager.get_state()
    
    print(f"Updated agent: {updated_state.current_agent}")
    
    # Test list operations
    manager.append_to_list("history", {"agent": "test", "message": "test message"})
    final_state = manager.get_state()
    
    print(f"History length: {len(final_state.history)}")
    print("State management test passed!")

def test_workflow_executor():
    """Test the workflow executor without actually running agents."""
    print("\nTesting workflow executor...")
    
    # Create a simple test workflow
    executor = WorkflowExecutor()
    
    def dummy_coordinator():
        print("Dummy coordinator executed")
        return "__end__"
    
    executor.add_node("coordinator", dummy_coordinator)
    executor.set_start_node("coordinator")
    
    # Create test state
    initial_state = WorkflowState(
        request="Test workflow",
        request_prompt="Test prompt"
    )
    
    # Execute workflow
    result = executor.execute(initial_state)
    
    print(f"Workflow result: {result['status']}")
    print(f"Final node: {result['final_node']}")
    print("Workflow executor test passed!")

def test_state_context():
    """Test the state context manager."""
    print("\nTesting state context...")
    
    initial_state = WorkflowState(request="Context test")
    
    with StateContext(initial_state) as manager:
        manager.update_state(current_agent="context_test")
        state = manager.get_state()
        print(f"Context state agent: {state.current_agent}")
    
    print("State context test passed!")

if __name__ == "__main__":
    test_state_management()
    test_workflow_executor()
    test_state_context()
    print("\n✅ All tests passed!")