import logging
import functools
from typing import Any, Callable, Type, TypeVar

# Simple logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

T = TypeVar("T")

class Colors:
    BLUE = '\033[94m'
    RED = '\033[91m'
    END = '\033[0m'

def log_io(func: Callable) -> Callable:
    """
    A decorator that logs the input parameters and output of a tool function.

    Args:
        func: The tool function to be decorated

    Returns:
        The wrapped function with input/output logging
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        func_name = func.__name__
        # Execute the function
        result = func(*args, **kwargs)

        # Log the output
        if len(result.split("||")) == 3:
            status, code, stdout = result.split("||")
            #logger.info(f"{Colors.RED}Python-REPL - {status}\n{code}{Colors.END}")
            #logger.info(f"{Colors.BLUE}\n{stdout}{Colors.END}")
        else:
            if len(result.split("||")) == 2:
                _, stdout = result.split("||")
            #logger.info(f"{Colors.RED}\nTool {func_name} returned:\n{result}{Colors.END}")

        # Put tool result in global queue
        try:
            from src.utils.event_queue import put_event
            from datetime import datetime
            
            put_event({
                "timestamp": datetime.now().isoformat(),
                "type": "tool_result",
                "event_type": "tool_result",
                "tool_name": func_name,
                "output": result,
                "source": "tool_execution"
            })
        except ImportError:
            pass  # Fallback if event_queue not available

        return result

    return wrapper


class LoggedToolMixin:
    """A mixin class that adds logging functionality to any tool."""

    def _log_operation(self, method_name: str, *args: Any, **kwargs: Any) -> None:
        """Helper method to log tool operations."""
        tool_name = self.__class__.__name__.replace("Logged", "")
        params = ", ".join(
            [*(str(arg) for arg in args), *(f"{k}={v}" for k, v in kwargs.items())]
        )
        logger.debug(f"{Colors.RED}Tool {tool_name}.{method_name} called with: {params}{Colors.END}")

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Override _run method to add logging."""
        self._log_operation("_run", *args, **kwargs)
        result = super()._run(*args, **kwargs)
        logger.debug(f"{Colors.BLUE}\nTool {self.__class__.__name__.replace('Logged', '')} returned: {result}{Colors.END}")
        return result


def create_logged_tool(base_tool_class: Type[T]) -> Type[T]:
    """
    Factory function to create a logged version of any tool class.

    Args:
        base_tool_class: The original tool class to be enhanced with logging

    Returns:
        A new class that inherits from both LoggedToolMixin and the base tool class
    """

    class LoggedTool(LoggedToolMixin, base_tool_class):
        pass

    # Set a more descriptive name for the class
    LoggedTool.__name__ = f"Logged{base_tool_class.__name__}"
    return LoggedTool
