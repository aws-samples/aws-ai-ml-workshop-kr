import logging
import subprocess
from typing import Annotated
from langchain_core.tools import tool
from .decorators import log_io

# Initialize logger
logger = logging.getLogger(__name__)
logger.propagate = False  # 상위 로거로 메시지 전파 중지
for handler in logger.handlers[:]: logger.removeHandler(handler)
handler = logging.StreamHandler()
formatter = logging.Formatter('\n%(levelname)s [%(name)s] %(message)s')  # 로그 레벨이 동적으로 표시되도록 변경
handler.setFormatter(formatter)
logger.addHandler(handler)
# DEBUG와 INFO 중 원하는 레벨로 설정
logger.setLevel(logging.INFO)  # DEBUG 이상 모든 로그 표시

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

@log_io
def handle_bash_tool(cmd: Annotated[str, "The bash command to be executed."],):
    """Use this to execute bash command and do necessary operations."""

    logger.info(f"{Colors.GREEN}===== Executing Bash ====={Colors.END}")
    logger.info(f"{Colors.BOLD}===== Coder - Command: {cmd} ====={Colors.END}")
    try:
        # Execute the command and capture output
        result = subprocess.run(
            cmd, shell=True, check=True, text=True, capture_output=True
        )
        # Return stdout as the result
        results = "||".join([cmd, result.stdout])
        return results
        #return result.stdout
    except subprocess.CalledProcessError as e:
        # If command fails, return error information
        error_message = f"Command failed with exit code {e.returncode}.\nStdout: {e.stdout}\nStderr: {e.stderr}"
        #logger.error(error_message)
        logger.error(f"{Colors.RED}Command failed with exit code {e.returncode}.\nStdout: {e.stdout}\nStderr: {e.stderr}{Colors.END}")
        return error_message
    except Exception as e:
        # Catch any other exceptions
        error_message = f"Error executing command: {str(e)}"
        #logger.error(error_message)
        logger.error(f"{Colors.RED}error_message{Colors.END}")
        return error_message

if __name__ == "__main__":
    print(bash_tool.invoke("ls -all"))
