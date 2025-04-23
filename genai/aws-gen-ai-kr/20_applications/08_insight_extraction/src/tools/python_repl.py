import logging
from typing import Annotated
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from .decorators import log_io

# Initialize REPL and logger
repl = PythonREPL()
# 새 핸들러와 포맷터 설정
logger = logging.getLogger(__name__)
logger.propagate = False  # 상위 로거로 메시지 전파 중지
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
handler = logging.StreamHandler()
formatter = logging.Formatter('\n%(levelname)s [%(name)s] %(message)s')  # 로그 레벨이 동적으로 표시되도록 변경
handler.setFormatter(formatter)
logger.addHandler(handler)
# DEBUG와 INFO 중 원하는 레벨로 설정
logger.setLevel(logging.DEBUG)  # DEBUG 이상 모든 로그 표시

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

@log_io
def handle_python_repl_tool(code: Annotated[str, "The python code to execute to do further analysis or calculation."]):
    """
    Use this to execute python code and do data analysis or calculation. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user.
    """

    #logger.info("Executing Python code")
    logger.info(f"{Colors.GREEN}===== Executing Python code ====={Colors.END}")
    try:
        result = repl.run(code)
        #logger.info("Code execution successful")
        logger.info(f"{Colors.GREEN}===== Code execution successful ====={Colors.END}")
    except BaseException as e:
        error_msg = f"Failed to execute. Error: {repr(e)}"
        #logger.error(error_msg)
        logger.error(f"{Colors.REF}Failed to execute. Error: {repr(e)}{Colors.END}")
        return error_msg
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return result_str

