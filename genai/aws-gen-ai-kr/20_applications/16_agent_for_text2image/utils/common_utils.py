import time
import pickle
import random
import logging
import functools
from IPython.display import Markdown, HTML, display
from botocore.exceptions import ClientError

logging.basicConfig()
logger = logging.getLogger('retry-bedrock-invocation')
logger.setLevel(logging.INFO)

def retry(total_try_cnt=5, sleep_in_sec=5, retryable_exceptions=(ClientError,)):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for cnt in range(total_try_cnt):
                logger.info(f"trying {func.__name__}() [{cnt+1}/{total_try_cnt}]")
                try:
                    result = func(*args, **kwargs)
                    logger.info(f"in retry(), {func.__name__}() returned '{result}'")
                    if result: return result
                except retryable_exceptions as e:
                    # Throttling 에러인지 확인
                    if isinstance(e, ClientError):
                        error_code = e.response['Error']['Code']
                        if error_code == 'ThrottlingException':
                            logger.info(f"in retry(), {func.__name__}() raised ThrottlingException")
                            time.sleep(sleep_in_sec)
                            continue
                    logger.info(f"in retry(), {func.__name__}() raised retryable exception '{e}'")
                    pass
                except Exception as e:
                    logger.info(f"in retry(), {func.__name__}() raised {e}")
                    raise e
                time.sleep(sleep_in_sec)
            logger.info(f"{func.__name__} finally has been failed")
        return wrapper
    return decorator

def to_pickle(obj, path):

    with open(file=path, mode="wb") as f:
        pickle.dump(obj, f)

    print (f'To_PICKLE: {path}')
    
def load_pickle(path):
    
    with open(file=path, mode="rb") as f:
        obj=pickle.load(f)

    print (f'Load from {path}')

    return obj

def to_markdown(obj, path):

    with open(file=path, mode="w") as f:
        f.write(obj)

    print (f'To_Markdown: {path}')
    
def print_html(input_html):

    html_string=""
    html_string = html_string + input_html

    display(HTML(html_string))


