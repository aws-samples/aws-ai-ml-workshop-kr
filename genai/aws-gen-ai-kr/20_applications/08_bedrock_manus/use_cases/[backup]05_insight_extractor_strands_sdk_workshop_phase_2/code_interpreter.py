from bedrock_agentcore.tools.code_interpreter_client import code_session
from strands import Agent, tool
import json


def execute_python(code: str, description: str = "") -> str:
    """Execute Python code in the sandbox."""
    
    if description:
        code = f"# {description}\n{code}"
    
    print(f"\n Generated Code: {code}")
    
    with code_session("us-west-2") as code_client:
        response = code_client.invoke("executeCode", {
            "code": code,
            "language": "python",
            "clearContext": False
        })
    
    for event in response["stream"]:
        print ("=====")
        print (event)
        return json.dumps(event["result"])
        #return event["result"]

code = '''
def is_prime(n):
    """Check if a number is prime"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    # Check odd divisors up to sqrt(n)
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# Find all prime numbers between 1 and 100 that are > 9 and < 84
primes_in_range = []
for num in range(10, 84):  # 10 to 83 (inclusive)
    if is_prime(num):
        primes_in_range.append(num)

print("Prime numbers between 9 and 84:")
print(primes_in_range)
print(f"Largest prime in this range: {max(primes_in_range)}")

# Verify that 83 is prime
def verify_prime(n):
    """Detailed verification that a number is prime"""
    print(f"Checking if {n} is prime:")

    if n < 2:
        print(f"{n} is less than 2, so not prime")
        return False

    print(f"Checking divisors from 2 to {int(n**0.5)}:")

    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            print(f"{n} is divisible by {i}, so not prime")
            return False
        else:
            print(f"{n} ÷ {i} = {n/i:.3f} (not divisible)")

    print(f"{n} is prime!")
    return True

verify_prime(83)

# Also verify our range constraints
print(f"Verifying constraints:")
print(f"83 > 9: {83 > 9}")
print(f"83 < 84: {83 < 84}")
print(f"83 is between 1 and 100: {1 <= 83 <= 100}")
'''

execute_python(
    code=code
)