import re
with open("/workspace/helios-rl/scripts/run_cartpole.py", "r") as f:
    text = f.read()

# We just want to print valid_returns.max() at the end
import sys
new_text = text.split("if valid_returns.max() >= 500:")[0] + """
    print(f"Absolute max return across entire run: {valid_returns.max()}")
    if valid_returns.max() >= 500:
        print("CartPole validation PASSED! Max reward >= 500 achieved.")
    else:
        print("Max reward target of 500 not reached.")
"""
with open("/workspace/helios-rl/scripts/run_cartpole.py", "w") as f:
    f.write(new_text)
