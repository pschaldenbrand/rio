"""List available policy configurations.

Usage:
    python -m examples.policy_cfgs
"""

from examples import policy_cfgs

print("Available policies (set with POLICY=<name>):\n")
for name in policy_cfgs.__all__:
    print(f"  {name}")
