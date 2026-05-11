"""List available station configurations.

Usage:
    python -m examples.cfg
"""

from examples import cfg as station_cfgs

print("Available stations (set with STATION=<name>):\n")
for name in station_cfgs.__all__:
    cls = getattr(station_cfgs, name)
    module = cls.__module__.split(".")[-1]
    print(f"  {name}  ({module}.py)")
