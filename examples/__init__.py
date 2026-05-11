import os


def get_station_cfg(station_cfgs=None):
    """Resolve a station config class from $STATION env var. Errors if not set."""
    if station_cfgs is None:
        from examples import cfg as station_cfgs
    name = os.environ.get("STATION")
    if name is None:
        raise RuntimeError(
            "STATION environment variable is not set. "
            f"Available stations: {station_cfgs.__all__}. "
            "Run `python -m examples.cfg` to list them."
        )
    if name not in station_cfgs.__all__:
        raise RuntimeError(
            f"Unknown station '{name}'. Available stations: {station_cfgs.__all__}. Run `python -m examples.cfg` to list them."
        )
    return getattr(station_cfgs, name)


def get_policy_cfg(policy_cfgs=None):
    """Resolve a policy config class from $POLICY env var. Errors if not set."""
    if policy_cfgs is None:
        from examples import policy_cfgs as _policy_cfgs

        policy_cfgs = _policy_cfgs
    name = os.environ.get("POLICY")
    if name is None:
        raise RuntimeError(f"POLICY environment variable is not set. Available policies: {policy_cfgs.__all__}.")
    if name not in policy_cfgs.__all__:
        raise RuntimeError(f"Unknown policy '{name}'. Available policies: {policy_cfgs.__all__}.")
    return getattr(policy_cfgs, name)
