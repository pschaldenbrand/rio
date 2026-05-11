__all__ = ["Pi05Cfg", "SmolVLACfg"]


def __getattr__(name):
    if name == "Pi05Cfg":
        from .pi05 import Pi05Cfg

        return Pi05Cfg
    if name == "SmolVLACfg":
        from .smolvla import SmolVLACfg

        return SmolVLACfg
    raise AttributeError(f"module 'examples.policy_cfgs' has no attribute {name!r}")
