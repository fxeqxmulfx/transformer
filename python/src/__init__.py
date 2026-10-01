"""Public model/optimizer API; domain imports do not initialize PyTorch."""

from importlib import import_module

__all__ = ["AMSGradW", "Config", "GPTMini", "amsgradw_update_"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    module = ".amsgradw" if name in ("AMSGradW", "amsgradw_update_") else ".gpt_mini"
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
