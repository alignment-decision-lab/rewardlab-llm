from __future__ import annotations

from typing import Any, Callable, Dict, Type


class Registry:
    def __init__(self):
        self._items: Dict[str, Any] = {}

    def register(self, name: str):
        def deco(obj):
            if name in self._items:
                raise KeyError(f"Duplicate registration: {name}")
            self._items[name] = obj
            return obj
        return deco

    def get(self, name: str):
        if name not in self._items:
            raise KeyError(f"Unknown item '{name}'. Available: {sorted(self._items.keys())}")
        return self._items[name]

    def create(self, name: str, **kwargs):
        cls = self.get(name)
        return cls(**kwargs)


TASKS = Registry()
JUDGES = Registry()
OBJECTIVES = Registry()
MODELS = Registry()
