"""Layered configuration for cognix: defaults, files, env vars, overrides."""

import json
import os

DEFAULTS = {
    "memory": {
        "working_capacity": 7,
        "decay_rate": 0.08,
        "activation_threshold": 0.05,
        "consolidation_threshold": 0.6,
        "min_rehearsals": 2,
        "semantic_support": 3,
        "episodic_max_age_days": 30,
        "replay_budget": 5,
        "replay_min_salience": 0.4,
    },
    "attention": {
        "bandwidth": 3,
        "salience_threshold": 0.3,
        "arousal_baseline": 0.5,
        "curiosity_bonus_scale": 0.15,
        "curiosity_boredom_threshold": 0.8,
    },
    "agent": {
        "max_retries": 2,
        "step_timeout": 30.0,
        "default_strategy": "auto",
        "max_attempts": 3,
    },
    "tools": {
        "workspace_dir": ".",
        "shell_enabled": True,
        "plugin_dir": "plugins",
    },
    "runtime": {
        "autosave": False,
        "save_path": "cognix_state.json",
        "consolidate_every": 10,
        "events_enabled": True,
        "event_log_size": 200,
    },
}


def _deep_copy(value):
    if isinstance(value, dict):
        return {k: _deep_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_copy(v) for v in value]
    return value


def _deep_merge(base, overrides):
    out = _deep_copy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = _deep_copy(value)
    return out


def _coerce(value, example):
    # coerce a string from env into the type of the default
    if isinstance(example, bool):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(example, int):
        return int(float(value))
    if isinstance(example, float):
        return float(value)
    return value


class Config:
    """Nested config with dotted-path access."""

    def __init__(self, data=None):
        self.data = _deep_copy(DEFAULTS)
        if data:
            self.data = _deep_merge(self.data, data)

    @classmethod
    def defaults(cls):
        return cls()

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as fh:
            return cls(json.load(fh))

    @classmethod
    def from_env(cls, prefix="COGNIX_"):
        # COGNIX_MEMORY__WORKING_CAPACITY=9 style overrides
        cfg = cls()
        for env_key, env_val in os.environ.items():
            if not env_key.startswith(prefix):
                continue
            path = env_key[len(prefix):].lower().split("__")
            node = cfg.data
            example = None
            ok = True
            for part in path[:-1]:
                node = node.get(part)
                if not isinstance(node, dict):
                    ok = False
                    break
            if not ok:
                continue
            leaf = path[-1]
            if leaf not in node:
                continue
            node[leaf] = _coerce(env_val, node[leaf])
        return cfg

    def get(self, dotted, default=None):
        node = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted, value):
        parts = dotted.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def merge(self, overrides):
        return Config(_deep_merge(self.data, overrides))

    def validate(self):
        # returns a list of human-readable problems, empty means fine
        problems = []
        checks = [
            ("memory.working_capacity", int, 1, 100),
            ("memory.decay_rate", float, 0.0, 1.0),
            ("memory.activation_threshold", float, 0.0, 1.0),
            ("memory.consolidation_threshold", float, 0.0, 1.0),
            ("memory.min_rehearsals", int, 0, 100),
            ("memory.semantic_support", int, 1, 1000),
            ("memory.replay_budget", int, 1, 100),
            ("memory.replay_min_salience", float, 0.0, 1.0),
            ("attention.bandwidth", int, 1, 20),
            ("attention.salience_threshold", float, 0.0, 1.0),
            ("attention.curiosity_bonus_scale", float, 0.0, 1.0),
            ("attention.curiosity_boredom_threshold", float, 0.0, 1.0),
            ("agent.max_retries", int, 0, 10),
            ("agent.step_timeout", float, 0.1, 3600.0),
            ("agent.max_attempts", int, 1, 10),
            ("runtime.consolidate_every", int, 1, 10000),
        ]
        for dotted, kind, lo, hi in checks:
            value = self.get(dotted)
            if not isinstance(value, kind) or isinstance(value, bool):
                problems.append("%s should be %s, got %r" % (dotted, kind.__name__, value))
            elif not (lo <= value <= hi):
                problems.append("%s out of range [%s, %s]: %r" % (dotted, lo, hi, value))
        if self.get("agent.default_strategy") not in ("auto", "decompose", "reactive", "means_ends"):
            problems.append("agent.default_strategy is not a known strategy")
        return problems

    def save(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=2)

    def to_dict(self):
        return _deep_copy(self.data)

    @classmethod
    def from_dict(cls, data):
        return cls(data)
