"""versioned state persistence: atomic json saves with migrations and a save store."""
import json
import os
import tempfile
from typing import Dict, List, Optional, Tuple

SAVE_VERSION = 1


class StateError(Exception):
    """raised for corrupt, missing, or unsupported-version save files."""


def _migrate_0_to_1(payload: dict) -> dict:
    # version 0 saves were a bare state dict with no envelope
    return {"version": 1, "state": payload}


# version -> function upgrading that version's payload by one step
MIGRATIONS = {0: _migrate_0_to_1}


def save_state(state: dict, path: str) -> str:
    """write state atomically (tmp file + os.replace); returns the path."""
    if not isinstance(state, dict):
        raise StateError("state must be a dict")
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    payload = {"version": SAVE_VERSION, "state": state}
    fd, tmp_path = tempfile.mkstemp(dir=parent, prefix=".cognix_save_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return path


def load_state(path: str) -> dict:
    """read a save file, verify version, run migrations, return the state dict."""
    if not os.path.isfile(path):
        raise StateError("save file not found: {!r}".format(path))
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise StateError("save file is corrupt: {!r} ({})".format(path, exc))
    if not isinstance(data, dict):
        raise StateError("save file is corrupt: {!r} (top level is not an object)".format(path))
    version = data.get("version", 0)
    if not isinstance(version, int) or version < 0:
        raise StateError("save file has bad version: {!r}".format(version))
    if version > SAVE_VERSION:
        raise StateError(
            "save version {} is newer than supported version {}".format(version, SAVE_VERSION)
        )
    while version < SAVE_VERSION:
        migrate = MIGRATIONS.get(version)
        if migrate is None:
            raise StateError("no migration available from save version {}".format(version))
        data = migrate(data)
        version = data.get("version", version + 1)
    state = data.get("state")
    if not isinstance(state, dict):
        raise StateError("save file is corrupt: {!r} (state is not an object)".format(path))
    return state


class StateStore:
    """named json saves inside one directory."""

    def __init__(self, directory: str):
        self.directory = os.path.abspath(directory)

    def _path(self, name: str) -> str:
        if not isinstance(name, str) or not name:
            raise StateError("save name must be a non-empty string")
        if "/" in name or "\\" in name or name.startswith("."):
            raise StateError("bad save name: {!r}".format(name))
        return os.path.join(self.directory, name + ".json")

    def save(self, name: str, state: dict) -> str:
        return save_state(state, self._path(name))

    def load(self, name: str) -> dict:
        path = self._path(name)
        if not os.path.isfile(path):
            raise StateError("no save named {!r}".format(name))
        return load_state(path)

    def list_saves(self) -> List[Tuple[str, float]]:
        # (name, mtime) pairs, newest first
        if not os.path.isdir(self.directory):
            return []
        saves = []
        for entry in os.listdir(self.directory):
            if not entry.endswith(".json") or entry.startswith("."):
                continue
            full = os.path.join(self.directory, entry)
            if os.path.isfile(full):
                saves.append((entry[:-5], os.path.getmtime(full)))
        saves.sort(key=lambda item: item[1], reverse=True)
        return saves

    def delete(self, name: str) -> bool:
        path = self._path(name)
        if not os.path.isfile(path):
            return False
        os.remove(path)
        return True

    def latest(self) -> Optional[str]:
        saves = self.list_saves()
        return saves[0][0] if saves else None
