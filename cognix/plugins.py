"""plugin loading: plain .py files exposing register_tools() -> list[Tool]."""
import importlib.util
import os
from typing import Dict, List

from .tools.registry import Tool, ToolRegistry


class PluginError(Exception):
    """raised when a plugin cannot be imported or is malformed."""


class PluginManager:
    """discovers and loads tool plugins from a directory of .py files."""

    def __init__(self, plugin_dir: str):
        self.plugin_dir = os.path.abspath(plugin_dir)

    def discover(self) -> List[str]:
        # .py files in the plugin dir, ignoring _-prefixed and non-files
        if not os.path.isdir(self.plugin_dir):
            return []
        names = []
        for entry in sorted(os.listdir(self.plugin_dir)):
            if entry.startswith("_") or not entry.endswith(".py"):
                continue
            full = os.path.join(self.plugin_dir, entry)
            if os.path.isfile(full):
                names.append(entry[:-3])
        return names

    def load(self, name: str) -> List[Tool]:
        # import the plugin module from its file path
        if "/" in name or "\\" in name or name.startswith("."):
            raise PluginError("bad plugin name: {!r}".format(name))
        path = os.path.join(self.plugin_dir, name + ".py")
        if not os.path.isfile(path):
            raise PluginError("plugin not found: {!r}".format(name))
        module_name = "cognix_plugin_{}".format(name)
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise PluginError("cannot build import spec for {!r}".format(name))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except PluginError:
            raise
        except Exception as exc:  # noqa: BLE001 - import errors become PluginError
            raise PluginError("plugin {!r} failed to import: {}: {}".format(name, type(exc).__name__, exc))
        factory = getattr(module, "register_tools", None)
        if not callable(factory):
            raise PluginError("plugin {!r} does not define register_tools()".format(name))
        try:
            tools = factory()
        except Exception as exc:  # noqa: BLE001 - factory errors become PluginError
            raise PluginError("plugin {!r} register_tools() failed: {}".format(name, exc))
        if not isinstance(tools, list) or not all(isinstance(t, Tool) for t in tools):
            raise PluginError("plugin {!r} register_tools() must return a list of Tool".format(name))
        return tools

    def load_all(self, registry: ToolRegistry) -> Dict[str, str]:
        # load every discovered plugin into the registry
        report = {}
        for name in self.discover():
            try:
                tools = self.load(name)
                count = 0
                for tool in tools:
                    registry.register(tool)
                    count += 1
                report[name] = "loaded {} tools".format(count)
            except (PluginError, ValueError) as exc:
                report[name] = "error: {}".format(exc)
        return report
