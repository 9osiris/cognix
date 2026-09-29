"""typed tool definitions and a registry that validates, coerces, and times calls."""
import time
from typing import Any, Callable, Dict, List, Optional

# allowed param types in tool specs
PARAM_TYPES = ("string", "number", "boolean", "integer")

# accepted truthy/falsy spellings for boolean coercion
_TRUE_WORDS = {"true", "1", "yes", "y", "on"}
_FALSE_WORDS = {"false", "0", "no", "n", "off"}


class Tool:
    """a named callable with a typed parameter spec."""

    def __init__(self, name: str, description: str, params: Dict[str, dict], func: Callable):
        if not isinstance(name, str) or not name:
            raise ValueError("tool name must be a non-empty string")
        if not isinstance(description, str) or not description:
            raise ValueError("tool description must be a non-empty string")
        if not isinstance(params, dict):
            raise ValueError("tool params must be a dict")
        if not callable(func):
            raise ValueError("tool func must be callable")
        self.name = name
        self.description = description
        self.params = params
        self.func = func

    def to_dict(self) -> Dict[str, Any]:
        # serializable form, func is deliberately excluded
        return {
            "name": self.name,
            "description": self.description,
            "params": self.params,
        }

    def schema(self) -> Dict[str, Any]:
        # json-schema flavored view of the params
        properties = {}
        required = []
        for pname, spec in self.params.items():
            properties[pname] = {
                "type": spec.get("type"),
                "description": spec.get("description", ""),
            }
            if "default" in spec:
                properties[pname]["default"] = spec["default"]
            if spec.get("required"):
                required.append(pname)
        params = {}
        for pname, spec in self.params.items():
            params[pname] = {
                "type": spec.get("type"),
                "description": spec.get("description", ""),
                "required": bool(spec.get("required")),
            }
            if "default" in spec:
                params[pname]["default"] = spec["default"]
        return {
            "name": self.name,
            "description": self.description,
            "params": params,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def __repr__(self) -> str:
        return "Tool(name={!r})".format(self.name)


def _check_spec(name: str, spec: dict) -> None:
    # validate one param spec entry
    if not isinstance(spec, dict):
        raise ValueError("param {!r} spec must be a dict".format(name))
    ptype = spec.get("type")
    if ptype not in PARAM_TYPES:
        raise ValueError(
            "param {!r} has bad type {!r}; must be one of {}".format(name, ptype, PARAM_TYPES)
        )
    if "description" in spec and not isinstance(spec["description"], str):
        raise ValueError("param {!r} description must be a string".format(name))
    if "required" in spec and not isinstance(spec["required"], bool):
        raise ValueError("param {!r} required must be a bool".format(name))


def _coerce(value: Any, ptype: str, pname: str) -> Any:
    # convert a raw arg value to the declared param type
    if ptype == "string":
        if isinstance(value, str):
            return value
        return str(value)
    if ptype == "number":
        if isinstance(value, bool):
            raise ValueError("param {!r} expects a number, got bool".format(pname))
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                pass
        raise ValueError("param {!r} expects a number, got {!r}".format(pname, value))
    if ptype == "integer":
        if isinstance(value, bool):
            raise ValueError("param {!r} expects an integer, got bool".format(pname))
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            raise ValueError("param {!r} expects an integer, got {!r}".format(pname, value))
        if isinstance(value, str):
            text = value.strip()
            try:
                return int(text)
            except ValueError:
                pass
            try:
                as_float = float(text)
            except ValueError:
                as_float = None
            if as_float is not None and as_float.is_integer():
                return int(as_float)
        raise ValueError("param {!r} expects an integer, got {!r}".format(pname, value))
    if ptype == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            word = value.strip().lower()
            if word in _TRUE_WORDS:
                return True
            if word in _FALSE_WORDS:
                return False
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value == 1:
                return True
            if value == 0:
                return False
        raise ValueError("param {!r} expects a boolean, got {!r}".format(pname, value))
    raise ValueError("unknown param type {!r}".format(ptype))


class ToolRegistry:
    """holds tools by name and dispatches validated, timed calls."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def register(self, tool: Tool) -> Tool:
        # validate param spec shape, reject duplicate names
        if not isinstance(tool, Tool):
            raise ValueError("can only register Tool instances")
        if tool.name in self._tools:
            raise ValueError("duplicate tool name: {!r}".format(tool.name))
        for pname, spec in tool.params.items():
            if not isinstance(pname, str) or not pname:
                raise ValueError("param names must be non-empty strings")
            _check_spec(pname, spec)
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return sorted(self._tools.keys())

    def schemas(self) -> List[Dict[str, Any]]:
        return [self._tools[name].schema() for name in self.names()]

    def call(self, name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # validate, coerce, dispatch; func exceptions become error strings
        started = time.perf_counter()
        tool = self._tools.get(name)
        if tool is None:
            return self._fail("unknown tool: {!r}".format(name), started)
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return self._fail("args must be a dict", started)
        unknown = sorted(set(args.keys()) - set(tool.params.keys()))
        if unknown:
            return self._fail("unknown params for tool {!r}: {}".format(name, ", ".join(unknown)), started)
        kwargs: Dict[str, Any] = {}
        for pname, spec in tool.params.items():
            value = args.get(pname)
            if value is None:
                if "default" in spec:
                    value = spec["default"]
                elif spec.get("required"):
                    return self._fail(
                        "missing required param {!r} for tool {!r}".format(pname, name), started
                    )
                else:
                    continue
            try:
                kwargs[pname] = _coerce(value, spec["type"], pname)
            except ValueError as exc:
                return self._fail(str(exc), started)
        try:
            result = tool.func(**kwargs)
        except Exception as exc:  # noqa: BLE001 - tool errors are reported, not raised
            return self._fail("tool {!r} raised {}: {}".format(name, type(exc).__name__, exc), started)
        return {
            "ok": True,
            "result": result,
            "error": None,
            "duration_ms": (time.perf_counter() - started) * 1000.0,
        }

    def _fail(self, message: str, started: float) -> Dict[str, Any]:
        return {
            "ok": False,
            "result": None,
            "error": message,
            "duration_ms": (time.perf_counter() - started) * 1000.0,
        }

    def help_text(self) -> str:
        # human-readable listing of every registered tool
        lines = []
        for name in self.names():
            tool = self._tools[name]
            lines.append("{} - {}".format(name, tool.description))
            for pname, spec in tool.params.items():
                marker = "required" if spec.get("required") else "optional"
                default = ""
                if "default" in spec:
                    default = ", default={!r}".format(spec["default"])
                lines.append(
                    "    {} ({}; {}{}): {}".format(
                        pname, spec.get("type"), marker, default, spec.get("description", "")
                    )
                )
        return "\n".join(lines)
