"""real tool implementations: calc, notes, files, shell, http, and small utilities."""
import ast
import json
import os
import re
import socket
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List

from .registry import Tool

# note names: letters, digits, dot, dash, underscore; must start alnum
_NOTE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")

# max chars of http body returned
_BODY_LIMIT = 4000

# longest expression accepted by calc
_CALC_MAX_LEN = 200


def _calc_eval(node: ast.AST) -> Any:
    # evaluate a whitelisted ast node, no names, no calls, no attribute access
    if isinstance(node, ast.Expression):
        return _calc_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("calc only supports numbers")
        return node.value
    if isinstance(node, ast.BinOp):
        left = _calc_eval(node.left)
        right = _calc_eval(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            # cap absurd exponents before they burn cpu
            if isinstance(right, (int, float)) and abs(right) > 1000:
                raise ValueError("calc exponent too large")
            return left ** right
        raise ValueError("calc does not support {!r}".format(type(node.op).__name__))
    if isinstance(node, ast.UnaryOp):
        operand = _calc_eval(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        raise ValueError("calc does not support {!r}".format(type(node.op).__name__))
    raise ValueError("calc does not support {!r}".format(type(node).__name__))


def _calc(expression: str) -> Any:
    text = expression.strip()
    if not text:
        raise ValueError("calc expression is empty")
    if len(text) > _CALC_MAX_LEN:
        raise ValueError("calc expression too long")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError("calc syntax error: {}".format(exc))
    return _calc_eval(tree)


def _check_note_name(name: str) -> str:
    if not isinstance(name, str) or not _NOTE_NAME_RE.fullmatch(name):
        raise ValueError(
            "invalid note name {!r}; use letters, digits, dot, dash, underscore".format(name)
        )
    return name


def _resolve(root: str, path: str) -> str:
    # resolve a workspace-relative path, refuse anything escaping root
    joined = os.path.join(root, path)
    real = os.path.realpath(os.path.abspath(joined))
    if os.path.commonpath([real, root]) != root:
        raise ValueError("path escapes workspace: {!r}".format(path))
    return real


def _decode_body(data: bytes, headers: Any) -> str:
    charset = "utf-8"
    content_type = headers.get("Content-Type", "") if headers else ""
    if "charset=" in content_type:
        charset = content_type.split("charset=")[-1].split(";")[0].strip() or "utf-8"
    try:
        return data.decode(charset, errors="replace")
    except (LookupError, ValueError):
        return data.decode("utf-8", errors="replace")


def _http_get(url: str, timeout: int) -> Dict[str, Any]:
    if not isinstance(url, str) or not url:
        raise ValueError("url must be a non-empty string")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("http_get only supports http/https urls, got {!r}".format(parsed.scheme))
    if not parsed.netloc:
        raise ValueError("url has no host: {!r}".format(url))
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    request = urllib.request.Request(url, headers={"User-Agent": "cognix/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            body = _decode_body(response.read(), response.headers)
    except urllib.error.HTTPError as exc:
        # http errors still carry a status and body worth returning
        status = exc.code
        try:
            body = _decode_body(exc.read(), exc.headers)
        except Exception:  # noqa: BLE001 - fall back to empty body
            body = ""
    except (urllib.error.URLError, socket.timeout, OSError) as exc:
        raise ValueError("http_get request failed: {}".format(exc))
    return {"status": status, "body": body[:_BODY_LIMIT]}


def _shell(command: str, timeout: int) -> Dict[str, Any]:
    # runs as the user; no filtering, the tool description says so
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError("command timed out after {}s".format(timeout))
    except OSError as exc:
        raise ValueError("shell failed to start: {}".format(exc))
    return {
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "returncode": completed.returncode,
    }


def make_builtin_tools(workspace_dir: str = None) -> List[Tool]:
    """build the standard toolset; file tools are rooted at workspace_dir."""
    root = os.path.realpath(os.path.abspath(workspace_dir or os.getcwd()))

    # notes live in a closure dict, no disk involved
    notes: Dict[str, str] = {}

    def note_write(name: str, text: str) -> str:
        notes[_check_note_name(name)] = text
        return "wrote note {!r}".format(name)

    def note_read(name: str) -> str:
        _check_note_name(name)
        if name not in notes:
            raise KeyError("no note named {!r}".format(name))
        return notes[name]

    def note_list() -> List[str]:
        return sorted(notes.keys())

    def note_delete(name: str) -> str:
        _check_note_name(name)
        if name not in notes:
            raise KeyError("no note named {!r}".format(name))
        del notes[name]
        return "deleted note {!r}".format(name)

    def file_read(path: str) -> str:
        target = _resolve(root, path)
        try:
            with open(target, "r", encoding="utf-8") as handle:
                return handle.read()
        except FileNotFoundError:
            raise ValueError("file not found: {!r}".format(path))
        except IsADirectoryError:
            raise ValueError("path is a directory: {!r}".format(path))
        except UnicodeDecodeError:
            raise ValueError("file is not text: {!r}".format(path))

    def file_write(path: str, content: str) -> str:
        target = _resolve(root, path)
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(content)
        return "wrote {} chars to {!r}".format(len(content), path)

    def file_append(path: str, content: str) -> str:
        target = _resolve(root, path)
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(content)
        return "appended {} chars to {!r}".format(len(content), path)

    def file_list(dir: str = ".") -> List[str]:
        target = _resolve(root, dir)
        if not os.path.isdir(target):
            raise ValueError("not a directory: {!r}".format(dir))
        return sorted(os.listdir(target))

    def file_delete(path: str) -> str:
        target = _resolve(root, path)
        if not os.path.exists(target) and not os.path.islink(target):
            raise ValueError("file not found: {!r}".format(path))
        if os.path.isdir(target) and not os.path.islink(target):
            raise ValueError("refusing to delete directory: {!r}".format(path))
        os.remove(target)
        return "deleted {!r}".format(path)

    def shell(command: str, timeout: int = 10) -> Dict[str, Any]:
        return _shell(command, timeout)

    def http_get(url: str, timeout: int = 10) -> Dict[str, Any]:
        return _http_get(url, timeout)

    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def echo(text: str) -> str:
        return text

    def word_count(text: str) -> Dict[str, int]:
        return {
            "words": len(text.split()),
            "chars": len(text),
            "lines": len(text.splitlines()),
        }

    def json_parse(text: str) -> Any:
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid json: {}".format(exc))

    def file_search(pattern: str, dir: str = ".", max_hits: int = 20) -> List[Dict[str, Any]]:
        target = _resolve(root, dir)
        if not os.path.isdir(target):
            raise ValueError("not a directory: {!r}".format(dir))
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ValueError("bad pattern: {}".format(exc))
        hits = []
        for base, _, files in os.walk(target):
            for name in sorted(files):
                if len(hits) >= max_hits:
                    return hits
                full = os.path.join(base, name)
                try:
                    with open(full, "r", encoding="utf-8") as handle:
                        for lineno, line in enumerate(handle, 1):
                            if regex.search(line):
                                hits.append({
                                    "path": os.path.relpath(full, root),
                                    "line": lineno,
                                    "text": line.rstrip("\n")[:200],
                                })
                                if len(hits) >= max_hits:
                                    return hits
                except (OSError, UnicodeDecodeError):
                    continue
        return hits

    def calc(expression: str) -> Any:
        return _calc(expression)

    tools = [
        Tool(
            name="calc",
            description="evaluate an arithmetic expression safely; supports + - * / // % ** and parentheses, no names or function calls",
            params={
                "expression": {
                    "type": "string",
                    "description": "arithmetic expression to evaluate",
                    "required": True,
                },
            },
            func=calc,
        ),
        Tool(
            name="note_write",
            description="store a named text note in memory for this session",
            params={
                "name": {"type": "string", "description": "note name", "required": True},
                "text": {"type": "string", "description": "note content", "required": True},
            },
            func=note_write,
        ),
        Tool(
            name="note_read",
            description="read a stored note by name",
            params={
                "name": {"type": "string", "description": "note name", "required": True},
            },
            func=note_read,
        ),
        Tool(
            name="note_list",
            description="list the names of all stored notes",
            params={},
            func=note_list,
        ),
        Tool(
            name="note_delete",
            description="delete a stored note by name",
            params={
                "name": {"type": "string", "description": "note name", "required": True},
            },
            func=note_delete,
        ),
        Tool(
            name="file_read",
            description="read a text file relative to the workspace",
            params={
                "path": {"type": "string", "description": "path relative to workspace", "required": True},
            },
            func=file_read,
        ),
        Tool(
            name="file_write",
            description="write text to a file relative to the workspace, creating parent dirs",
            params={
                "path": {"type": "string", "description": "path relative to workspace", "required": True},
                "content": {"type": "string", "description": "text to write", "required": True},
            },
            func=file_write,
        ),
        Tool(
            name="file_append",
            description="append text to a file relative to the workspace",
            params={
                "path": {"type": "string", "description": "path relative to workspace", "required": True},
                "content": {"type": "string", "description": "text to append", "required": True},
            },
            func=file_append,
        ),
        Tool(
            name="file_list",
            description="list entries in a workspace directory",
            params={
                "dir": {"type": "string", "description": "directory relative to workspace", "required": False, "default": "."},
            },
            func=file_list,
        ),
        Tool(
            name="file_delete",
            description="delete a file relative to the workspace; refuses directories",
            params={
                "path": {"type": "string", "description": "path relative to workspace", "required": True},
            },
            func=file_delete,
        ),
        Tool(
            name="shell",
            description="run a shell command as the user and capture output; runs with full user privileges",
            params={
                "command": {"type": "string", "description": "shell command to run", "required": True},
                "timeout": {"type": "integer", "description": "max seconds to wait", "required": False, "default": 10},
            },
            func=shell,
        ),
        Tool(
            name="http_get",
            description="fetch an http/https url, returning status and the first 4000 chars of the body",
            params={
                "url": {"type": "string", "description": "url to fetch", "required": True},
                "timeout": {"type": "integer", "description": "max seconds to wait", "required": False, "default": 10},
            },
            func=http_get,
        ),
        Tool(
            name="now_iso",
            description="current utc time as an iso 8601 string",
            params={},
            func=now_iso,
        ),
        Tool(
            name="echo",
            description="return the given text unchanged",
            params={
                "text": {"type": "string", "description": "text to return", "required": True},
            },
            func=echo,
        ),
        Tool(
            name="word_count",
            description="count words, characters, and lines in the given text",
            params={
                "text": {"type": "string", "description": "text to count", "required": True},
            },
            func=word_count,
        ),
        Tool(
            name="json_parse",
            description="parse a json string, returning the parsed value or an error",
            params={
                "text": {"type": "string", "description": "json text to parse", "required": True},
            },
            func=json_parse,
        ),
        Tool(
            name="file_search",
            description="search workspace files for a regex pattern, returning path/line/text hits",
            params={
                "pattern": {"type": "string", "description": "regex to search for", "required": True},
                "dir": {"type": "string", "description": "directory to search under the workspace", "required": False},
                "max_hits": {"type": "integer", "description": "maximum hits to return", "required": False},
            },
            func=file_search,
        ),
    ]
    return tools
