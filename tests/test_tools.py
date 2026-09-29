"""tests for the tool registry and builtin tools."""
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from cognix.tools import Tool, ToolRegistry, make_builtin_tools


def _sample_tool(**kwargs):
    def func(**inner):
        return inner
    spec = {"name": "sample", "description": "a sample", "params": {}, "func": func}
    spec.update(kwargs)
    return Tool(**spec)


# registry registration


def test_register_and_get():
    reg = ToolRegistry()
    tool = _sample_tool()
    reg.register(tool)
    assert reg.get("sample") is tool
    assert reg.get("missing") is None
    assert reg.names() == ["sample"]


def test_register_rejects_duplicates():
    reg = ToolRegistry()
    reg.register(_sample_tool())
    with pytest.raises(ValueError, match="duplicate"):
        reg.register(_sample_tool())


def test_register_validates_param_spec_shape():
    reg = ToolRegistry()
    with pytest.raises(ValueError, match="bad type"):
        reg.register(_sample_tool(params={"x": {"type": "weird", "required": True}}))
    with pytest.raises(ValueError, match="must be a dict"):
        reg.register(_sample_tool(params={"x": "string"}))
    with pytest.raises(ValueError, match="required must be a bool"):
        reg.register(_sample_tool(params={"x": {"type": "string", "required": "yes"}}))


def test_register_rejects_non_tool():
    reg = ToolRegistry()
    with pytest.raises(ValueError):
        reg.register("not a tool")


def test_schemas_shape():
    reg = ToolRegistry()
    reg.register(
        Tool(
            name="adder",
            description="adds",
            params={
                "a": {"type": "number", "description": "first", "required": True},
                "b": {"type": "number", "description": "second", "required": False, "default": 1},
            },
            func=lambda a, b=1: a + b,
        )
    )
    (schema,) = reg.schemas()
    assert schema["name"] == "adder"
    assert schema["parameters"]["required"] == ["a"]
    assert schema["parameters"]["properties"]["b"]["default"] == 1
    # to_dict excludes func
    as_dict = reg.get("adder").to_dict()
    assert "func" not in as_dict
    assert as_dict["name"] == "adder"


# registry call: validation and coercion


def _math_registry():
    reg = ToolRegistry()
    reg.register(
        Tool(
            name="add",
            description="add two ints",
            params={
                "a": {"type": "integer", "description": "a", "required": True},
                "b": {"type": "integer", "description": "b", "required": True},
            },
            func=lambda a, b: a + b,
        )
    )
    reg.register(
        Tool(
            name="greet",
            description="greet",
            params={
                "name": {"type": "string", "description": "who", "required": True},
                "loud": {"type": "boolean", "description": "shout", "required": False, "default": False},
            },
            func=lambda name, loud=False: name.upper() if loud else name,
        )
    )
    return reg


def test_call_happy_path():
    reg = _math_registry()
    out = reg.call("add", {"a": 2, "b": 3})
    assert out["ok"] is True
    assert out["result"] == 5
    assert out["error"] is None
    assert out["duration_ms"] >= 0


def test_call_unknown_tool():
    reg = _math_registry()
    out = reg.call("nope", {})
    assert out["ok"] is False
    assert "unknown tool" in out["error"]


def test_call_missing_required():
    reg = _math_registry()
    out = reg.call("add", {"a": 1})
    assert out["ok"] is False
    assert "missing required param 'b'" in out["error"]


def test_call_rejects_unknown_params():
    reg = _math_registry()
    out = reg.call("add", {"a": 1, "b": 2, "c": 3})
    assert out["ok"] is False
    assert "unknown params" in out["error"]


def test_call_applies_defaults():
    reg = _math_registry()
    out = reg.call("greet", {"name": "osiris"})
    assert out["ok"] is True and out["result"] == "osiris"
    out = reg.call("greet", {"name": "osiris", "loud": "yes"})
    assert out["result"] == "OSIRIS"


def test_call_coerces_integers():
    reg = _math_registry()
    assert reg.call("add", {"a": "2", "b": "3.0"})["result"] == 5
    assert reg.call("add", {"a": 2.0, "b": 3})["result"] == 5
    out = reg.call("add", {"a": "two", "b": 3})
    assert out["ok"] is False and "expects an integer" in out["error"]
    out = reg.call("add", {"a": 2.5, "b": 3})
    assert out["ok"] is False and "expects an integer" in out["error"]
    out = reg.call("add", {"a": True, "b": 3})
    assert out["ok"] is False and "got bool" in out["error"]


def test_call_coerces_numbers():
    reg = ToolRegistry()
    reg.register(
        Tool("half", "halve", {"x": {"type": "number", "description": "x", "required": True}},
             func=lambda x: x / 2)
    )
    assert reg.call("half", {"x": "5"})["result"] == 2.5
    assert reg.call("half", {"x": 5})["result"] == 2.5
    out = reg.call("half", {"x": "abc"})
    assert out["ok"] is False and "expects a number" in out["error"]
    out = reg.call("half", {"x": True})
    assert out["ok"] is False and "got bool" in out["error"]


def test_call_coerces_booleans():
    reg = ToolRegistry()
    reg.register(
        Tool("flag", "flag", {"on": {"type": "boolean", "description": "on", "required": True}},
             func=lambda on: on)
    )
    for raw in (True, "true", "YES", "1", 1):
        assert reg.call("flag", {"on": raw})["result"] is True
    for raw in (False, "false", "No", "0", 0):
        assert reg.call("flag", {"on": raw})["result"] is False
    out = reg.call("flag", {"on": "maybe"})
    assert out["ok"] is False and "expects a boolean" in out["error"]


def test_call_coerces_strings():
    reg = ToolRegistry()
    reg.register(
        Tool("say", "say", {"t": {"type": "string", "description": "t", "required": True}},
             func=lambda t: t)
    )
    assert reg.call("say", {"t": "hi"})["result"] == "hi"
    assert reg.call("say", {"t": 42})["result"] == "42"


def test_call_catches_func_exceptions():
    reg = ToolRegistry()
    def boom():
        raise RuntimeError("kaboom")
    reg.register(Tool("boom", "booms", {}, boom))
    out = reg.call("boom", {})
    assert out["ok"] is False
    assert "kaboom" in out["error"]
    assert "duration_ms" in out


def test_help_text_lists_tools_and_params():
    reg = _math_registry()
    text = reg.help_text()
    assert "add - add two ints" in text
    assert "a (integer; required)" in text
    assert "loud (boolean; optional, default=False)" in text


# builtin tools


def _builtins(tmp_path):
    reg = ToolRegistry()
    for tool in make_builtin_tools(workspace_dir=str(tmp_path)):
        reg.register(tool)
    return reg


def test_builtin_names():
    reg = ToolRegistry()
    for tool in make_builtin_tools():
        reg.register(tool)
    expected = {
        "calc", "note_write", "note_read", "note_list", "note_delete",
        "file_read", "file_write", "file_append", "file_list", "file_delete",
        "file_search",
        "shell", "http_get", "now_iso", "echo", "word_count", "json_parse",
    }
    assert set(reg.names()) == expected


def test_calc_happy_path(tmp_path):
    reg = _builtins(tmp_path)
    cases = {
        "2 + 3 * 4": 14,
        "(2 + 3) * 4": 20,
        "-5 + 2": -3,
        "2 ** 10": 1024,
        "7 // 2": 3,
        "7 % 3": 1,
        "1.5 * 2": 3.0,
        "10 / 4": 2.5,
    }
    for expr, want in cases.items():
        out = reg.call("calc", {"expression": expr})
        assert out["ok"] is True, (expr, out)
        assert out["result"] == want


def test_calc_rejects_non_arithmetic(tmp_path):
    reg = _builtins(tmp_path)
    for expr in ("__import__('os')", "open('x')", "x + 1", "[1, 2][0]",
                 "1 if True else 2", "len('ab')", "2 ** 1000000", ""):
        out = reg.call("calc", {"expression": expr})
        assert out["ok"] is False, expr
        assert out["error"]


def test_calc_division_by_zero_is_error(tmp_path):
    reg = _builtins(tmp_path)
    out = reg.call("calc", {"expression": "1 / 0"})
    assert out["ok"] is False
    assert "ZeroDivision" in out["error"]


def test_notes_roundtrip(tmp_path):
    reg = _builtins(tmp_path)
    assert reg.call("note_write", {"name": "todo", "text": "ship it"})["ok"] is True
    assert reg.call("note_read", {"name": "todo"})["result"] == "ship it"
    assert reg.call("note_list", {})["result"] == ["todo"]
    assert reg.call("note_delete", {"name": "todo"})["ok"] is True
    assert reg.call("note_list", {})["result"] == []


def test_notes_reject_bad_names(tmp_path):
    reg = _builtins(tmp_path)
    for bad in ("../evil", "a/b", "", "has space"):
        out = reg.call("note_write", {"name": bad, "text": "x"})
        assert out["ok"] is False, bad


def test_notes_missing_name_errors(tmp_path):
    reg = _builtins(tmp_path)
    out = reg.call("note_read", {"name": "ghost"})
    assert out["ok"] is False and "no note" in out["error"]
    out = reg.call("note_delete", {"name": "ghost"})
    assert out["ok"] is False and "no note" in out["error"]


def test_files_roundtrip(tmp_path):
    reg = _builtins(tmp_path)
    assert reg.call("file_write", {"path": "sub/a.txt", "content": "hello"})["ok"] is True
    assert reg.call("file_read", {"path": "sub/a.txt"})["result"] == "hello"
    assert reg.call("file_append", {"path": "sub/a.txt", "content": " world"})["ok"] is True
    assert reg.call("file_read", {"path": "sub/a.txt"})["result"] == "hello world"
    assert reg.call("file_list", {"dir": "sub"})["result"] == ["a.txt"]
    assert reg.call("file_delete", {"path": "sub/a.txt"})["ok"] is True
    out = reg.call("file_read", {"path": "sub/a.txt"})
    assert out["ok"] is False and "not found" in out["error"]


def test_files_refuse_escape(tmp_path):
    reg = _builtins(tmp_path)
    for bad in ("../escape.txt", "../../escape.txt", "sub/../../escape.txt"):
        out = reg.call("file_write", {"path": bad, "content": "x"})
        assert out["ok"] is False and "escapes workspace" in out["error"], bad
        out = reg.call("file_read", {"path": bad})
        assert out["ok"] is False and "escapes workspace" in out["error"], bad
    # absolute paths outside the workspace are refused too
    out = reg.call("file_read", {"path": "/etc/hostname"})
    assert out["ok"] is False and "escapes workspace" in out["error"]


def test_file_delete_refuses_directories(tmp_path):
    reg = _builtins(tmp_path)
    (tmp_path / "adir").mkdir()
    out = reg.call("file_delete", {"path": "adir"})
    assert out["ok"] is False and "directory" in out["error"]
    out = reg.call("file_delete", {"path": "missing.txt"})
    assert out["ok"] is False and "not found" in out["error"]
    out = reg.call("file_list", {"dir": "missing"})
    assert out["ok"] is False and "not a directory" in out["error"]


def test_shell_happy_path(tmp_path):
    reg = _builtins(tmp_path)
    out = reg.call("shell", {"command": "echo hello"})
    assert out["ok"] is True
    assert out["result"]["stdout"].strip() == "hello"
    assert out["result"]["returncode"] == 0


def test_shell_reports_nonzero_exit(tmp_path):
    reg = _builtins(tmp_path)
    out = reg.call("shell", {"command": "exit 3"})
    assert out["ok"] is True
    assert out["result"]["returncode"] == 3


def test_shell_timeout(tmp_path):
    reg = _builtins(tmp_path)
    out = reg.call("shell", {"command": "sleep 5", "timeout": 1})
    assert out["ok"] is False
    assert "timed out" in out["error"]


def test_http_get_rejects_bad_scheme(tmp_path):
    reg = _builtins(tmp_path)
    for bad in ("ftp://example.com/x", "gopher://x", "not a url", ""):
        out = reg.call("http_get", {"url": bad})
        assert out["ok"] is False, bad


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = ("hello cognix " * 500).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_http_get_happy_path(tmp_path):
    reg = _builtins(tmp_path)
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = "http://127.0.0.1:{}/page".format(server.server_port)
        out = reg.call("http_get", {"url": url})
    finally:
        server.shutdown()
        thread.join()
    assert out["ok"] is True
    assert out["result"]["status"] == 200
    assert len(out["result"]["body"]) == 4000
    assert out["result"]["body"].startswith("hello cognix")


def test_now_iso_parses(tmp_path):
    reg = _builtins(tmp_path)
    out = reg.call("now_iso", {})
    assert out["ok"] is True
    parsed = datetime.fromisoformat(out["result"])
    assert parsed.tzinfo is not None


def test_echo_and_word_count(tmp_path):
    reg = _builtins(tmp_path)
    assert reg.call("echo", {"text": "hi"})["result"] == "hi"
    out = reg.call("word_count", {"text": "one two\nthree"})
    assert out["ok"] is True
    assert out["result"] == {"words": 3, "chars": 13, "lines": 2}


def test_json_parse(tmp_path):
    reg = _builtins(tmp_path)
    out = reg.call("json_parse", {"text": '{"a": [1, 2]}'})
    assert out["ok"] is True and out["result"] == {"a": [1, 2]}
    out = reg.call("json_parse", {"text": "{bad"})
    assert out["ok"] is False and "invalid json" in out["error"]


def test_builtin_descriptions_and_params_typed(tmp_path):
    for tool in make_builtin_tools(workspace_dir=str(tmp_path)):
        assert tool.description, tool.name
        for pname, spec in tool.params.items():
            assert spec["type"] in ("string", "number", "boolean", "integer"), (tool.name, pname)


# extra registry edge cases


def test_tool_constructor_validation():
    def func():
        return 1
    with pytest.raises(ValueError, match="non-empty string"):
        Tool("", "d", {}, func)
    with pytest.raises(ValueError, match="non-empty string"):
        Tool("n", "", {}, func)
    with pytest.raises(ValueError, match="must be a dict"):
        Tool("n", "d", [], func)
    with pytest.raises(ValueError, match="callable"):
        Tool("n", "d", {}, "not callable")


def test_registry_len_and_contains():
    reg = ToolRegistry()
    assert len(reg) == 0
    assert "sample" not in reg
    reg.register(_sample_tool())
    assert len(reg) == 1
    assert "sample" in reg


def test_call_with_no_args_dict():
    reg = ToolRegistry()
    reg.register(Tool("ping", "ping", {}, lambda: "pong"))
    out = reg.call("ping")
    assert out["ok"] is True and out["result"] == "pong"
    out = reg.call("ping", "not a dict")
    assert out["ok"] is False and "must be a dict" in out["error"]


def test_call_coercion_whitespace_and_edge_values():
    reg = ToolRegistry()
    reg.register(
        Tool("half", "halve", {"x": {"type": "number", "description": "x", "required": True}},
             func=lambda x: x / 2)
    )
    assert reg.call("half", {"x": "  5  "})["result"] == 2.5
    assert reg.call("half", {"x": "-3"})["result"] == -1.5
    out = reg.call("half", {"x": ""})
    assert out["ok"] is False


def test_call_integer_rejects_float_strings():
    reg = _math_registry()
    out = reg.call("add", {"a": "3.7", "b": 1})
    assert out["ok"] is False and "expects an integer" in out["error"]
    out = reg.call("add", {"a": " 4 ", "b": 1})
    assert out["result"] == 5


def test_call_boolean_rejects_out_of_range_numbers():
    reg = ToolRegistry()
    reg.register(
        Tool("flag", "flag", {"on": {"type": "boolean", "description": "on", "required": True}},
             func=lambda on: on)
    )
    out = reg.call("flag", {"on": 2})
    assert out["ok"] is False and "expects a boolean" in out["error"]
    out = reg.call("flag", {"on": [1]})
    assert out["ok"] is False and "expects a boolean" in out["error"]


def test_call_optional_without_default_skipped():
    reg = ToolRegistry()
    seen = {}
    def func(name, nick=None):
        seen["nick"] = nick
        return name
    reg.register(
        Tool("who", "who",
             {"name": {"type": "string", "description": "n", "required": True},
              "nick": {"type": "string", "description": "n", "required": False}},
             func)
    )
    out = reg.call("who", {"name": "osiris"})
    assert out["ok"] is True and seen["nick"] is None


def test_help_text_empty_registry():
    assert ToolRegistry().help_text() == ""


# extra builtin edge cases


def test_calc_more_operators(tmp_path):
    reg = _builtins(tmp_path)
    cases = {
        "+5": 5,
        "--5": 5,
        "((2 + 3) * (4 - 1)) / 5": 3.0,
        "-7 % 3": 2,
        "2 * -3": -6,
        "100 // 3": 33,
    }
    for expr, want in cases.items():
        out = reg.call("calc", {"expression": expr})
        assert out["ok"] is True, (expr, out)
        assert out["result"] == want


def test_calc_rejects_attribute_and_tuple(tmp_path):
    reg = _builtins(tmp_path)
    for expr in ("(1).real", "(1, 2)", "{1: 2}", "'a' + 'b'"):
        out = reg.call("calc", {"expression": expr})
        assert out["ok"] is False, expr


def test_notes_overwrite_and_dotted_names(tmp_path):
    reg = _builtins(tmp_path)
    reg.call("note_write", {"name": "n", "text": "one"})
    reg.call("note_write", {"name": "n", "text": "two"})
    assert reg.call("note_read", {"name": "n"})["result"] == "two"
    assert reg.call("note_write", {"name": "my.note-1", "text": "x"})["ok"] is True
    assert reg.call("note_list", {})["result"] == ["my.note-1", "n"]


def test_file_write_overwrites_and_append_creates(tmp_path):
    reg = _builtins(tmp_path)
    reg.call("file_write", {"path": "f.txt", "content": "a"})
    reg.call("file_write", {"path": "f.txt", "content": "b"})
    assert reg.call("file_read", {"path": "f.txt"})["result"] == "b"
    reg.call("file_append", {"path": "new.txt", "content": "z"})
    assert reg.call("file_read", {"path": "new.txt"})["result"] == "z"


def test_file_list_defaults_to_workspace_root(tmp_path):
    reg = _builtins(tmp_path)
    reg.call("file_write", {"path": "root.txt", "content": "x"})
    out = reg.call("file_list", {})
    assert out["ok"] is True and "root.txt" in out["result"]


def test_file_read_directory_errors(tmp_path):
    reg = _builtins(tmp_path)
    (tmp_path / "d").mkdir()
    out = reg.call("file_read", {"path": "d"})
    assert out["ok"] is False and "directory" in out["error"]


def test_shell_captures_stderr_and_rejects_empty(tmp_path):
    reg = _builtins(tmp_path)
    out = reg.call("shell", {"command": "echo oops >&2"})
    assert out["ok"] is True
    assert out["result"]["stderr"].strip() == "oops"
    out = reg.call("shell", {"command": "   "})
    assert out["ok"] is False and "non-empty" in out["error"]
    out = reg.call("shell", {"command": "echo hi", "timeout": 0})
    assert out["ok"] is False and "positive" in out["error"]


def test_http_get_connection_refused_is_error(tmp_path):
    reg = _builtins(tmp_path)
    # port 1 is effectively never listening
    out = reg.call("http_get", {"url": "http://127.0.0.1:1/", "timeout": 2})
    assert out["ok"] is False and "failed" in out["error"]


def test_http_get_rejects_nonpositive_timeout(tmp_path):
    reg = _builtins(tmp_path)
    out = reg.call("http_get", {"url": "http://127.0.0.1/", "timeout": 0})
    assert out["ok"] is False and "positive" in out["error"]


def test_word_count_empty_and_json_values(tmp_path):
    reg = _builtins(tmp_path)
    out = reg.call("word_count", {"text": ""})
    assert out["result"] == {"words": 0, "chars": 0, "lines": 0}
    assert reg.call("json_parse", {"text": "[1, 2]"})["result"] == [1, 2]
    assert reg.call("json_parse", {"text": "42"})["result"] == 42
    out = reg.call("json_parse", {"text": ""})
    assert out["ok"] is False
