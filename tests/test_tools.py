import json
import math

import pytest

from manas.tools.registry import EXECUTORS, execute, is_valid_call, parse_tool_calls
from manas.tools.safe_math import safe_math_eval


def test_arithmetic_and_math_functions():
    assert safe_math_eval("23 * 7 + 11") == 172
    assert safe_math_eval("(2 + 3) * 4") == 20
    assert safe_math_eval("7 // 2") == 3 and safe_math_eval("7 % 2") == 1
    assert safe_math_eval("sqrt(16)") == 4.0
    assert safe_math_eval("math.sqrt(16)") == 4.0
    assert math.isclose(safe_math_eval("pi"), math.pi)
    assert safe_math_eval("−5 + 2") == -3
    assert safe_math_eval("3²") == 9
    assert safe_math_eval("2 × 3") == 6


def test_it_refuses_anything_that_is_not_arithmetic():
    for dangerous in ["__import__('os').system('ls')", "open('/etc/passwd')", "[].__class__", "x = 1", "print(1)"]:
        with pytest.raises(ValueError):
            safe_math_eval(dangerous)


def test_the_power_guard_stops_a_process_killer():
    with pytest.raises(ValueError, match="too large"):
        safe_math_eval("9**9**9")
    with pytest.raises(ValueError, match="too large"):
        safe_math_eval("10**99999")
    assert safe_math_eval("2**10") == 1024


def test_empty_and_oversized_expressions_are_rejected():
    with pytest.raises(ValueError, match="empty or too long"):
        safe_math_eval("   ")
    with pytest.raises(ValueError, match="empty or too long"):
        safe_math_eval("1+" * 400 + "1")


def test_parse_tool_calls_survives_broken_json():
    text = (
        '<tool_call>\n{"name": "calculate_math", "arguments": {"expression": "2+2"}}\n</tool_call>'
        "<tool_call>\nnot json\n</tool_call>"
        '<tool_call>\n{"name": "get_current_time", "arguments": {"timezone": "Europe/London"}}\n</tool_call>'
    )
    calls = parse_tool_calls(text)
    assert [c["name"] for c in calls] == ["calculate_math", "get_current_time"]
    assert calls[0]["arguments"] == {"expression": "2+2"}
    assert parse_tool_calls("no tools here") == []


def test_validation_checks_the_offered_set_and_required_arguments():
    call = {"name": "calculate_math", "arguments": {"expression": "1+1"}}
    assert is_valid_call(call, {"calculate_math"})
    assert not is_valid_call(call, {"get_current_time"})
    assert not is_valid_call({"name": "calculate_math", "arguments": {}}, {"calculate_math"})
    assert not is_valid_call({"name": "no_such_tool", "arguments": {}}, {"no_such_tool"})


def test_every_tool_executes_deterministically():
    assert len(EXECUTORS) == 6
    results = {
        "calculate_math": ({"expression": "23*7"}, "161"),
        "get_current_weather": ({"city": "London"}, "14C, cloudy"),
        "get_current_time": ({"timezone": "Europe/London"}, "14:30"),
        "get_exchange_rate": ({"from_currency": "USD", "to_currency": "EUR", "amount": 100}, "92"),
        "unit_converter": ({"value": 10, "from_unit": "km", "to_unit": "miles"}, "6.2137"),
        "translate_text": ({"text": "hello", "target_language": "Spanish"}, "hola"),
    }
    for name, (arguments, expected) in results.items():
        assert execute({"name": name, "arguments": arguments}) == expected
        assert execute({"name": name, "arguments": arguments}) == expected


def test_execution_errors_become_observations_not_crashes():
    assert "tool error" in execute({"name": "calculate_math", "arguments": {"expression": "9**9**9"}})
    assert "unknown tool" in execute({"name": "nope", "arguments": {}})
    assert "no weather data" in execute({"name": "get_current_weather", "arguments": {"city": "Atlantis"}})
    assert json.loads('{"ok": 1}')["ok"] == 1
