import json
import re

from datapipe.tools_en import (
    EXCHANGE_RATES,
    TIME_DATA,
    TOOLS,
    TRANSLATE_DATA,
    UNIT_FACTORS,
    WEATHER_DATA,
    tool_list,
)
from manas.tools.safe_math import safe_math_eval

TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
MAX_OBSERVATION_CHARS = 2048


def _number(value):
    text = f"{value:.4f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)
    return text


def run_calculate_math(arguments):
    return _number(safe_math_eval(arguments["expression"]))


def run_get_current_weather(arguments):
    city = str(arguments["city"]).strip().lower()
    if city not in WEATHER_DATA:
        return f"no weather data for {arguments['city']}"
    data = WEATHER_DATA[city]
    return f"{data['temp']}C, {data['condition']}"


def run_get_current_time(arguments):
    zone = str(arguments["timezone"]).strip().lower()
    if zone not in TIME_DATA:
        return f"no time data for {arguments['timezone']}"
    return TIME_DATA[zone]


def run_get_exchange_rate(arguments):
    pair = (str(arguments["from_currency"]).lower(), str(arguments["to_currency"]).lower())
    if pair not in EXCHANGE_RATES:
        return f"no rate for {pair[0]} to {pair[1]}"
    rate = EXCHANGE_RATES[pair]
    amount = arguments.get("amount")
    if amount is None:
        return _number(rate)
    return _number(round(float(amount) * rate, 2))


def run_unit_converter(arguments):
    pair = (str(arguments["from_unit"]).lower(), str(arguments["to_unit"]).lower())
    if pair not in UNIT_FACTORS:
        return f"cannot convert {pair[0]} to {pair[1]}"
    return _number(round(float(arguments["value"]) * UNIT_FACTORS[pair], 4))


def run_translate_text(arguments):
    key = (str(arguments["text"]).strip().lower(), str(arguments["target_language"]).strip().lower())
    if key not in TRANSLATE_DATA:
        return f"no translation for '{arguments['text']}'"
    return TRANSLATE_DATA[key]


EXECUTORS = {
    "calculate_math": run_calculate_math,
    "get_current_weather": run_get_current_weather,
    "get_current_time": run_get_current_time,
    "get_exchange_rate": run_get_exchange_rate,
    "unit_converter": run_unit_converter,
    "translate_text": run_translate_text,
}

REQUIRED_ARGS = {
    name: tuple(spec["function"]["parameters"]["required"]) for name, spec in TOOLS.items()
}


def parse_tool_calls(text):
    calls = []
    for block in TOOL_CALL_BLOCK.findall(text):
        try:
            call = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(call, dict) and isinstance(call.get("name"), str):
            calls.append({"name": call["name"], "arguments": call.get("arguments") or {}})
    return calls


def is_valid_call(call, offered_names):
    if call["name"] not in offered_names or call["name"] not in REQUIRED_ARGS:
        return False
    arguments = call["arguments"]
    return isinstance(arguments, dict) and all(key in arguments for key in REQUIRED_ARGS[call["name"]])


def execute(call):
    executor = EXECUTORS.get(call["name"])
    if executor is None:
        return f"unknown tool {call['name']}"
    try:
        result = executor(call["arguments"])
    except (KeyError, ValueError, TypeError) as error:
        result = f"tool error: {error}"
    return str(result)[:MAX_OBSERVATION_CHARS]


def tools_for(names):
    return tool_list(list(names))
