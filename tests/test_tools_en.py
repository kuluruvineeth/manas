import json

from datapipe.tools_en import (
    EXCHANGE_RATES,
    TIME_DATA,
    TOOLS,
    TRANSLATE_DATA,
    UNIT_FACTORS,
    WEATHER_DATA,
    tool_list,
    tools_json,
)


def test_tool_schemas_well_formed():
    for name, tool in TOOLS.items():
        assert tool["type"] == "function"
        fn = tool["function"]
        assert fn["name"] == name
        assert fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        for required in params["required"]:
            assert required in params["properties"]


def test_tables_consistent():
    for city, data in WEATHER_DATA.items():
        assert city == city.lower()
        assert isinstance(data["temp"], int) and isinstance(data["condition"], str)
    assert all(tz == tz.lower() for tz in TIME_DATA)
    assert all(rate > 0 for rate in EXCHANGE_RATES.values())
    assert all(factor > 0 for factor in UNIT_FACTORS.values())
    assert all(isinstance(v, str) and v for v in TRANSLATE_DATA.values())


def test_unit_factors_are_inverses():
    for (frm, to), factor in UNIT_FACTORS.items():
        assert abs(factor * UNIT_FACTORS[(to, frm)] - 1.0) < 1e-4


def test_tools_json_round_trip():
    names = ["calculate_math", "get_current_weather"]
    parsed = json.loads(tools_json(names))
    assert parsed == tool_list(names)
    assert [t["function"]["name"] for t in parsed] == names
