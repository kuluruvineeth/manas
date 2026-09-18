import json

WEATHER_DATA = {
    "new york": {"temp": 22, "condition": "sunny"},
    "london": {"temp": 14, "condition": "cloudy"},
    "paris": {"temp": 18, "condition": "partly cloudy"},
    "berlin": {"temp": 16, "condition": "rainy"},
    "tokyo": {"temp": 26, "condition": "clear"},
    "sydney": {"temp": 20, "condition": "windy"},
    "mumbai": {"temp": 31, "condition": "humid"},
    "delhi": {"temp": 33, "condition": "hazy"},
    "bangalore": {"temp": 24, "condition": "pleasant"},
    "san francisco": {"temp": 17, "condition": "foggy"},
    "seattle": {"temp": 13, "condition": "drizzly"},
    "toronto": {"temp": 11, "condition": "overcast"},
    "singapore": {"temp": 30, "condition": "thunderstorms"},
    "dubai": {"temp": 39, "condition": "hot"},
    "amsterdam": {"temp": 15, "condition": "breezy"},
}

TIME_DATA = {
    "america/new_york": "09:30",
    "europe/london": "14:30",
    "europe/paris": "15:30",
    "asia/kolkata": "19:00",
    "asia/tokyo": "23:30",
    "australia/sydney": "00:30",
}

EXCHANGE_RATES = {
    ("usd", "eur"): 0.92,
    ("usd", "inr"): 83.20,
    ("usd", "gbp"): 0.79,
    ("usd", "jpy"): 149.50,
    ("eur", "usd"): 1.09,
    ("gbp", "usd"): 1.27,
    ("inr", "usd"): 0.012,
    ("eur", "gbp"): 0.86,
}

UNIT_FACTORS = {
    ("km", "miles"): 0.621371,
    ("miles", "km"): 1.609344,
    ("kg", "lbs"): 2.20462,
    ("lbs", "kg"): 0.453592,
    ("m", "ft"): 3.28084,
    ("ft", "m"): 0.3048,
    ("l", "gal"): 0.264172,
    ("gal", "l"): 3.785412,
}

TRANSLATE_DATA = {
    ("hello", "spanish"): "hola",
    ("thank you", "french"): "merci",
    ("good morning", "german"): "guten Morgen",
    ("goodbye", "spanish"): "adiós",
    ("please", "french"): "s'il vous plaît",
    ("welcome", "german"): "willkommen",
}

TOOLS = {
    "calculate_math": {
        "type": "function",
        "function": {
            "name": "calculate_math",
            "description": "Evaluate a mathematical expression and return the numeric result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "The math expression to evaluate, e.g. '23*7+11'"}
                },
                "required": ["expression"],
            },
        },
    },
    "get_current_weather": {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather (temperature in Celsius and condition) for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. 'London'"}
                },
                "required": ["city"],
            },
        },
    },
    "get_current_time": {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current local time for an IANA timezone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "IANA timezone, e.g. 'Europe/London'"}
                },
                "required": ["timezone"],
            },
        },
    },
    "get_exchange_rate": {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "Get the exchange rate between two currencies and optionally convert an amount.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_currency": {"type": "string", "description": "Source currency code, e.g. 'USD'"},
                    "to_currency": {"type": "string", "description": "Target currency code, e.g. 'EUR'"},
                    "amount": {"type": "number", "description": "Amount to convert (optional)"},
                },
                "required": ["from_currency", "to_currency"],
            },
        },
    },
    "unit_converter": {
        "type": "function",
        "function": {
            "name": "unit_converter",
            "description": "Convert a value between units (km/miles, kg/lbs, m/ft, l/gal).",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "The numeric value to convert"},
                    "from_unit": {"type": "string", "description": "Source unit, e.g. 'km'"},
                    "to_unit": {"type": "string", "description": "Target unit, e.g. 'miles'"},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
    "translate_text": {
        "type": "function",
        "function": {
            "name": "translate_text",
            "description": "Translate a short English phrase into a target language.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "English text to translate"},
                    "target_language": {"type": "string", "description": "Target language, e.g. 'Spanish'"},
                },
                "required": ["text", "target_language"],
            },
        },
    },
}


def tool_list(names):
    return [TOOLS[n] for n in names]


def tools_json(names):
    return json.dumps(tool_list(names), ensure_ascii=False)
