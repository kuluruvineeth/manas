import random

from datapipe.tools_en import (
    EXCHANGE_RATES,
    TIME_DATA,
    TOOLS,
    TRANSLATE_DATA,
    UNIT_FACTORS,
    WEATHER_DATA,
    tools_json,
)
from datapipe.transforms import AGENT_SYSTEM_PROMPT

MATH_TEMPLATES = [
    "What is {expr}?", "Calculate {expr} for me.", "Compute {expr}.",
    "I need the result of {expr}.", "Work out {expr}, please.",
    "Can you evaluate {expr}?", "Quick math: {expr} = ?",
]
WEATHER_TEMPLATES = [
    "What's the weather like in {city} right now?", "How's the weather in {city}?",
    "Is it nice out in {city} today?", "Tell me the current weather in {city}.",
    "I'm heading to {city} — what's the weather there?", "Current conditions in {city}?",
]
TIME_TEMPLATES = [
    "What time is it in {city}?", "Tell me the current time in {city}.",
    "What's the local time in {city} right now?", "Current time in {city}?",
]
EXCHANGE_TEMPLATES = [
    "How much is {amount} {frm} in {to}?", "Convert {amount} {frm} to {to}.",
    "What's {amount} {frm} worth in {to}?", "Exchange {amount} {frm} into {to} for me.",
]
UNIT_TEMPLATES = [
    "Convert {value} {frm} to {to}.", "How many {to} is {value} {frm}?",
    "What's {value} {frm} in {to}?", "Turn {value} {frm} into {to}.",
]
TRANSLATE_TEMPLATES = [
    "How do you say '{text}' in {lang}?", "Translate '{text}' to {lang}.",
    "What's '{text}' in {lang}?", "Give me the {lang} word for '{text}'.",
]

TIMEZONE_CITY = {
    "america/new_york": "New York", "europe/london": "London", "europe/paris": "Paris",
    "asia/kolkata": "Kolkata", "asia/tokyo": "Tokyo", "australia/sydney": "Sydney",
}
ALL_TOOL_NAMES = list(TOOLS)


def format_number(value):
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def gen_math(rng):
    a, b, c = rng.randint(2, 999), rng.randint(2, 99), rng.randint(2, 49)
    expression, value = rng.choice([
        (f"{a} + {b} * {c}", a + b * c),
        (f"({a} + {b}) * {c}", (a + b) * c),
        (f"{a} * {b} - {c}", a * b - c),
        (f"{a} * {b} + {c}", a * b + c),
        (f"{a} - {b} * {c}", a - b * c),
    ])
    return rng.choice(MATH_TEMPLATES).format(expr=expression), "calculate_math", [format_number(value)]


def gen_weather(rng):
    city = rng.choice(list(WEATHER_DATA))
    data = WEATHER_DATA[city]
    question = rng.choice(WEATHER_TEMPLATES).format(city=city.title())
    return question, "get_current_weather", [str(data["temp"]), data["condition"]]


def gen_time(rng):
    timezone = rng.choice(list(TIME_DATA))
    question = rng.choice(TIME_TEMPLATES).format(city=TIMEZONE_CITY[timezone])
    return question, "get_current_time", [TIME_DATA[timezone]]


def gen_exchange(rng):
    frm, to = rng.choice(list(EXCHANGE_RATES))
    amount = rng.choice([10, 25, 50, 100, 200, 500, 1000, 2500])
    value = amount * EXCHANGE_RATES[(frm, to)]
    question = rng.choice(EXCHANGE_TEMPLATES).format(amount=amount, frm=frm.upper(), to=to.upper())
    return question, "get_exchange_rate", [format_number(round(value, 2))]


def gen_unit(rng):
    frm, to = rng.choice(list(UNIT_FACTORS))
    value = rng.choice([1, 2, 5, 8, 10, 12, 25, 42, 60, 75, 100, 150, 250])
    converted = value * UNIT_FACTORS[(frm, to)]
    question = rng.choice(UNIT_TEMPLATES).format(value=value, frm=frm, to=to)
    return question, "unit_converter", [format_number(round(converted, 4))]


def gen_translate(rng):
    text, language = rng.choice(list(TRANSLATE_DATA))
    question = rng.choice(TRANSLATE_TEMPLATES).format(text=text, lang=language.title())
    return question, "translate_text", [TRANSLATE_DATA[(text, language)]]


GENERATORS = [gen_math, gen_weather, gen_time, gen_exchange, gen_unit, gen_translate]


def agent_row(rng, question, tool_name, gt):
    offered = {tool_name, *rng.sample(ALL_TOOL_NAMES, rng.randint(1, 3))}
    names = sorted(offered)
    rng.shuffle(names)
    return {
        "conversations": [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT, "tools": tools_json(names)},
            {"role": "user", "content": question},
            {"role": "assistant", "content": "none"},
        ],
        "gt": gt,
    }


def multi_agent_row(rng):
    first, second = rng.sample(GENERATORS, 2)
    question_1, tool_1, gt_1 = first(rng)
    question_2, tool_2, gt_2 = second(rng)
    question = f"{question_1} Also, {question_2[0].lower()}{question_2[1:]}"
    names = sorted({tool_1, tool_2, *rng.sample(ALL_TOOL_NAMES, 1)})
    rng.shuffle(names)
    return {
        "conversations": [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT, "tools": tools_json(names)},
            {"role": "user", "content": question},
            {"role": "assistant", "content": "none"},
        ],
        "gt": gt_1 + gt_2,
    }


def iter_agent_rows(count_single=42000, count_multi=8000, seed=42):
    rng = random.Random(seed)
    for _ in range(count_single):
        question, tool_name, gt = rng.choice(GENERATORS)(rng)
        yield agent_row(rng, question, tool_name, gt)
    for _ in range(count_multi):
        yield multi_agent_row(rng)


IDENTITY_QA = [
    ("Who are you?",
     "I am Manas, a lightweight language model trained entirely from scratch."),
    ("What is your name?",
     "My name is Manas. 'Manas' is a Sanskrit word meaning 'mind'."),
    ("What are you?",
     "I'm Manas, a small open-source language model — every part of me, from the tokenizer "
     "to the training loop, was built from the ground up."),
    ("Who made you?",
     "I'm Manas, built by Kuluru Vineeth as part of an open-source project that trains a "
     "language model completely from scratch."),
    ("Who created you?",
     "Kuluru Vineeth created me — I'm Manas, the result of building a full LLM training "
     "pipeline from scratch: pretraining, fine-tuning, and reinforcement learning included."),
    ("Are you ChatGPT?",
     "No, I'm not ChatGPT. I'm Manas, a much smaller independent model trained from scratch "
     "as an open-source project."),
    ("Are you GPT-4?",
     "No. I'm Manas — a tiny language model with millions, not trillions, of parameters, "
     "trained from scratch."),
    ("What does your name mean?",
     "Manas is a Sanskrit word that means 'mind'. A fitting name for a small artificial mind "
     "built from scratch."),
    ("How big are you?",
     "Manas is deliberately tiny — my dense version has about 64 million parameters, small "
     "enough to train on a single consumer GPU."),
    ("What can you do?",
     "As Manas, I can chat, answer questions, reason step by step inside my thinking tags, "
     "and call tools when they're provided to me."),
    ("Can you think?",
     "Yes — Manas reasons step by step. When it helps, I write my reasoning inside <think> "
     "tags before giving you the final answer."),
    ("What language model are you based on?",
     "I'm not based on any existing model — Manas was pretrained from random weights on an "
     "open English corpus."),
    ("Are you open source?",
     "Yes! My architecture, training code, and datasets are all open. You can train a Manas "
     "of your own from scratch."),
    ("Tell me about yourself.",
     "I'm Manas, a compact language model. I was pretrained on English web and textbook data, "
     "fine-tuned to chat, aligned with preference learning, and taught to use tools with "
     "reinforcement learning."),
    ("Where do you run?",
     "Manas is small enough to run on a laptop — and I can be exported to formats that "
     "llama.cpp, vLLM, and Ollama understand."),
]

PARAPHRASES = [
    lambda q: q,
    lambda q: q.lower(),
    lambda q: "Hey, " + q[0].lower() + q[1:],
    lambda q: q + " Please be brief.",
    lambda q: "I'm curious — " + q[0].lower() + q[1:],
]


def iter_identity_rows():
    for question, answer in IDENTITY_QA:
        for paraphrase in PARAPHRASES:
            yield {"conversations": [
                {"role": "user", "content": paraphrase(question)},
                {"role": "assistant", "content": answer},
            ]}
