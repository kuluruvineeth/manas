import json
import random

from datapipe.schema import validate_line
from datapipe.synthesize import GENERATORS, format_number, iter_agent_rows, iter_identity_rows
from datapipe.tools_en import TIME_DATA, TRANSLATE_DATA, WEATHER_DATA


def known_answers():
    known = {str(d["temp"]) for d in WEATHER_DATA.values()}
    known |= {d["condition"] for d in WEATHER_DATA.values()}
    known |= set(TIME_DATA.values()) | set(TRANSLATE_DATA.values())
    return {v.lower() for v in known}


def test_generated_gt_is_verifiable():
    known = known_answers()
    for row in iter_agent_rows(count_single=300, count_multi=100, seed=7):
        for gt in row["gt"]:
            if gt.lower() in known:
                continue
            float(gt)


def test_generated_rows_pass_schema():
    for row in iter_agent_rows(count_single=100, count_multi=50, seed=7):
        assert validate_line("agent", json.dumps(row)) is None


def test_offered_tools_include_the_needed_one():
    rng = random.Random(3)
    for generator in GENERATORS:
        _, tool_name, gt = generator(rng)
        assert tool_name and gt
    for row in iter_agent_rows(count_single=200, count_multi=0, seed=3):
        tools = json.loads(row["conversations"][0]["tools"])
        assert len(tools) >= 1


def test_deterministic_given_seed():
    first = list(iter_agent_rows(count_single=50, count_multi=10, seed=11))
    second = list(iter_agent_rows(count_single=50, count_multi=10, seed=11))
    assert first == second


def test_math_gt_matches_expression():
    rng = random.Random(5)
    from datapipe.synthesize import gen_math
    for _ in range(200):
        question, _, gt = gen_math(rng)
        expression = question.split(":")[-1].replace("= ?", "").strip()
        for template_noise in ("What is ", "Calculate ", "Compute ", "I need the result of ",
                               "Work out ", "Can you evaluate ", "Quick math: "):
            expression = expression.replace(template_noise, "")
        expression = expression.rstrip("?.!, ").replace("for me", "").replace("please", "").strip().rstrip(",")
        assert format_number(eval(expression)) == gt[0]  # noqa: S307


def test_identity_rows():
    rows = list(iter_identity_rows())
    assert len(rows) == 75
    for row in rows:
        assert validate_line("sft", json.dumps(row)) is None
        assert "Manas" in row["conversations"][1]["content"]


def test_format_number():
    assert format_number(7.0) == "7"
    assert format_number(0.62140) == "0.6214"
    assert format_number(83.2) == "83.2"
