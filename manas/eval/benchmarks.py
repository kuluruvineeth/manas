LETTERS = ["A", "B", "C", "D", "E"]


def mmlu_records(rows):
    records = []
    for row in rows:
        choices = list(row.get("choices") or [])
        answer = row.get("answer")
        if not choices or not isinstance(answer, int) or answer >= len(choices):
            continue
        records.append({
            "context": f"Question: {row['question'].strip()}\nAnswer:",
            "choices": [f" {choice}" for choice in choices],
            "answer": answer,
        })
    return records


def labelled_records(rows, question_key="question"):
    records = []
    for row in rows:
        choices = row.get("choices") or {}
        texts, labels = list(choices.get("text") or []), list(choices.get("label") or [])
        key = row.get("answerKey")
        if not texts or key not in labels:
            continue
        records.append({
            "context": f"Question: {row[question_key].strip()}\nAnswer:",
            "choices": [f" {text}" for text in texts],
            "answer": labels.index(key),
        })
    return records


def hellaswag_records(rows):
    records = []
    for row in rows:
        endings = list(row.get("endings") or [])
        try:
            answer = int(row.get("label"))
        except (TypeError, ValueError):
            continue
        if not endings or answer >= len(endings):
            continue
        context = f"{row.get('activity_label', '').strip()}: {row['ctx'].strip()}".strip(": ")
        records.append({
            "context": context,
            "choices": [f" {ending.strip()}" for ending in endings],
            "answer": answer,
        })
    return records


BENCHMARKS = {
    "mmlu": {"repo": "cais/mmlu", "config": "all", "split": "test", "builder": mmlu_records},
    "arc_easy": {"repo": "allenai/ai2_arc", "config": "ARC-Easy", "split": "test", "builder": labelled_records},
    "arc_challenge": {
        "repo": "allenai/ai2_arc", "config": "ARC-Challenge", "split": "test", "builder": labelled_records
    },
    "openbookqa": {
        "repo": "allenai/openbookqa", "config": "main", "split": "test",
        "builder": lambda rows: labelled_records(rows, question_key="question_stem"),
    },
    "hellaswag": {"repo": "Rowan/hellaswag", "config": None, "split": "validation", "builder": hellaswag_records},
}


def load_benchmark(name, limit=None):
    from datasets import load_dataset

    spec = BENCHMARKS[name]
    dataset = load_dataset(spec["repo"], name=spec["config"], split=spec["split"])
    rows = list(dataset.select(range(min(limit, len(dataset))))) if limit else list(dataset)
    return spec["builder"](rows)
