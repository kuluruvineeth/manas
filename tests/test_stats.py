import json

from datapipe.stats import FILES, file_stats, report
from datapipe.synthesize import iter_identity_rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_files_cover_all_bundle_outputs():
    assert len(FILES) == 11
    assert set(FILES.values()) == {"pretrain", "sft", "dpo", "rlaif", "agent"}


def test_file_stats_counts_errors(tmp_path):
    path = tmp_path / "bad.jsonl"
    write_jsonl(path, [{"text": "fine document here"}, {"text": "   "}])
    stats = file_stats(str(path), "pretrain", sample_size=50)
    assert stats["errors"] > 0
    assert stats["sampled"] == 50


def test_report_flags_missing_and_passes_valid(tmp_path):
    write_jsonl(tmp_path / "lora_identity.jsonl", list(iter_identity_rows()))
    text, failed = report(str(tmp_path), sample_size=20)
    assert failed
    assert "MISSING" in text
    assert "lora_identity.jsonl" in text
