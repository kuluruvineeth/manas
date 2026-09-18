import pytest

from datapipe.push import FILE_NOTES, build_card, push
from datapipe.sources import SOURCES
from datapipe.stats import FILES


def test_card_covers_every_file_and_source():
    card = build_card()
    assert set(FILE_NOTES) == set(FILES)
    for name in FILES:
        assert f"`{name}`" in card
    for source in SOURCES:
        assert source.repo in card
        assert source.license in card
    assert "test split" in card


def test_push_refuses_invalid_bundle(tmp_path):
    with pytest.raises(SystemExit, match="not pushing"):
        push(dataset_dir=str(tmp_path), dry_run=True)
