from datapipe.sources import SOURCES, by_kind, kinds


def test_registry_shape():
    assert len(SOURCES) >= 15
    for source in SOURCES:
        assert source.repo and source.kind and source.transform
        assert source.license and source.why


def test_pretrain_weights_sum_to_one():
    weights = [s.weight for s in by_kind("pretrain")]
    assert all(w is not None for w in weights)
    assert abs(sum(weights) - 1.0) < 1e-9


def test_no_test_split_leakage():
    assert all(s.split != "test" for s in SOURCES)


def test_kinds_cover_pipeline():
    assert {"pretrain", "sft", "dpo", "exam", "medical", "math"} <= set(kinds())
