import contextlib
import os

import torch

from manas.config import ManasConfig
from manas.model.causal_lm import ManasForCausalLM
from manas.training.checkpoint import SkipBatchSampler, load_resume, restore, resume_path, save_resume

CONFIG = ManasConfig(hidden_size=64, num_hidden_layers=2)


def build(seed=0):
    torch.manual_seed(seed)
    model = ManasForCausalLM(CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return model, optimizer, scaler


def test_resume_path_matches_the_checkpoint_naming():
    assert resume_path("out", "pretrain", CONFIG).endswith("pretrain_64_resume.pth")
    moe = ManasConfig(hidden_size=64, num_hidden_layers=2, use_moe=True)
    assert resume_path("out", "pretrain", moe).endswith("pretrain_64_moe_resume.pth")


def test_bundle_round_trip_restores_optimizer_state(tmp_path):
    model, optimizer, scaler = build()
    ids = torch.randint(3, CONFIG.vocab_size, (2, 8))
    model(ids, labels=ids).loss.backward()
    optimizer.step()
    path = save_resume(resume_path(str(tmp_path), "pretrain", CONFIG), model, optimizer, scaler, epoch=1, step=250)
    assert not os.path.exists(path + ".tmp")

    fresh_model, fresh_optimizer, fresh_scaler = build(seed=1)
    assert not torch.allclose(fresh_model.lm_head.weight, model.lm_head.weight)
    bundle = load_resume(path)
    epoch, step = restore(bundle, fresh_model, fresh_optimizer, fresh_scaler)
    assert (epoch, step) == (1, 250)
    torch.testing.assert_close(fresh_model.lm_head.weight, model.lm_head.weight)
    restored_state = fresh_optimizer.state_dict()["state"]
    original_state = optimizer.state_dict()["state"]
    assert set(restored_state) == set(original_state) and restored_state
    for key, values in original_state.items():
        for name, value in values.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(restored_state[key][name], value)
            else:
                assert restored_state[key][name] == value


def test_missing_bundle_is_not_an_error(tmp_path):
    assert load_resume(str(tmp_path / "nothing.pth")) is None


def test_step_is_rescaled_when_the_gpu_count_changes(tmp_path):
    model, optimizer, scaler = build()
    path = save_resume(
        resume_path(str(tmp_path), "pretrain", CONFIG), model, optimizer, scaler, epoch=0, step=800, world_size=4
    )
    assert load_resume(path, world_size=4)["step"] == 800
    assert load_resume(path, world_size=2)["step"] == 1600
    assert load_resume(path, world_size=8)["step"] == 400


def test_skip_batch_sampler_drops_consumed_batches():
    sampler = SkipBatchSampler(list(range(10)), batch_size=3, skip_batches=2)
    assert len(sampler) == 2
    assert list(sampler) == [[6, 7, 8], [9]]
    assert list(SkipBatchSampler(list(range(10)), 3, 0)) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert list(SkipBatchSampler(list(range(4)), 2, 5)) == []


def test_atomic_write_leaves_the_old_bundle_intact_on_failure(tmp_path):
    model, optimizer, scaler = build()
    path = resume_path(str(tmp_path), "pretrain", CONFIG)
    save_resume(path, model, optimizer, scaler, epoch=0, step=10)
    original = torch.load(path, weights_only=False)["step"]
    with contextlib.suppress(AttributeError):
        save_resume(path, model, optimizer, "not a scaler", epoch=1, step=99)
    assert torch.load(path, weights_only=False)["step"] == original
