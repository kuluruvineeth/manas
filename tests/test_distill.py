import json
import math
import sys
from pathlib import Path

import torch

from manas.config import ManasConfig
from manas.model.causal_lm import ManasForCausalLM
from manas.training.distill import distill_step, distillation_loss
from manas.training.utils import save_weights, weight_path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trainer"))

from train_distillation import parse_args, train  # noqa: E402

STUDENT = ManasConfig(hidden_size=64, num_hidden_layers=2)
TEACHER = ManasConfig(hidden_size=64, num_hidden_layers=2, use_moe=True)


def test_distillation_loss_is_zero_for_identical_logits_and_grows_with_disagreement():
    logits = torch.randn(5, 10)
    assert math.isclose(distillation_loss(logits, logits, temperature=2.0).item(), 0.0, abs_tol=1e-6)
    other = torch.randn(5, 10)
    assert distillation_loss(logits, other).item() > 0
    assert distillation_loss(logits, other, temperature=2.0).item() > 0


def test_distill_step_blends_hard_and_soft_losses():
    torch.manual_seed(0)
    student = ManasForCausalLM(STUDENT)
    teacher = ManasForCausalLM(TEACHER).eval()
    ids = torch.randint(3, STUDENT.vocab_size, (2, 8))
    labels = ids.clone()
    labels[:, :3] = -100
    hard_only = distill_step(student, (ids, labels), teacher=None, alpha=0.5, temperature=1.5)
    blended = distill_step(student, (ids, labels), teacher=teacher, alpha=0.5, temperature=1.5)
    soft_only = distill_step(student, (ids, labels), teacher=teacher, alpha=0.0, temperature=1.5)
    assert hard_only.item() > 0 and blended.item() > 0 and soft_only.item() > 0
    torch.testing.assert_close(blended, 0.5 * hard_only + 0.5 * soft_only, atol=1e-5, rtol=1e-5)


def test_distillation_trainer_pulls_student_towards_teacher(tmp_path):
    torch.manual_seed(0)
    save_weights(ManasForCausalLM(STUDENT), weight_path(str(tmp_path), "full_sft", STUDENT))
    save_weights(ManasForCausalLM(TEACHER), weight_path(str(tmp_path), "full_sft", TEACHER))
    data = tmp_path / "sft.jsonl"
    with open(data, "w", encoding="utf-8") as f:
        for _ in range(8):
            f.write(json.dumps({"conversations": [
                {"role": "user", "content": "name a fruit"}, {"role": "assistant", "content": "apple"},
            ]}) + "\n")
    args = parse_args(["--data_path", str(data), "--save_dir", str(tmp_path), "--device", "cpu",
                       "--hidden_size", "64", "--num_hidden_layers", "2", "--teacher_hidden_size", "64",
                       "--teacher_num_layers", "2", "--max_seq_len", "32", "--batch_size", "4", "--epochs", "2",
                       "--learning_rate", "1e-3", "--num_workers", "0", "--log_interval", "1000",
                       "--save_interval", "1000", "--eval_interval", "1000"])
    assert args.teacher_use_moe == 1 and args.alpha == 0.5 and args.temperature == 1.5
    final = train(args)
    assert final > 0
    assert (tmp_path / "full_dist_64.pth").exists()
