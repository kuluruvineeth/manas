import torch

from manas.config import ManasConfig
from manas.model.causal_lm import ManasForCausalLM
from manas.training.loop import build_parser, fit, language_model_loss


class Repeat(torch.utils.data.Dataset):
    def __init__(self, ids, n):
        self.ids, self.n = ids, n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return self.ids, self.ids


def test_build_parser_merges_stage_defaults():
    parser = build_parser("x", save_weight="demo", batch_size=4, learning_rate=1e-3, accumulation_steps=2,
                          max_seq_len=64, data_path="d.jsonl", from_weight="none")
    args = parser.parse_args(["--epochs", "3", "--learning_rate", "2e-3"])
    assert (args.save_weight, args.batch_size, args.epochs, args.learning_rate) == ("demo", 4, 3, 2e-3)
    assert args.hidden_size == 768 and args.max_steps == 0


def test_fit_learns_a_repeated_sequence(tmp_path):
    torch.manual_seed(0)
    config = ManasConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2)
    model = ManasForCausalLM(config)
    ids = torch.randint(3, config.vocab_size, (24,))
    parser = build_parser("x", save_weight="demo", batch_size=4, learning_rate=3e-3, accumulation_steps=1,
                          max_seq_len=24, data_path="", from_weight="none")
    args = parser.parse_args(["--epochs", "10", "--device", "cpu", "--num_workers", "0",
                              "--save_dir", str(tmp_path), "--log_interval", "1000", "--save_interval", "1000"])
    before = language_model_loss(model, (ids.unsqueeze(0), ids.unsqueeze(0))).item()
    after = fit(args, model, Repeat(ids, 16), language_model_loss)
    assert after < before * 0.7
    assert (tmp_path / "demo_64.pth").exists()


def test_max_steps_stops_early_and_saves(tmp_path):
    config = ManasConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2)
    model = ManasForCausalLM(config)
    ids = torch.randint(3, config.vocab_size, (16,))
    parser = build_parser("x", save_weight="demo", batch_size=4, learning_rate=1e-3, accumulation_steps=1,
                          max_seq_len=16, data_path="", from_weight="none")
    args = parser.parse_args(["--epochs", "5", "--device", "cpu", "--num_workers", "0", "--max_steps", "2",
                              "--save_dir", str(tmp_path), "--log_interval", "1", "--save_interval", "100"])
    fit(args, model, Repeat(ids, 40), language_model_loss)
    assert (tmp_path / "demo_64.pth").exists()
