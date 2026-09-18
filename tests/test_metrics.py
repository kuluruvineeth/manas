import torch

from manas.config import ManasConfig
from manas.model.causal_lm import ManasForCausalLM
from manas.training.loop import batch_tokens, build_parser, fit, language_model_loss, split_validation
from manas.training.metrics import MetricsLogger, evaluate, read_metrics


class Repeat(torch.utils.data.Dataset):
    def __init__(self, ids, n):
        self.ids, self.n = ids, n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return self.ids, self.ids


def small_model():
    torch.manual_seed(0)
    config = ManasConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2)
    return ManasForCausalLM(config), config


def test_logger_writes_jsonl_records(tmp_path):
    path = tmp_path / "run" / "metrics.jsonl"
    logger = MetricsLogger(str(path))
    logger.log(1, loss=2.5, lr=1e-3)
    logger.log(2, val_loss=2.4)
    logger.close()
    rows = read_metrics(path)
    assert [r["step"] for r in rows] == [1, 2]
    assert rows[0]["loss"] == 2.5 and rows[1]["val_loss"] == 2.4
    assert all("elapsed" in r for r in rows)


def test_split_validation_holds_out_a_slice():
    dataset = Repeat(torch.zeros(4, dtype=torch.long), 100)
    train, val = split_validation(dataset, 256, seed=1)
    assert len(train) == 90 and len(val) == 10
    tiny = Repeat(torch.zeros(4, dtype=torch.long), 5)
    assert split_validation(tiny, 256, seed=1) == (tiny, None)


def test_evaluate_restores_train_mode_and_averages():
    model, config = small_model()
    ids = torch.randint(3, config.vocab_size, (8,))
    loader = torch.utils.data.DataLoader(Repeat(ids, 6), batch_size=2)
    model.train()
    from contextlib import nullcontext

    loss = evaluate(model, loader, language_model_loss, nullcontext(), max_batches=2)
    assert loss > 0 and model.training


def test_fit_logs_train_and_validation_curves(tmp_path):
    model, config = small_model()
    ids = torch.randint(3, config.vocab_size, (16,))
    parser = build_parser("x", save_weight="demo", batch_size=4, learning_rate=2e-3, accumulation_steps=1,
                          max_seq_len=16, data_path="", from_weight="none")
    args = parser.parse_args(["--epochs", "2", "--device", "cpu", "--num_workers", "0", "--save_dir", str(tmp_path),
                              "--log_interval", "2", "--eval_interval", "4", "--save_interval", "100",
                              "--val_samples", "8"])
    fit(args, model, Repeat(ids, 48), language_model_loss)
    rows = read_metrics(tmp_path / "demo_64_metrics.jsonl")
    train_rows = [r for r in rows if "loss" in r]
    val_rows = [r for r in rows if "val_loss" in r]
    assert len(train_rows) >= 4 and len(val_rows) >= 2
    assert all(r["grad_norm"] is not None and r["tokens_per_sec"] > 0 for r in train_rows)
    assert val_rows[-1]["val_loss"] < val_rows[0]["val_loss"]


def test_batch_tokens_counts_ids():
    assert batch_tokens((torch.zeros(2, 5), torch.zeros(2, 5))) == 10
    assert batch_tokens({"a": torch.zeros(3), "b": 7}) == 3
