import argparse

from manas.training.metrics import read_metrics

PAPER, INK, MUTED = "#f8f1df", "#3a3026", "#6b6152"
BLUE, ORANGE, GREEN, PURPLE = "#54b0d6", "#eb9c3f", "#5cb878", "#8d84c6"


def series(rows, key):
    points = [(r["step"], r[key]) for r in rows if r.get(key) is not None]
    return [p[0] for p in points], [p[1] for p in points]


def plot(rows, title, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), facecolor=PAPER)
    fig.suptitle(title, color=INK, fontsize=18)
    panels = [
        (axes[0][0], [("loss", BLUE, "train"), ("val_loss", ORANGE, "held-out")], "loss"),
        (axes[0][1], [("lr", GREEN, "learning rate")], "learning rate"),
        (axes[1][0], [("grad_norm", PURPLE, "gradient norm")], "gradient norm"),
        (axes[1][1], [("tokens_per_sec", INK, "tokens / s")], "throughput"),
    ]
    for ax, curves, label in panels:
        ax.set_facecolor(PAPER)
        for key, color, name in curves:
            steps, values = series(rows, key)
            if steps:
                ax.plot(steps, values, color=color, linewidth=2, label=name, marker="o" if len(steps) < 40 else None)
        ax.set_title(label, color=INK, loc="left")
        ax.set_xlabel("step", color=MUTED)
        ax.tick_params(colors=MUTED)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(MUTED)
        if len(curves) > 1:
            ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, facecolor=PAPER)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from a metrics.jsonl")
    parser.add_argument("metrics", help="path to <stage>_<hidden>_metrics.jsonl")
    parser.add_argument("--out", default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    rows = read_metrics(args.metrics)
    out = args.out or args.metrics.replace("_metrics.jsonl", "_curves.png")
    plot(rows, args.title or args.metrics.split("/")[-1].replace("_metrics.jsonl", ""), out)
    print(out)


if __name__ == "__main__":
    main()
