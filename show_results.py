"""
Show evaluation results from experiments/train_task1_config or train_task2_config.
Scores are read from each run's scores.tsv (Task, score, weight).
"""
import os
import argparse

# Paths under experiments/
TASK1_CONFIG_DIR = "experiments/train_task1_config"
TASK2_CONFIG_DIR = "experiments/train_task2_config"

# Task1: order with blank lines between groups. Use TSV Task names (eval_* or Overall).
TASK1_ORDER = [
    ["eval_esc-50", "eval_fsd50k", "eval_urbansound8k", "eval_fsdkaggle2018"],
    [],  # blank line
    ["eval_gtzan", "eval_nsynth", "eval_freemusicarchive"],
    [],  # blank line
    [
        "eval_speechcommandsv1",
        "eval_libricount",
        "eval_voxlingua33",
        "eval_voxceleb1",
        "eval_asvspoof2015",
        "eval_fluentspeechcommands",
        "eval_vocalsound",
        "eval_cremad",
    ],
    [],  # blank line
    ["Overall"],
]

# Task2: order (no blank lines between)
TASK2_ORDER = [
    "eval_librispeech",
    "eval_aishell-1",
    "eval_clotho",
    "eval_songdescriber",
    "eval_mecat",
    "Overall",
]


def load_scores_tsv(path):
    """Load scores.tsv into a dict: Task -> score (float)."""
    out = {}
    with open(path) as f:
        lines = f.readlines()
    if not lines:
        return out
    # header: Task\tscore\tweight
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            task, score = parts[0], parts[1]
            try:
                out[task] = float(score)
            except ValueError:
                out[task] = score
    return out


def show_task1(base_dir, model=None):
    """Print task1 run(s) with scores in the requested order (with blank lines)."""
    if not os.path.isdir(base_dir):
        print(f"Error: directory not found: {base_dir}")
        return
    if model is not None:
        run_dir = os.path.join(base_dir, model)
        if not os.path.isdir(run_dir):
            print(f"Error: model run not found: {run_dir}")
            return
        runs = [model]
    else:
        runs = [
            d
            for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]
        runs.sort()
    for run in runs:
        scores_path = os.path.join(base_dir, run, "scores.tsv")
        if not os.path.isfile(scores_path):
            continue
        scores = load_scores_tsv(scores_path)
        print(f"\n--- {run} ---")
        for group in TASK1_ORDER:
            if not group:
                print()
                continue
            for task in group:
                val = scores.get(task, "—")
                if isinstance(val, float):
                    val = f"{val:.3f}"
                # display name: strip eval_ for brevity
                name = task.replace("eval_", "") if task != "Overall" else "Overall"
                print(f"  {name:25} {val}")
    print()


def show_task2(base_dir, model=None):
    """Print task2 run(s) with scores in the requested order."""
    if not os.path.isdir(base_dir):
        print(f"Error: directory not found: {base_dir}")
        return
    if model is not None:
        run_dir = os.path.join(base_dir, model)
        if not os.path.isdir(run_dir):
            print(f"Error: model run not found: {run_dir}")
            return
        runs = [model]
    else:
        runs = [
            d
            for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]
        runs.sort()
    for run in runs:
        scores_path = os.path.join(base_dir, run, "scores.tsv")
        if not os.path.isfile(scores_path):
            continue
        scores = load_scores_tsv(scores_path)
        print(f"\n--- {run} ---")
        for task in TASK2_ORDER:
            val = scores.get(task, "—")
            if isinstance(val, float):
                val = f"{val:.3f}"
            name = task.replace("eval_", "") if task != "Overall" else "Overall"
            print(f"  {name:25} {val}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Show evaluation results from experiments (task1 or task2)."
    )
    parser.add_argument(
        "task",
        type=str,
        choices=["1", "2", "task1", "task2"],
        help="Which task config to show: 1 (train_task1_config) or 2 (train_task2_config)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Show only this run/model (folder name, e.g. 400k_mae_large_laionss_captionstew_random_mask08_lr2e-4_lrb20k_warm500_bs1000).",
    )
    parser.add_argument(
        "--experiments-dir",
        type=str,
        default=".",
        help="Path to repo root (default: current dir). Config paths are relative to this.",
    )
    args = parser.parse_args()
    root = os.path.abspath(args.experiments_dir)
    if args.task in ("1", "task1"):
        config_dir = os.path.join(root, TASK1_CONFIG_DIR)
        show_task1(config_dir, model=args.model)
    else:
        config_dir = os.path.join(root, TASK2_CONFIG_DIR)
        show_task2(config_dir, model=args.model)


if __name__ == "__main__":
    main()
