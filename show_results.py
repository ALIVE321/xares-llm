"""
Show evaluation results from experiments/train_task1_config or train_task2_config.
Scores are read from each run's scores.tsv (Task, score, weight).
If scores.tsv is incomplete or missing, fallback to score_*.yaml per-task cache files.
"""
import os
import re
import glob
import argparse
import yaml

# Paths under experiments/
TASK1_CONFIG_DIR = "experiments/train_task1_config"
TASK2_CONFIG_DIR = "experiments/train_task2_config"

# Task1: order with blank lines between groups. Use TSV Task names (eval_* or Overall).
TASK1_ORDER = [
    ["eval_esc-50", "eval_fsd50k", "eval_urbansound8k", "eval_fsdkaggle2018"],
    ["eval_gtzan", "eval_nsynth", "eval_freemusicarchive"],
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
    if not os.path.isfile(path):
        return out
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


# 匹配 score_*.yaml 中的浮点数值（兼容 numpy yaml 序列化）
_SCORE_RE = re.compile(r"score:\s*([\d.eE+\-]+)")


def load_scores_from_yaml(run_dir):
    """从 score_*.yaml 文件中收集已完成的 per-task 分数。
    使用正则提取数值，避免 yaml.safe_load 无法处理 numpy scalar 等问题。"""
    out = {}
    for yaml_path in glob.glob(os.path.join(run_dir, "score_*.yaml")):
        basename = os.path.basename(yaml_path)  # score_eval_xxx.yaml
        task_name = basename[len("score_"):-len(".yaml")]  # eval_xxx
        with open(yaml_path) as f:
            content = f.read()
        m = _SCORE_RE.search(content)
        if m:
            out[task_name] = float(m.group(1))
    return out


def collect_scores(run_dir):
    """优先从 scores.tsv 加载，缺失的部分用 score_*.yaml 补充。"""
    scores = load_scores_tsv(os.path.join(run_dir, "scores.tsv"))
    yaml_scores = load_scores_from_yaml(run_dir)
    # yaml 补充 tsv 中缺失的 task
    for task, score in yaml_scores.items():
        if task not in scores:
            scores[task] = score
    return scores


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
        run_dir = os.path.join(base_dir, run)
        scores = collect_scores(run_dir)
        if not scores:
            continue
        # 有结果但不全时标记 (partial)
        all_tasks = [t for g in TASK1_ORDER for t in g]
        has_all = all(t in scores for t in all_tasks if t != "Overall")
        suffix = "" if has_all else " (partial)"
        print(f"\n--- {run}{suffix} ---")
        for group in TASK1_ORDER:
            if not group:
                print()
                continue
            for task in group:
                val = scores.get(task, "—")
                if isinstance(val, float):
                    val = f"{val:.3f}"
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
        run_dir = os.path.join(base_dir, run)
        scores = collect_scores(run_dir)
        if not scores:
            continue
        all_tasks = [t for t in TASK2_ORDER if t != "Overall"]
        has_all = all(t in scores for t in all_tasks)
        suffix = "" if has_all else " (partial)"
        print(f"\n--- {run}{suffix} ---")
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
