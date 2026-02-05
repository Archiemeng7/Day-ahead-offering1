# ro/scripts/dat_scripts/offering_network_gen_eval_table.py
"""
Generate evaluation commands and tables for offering_network problem.

Usage:
    python -m ro.scripts.dat_scripts.offering_network_gen_eval_table --cmd_type run
    python -m ro.scripts.dat_scripts.offering_network_gen_eval_table --cmd_type eval
    python -m ro.scripts.dat_scripts.offering_network_gen_eval_table --cmd_type all --n_instances 25
"""

import os
import sys
import glob
import pickle as pkl
import numpy as np
import pandas as pd
from argparse import ArgumentParser

# Add project root to path (same pattern as original script)
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

import ro.params as params
from ro.utils import factory_get_path


def run_evaluation(args):
    """Generate and run evaluation instances."""
    print("=" * 60)
    print("Step 1: Generating test instances...")
    print("=" * 60)

    # Generate test instances
    cmd = (
        f"python -m ro.scripts.05_generate_instances_offering "
        f"--problem {args.problem} --n_instances {args.n_instances}"
    )
    print(f"Running: {cmd}")
    os.system(cmd)

    print("\n" + "=" * 60)
    print("Step 2: Running ML-CCG evaluation...")
    print("=" * 60)

    # Run ML-CCG evaluation
    cmd = f"python -m ro.scripts.06_eval_ml_ccg_offering --problem {args.problem}"
    cmd += f" --max_iterations {args.max_iterations}"
    cmd += f" --baseline_time_limit {args.baseline_time_limit}"
    cmd += f" --verbose {args.verbose}"
    print(f"Running: {cmd}")
    os.system(cmd)

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


def _get_eval_dir(problem: str):
    cfg = getattr(params, problem)
    get_path = factory_get_path(problem)

    # test_instances directory (created by 05_generate_instances_offering)
    test_inst_dir = get_path(cfg.data_path, cfg, "test_instances/")
    prob_dir = os.path.dirname(test_inst_dir.rstrip("/"))
    eval_dir = os.path.join(prob_dir, "eval_results")
    return eval_dir, prob_dir


def load_results(args):
    """Load evaluation results."""
    eval_dir, _ = _get_eval_dir(args.problem)

    if not os.path.isdir(eval_dir):
        print(f"Evaluation directory not found: {eval_dir}")
        return None

    result_files = sorted(glob.glob(os.path.join(eval_dir, "results_*.pkl")))
    if not result_files:
        print(f"No result files found in: {eval_dir}")
        return None

    results = []
    for fp in result_files:
        try:
            with open(fp, "rb") as f:
                res = pkl.load(f)
                results.append(res)
        except Exception as e:
            print(f"Error loading {fp}: {e}")

    return results


def generate_eval_table(args):
    """Generate evaluation summary table."""
    print("Loading evaluation results...")
    results = load_results(args)

    if not results:
        print("No results to analyze!")
        print("Run evaluation first:")
        print(
            "  python -m ro.scripts.dat_scripts.offering_network_gen_eval_table --cmd_type run"
        )
        return

    print(f"Loaded {len(results)} result files")

    rows = []
    for res in results:
        inst = res.get("instance", {}) if isinstance(res.get("instance", {}), dict) else {}

        row = {
            "instance": res.get("inst_name", "unknown"),
            "scenario_id": inst.get("scenario_id", -1),
            "algo_obj": res.get("algo_obj"),
            "baseline_obj": res.get("baseline_obj"),
            "algo_time": res.get("algo_time"),
            "baseline_time": res.get("baseline_time"),
            "algo_status": res.get("algo_status"),
            "baseline_status": res.get("baseline_status"),
        }

        # gap (%): positive means algo better if objective is minimization
        if row["algo_obj"] is not None and row["baseline_obj"] is not None:
            base = float(row["baseline_obj"])
            algo = float(row["algo_obj"])
            if abs(base) > 1e-10:
                row["gap_pct"] = 100.0 * (base - algo) / abs(base)
            else:
                row["gap_pct"] = 0.0

            best = min(algo, base)
            if abs(best) > 1e-10:
                row["re_algo"] = 100.0 * abs(algo - best) / abs(best)
                row["re_baseline"] = 100.0 * abs(base - best) / abs(best)
            else:
                row["re_algo"] = 0.0
                row["re_baseline"] = 0.0

        # speedup
        if row["algo_time"] is not None and row["baseline_time"] is not None:
            at = float(row["algo_time"])
            bt = float(row["baseline_time"])
            if at > 1e-12:
                row["speedup"] = bt / at

        # iterations / scenarios from opt_stats (if present)
        opt_stats = res.get("opt_stats", {}) if isinstance(res.get("opt_stats", {}), dict) else {}
        if "n_iterations" in opt_stats:
            row["n_iterations"] = opt_stats["n_iterations"]
        if "xi_added" in opt_stats and isinstance(opt_stats["xi_added"], dict):
            row["n_scenarios"] = len(opt_stats["xi_added"].get("obj", []))

        rows.append(row)

    df = pd.DataFrame(rows)

    # detailed print
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)
    with pd.option_context("display.max_rows", 200, "display.max_columns", 200):
        print(df.to_string(index=False))

    # summary
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    valid = df.dropna(subset=["algo_obj", "baseline_obj"])
    if len(valid) == 0:
        print("No valid results to summarize!")
    else:
        summary = {
            "Total instances": len(df),
            "Valid results": len(valid),
            "Mean algo time (s)": valid["algo_time"].mean() if "algo_time" in valid else np.nan,
            "Mean baseline time (s)": valid["baseline_time"].mean() if "baseline_time" in valid else np.nan,
            "Mean speedup": valid["speedup"].mean() if "speedup" in valid.columns else np.nan,
            "Mean gap (%)": valid["gap_pct"].mean() if "gap_pct" in valid.columns else np.nan,
            "Mean RE algo (%)": valid["re_algo"].mean() if "re_algo" in valid.columns else np.nan,
            "Mean RE baseline (%)": valid["re_baseline"].mean() if "re_baseline" in valid.columns else np.nan,
            "Mean iterations": valid["n_iterations"].mean() if "n_iterations" in valid.columns else np.nan,
            "Mean scenarios": valid["n_scenarios"].mean() if "n_scenarios" in valid.columns else np.nan,
        }

        for k, v in summary.items():
            if isinstance(v, float) and not np.isnan(v):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        # LaTeX table
        print("\n" + "=" * 80)
        print("LATEX TABLE")
        print("=" * 80)

        mean_algo_obj = valid["algo_obj"].mean()
        mean_base_obj = valid["baseline_obj"].mean()
        mean_gap = valid["gap_pct"].mean() if "gap_pct" in valid.columns else 0.0
        mean_algo_t = valid["algo_time"].mean() if "algo_time" in valid.columns else np.nan
        mean_base_t = valid["baseline_time"].mean() if "baseline_time" in valid.columns else np.nan
        mean_sp = valid["speedup"].mean() if "speedup" in valid.columns else np.nan
        mean_re_a = valid["re_algo"].mean() if "re_algo" in valid.columns else 0.0
        mean_re_b = valid["re_baseline"].mean() if "re_baseline" in valid.columns else 0.0

        latex = r"""
\begin{table}[h]
\centering
\caption{Computational Results for Day-Ahead Offering with Distribution-Network Constraints}
\label{tab:offering_network_results}
\begin{tabular}{lrrrr}
\toprule
Metric & ML-CCG & Baseline & Gap (\%) & Speedup \\
\midrule
"""
        latex += f"Mean Obj & {mean_algo_obj:.2f} & {mean_base_obj:.2f} & {mean_gap:.2f} & - \\\\\n"
        if not np.isnan(mean_sp):
            latex += f"Mean Time (s) & {mean_algo_t:.2f} & {mean_base_t:.2f} & - & {mean_sp:.1f}$\\times$ \\\\\n"
        else:
            latex += f"Mean Time (s) & {mean_algo_t:.2f} & {mean_base_t:.2f} & - & - \\\\\n"
        latex += f"Mean RE (\\%) & {mean_re_a:.3f} & {mean_re_b:.3f} & - & - \\\\\n"
        latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        print(latex)

    # save CSV
    _, prob_dir = _get_eval_dir(args.problem)
    summary_path = os.path.join(prob_dir, "eval_summary_offering_network.csv")
    df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")


def main(args):
    if args.cmd_type == "run":
        run_evaluation(args)
    elif args.cmd_type == "eval":
        generate_eval_table(args)
    elif args.cmd_type == "all":
        run_evaluation(args)
        print("\n")
        generate_eval_table(args)
    else:
        print(f"Unknown cmd_type: {args.cmd_type}")
        print("Use: run, eval, or all")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--problem", type=str, default="offering_network")
    parser.add_argument(
        "--cmd_type",
        type=str,
        default="eval",
        choices=["run", "eval", "all"],
        help="run=generate instances and evaluate, eval=analyze results, all=both",
    )

    # Evaluation parameters
    parser.add_argument("--n_instances", type=int, default=25)
    parser.add_argument("--max_iterations", type=int, default=20)
    parser.add_argument("--baseline_time_limit", type=float, default=600)
    parser.add_argument("--verbose", type=int, default=1)

    args = parser.parse_args()
    main(args)
