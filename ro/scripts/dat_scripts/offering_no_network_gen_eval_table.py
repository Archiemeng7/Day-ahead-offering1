# ro/scripts/dat_scripts/offering_no_network_gen_eval_table.py
"""
Generate evaluation commands and tables for offering_no_network problem.
Similar to kp_gen_eval_table.py but for offering problem.

Usage:
    python -m ro.scripts.dat_scripts.offering_no_network_gen_eval_table --cmd_type run
    python -m ro.scripts.dat_scripts.offering_no_network_gen_eval_table --cmd_type eval
"""

import os
import sys
import glob
import pickle as pkl
import numpy as np
import pandas as pd
from argparse import ArgumentParser

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import ro.params as params
from ro.utils import factory_get_path


def run_evaluation(args):
    """Generate and run evaluation instances."""
    print("=" * 60)
    print("Step 1: Generating test instances...")
    print("=" * 60)
    
    # Generate test instances
    cmd = f"python -m ro.scripts.05_generate_instances_offering --problem {args.problem} --n_instances {args.n_instances}"
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


def load_results(args):
    """Load evaluation results."""
    cfg = getattr(params, args.problem)
    get_path = factory_get_path(args.problem)
    
    # Get eval results directory
    test_inst_dir = get_path(cfg.data_path, cfg, "test_instances/")
    eval_dir = os.path.join(os.path.dirname(test_inst_dir.rstrip('/')), 'eval_results')
    
    if not os.path.isdir(eval_dir):
        print(f"Evaluation directory not found: {eval_dir}")
        return None
    
    result_files = sorted(glob.glob(os.path.join(eval_dir, 'results_*.pkl')))
    
    if not result_files:
        print(f"No result files found in: {eval_dir}")
        return None
    
    results = []
    for fp in result_files:
        try:
            with open(fp, 'rb') as f:
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
        print(f"  python -m ro.scripts.dat_scripts.offering_no_network_gen_eval_table --cmd_type run")
        return
    
    print(f"Loaded {len(results)} result files")
    
    # Convert to DataFrame
    rows = []
    for res in results:
        row = {
            'instance': res.get('inst_name', 'unknown'),
            'scenario_id': res.get('instance', {}).get('scenario_id', -1),
            'algo_obj': res.get('algo_obj'),
            'baseline_obj': res.get('baseline_obj'),
            'algo_time': res.get('algo_time'),
            'baseline_time': res.get('baseline_time'),
            'algo_status': res.get('algo_status'),
            'baseline_status': res.get('baseline_status'),
        }
        
        # Calculate metrics
        if row['algo_obj'] is not None and row['baseline_obj'] is not None:
            if abs(row['baseline_obj']) > 1e-10:
                row['gap_pct'] = 100 * (row['baseline_obj'] - row['algo_obj']) / abs(row['baseline_obj'])
            else:
                row['gap_pct'] = 0
            
            # Relative error (to best known)
            best = min(row['algo_obj'], row['baseline_obj'])
            row['re_algo'] = 100 * abs(row['algo_obj'] - best) / abs(best) if abs(best) > 1e-10 else 0
            row['re_baseline'] = 100 * abs(row['baseline_obj'] - best) / abs(best) if abs(best) > 1e-10 else 0
        
        # Speedup
        if row['algo_time'] and row['baseline_time'] and row['algo_time'] > 0:
            row['speedup'] = row['baseline_time'] / row['algo_time']
        
        # Iterations
        opt_stats = res.get('opt_stats', {})
        if 'n_iterations' in opt_stats:
            row['n_iterations'] = opt_stats['n_iterations']
        if 'xi_added' in opt_stats:
            row['n_scenarios'] = len(opt_stats['xi_added'].get('obj', []))
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Print detailed results
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # Filter valid results
    valid = df.dropna(subset=['algo_obj', 'baseline_obj'])
    
    if len(valid) > 0:
        summary = {
            'Total instances': len(df),
            'Valid results': len(valid),
            'Mean algo time (s)': valid['algo_time'].mean(),
            'Mean baseline time (s)': valid['baseline_time'].mean(),
            'Mean speedup': valid['speedup'].mean() if 'speedup' in valid.columns else 'N/A',
            'Mean gap (%)': valid['gap_pct'].mean() if 'gap_pct' in valid.columns else 'N/A',
            'Mean RE algo (%)': valid['re_algo'].mean() if 're_algo' in valid.columns else 'N/A',
            'Mean RE baseline (%)': valid['re_baseline'].mean() if 're_baseline' in valid.columns else 'N/A',
            'Mean iterations': valid['n_iterations'].mean() if 'n_iterations' in valid.columns else 'N/A',
            'Mean scenarios': valid['n_scenarios'].mean() if 'n_scenarios' in valid.columns else 'N/A',
        }
        
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        
        # Generate LaTeX table
        print("\n" + "=" * 80)
        print("LATEX TABLE")
        print("=" * 80)
        
        latex = r"""
\begin{table}[h]
\centering
\caption{Computational Results for Day-Ahead Offering Problem}
\label{tab:offering_results}
\begin{tabular}{lrrrr}
\toprule
Metric & Neur2RO & Baseline & Gap (\%) & Speedup \\
\midrule
"""
        latex += f"Mean Obj & {valid['algo_obj'].mean():.2f} & {valid['baseline_obj'].mean():.2f} & {valid['gap_pct'].mean():.2f} & - \\\\\n"
        latex += f"Mean Time (s) & {valid['algo_time'].mean():.2f} & {valid['baseline_time'].mean():.2f} & - & {valid['speedup'].mean():.1f}$\\times$ \\\\\n"
        latex += f"Mean RE (\\%) & {valid['re_algo'].mean():.3f} & {valid['re_baseline'].mean():.3f} & - & - \\\\\n"
        latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        print(latex)
        
    else:
        print("No valid results to summarize!")
    
    # Save summary to file
    cfg = getattr(params, args.problem)
    get_path = factory_get_path(args.problem)
    test_inst_dir = get_path(cfg.data_path, cfg, "test_instances/")
    prob_dir = os.path.dirname(test_inst_dir.rstrip('/'))
    
    summary_path = os.path.join(prob_dir, 'eval_summary.csv')
    df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")


def main(args):
    if args.cmd_type == 'run':
        run_evaluation(args)
    elif args.cmd_type == 'eval':
        generate_eval_table(args)
    elif args.cmd_type == 'all':
        run_evaluation(args)
        print("\n")
        generate_eval_table(args)
    else:
        print(f"Unknown cmd_type: {args.cmd_type}")
        print("Use: run, eval, or all")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--problem', type=str, default='offering_no_network')
    parser.add_argument('--cmd_type', type=str, default='eval',
                        choices=['run', 'eval', 'all'],
                        help='run=generate instances and evaluate, eval=analyze results, all=both')
    
    # Evaluation parameters
    parser.add_argument('--n_instances', type=int, default=25)
    parser.add_argument('--max_iterations', type=int, default=20)
    parser.add_argument('--baseline_time_limit', type=float, default=600)
    parser.add_argument('--verbose', type=int, default=1)
    
    args = parser.parse_args()
    main(args)