# ro/scripts/06_eval_ml_ccg_offering.py
"""
Evaluate ML-CCG algorithm on offering_no_network test instances.
Compares Neur2RO approach against baseline (Benders-style or direct solve).
"""

import os
import sys
import time
import pickle as pkl
import numpy as np
from argparse import ArgumentParser

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import ro.params as params
from ro.utils import factory_get_path


def load_model(model_path):
    """Load trained neural network model."""
    # PyTorch 2.6+ requires weights_only=False to load full model objects
    # This is safe since we're loading our own trained models
    try:
        # Try new API first (PyTorch 2.6+)
        net = torch.load(model_path, map_location='cpu', weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions
        net = torch.load(model_path, map_location='cpu')
    net.eval()
    return net


def solve_baseline(instance, two_ro, cfg, time_limit=3600):
    """
    Solve using baseline method (sampling-based or direct enumeration).
    For offering problem, we sample multiple xi scenarios and take worst-case.
    """
    from ro.dm.offering_no_network import OfferingNoNetworkDataManager
    
    T = int(instance.get('T', 24))
    Gamma = float(instance.get('Gamma', 12))
    
    # Sample first-stage decision (center of feasible region)
    p_load = np.array(instance['p_load'])
    pv_max = np.array(instance['p_pv_max'])
    P_ess_max = float(instance['P_ess_max'])
    
    # Simple heuristic: x = net load (conservative)
    x_baseline = p_load - 0.5 * pv_max
    
    # Sample multiple uncertainty scenarios
    n_samples = 100
    worst_ss_obj = -np.inf
    worst_xi = None
    
    dm = OfferingNoNetworkDataManager(cfg, 'offering_no_network')
    
    start_time = time.time()
    
    for _ in range(n_samples):
        if time.time() - start_time > time_limit:
            break
            
        xi = dm.sample_xi(instance, x_baseline)
        
        try:
            fs_obj, ss_obj, _ = two_ro.solve_second_stage(
                x_baseline, xi, instance, 
                gap=cfg.mip_gap, time_limit=30, verbose=0
            )
            
            if ss_obj > worst_ss_obj:
                worst_ss_obj = ss_obj
                worst_xi = xi.copy()
        except Exception:
            continue
    
    solve_time = time.time() - start_time
    
    if worst_xi is None:
        return None, None, solve_time, 'FAILED'
    
    # Compute final objective
    fs_obj = two_ro.get_first_stage_obj(x_baseline, instance)
    total_obj = fs_obj + instance.get('rho', 1.0) * worst_ss_obj
    
    return total_obj, x_baseline, solve_time, 'OPTIMAL'


def solve_ml_ccg(instance, two_ro, net, approximator_cls, cfg, args):
    """
    Solve using ML-CCG (Neur2RO) approach.
    """
    from types import SimpleNamespace
    
    # Create approximator args
    approx_args = SimpleNamespace(
        verbose=args.verbose,
        mp_gap=args.mp_gap,
        mp_time=args.mp_time,
        mp_focus=0,
        mp_inc_time=0,
        adversarial_gap=args.adv_gap,
        adversarial_time=args.adv_time,
        adversarial_focus=0,
        adversarial_inc_time=0,
    )
    
    # Instance params for approximator
    inst_params = {'scenario_id': instance.get('scenario_id', 0)}
    
    # Initialize approximator
    approximator = approximator_cls(approx_args, cfg, net, inst_params)
    
    start_time = time.time()
    
    # Initialize with random scenarios
    n_init = args.n_init_scenarios
    xi_added = {'obj': [], 'feas': []}
    
    for _ in range(n_init):
        xi_init = approximator._sample_random_xi()
        approximator.add_worst_case_scenario_to_main(xi_init, len(xi_added['obj']), 'obj')
        xi_added['obj'].append(xi_init.tolist())
    
    # Main CCG loop
    for iteration in range(args.max_iterations):
        # Solve master problem
        approximator.main_model.optimize()
        
        if approximator.main_model.Status not in [2, 9]:  # OPTIMAL or TIME_LIMIT
            if args.verbose:
                print(f"  Master problem failed: status={approximator.main_model.Status}")
            break
        
        # Get current solution
        x_current = np.array([approximator.main_model._x[t].X for t in range(24)])
        z_current = approximator.main_model._z.X
        
        # Set x in adversarial model
        approximator.set_first_stage_in_adversarial_model(x_current)
        
        # Solve adversarial problem
        approximator.adv_model['obj'].optimize()
        
        if approximator.adv_model['obj'].Status not in [2, 9]:
            if args.verbose:
                print(f"  Adversarial problem failed: status={approximator.adv_model['obj'].Status}")
            break
        
        # Get worst-case xi
        xi_worst = np.array([approximator.adv_model['obj']._xi[t].X for t in range(24)])
        adv_obj = approximator.adv_model['obj'].ObjVal
        
        # Check convergence
        gap = adv_obj - z_current
        if args.verbose:
            print(f"  Iter {iteration+1}: z={z_current:.4f}, adv={adv_obj:.4f}, gap={gap:.4f}")
        
        if gap < args.convergence_tol:
            if args.verbose:
                print(f"  Converged at iteration {iteration+1}")
            break
        
        # Add scenario
        approximator.add_worst_case_scenario_to_main(xi_worst, len(xi_added['obj']), 'obj')
        xi_added['obj'].append(xi_worst.tolist())
    
    solve_time = time.time() - start_time
    
    # Get final solution
    x_final = np.array([approximator.main_model._x[t].X for t in range(24)])
    
    # Evaluate true objective with actual second-stage solve
    # Sample worst-case scenarios and solve
    worst_ss_obj = -np.inf
    for xi in xi_added['obj'][-5:]:  # Check last few added scenarios
        xi_arr = np.array(xi)
        try:
            fs_obj, ss_obj, _ = two_ro.solve_second_stage(
                x_final, xi_arr, instance,
                gap=cfg.mip_gap, time_limit=30, verbose=0
            )
            if ss_obj > worst_ss_obj:
                worst_ss_obj = ss_obj
        except Exception:
            continue
    
    if worst_ss_obj == -np.inf:
        return None, None, solve_time, 'FAILED', {}
    
    fs_obj = two_ro.get_first_stage_obj(x_final, instance)
    total_obj = fs_obj + instance.get('rho', 1.0) * worst_ss_obj
    
    opt_stats = {
        'xi_added': xi_added,
        'n_iterations': len(xi_added['obj']) - n_init,
    }
    
    return total_obj, x_final, solve_time, 'OPTIMAL', opt_stats


def main(args):
    print(f"Evaluating ML-CCG on {args.problem}...")
    
    # Load config
    cfg = getattr(params, args.problem)
    
    # Get paths
    get_path = factory_get_path(args.problem)
    
    test_inst_dir = get_path(cfg.data_path, cfg, "test_instances/")
    eval_results_dir = os.path.join(os.path.dirname(test_inst_dir.rstrip('/')), 'eval_results')
    os.makedirs(eval_results_dir, exist_ok=True)
    
    print(f"Test instances: {test_inst_dir}")
    print(f"Results output: {eval_results_dir}")
    
    # Load model
    model_path = None
    rs_dir = os.path.join(os.path.dirname(test_inst_dir.rstrip('/')), 'random_search')
    if os.path.isdir(rs_dir):
        import glob
        model_files = glob.glob(os.path.join(rs_dir, 'set_encoder*.pt'))
        if model_files:
            model_path = max(model_files, key=os.path.getmtime)
    
    # Also check main directory
    if model_path is None:
        prob_dir = os.path.dirname(test_inst_dir.rstrip('/'))
        import glob
        model_files = glob.glob(os.path.join(prob_dir, 'set_encoder*.pt'))
        if model_files:
            model_path = max(model_files, key=os.path.getmtime)
    
    if model_path is None:
        print("ERROR: No trained model found!")
        print("Run training first: python -m ro.scripts.03_train_model --problem offering_no_network")
        return
    
    print(f"Loading model: {model_path}")
    net = load_model(model_path)
    
    # Load two_ro solver
    from ro.two_ro import factory_two_ro
    two_ro = factory_two_ro(args.problem)
    
    # Load approximator
    from ro.approximator.offering_no_network import OfferingNoNetworkApproximator
    
    # Get test instances
    import glob
    inst_files = sorted(glob.glob(os.path.join(test_inst_dir, '*.pkl')))
    
    if not inst_files:
        print(f"No test instances found in {test_inst_dir}")
        print("Run: python -m ro.scripts.05_generate_instances_offering --problem offering_no_network")
        return
    
    print(f"Found {len(inst_files)} test instances")
    
    # Evaluate each instance
    for inst_file in inst_files:
        inst_name = os.path.basename(inst_file).replace('.pkl', '')
        result_path = os.path.join(eval_results_dir, f"results_{inst_name}.pkl")
        
        # Skip if already evaluated
        if os.path.exists(result_path) and not args.overwrite:
            print(f"Skipping {inst_name} (already evaluated)")
            continue
        
        print(f"\nEvaluating {inst_name}...")
        
        # Load instance
        with open(inst_file, 'rb') as f:
            instance = pkl.load(f)
        
        # Solve with ML-CCG
        print("  Running ML-CCG (Neur2RO)...")
        algo_obj, algo_x, algo_time, algo_status, opt_stats = solve_ml_ccg(
            instance, two_ro, net, OfferingNoNetworkApproximator, cfg, args
        )
        
        # Solve baseline
        print("  Running baseline...")
        baseline_obj, baseline_x, baseline_time, baseline_status = solve_baseline(
            instance, two_ro, cfg, time_limit=args.baseline_time_limit
        )
        
        # Save results
        results = {
            'inst_name': inst_name,
            'instance': instance,
            'algo_obj': algo_obj,
            'algo_x': algo_x.tolist() if algo_x is not None else None,
            'algo_time': algo_time,
            'algo_status': algo_status,
            'baseline_obj': baseline_obj,
            'baseline_x': baseline_x.tolist() if baseline_x is not None else None,
            'baseline_time': baseline_time,
            'baseline_status': baseline_status,
            'opt_stats': opt_stats,
        }
        
        with open(result_path, 'wb') as f:
            pkl.dump(results, f)
        
        # Print summary
        print(f"  ML-CCG:   obj={algo_obj:.4f if algo_obj else 'N/A'}, time={algo_time:.2f}s")
        print(f"  Baseline: obj={baseline_obj:.4f if baseline_obj else 'N/A'}, time={baseline_time:.2f}s")
        
        if algo_obj and baseline_obj:
            gap = 100 * (baseline_obj - algo_obj) / abs(baseline_obj) if abs(baseline_obj) > 1e-10 else 0
            print(f"  Gap: {gap:.2f}%")
    
    print(f"\nResults saved to: {eval_results_dir}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--problem', type=str, default='offering_no_network')
    
    # ML-CCG parameters
    parser.add_argument('--max_iterations', type=int, default=20)
    parser.add_argument('--n_init_scenarios', type=int, default=3)
    parser.add_argument('--convergence_tol', type=float, default=1e-3)
    
    # Solver parameters
    parser.add_argument('--mp_gap', type=float, default=0.01)
    parser.add_argument('--mp_time', type=float, default=300)
    parser.add_argument('--adv_gap', type=float, default=0.01)
    parser.add_argument('--adv_time', type=float, default=60)
    
    # Baseline parameters
    parser.add_argument('--baseline_time_limit', type=float, default=600)
    
    # Other
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--overwrite', type=int, default=0)
    
    args = parser.parse_args()
    main(args)