# /content/drive/MyDrive/Neur2RO/ro/scripts/06_eval_ml_ccg_offering.py
"""
Evaluate ML-CCG algorithm on offering test instances.
Supports:
  - offering_no_network
  - offering_network
"""

import os
import sys
import time
import pickle as pkl
import numpy as np
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import ro.params as params
from ro.utils import factory_get_path


def load_model(model_path):
    try:
        net = torch.load(model_path, map_location='cpu', weights_only=False)
    except TypeError:
        net = torch.load(model_path, map_location='cpu')
    net.eval()
    return net


def solve_baseline(instance, two_ro, cfg, time_limit=3600):
    """
    Baseline: fix a heuristic x and sample multiple xi, take worst ss_obj.
    """
    T = int(instance.get('T', 24))

    # heuristic x: net load minus half pv_max (aggregate proxy)
    if 'p_load' in instance and 'p_pv_max' in instance:
        p_load = np.array(instance['p_load'])
        pv_max = np.array(instance['p_pv_max'])
        x_baseline = p_load - 0.5 * pv_max
    else:
        # offering_network: build aggregate proxy from bus loads
        p_load_bus = np.asarray(instance['p_load_bus'], dtype=float)  # (nb,T)
        load_sum = np.sum(p_load_bus, axis=0)
        pv_max = np.asarray(instance['p_pv_max'], dtype=float)        # (n_der,T)
        pv_sum = np.sum(pv_max, axis=0)

        enable_fload = bool(instance.get("enable_fload", False))
        if enable_fload:
            fbase_sum = np.sum(np.asarray(instance["fload_base"], dtype=float), axis=0)
        else:
            fbase_sum = np.zeros(T)

        # net injection baseline: -(load+fbase) + 0.5*pv
        x_baseline = -(load_sum + fbase_sum) + 0.5 * pv_sum

    # Sample multiple uncertainty scenarios
    n_samples = 50
    worst_ss_obj = -np.inf
    worst_xi = None

    # DM sampler (problem-specific)
    if cfg.__class__.__name__ == 'SimpleNamespace':
        # pick dm by problem name
        from ro.dm import factory_dm
        dm = factory_dm(cfg.problem_name if hasattr(cfg, "problem_name") else "offering_no_network")
    else:
        dm = None

    start_time = time.time()
    for _ in range(n_samples):
        if time.time() - start_time > time_limit:
            break

        if dm is not None and hasattr(dm, "sample_xi"):
            xi = dm.sample_xi(instance, x_baseline)
        else:
            xi = np.random.uniform(0, 1, size=T)

        try:
            fs_obj, ss_obj, _ = two_ro.solve_second_stage(
                x_baseline, xi, instance,
                gap=cfg.mip_gap, time_limit=min(60, int(time_limit)), verbose=0
            )
            if ss_obj > worst_ss_obj:
                worst_ss_obj = ss_obj
                worst_xi = xi.copy()
        except Exception:
            continue

    solve_time = time.time() - start_time
    if worst_xi is None:
        return None, None, solve_time, 'FAILED'

    fs_obj = two_ro.get_first_stage_obj(x_baseline, instance)
    total_obj = fs_obj + instance.get('rho', 1.0) * worst_ss_obj
    return total_obj, x_baseline, solve_time, 'OPTIMAL'


def _get_approximator_cls(problem):
    if problem == "offering_no_network":
        from ro.approximator.offering_no_network import OfferingNoNetworkApproximator
        return OfferingNoNetworkApproximator
    if problem == "offering_network":
        from ro.approximator.offering_network import OfferingNetworkApproximator
        return OfferingNetworkApproximator
    raise ValueError(f"Unsupported problem: {problem}")


def solve_ml_ccg(instance, two_ro, net, cfg, args):
    from types import SimpleNamespace

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

    approximator_cls = _get_approximator_cls(args.problem)

    # IMPORTANT: pass instance directly (robust, avoids rebuilding from mat)
    inst_params = {'scenario_id': instance.get('scenario_id', 0), 'instance': instance}

    approximator = approximator_cls(approx_args, cfg, net, inst_params)

    start_time = time.time()

    n_init = args.n_init_scenarios
    xi_added = {'obj': [], 'feas': []}

    for _ in range(n_init):
        xi_init = approximator._sample_random_xi()
        approximator.add_worst_case_scenario_to_main(xi_init, len(xi_added['obj']), 'obj')
        xi_added['obj'].append(xi_init.tolist())

    for iteration in range(args.max_iterations):
        approximator.main_model.optimize()

        if approximator.main_model.Status not in [2, 9]:
            if args.verbose:
                print(f"  Master problem failed: status={approximator.main_model.Status}")
            break

        x_current = np.array([approximator.main_model._x[t].X for t in range(24)])
        z_current = approximator.main_model._z.X

        approximator.set_first_stage_in_adversarial_model(x_current)
        approximator.adv_model['obj'].optimize()

        if approximator.adv_model['obj'].Status not in [2, 9]:
            if args.verbose:
                print(f"  Adversarial problem failed: status={approximator.adv_model['obj'].Status}")
            break

        xi_worst = np.array([approximator.adv_model['obj']._xi[t].X for t in range(24)])
        adv_obj = approximator.adv_model['obj'].ObjVal

        gap = adv_obj - z_current
        if args.verbose:
            print(f"  Iter {iteration+1}: z={z_current:.4f}, adv={adv_obj:.4f}, gap={gap:.4f}")

        if gap < args.convergence_tol:
            if args.verbose:
                print(f"  Converged at iteration {iteration+1}")
            break

        approximator.add_worst_case_scenario_to_main(xi_worst, len(xi_added['obj']), 'obj')
        xi_added['obj'].append(xi_worst.tolist())

    solve_time = time.time() - start_time
    x_final = np.array([approximator.main_model._x[t].X for t in range(24)])

    worst_ss_obj = -np.inf
    for xi in xi_added['obj'][-5:]:
        xi_arr = np.array(xi)
        try:
            fs_obj, ss_obj, _ = two_ro.solve_second_stage(
                x_final, xi_arr, instance,
                gap=cfg.mip_gap, time_limit=60, verbose=0
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

    cfg = getattr(params, args.problem)
    # tag cfg with problem_name for baseline helper (optional)
    cfg.problem_name = args.problem

    get_path = factory_get_path(args.problem)

    test_inst_dir = get_path(cfg.data_path, cfg, "test_instances/")
    eval_results_dir = os.path.join(os.path.dirname(test_inst_dir.rstrip('/')), 'eval_results')
    os.makedirs(eval_results_dir, exist_ok=True)

    print(f"Test instances: {test_inst_dir}")
    print(f"Results output: {eval_results_dir}")

    model_path = None
    rs_dir = os.path.join(os.path.dirname(test_inst_dir.rstrip('/')), 'random_search')
    if os.path.isdir(rs_dir):
        import glob
        model_files = glob.glob(os.path.join(rs_dir, 'set_encoder*.pt'))
        if model_files:
            model_path = max(model_files, key=os.path.getmtime)

    if model_path is None:
        prob_dir = os.path.dirname(test_inst_dir.rstrip('/'))
        import glob
        model_files = glob.glob(os.path.join(prob_dir, 'set_encoder*.pt'))
        if model_files:
            model_path = max(model_files, key=os.path.getmtime)

    if model_path is None:
        print("ERROR: No trained model found!")
        print(f"Run training first: python -m ro.scripts.03_train_model --problem {args.problem}")
        return

    print(f"Loading model: {model_path}")
    net = load_model(model_path)

    from ro.two_ro import factory_two_ro
    two_ro = factory_two_ro(args.problem)

    import glob
    inst_files = sorted(glob.glob(os.path.join(test_inst_dir, '*.pkl')))
    if not inst_files:
        print(f"No test instances found in {test_inst_dir}")
        print(f"Run: python -m ro.scripts.05_generate_instances_offering --problem {args.problem}")
        return

    print(f"Found {len(inst_files)} test instances")

    for inst_file in inst_files:
        inst_name = os.path.basename(inst_file).replace('.pkl', '')
        result_path = os.path.join(eval_results_dir, f"results_{inst_name}.pkl")

        if os.path.exists(result_path) and not args.overwrite:
            print(f"Skipping {inst_name} (already evaluated)")
            continue

        print(f"\nEvaluating {inst_name}...")

        with open(inst_file, 'rb') as f:
            instance = pkl.load(f)

        print("  Running ML-CCG (Neur2RO)...")
        algo_obj, algo_x, algo_time, algo_status, opt_stats = solve_ml_ccg(
            instance, two_ro, net, cfg, args
        )

        print("  Running baseline...")
        baseline_obj, baseline_x, baseline_time, baseline_status = solve_baseline(
            instance, two_ro, cfg, time_limit=args.baseline_time_limit
        )

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

        print(f"  ML-CCG:   obj={algo_obj if algo_obj is not None else 'N/A'}, time={algo_time:.2f}s")
        print(f"  Baseline: obj={baseline_obj if baseline_obj is not None else 'N/A'}, time={baseline_time:.2f}s")

        if (algo_obj is not None) and (baseline_obj is not None) and abs(baseline_obj) > 1e-10:
            gap = 100 * (baseline_obj - algo_obj) / abs(baseline_obj)
            print(f"  Gap: {gap:.2f}%")

    print(f"\nResults saved to: {eval_results_dir}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--problem', type=str, default='offering_no_network')

    parser.add_argument('--max_iterations', type=int, default=20)
    parser.add_argument('--n_init_scenarios', type=int, default=3)
    parser.add_argument('--convergence_tol', type=float, default=1e-3)

    parser.add_argument('--mp_gap', type=float, default=0.01)
    parser.add_argument('--mp_time', type=float, default=300)
    parser.add_argument('--adv_gap', type=float, default=0.01)
    parser.add_argument('--adv_time', type=float, default=60)

    parser.add_argument('--baseline_time_limit', type=float, default=600)

    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--overwrite', type=int, default=0)

    args = parser.parse_args()
    main(args)
