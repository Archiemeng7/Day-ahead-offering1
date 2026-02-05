# ro/scripts/05_generate_instances_offering.py
"""
Generate test instances for offering_no_network problem evaluation.
Similar to 05_generate_instances.py but adapted for offering problem.
"""

import os
import sys
import pickle as pkl
import numpy as np
from argparse import ArgumentParser

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import ro.params as params
from ro.utils import factory_get_path
from ro.dm import factory_dm


def main(args):
    print(f"Generating test instances for {args.problem}...")
    
    # Load config
    cfg = getattr(params, args.problem)
    
    # Get path function
    get_path = factory_get_path(args.problem)
    
    # Create output directory
    test_inst_dir = get_path(cfg.data_path, cfg, "test_instances/")
    os.makedirs(test_inst_dir, exist_ok=True)
    print(f"Output directory: {test_inst_dir}")
    
    # Load data manager
    dm = factory_dm(args.problem)
    
    # Get two_ro solver
    from ro.two_ro import factory_two_ro
    two_ro = factory_two_ro(args.problem)
    
    # Sample instances
    instances = dm.sample_instances(two_ro)
    print(f"Total instances available: {len(instances)}")
    
    # Select subset for evaluation
    n_eval = min(args.n_instances, len(instances))
    
    if args.seed is not None:
        np.random.seed(args.seed)
    
    if n_eval < len(instances):
        selected_idx = np.random.choice(len(instances), size=n_eval, replace=False)
        selected_instances = [instances[i] for i in selected_idx]
    else:
        selected_instances = instances
    
    print(f"Selected {len(selected_instances)} instances for evaluation")
    
    # Save each instance
    for i, inst in enumerate(selected_instances):
        scenario_id = inst.get('scenario_id', i)
        inst_name = f"inst_s{scenario_id:03d}"
        inst_path = os.path.join(test_inst_dir, f"{inst_name}.pkl")
        
        # Add metadata
        inst['_inst_name'] = inst_name
        inst['_inst_id'] = i
        
        with open(inst_path, 'wb') as f:
            pkl.dump(inst, f)
        
        if args.verbose:
            print(f"  Saved: {inst_name}.pkl")
    
    print(f"\nGenerated {len(selected_instances)} test instances in {test_inst_dir}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--problem', type=str, default='offering_no_network')
    parser.add_argument('--n_instances', type=int, default=25, 
                        help='Number of instances to generate')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', type=int, default=1)
    args = parser.parse_args()
    main(args)