# ro/dm/offering_no_network.py
# - Adds MATLAB v7.3 (.mat HDF5) fallback via h5py
# - Keeps your “monotone offer curve” sampling design in sample_procs()
# - Uses cfg profiles (pv_min_kw, pv_max_kw, load_sum_kw, etc.) instead of hard-coded arrays

import time
import numpy as np
from multiprocessing import Manager, Pool

from .dm import DataManager

try:
    from scipy.io import loadmat
except Exception:
    loadmat = None


class OfferingNoNetworkDataManager(DataManager):
    """
    Data manager for two-stage offering strategy (no network constraints).
    Each DA price scenario s is treated as one instance.
    """

    def __init__(self, cfg, problem):
        super(OfferingNoNetworkDataManager, self).__init__(cfg, problem)

    # ----------------------------
    # Helpers
    # ----------------------------
    def _get_pu_profiles(self):
        """Return (pv_min_pu, pv_max_pu, load_pu, P_ess_max_pu, E_ess_max_pu)"""
        Sbase = float(self.cfg.Sbase)

        pv_min = np.array(self.cfg.pv_min_kw, dtype=float) / Sbase
        pv_max = (np.array(self.cfg.pv_max_kw, dtype=float) + float(self.cfg.pv_max_shift_kw)) / Sbase
        load = np.array(self.cfg.load_sum_kw, dtype=float) / Sbase

        P_ess_max = (float(self.cfg.N_es) * float(self.cfg.P_ess_per_unit_kw)) / Sbase
        E_ess_max = (float(self.cfg.N_es) * float(self.cfg.E_ess_per_unit_kwh)) / Sbase

        return pv_min, pv_max, load, P_ess_max, E_ess_max

    def _load_price_matrix_any(self, path: str, varname: str | None = None) -> np.ndarray:
        """
        Load 24xS price matrix from .mat, supporting both:
          - MATLAB <= v7.2 (scipy.io.loadmat)
          - MATLAB v7.3 (HDF5) (h5py fallback)

        Returns:
          price_matrix: np.ndarray shape (24, S)
        """
        # 1) Try scipy for non-v7.3
        if loadmat is not None:
            try:
                mat = loadmat(path)
                if varname and varname in mat:
                    arr = mat[varname]
                elif "price_matrix_revised_1000" in mat:
                    arr = mat["price_matrix_revised_1000"]
                elif "price_matrix" in mat:
                    arr = mat["price_matrix"]
                else:
                    arr = None
                    for k, v in mat.items():
                        if k.startswith("__"):
                            continue
                        if isinstance(v, np.ndarray) and v.ndim == 2:
                            arr = v
                            break
                if arr is None:
                    raise KeyError("Cannot find 2D price matrix in .mat (scipy branch).")

                arr = np.asarray(arr, dtype=float)

                # Some .mat store as (S,24); transpose if needed
                if arr.shape[0] != 24 and arr.shape[1] == 24:
                    arr = arr.T

                if arr.shape[0] != 24:
                    raise ValueError(f"Price matrix must be 24xS. Got {arr.shape} (scipy branch).")

                return arr

            except NotImplementedError:
                # MATLAB v7.3 -> fallback to h5py
                pass

        # 2) Fallback: h5py for v7.3
        import h5py  # requires: pip install h5py

        with h5py.File(path, "r") as f:
            if varname and varname in f:
                arr = f[varname][()]
            elif "price_matrix_revised_1000" in f:
                arr = f["price_matrix_revised_1000"][()]
            elif "price_matrix" in f:
                arr = f["price_matrix"][()]
            else:
                arr = None

                def _walk(name, obj):
                    nonlocal arr
                    if arr is not None:
                        return
                    if isinstance(obj, h5py.Dataset) and obj.ndim == 2:
                        arr = obj[()]

                f.visititems(_walk)
                if arr is None:
                    raise KeyError("Cannot find 2D price matrix in .mat (h5py branch).")

        arr = np.asarray(arr, dtype=float)

        # Some v7.3 datasets load as (S,24)
        if arr.shape[0] != 24 and arr.shape[1] == 24:
            arr = arr.T

        if arr.shape[0] != 24:
            raise ValueError(f"Price matrix must be 24xS. Got {arr.shape} (h5py branch).")

        return arr

    def _sample_offer_curve_monotone(self, lambda_da: np.ndarray) -> np.ndarray:
        """
        lambda_da: (T,S) —— per-hour sort scenarios by price, enforce p_DA(t,·) nondecreasing.
        Returns: p_DA (T,S)
        """
        T, S = lambda_da.shape
        pv_min, pv_max, load, P_ess_max, _ = self._get_pu_profiles()

        # Safe physical-ish bounds (you can tune margin)
        lb = -load - P_ess_max
        ub = pv_max + P_ess_max - load

        m = float(getattr(self.cfg, "p_da_bound_margin_pu", getattr(self.cfg, "x_margin", 0.0)))
        lb = lb - m
        ub = ub + m

        p_DA = np.zeros((T, S), dtype=float)
        for t in range(T):
            order = np.argsort(lambda_da[t, :], kind="stable")  # low -> high
            q_sorted = np.sort(np.random.uniform(lb[t], ub[t], size=S))
            p_DA[t, order] = q_sorted
        return p_DA

    # ----------------------------
    # Required abstract methods
    # ----------------------------
    def get_problem_data(self):
        prob = {}
        prob["data_type"] = self.cfg.data_type
        prob["seed"] = self.cfg.seed
        prob["data_path"] = self.cfg.data_path

        prob["time_limit"] = self.cfg.time_limit
        prob["mip_gap"] = self.cfg.mip_gap
        prob["verbose"] = self.cfg.verbose
        prob["threads"] = self.cfg.threads
        prob["tr_split"] = self.cfg.tr_split
        prob["n_samples_inst"] = self.cfg.n_samples_inst
        prob["n_samples_fs"] = self.cfg.n_samples_fs
        prob["n_samples_per_fs"] = self.cfg.n_samples_per_fs

        prob["Gamma"] = self.cfg.Gamma
        prob["lambda_rt_value"] = self.cfg.lambda_rt_value
        prob["price_mat_path"] = self.cfg.price_mat_path
        prob["price_mat_var"] = getattr(self.cfg, "price_mat_var", None)

        prob["x_margin"] = getattr(self.cfg, "x_margin", 0.2)
        prob["xi_allow_positive"] = getattr(self.cfg, "xi_allow_positive", True)

        prob["offer_curve_sampling"] = getattr(self.cfg, "offer_curve_sampling", "none")

        prob["cfg"] = self.cfg
        return prob

    def sample_instances(self, two_ro):
        """
        Build one instance per DA price scenario column.
        """
        T_cfg = int(getattr(self.cfg, "T", 24))
        if T_cfg != 24:
            raise ValueError(f"Expected T=24 for this problem. Got cfg.T={T_cfg}")

        Sbase = float(self.cfg.Sbase)

        pv_min, pv_max, load, P_ess_max, E_ess_max = self._get_pu_profiles()
        if pv_min.shape[0] != 24 or pv_max.shape[0] != 24 or load.shape[0] != 24:
            raise ValueError("pv_min_kw/pv_max_kw/load_sum_kw must be length 24 in cfg.")

        eta = float(self.cfg.eta)
        Gamma = float(self.cfg.Gamma)

        # costs (keep same scaling as your original; adjust if needed)
        ESS_cost = float(getattr(self.cfg, "ESS_cost", 0.0))
        PV_cost = float(getattr(self.cfg, "PV_cost", 0.0))

        # Load price matrix
        price_var = getattr(self.cfg, "price_mat_var", None)
        price_matrix = self._load_price_matrix_any(self.cfg.price_mat_path, varname=price_var)  # (24,S)

        lambda_da = (Sbase * np.asarray(price_matrix, dtype=float)) / 1000.0  # (24,S)
        if lambda_da.shape[0] != 24:
            raise ValueError(f"lambda_da must be 24xS, got {lambda_da.shape}")

        S = int(lambda_da.shape[1])
        rho = 1.0 / S

        lambda_rt = float(self.cfg.lambda_rt_value) * np.ones(24, dtype=float)

        # SOC initial: Matlab uses 0.5 * E_ess_0
        soc0 = float(0.5 * E_ess_max)

        instances = []
        for s in range(S):
            inst = {
                "T": 24,
                "scenario_id": int(s),
                "rho": float(rho),
                "lambda_da": lambda_da[:, s].reshape(-1),
                "lambda_rt": lambda_rt.reshape(-1),
                "p_load": load.reshape(-1),
                "p_pv_min": pv_min.reshape(-1),
                "p_pv_max": pv_max.reshape(-1),
                "P_ess_max": float(P_ess_max),
                "E_ess_max": float(E_ess_max),
                "eta": float(eta),
                "soc0": float(soc0),
                "ESS_cost": float(ESS_cost),
                "PV_cost": float(PV_cost),
                "Gamma": float(Gamma),
            }
            instances.append(inst)

        return instances

    def sample_procs(self, two_ro):
        """
        Generates “global monotone offer curve across selected scenarios”, then slice x per scenario.
        """
        procs_to_run = []
        instances = self.sample_instances(two_ro)

        # optional: subsample scenarios
        if self.cfg.n_samples_inst is not None and self.cfg.n_samples_inst > 0:
            if self.cfg.n_samples_inst < len(instances):
                rng = np.random.default_rng(self.cfg.seed)
                idx = rng.choice(len(instances), size=self.cfg.n_samples_inst, replace=False)
                instances = [instances[i] for i in idx.tolist()]

        offer_sampling = getattr(self.cfg, "offer_curve_sampling", "none")
        if offer_sampling == "monotone_by_price":
            lambda_da_mat = np.column_stack([inst["lambda_da"] for inst in instances])  # (24,S_sel)
        else:
            lambda_da_mat = None

        for _ in range(int(self.cfg.n_samples_fs)):
            if offer_sampling == "monotone_by_price":
                p_DA_mat = self._sample_offer_curve_monotone(lambda_da_mat)  # (24,S_sel)
            else:
                p_DA_mat = None

            for inst_id, instance in enumerate(instances):
                if p_DA_mat is not None:
                    x = p_DA_mat[:, inst_id].astype(float)
                else:
                    x = self.sample_x(instance)

                for _ in range(int(self.cfg.n_samples_per_fs)):
                    xi = self.sample_xi(instance, x)
                    procs_to_run.append((instance, inst_id, x, xi))

        return procs_to_run

    def sample_x(self, instance):
        """
        Fallback sampling if offer_curve_sampling != monotone_by_price.
        """
        T = int(instance["T"])
        p_load = np.asarray(instance["p_load"], dtype=float)
        pv_max = np.asarray(instance["p_pv_max"], dtype=float)
        Pmax = float(instance["P_ess_max"])

        lb = -(p_load + Pmax)
        ub = (pv_max + Pmax - p_load)

        # Use either x_margin or p_da_bound_margin_pu
        margin = float(getattr(self.cfg, "x_margin", getattr(self.cfg, "p_da_bound_margin_pu", 0.05)))
        width = np.maximum(ub - lb, 1e-6)
        lb2 = lb - margin * width
        ub2 = ub + margin * width

        x = np.random.uniform(lb2, ub2, size=T).astype(float)
        return x

    def sample_xi(self, instance, x):
        """
        Samples xi in [0,1]^T with budget on delta=2xi-1:
            sum(|delta|) <= Gamma
        """
        T = int(instance["T"])
        pv_min = np.asarray(instance["p_pv_min"], dtype=float).reshape(-1)
        pv_max = np.asarray(instance["p_pv_max"], dtype=float).reshape(-1)
        Gamma = float(instance["Gamma"])

        if pv_min.shape[0] != T or pv_max.shape[0] != T:
            raise ValueError("p_pv_min / p_pv_max must be length T")

        d = 0.5 * (pv_max - pv_min)
        active = d > 1e-12

        allow_pos = bool(getattr(self.cfg, "xi_allow_positive", True))

        delta = np.random.uniform(-1.0, 1.0, size=T)
        if not allow_pos:
            delta = -np.abs(delta)

        target = np.random.uniform(0.0, Gamma)
        denom = np.sum(np.abs(delta[active]))
        if denom < 1e-12:
            delta[:] = 0.0
        else:
            delta = delta * (target / denom)

        delta = np.clip(delta, -1.0, 1.0)

        # xi = (delta+1)/2 so delta = 2xi-1
        xi = 0.5 * (delta + 1.0)
        xi[~active] = 0.5
        return xi.astype(float)

    def solve_second_stage(self, x, xi, instance, two_ro, inst_id, mp_time, mp_count, n_samples):
        t0 = time.time()
        fs_obj, ss_obj, _ = two_ro.solve_second_stage(
            x,
            xi,
            instance,
            gap=self.cfg.mip_gap,
            time_limit=self.cfg.time_limit,
            verbose=self.cfg.verbose,
            threads=self.cfg.threads,
        )

        res = {
            "x": x,
            "xi": xi,
            "instance": instance,
            "inst_id": inst_id,
            "concat_x_xi": list(map(float, x)) + list(map(float, xi)),
            "fs_obj": float(fs_obj),
            "ss_obj": float(ss_obj),
            "time": float(time.time() - t0),
        }

        self.update_mp_status(mp_count, mp_time, n_samples)
        return res




# # ro/dm/offering_no_network.py
# """
# Data manager for offering_no_network problem.
# Updated: supports separate RT buy/sell prices.
#   - lambda_rt_buy: price for buying from RT market (e.g., 800)
#   - lambda_rt_sell: price for selling to RT market (e.g., 0)
# """

# import os
# import time
# import pickle as pkl
# import numpy as np
# from scipy.io import loadmat

# import ro.params as ro_params
# from .data_manager import DataManager
# from ro.utils.offering_no_network import get_path


# class OfferingNoNetworkDataManager(DataManager):
#     """
#     Data manager for offering_no_network.
#     Generates instances and samples for ML training.
#     """

#     def __init__(self, cfg=None, problem='offering_no_network'):
#         if cfg is None:
#             cfg = getattr(ro_params, problem)
#         super().__init__(cfg, problem)
#         self.cfg = cfg
#         self.problem = problem

#         # Load price matrix
#         self._load_price_matrix()

#     def _load_price_matrix(self):
#         """Load DA price matrix from .mat file."""
#         mat_path = getattr(self.cfg, 'price_mat_path', None) or getattr(self.cfg, 'price_mat_file', None)
#         mat_var = getattr(self.cfg, 'price_mat_var', 'price_matrix_revised_1000')

#         if mat_path is None or not os.path.exists(mat_path):
#             print(f"Warning: Price matrix not found at {mat_path}")
#             self.price_matrix = None
#             return

#         try:
#             mat = loadmat(mat_path)
#             if mat_var in mat:
#                 self.price_matrix = np.asarray(mat[mat_var], dtype=float)
#             else:
#                 # Try to find any 24xS array
#                 for k, v in mat.items():
#                     if k.startswith('__'):
#                         continue
#                     if isinstance(v, np.ndarray) and v.ndim == 2 and 24 in v.shape:
#                         self.price_matrix = np.asarray(v, dtype=float)
#                         if self.price_matrix.shape[0] != 24:
#                             self.price_matrix = self.price_matrix.T
#                         break
#                 else:
#                     print(f"Warning: Could not find price matrix variable in {mat_path}")
#                     self.price_matrix = None
#                     return

#             # Ensure shape is (24, S)
#             if self.price_matrix.shape[0] != 24 and self.price_matrix.shape[1] == 24:
#                 self.price_matrix = self.price_matrix.T

#             print(f"Loaded price matrix: {self.price_matrix.shape}")
#         except Exception as e:
#             print(f"Error loading price matrix: {e}")
#             self.price_matrix = None

#     def sample_instances(self, two_ro=None):
#         """
#         Generate problem instances from price scenarios.
#         Each scenario s gives one instance with a specific lambda_da profile.
#         """
#         if self.price_matrix is None:
#             raise RuntimeError("Price matrix not loaded. Cannot generate instances.")

#         T = int(getattr(self.cfg, 'T', 24))
#         Sbase = float(getattr(self.cfg, 'Sbase', 1000.0))
#         n_scenarios = self.price_matrix.shape[1]

#         # Get PV bounds (convert kW to p.u.)
#         pv_min_kw = np.array(getattr(self.cfg, 'pv_min_kw', [0]*T))
#         pv_max_kw = np.array(getattr(self.cfg, 'pv_max_kw', [0]*T))
#         pv_shift = float(getattr(self.cfg, 'pv_max_shift_kw', 0))

#         pv_min = pv_min_kw / Sbase
#         pv_max = (pv_max_kw + pv_shift) / Sbase

#         # Get load (convert kW to p.u.)
#         load_kw = np.array(getattr(self.cfg, 'load_sum_kw', [0]*T))
#         p_load = load_kw / Sbase

#         # ESS parameters
#         N_es = int(getattr(self.cfg, 'N_es', 10))
#         E_ess_per_unit = float(getattr(self.cfg, 'E_ess_per_unit_kwh', 14.5))
#         P_ess_per_unit = float(getattr(self.cfg, 'P_ess_per_unit_kw', 11.3))
#         eta = float(getattr(self.cfg, 'eta', 0.95))

#         E_ess_max = N_es * E_ess_per_unit / Sbase
#         P_ess_max = N_es * P_ess_per_unit / Sbase
#         soc0 = 0.5 * E_ess_max

#         # RT prices - UPDATED: separate buy and sell
#         lambda_rt_buy = float(getattr(self.cfg, 'lambda_rt_buy', 800.0))
#         lambda_rt_sell = float(getattr(self.cfg, 'lambda_rt_sell', 0.0))

#         # Other parameters
#         Gamma = float(getattr(self.cfg, 'Gamma', 12.0))
#         ESS_cost = float(getattr(self.cfg, 'ESS_cost', 1e-6))
#         PV_cost = float(getattr(self.cfg, 'PV_cost', 1e-6))

#         # Probability (uniform by default)
#         rho = 1.0 / n_scenarios

#         instances = []
#         for s in range(n_scenarios):
#             # DA price for this scenario (convert to consistent units)
#             lambda_da = Sbase * self.price_matrix[:, s] / 1000.0  # Adjust scaling as needed

#             inst = {
#                 'scenario_id': s,
#                 'T': T,
#                 'lambda_da': lambda_da.tolist(),
#                 'lambda_rt_buy': [lambda_rt_buy] * T,    # NEW: RT buy price array
#                 'lambda_rt_sell': [lambda_rt_sell] * T,  # NEW: RT sell price array
#                 'lambda_rt': [lambda_rt_buy] * T,        # Keep for backward compatibility
#                 'p_load': p_load.tolist(),
#                 'p_pv_min': pv_min.tolist(),
#                 'p_pv_max': pv_max.tolist(),
#                 'P_ess_max': P_ess_max,
#                 'E_ess_max': E_ess_max,
#                 'eta': eta,
#                 'soc0': soc0,
#                 'Gamma': Gamma,
#                 'ESS_cost': ESS_cost,
#                 'PV_cost': PV_cost,
#                 'rho': rho,
#                 'Sbase': Sbase,
#             }
#             instances.append(inst)

#         return instances

#     def sample_x(self, instance):
#         """
#         Sample a first-stage decision x = p_DA for given instance.
#         Uses monotone offer curve sampling based on price ordering.
#         """
#         T = int(instance['T'])
#         p_load = np.array(instance['p_load'])
#         pv_max = np.array(instance['p_pv_max'])
#         P_ess_max = float(instance['P_ess_max'])

#         # Bounds for p_DA
#         margin = float(getattr(self.cfg, 'p_da_bound_margin_pu', 0.05))
#         lb = -(p_load + P_ess_max) - margin
#         ub = (pv_max + P_ess_max - p_load) + margin

#         sampling_mode = getattr(self.cfg, 'offer_curve_sampling', 'monotone_by_price')

#         if sampling_mode == 'monotone_by_price':
#             # Sort hours by DA price and sample monotonically
#             lambda_da = np.array(instance['lambda_da'])
#             sorted_idx = np.argsort(lambda_da)

#             # Sample cumulative values
#             x = np.zeros(T)
#             prev_val = lb[sorted_idx[0]]

#             for i, t in enumerate(sorted_idx):
#                 # Must be >= previous value (monotone increasing with price)
#                 low = max(lb[t], prev_val)
#                 high = ub[t]
#                 if low > high:
#                     low = high = (low + high) / 2
#                 x[t] = np.random.uniform(low, high)
#                 prev_val = x[t]
#         else:
#             # Simple uniform sampling
#             x = np.random.uniform(lb, ub)

#         return x

#     def sample_xi(self, instance, x=None):
#         """
#         Sample uncertainty realization xi in [0,1]^T.
#         xi controls PV upper bound: pv_upper(t) = pv_min(t) + xi(t)*(pv_max(t)-pv_min(t))
        
#         With budget constraint: sum_t |2*xi(t) - 1| <= Gamma
#         """
#         T = int(instance['T'])
#         Gamma = float(instance.get('Gamma', T/2))

#         # Sample xi satisfying budget constraint
#         # Start with random values and project
#         xi = np.random.uniform(0, 1, T)

#         # Project to satisfy budget constraint
#         # sum |2*xi - 1| <= Gamma
#         deviation = np.abs(2 * xi - 1)
#         total_dev = np.sum(deviation)

#         if total_dev > Gamma:
#             # Scale down deviations proportionally
#             scale = Gamma / total_dev
#             # xi_new such that |2*xi_new - 1| = scale * |2*xi - 1|
#             signs = np.sign(2 * xi - 1)
#             xi = 0.5 + 0.5 * signs * scale * np.abs(2 * xi - 1)
#             xi = np.clip(xi, 0, 1)

#         return xi

#     def generate_dataset(self, n_procs=1, debug=False, name_suffix=""):
#         """Generate ML training dataset."""
#         from ro.two_ro import factory_two_ro

#         print(f"Loading offering_no_network data manager...")
#         two_ro = factory_two_ro(self.problem)

#         # Get instances
#         instances = self.sample_instances(two_ro)
#         print(f"Generated {len(instances)} instances from price scenarios")

#         # Sampling parameters
#         n_samples_inst = int(getattr(self.cfg, 'n_samples_inst', 50))
#         n_samples_fs = int(getattr(self.cfg, 'n_samples_fs', 10))
#         n_samples_per_fs = int(getattr(self.cfg, 'n_samples_per_fs', 50))

#         # Limit instances if needed
#         if n_samples_inst < len(instances):
#             np.random.seed(self.cfg.seed)
#             selected_idx = np.random.choice(len(instances), size=n_samples_inst, replace=False)
#             instances = [instances[i] for i in selected_idx]

#         all_samples = []
#         start_time = time.time()

#         for inst_idx, inst in enumerate(instances):
#             if debug and inst_idx >= 2:
#                 break

#             print(f"Processing instance {inst_idx+1}/{len(instances)} (scenario {inst.get('scenario_id', inst_idx)})")

#             for fs_idx in range(n_samples_fs):
#                 # Sample first-stage decision
#                 x = self.sample_x(inst)

#                 for xi_idx in range(n_samples_per_fs):
#                     # Sample uncertainty
#                     xi = self.sample_xi(inst, x)

#                     # Solve second stage
#                     try:
#                         fs_obj, ss_obj, model = two_ro.solve_second_stage(
#                             x, xi, inst,
#                             gap=self.cfg.mip_gap,
#                             time_limit=self.cfg.time_limit,
#                             verbose=0
#                         )

#                         sample = {
#                             'instance': inst,
#                             'x': x.tolist(),
#                             'xi': xi.tolist(),
#                             'fs_obj': fs_obj,
#                             'ss_obj': ss_obj,
#                         }
#                         all_samples.append(sample)

#                     except Exception as e:
#                         if debug:
#                             print(f"  Solve failed: {e}")
#                         continue

#         total_time = time.time() - start_time
#         print(f"Generated {len(all_samples)} samples in {total_time:.2f}s")

#         # Split train/val
#         tr_split = float(getattr(self.cfg, 'tr_split', 0.8))
#         np.random.shuffle(all_samples)
#         n_train = int(len(all_samples) * tr_split)

#         tr_data = all_samples[:n_train]
#         val_data = all_samples[n_train:]

#         print(f"Train: {len(tr_data)}, Val: {len(val_data)}")

#         # Save dataset
#         dataset = {
#             'tr_data': tr_data,
#             'val_data': val_data,
#             'total_time': total_time,
#         }

#         fp = get_path(self.cfg.data_path, self.cfg, f"ml_data{name_suffix}")
#         os.makedirs(os.path.dirname(fp), exist_ok=True)

#         with open(fp, 'wb') as f:
#             pkl.dump(dataset, f)
#         print(f"Saved dataset to: {fp}")

#         return dataset

#     def generate_dataset_by_inst(self, n_procs=1, debug=False, name_suffix=""):
#         """Alias for generate_dataset (no instance-based split needed for this problem)."""
#         return self.generate_dataset(n_procs, debug, name_suffix)

