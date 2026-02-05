# /content/drive/MyDrive/Neur2RO/ro/dm/offering_network.py
import time
import os
import re
import numpy as np
from multiprocessing import Pool

from .dm import DataManager

# =========================
# MAT helpers (v5 + v7.3)
# =========================
try:
    from scipy.io import loadmat as _scipy_loadmat
except Exception:
    _scipy_loadmat = None

try:
    import h5py
except Exception:
    h5py = None


def _to_str(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    if isinstance(x, np.str_):
        return str(x)
    return str(x)


def _mat_keys(d: dict):
    return [k for k in d.keys() if not k.startswith("__")]


def _get_first_existing_key(d: dict, candidates):
    for k in candidates:
        if k in d:
            return k
    return None


def _load_mat_any(path: str, squeeze: bool = True) -> dict:
    """
    Load .mat file in both:
      - MATLAB v5/v7 (scipy.io.loadmat)
      - MATLAB v7.3 (HDF5) (h5py)
    Return a python dict of variables.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"MAT file not found: {path}")

    # Try v5/v7
    if _scipy_loadmat is not None:
        try:
            return _scipy_loadmat(path, squeeze_me=squeeze, struct_as_record=False)
        except Exception:
            pass

    # Try v7.3 (HDF5)
    if h5py is None:
        raise RuntimeError(
            "Failed to read .mat via scipy.io.loadmat; and h5py is not available for v7.3 MAT."
        )

    out = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            obj = f[k]
            if isinstance(obj, h5py.Dataset):
                out[k] = obj[()]
            else:
                def _walk(name, item):
                    if isinstance(item, h5py.Dataset):
                        out[name] = item[()]
                obj.visititems(_walk)
    return out


def _mat_cellstr_to_list(arr):
    """
    Convert MATLAB cellstr / string-ish arrays loaded by scipy into python list[str].
    """
    arr = np.asarray(arr)
    out = []
    if arr.dtype == object:
        for v in arr.reshape(-1):
            if isinstance(v, np.ndarray):
                if v.size == 0:
                    out.append("")
                else:
                    out.append(_to_str(v.reshape(-1)[0]))
            else:
                out.append(_to_str(v))
    else:
        for v in arr.reshape(-1):
            out.append(_to_str(v))
    return out


def _normalize_bus_name(s: str) -> str:
    s = _to_str(s).strip()
    return s


def _alt_bus_names(s: str):
    """
    Provide alternate spellings to match S-prefix inconsistencies:
      S1a <-> 1a
    """
    s = _normalize_bus_name(s)
    alts = [s]
    if len(s) >= 2 and (s[0].isalpha()):
        # strip leading letters (e.g., 'S')
        alts.append(re.sub(r"^[A-Za-z]+", "", s))
    else:
        alts.append("S" + s)
    # unique preserve order
    seen = set()
    out = []
    for a in alts:
        if a not in seen and a != "":
            seen.add(a)
            out.append(a)
    return out


class OfferingNetworkDataManager(DataManager):
    """
    Data manager for offering_network (linear distribution network constraints).
    Each DA price scenario column is treated as one instance.
    """

    def __init__(self, cfg, problem):
        super(OfferingNetworkDataManager, self).__init__(cfg, problem)

    # ----------------------------
    # Price matrix loader (v5 + v7.3)
    # ----------------------------
    def _load_price_matrix_any(self, path: str, varname=None) -> np.ndarray:
        # scipy
        if _scipy_loadmat is not None:
            try:
                mat = _scipy_loadmat(path, squeeze_me=True, struct_as_record=False)
                if varname and varname in mat:
                    arr = mat[varname]
                else:
                    arr = None
                    for k, v in mat.items():
                        if k.startswith("__"):
                            continue
                        if isinstance(v, np.ndarray) and v.ndim == 2:
                            arr = v
                            break
                if arr is None:
                    raise KeyError("No 2D array found in .mat for price matrix (scipy).")
                arr = np.asarray(arr, dtype=float)
                if arr.shape[0] != 24 and arr.shape[1] == 24:
                    arr = arr.T
                if arr.shape[0] != 24:
                    raise ValueError(f"Price matrix must be 24xS. Got {arr.shape}.")
                return arr
            except NotImplementedError:
                pass
            except Exception:
                pass

        # h5py (v7.3)
        if h5py is None:
            raise RuntimeError("h5py not available to read v7.3 price matrix.")
        with h5py.File(path, "r") as f:
            if varname and varname in f:
                arr = f[varname][()]
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
                    raise KeyError("No 2D dataset found in .mat for price matrix (h5py).")

        arr = np.asarray(arr, dtype=float)
        if arr.shape[0] != 24 and arr.shape[1] == 24:
            arr = arr.T
        if arr.shape[0] != 24:
            raise ValueError(f"Price matrix must be 24xS. Got {arr.shape}.")
        return arr

    # ----------------------------
    # Network loader
    # ----------------------------
    def _load_network(self):
        ldf = _load_mat_any(self.cfg.ldf_mat_path)

        def _pick(mat, key):
            if key in mat:
                return np.asarray(mat[key], dtype=float)
            raise KeyError(f"Missing {key} in {self.cfg.ldf_mat_path}")

        A = _pick(ldf, "A")
        A0 = _pick(ldf, "A0")
        RD = _pick(ldf, "RD")
        XD = _pick(ldf, "XD")
        return A, A0, RD, XD

    # ----------------------------
    # Bus-phase names (ROBUST)
    # ----------------------------
    def _load_bus_phase_names(self):
        """
        Goal: produce `bus_phase_names` with length nb (RD.shape[0]) so that:
          - names contain 'S1a' style strings (so cfg.DER_BusName matches)
          - if load_name_bus_phase.mat is unreadable (MatlabOpaque), fallback uses 123_load.mat load_Name.

        Note:
          - If your load_name_bus_phase.mat is a MATLAB "string" object, scipy often returns MatlabOpaque.
            In that case we DO NOT crash; we fallback.
        """
        # node dimension from LDF
        ldf = _load_mat_any(self.cfg.ldf_mat_path)
        rd_key = _get_first_existing_key(ldf, ["RD", "Rd", "R_D", "R"])
        xd_key = _get_first_existing_key(ldf, ["XD", "Xd", "X_D", "X"])
        if rd_key is None and xd_key is None:
            raise KeyError(f"Cannot find RD/XD in {self.cfg.ldf_mat_path}. Keys: {_mat_keys(ldf)}")
        n_nodes = int(ldf[rd_key].shape[0] if rd_key is not None else ldf[xd_key].shape[0])

        bus_phase_mat_path = getattr(self.cfg, "bus_phase_mat_path", None)

        # ---- Try reading from bus_phase_mat_path if possible ----
        if bus_phase_mat_path is not None and os.path.exists(bus_phase_mat_path):
            try:
                bp = _load_mat_any(bus_phase_mat_path)
                # try common keys
                k0 = _get_first_existing_key(
                    bp,
                    ["load_name_bus_phase", "loadNameBusPhase", "load_bus_phase", "load2busphase"],
                )
                if k0 is not None:
                    raw = bp[k0]
                else:
                    # sometimes only one non-__ key exists
                    ks = [k for k in bp.keys() if not k.startswith("__")]
                    raw = bp[ks[0]] if len(ks) == 1 else None

                if raw is not None:
                    arr = np.asarray(raw)
                    # Expected MATLAB: Nx2 cell, first col are names (S1a, ...)
                    if arr.ndim == 2 and arr.shape[0] >= 1:
                        col0 = arr[:, 0]
                        names = _mat_cellstr_to_list(col0)
                        names = [_normalize_bus_name(s) for s in names if _normalize_bus_name(s) != ""]
                        if len(names) > 0:
                            if len(names) < n_nodes:
                                names += [f"node_{i:04d}" for i in range(len(names), n_nodes)]
                            else:
                                names = names[:n_nodes]
                            self.bus_phase_names = names
                            return names
            except Exception:
                # fallthrough to 123_load.mat fallback
                pass

        # ---- Fallback: use 123_load.mat load_Name as "known" nodes, then pad ----
        ld = _load_mat_any(self.cfg.load_mat_path)
        name_key = _get_first_existing_key(ld, ["load_Name", "load_name", "loadName", "loadNames"])
        if name_key is None:
            raise KeyError(f"Cannot find load_Name in {self.cfg.load_mat_path}. Keys: {_mat_keys(ld)}")

        load_names = np.array(ld[name_key]).reshape(-1)
        load_names = [_normalize_bus_name(x) for x in _mat_cellstr_to_list(load_names)]

        # ensure uniqueness while keeping order
        seen = set()
        names = []
        for s in load_names:
            if s and s not in seen:
                seen.add(s)
                names.append(s)

        # pad to n_nodes
        if len(names) < n_nodes:
            names += [f"node_{i:04d}" for i in range(len(names), n_nodes)]
        else:
            names = names[:n_nodes]

        self.bus_phase_names = names
        return names

    # ----------------------------
    # Loads
    # ----------------------------
    def _load_loads(self):
        d = _load_mat_any(self.cfg.load_mat_path)

        def _get(key):
            if key not in d:
                raise KeyError(f"Missing {key} in {self.cfg.load_mat_path}")
            return d[key]

        load_Name = _get("load_Name")
        load_kW = np.asarray(_get("load_kW"), dtype=float).reshape(-1)
        load_kVar = np.asarray(_get("load_kVar"), dtype=float).reshape(-1)

        load_Name_list = _mat_cellstr_to_list(load_Name)
        load_Name_list = [_normalize_bus_name(s) for s in load_Name_list]

        return load_Name_list, load_kW, load_kVar

    def _build_bus_load_profiles(self, bus_names, load_names, load_kW, load_kVar, RD):
        Sbase = float(self.cfg.Sbase)
        T = int(self.cfg.T)
        factors = np.asarray(self.cfg.load_hour_factors, dtype=float).reshape(-1)
        if factors.size != T:
            raise ValueError("cfg.load_hour_factors must be length T (e.g., 24)")

        nb = int(RD.shape[0])
        p_load = np.zeros((nb, T), dtype=float)
        q_load = np.zeros((nb, T), dtype=float)

        name_to_idx = {bus_names[i]: i for i in range(len(bus_names))}

        for i, lname in enumerate(load_names):
            # allow S-prefix mismatch
            idx = None
            for cand in _alt_bus_names(lname):
                if cand in name_to_idx:
                    idx = name_to_idx[cand]
                    break
            if idx is None:
                continue
            p_load[idx, :] += (load_kW[i] * factors) / Sbase
            q_load[idx, :] += (load_kVar[i] * factors) / Sbase

        return p_load, q_load

    def _der_bus_indices(self, bus_names):
        der_names = list(getattr(self.cfg, "DER_BusName", []))
        name_to_idx = {bus_names[i]: i for i in range(len(bus_names))}

        idx = []
        for nm in der_names:
            found = None
            for cand in _alt_bus_names(nm):
                if cand in name_to_idx:
                    found = int(name_to_idx[cand])
                    break
            if found is None:
                raise KeyError(f"DER bus name {nm} not found in bus_phase_names (or alt forms).")
            idx.append(found)
        return np.asarray(idx, dtype=int)

    # ----------------------------
    # PV bounds + flexible load (same as你之前版本)
    # ----------------------------
    def _pv_bounds_pu(self, n_der: int):
        Sbase = float(self.cfg.Sbase)
        T = int(self.cfg.T)

        pv_min = np.asarray(self.cfg.pv_min_kw, dtype=float).reshape(-1)
        pv_max = np.asarray(self.cfg.pv_max_kw, dtype=float).reshape(-1)
        if pv_min.size != T or pv_max.size != T:
            raise ValueError("cfg.pv_min_kw/pv_max_kw must be length T")

        pv_max = pv_max + float(getattr(self.cfg, "pv_max_shift_kw", 0.0))

        pv_min_pu_1 = pv_min / Sbase
        pv_max_pu_1 = pv_max / Sbase

        pv_min_pu = np.repeat(pv_min_pu_1.reshape(1, T), n_der, axis=0)
        pv_max_pu = np.repeat(pv_max_pu_1.reshape(1, T), n_der, axis=0)
        return pv_min_pu, pv_max_pu

    def _build_fload_arrays(self, n_der: int):
        if not bool(getattr(self.cfg, "enable_fload", True)):
            T = int(self.cfg.T)
            return (
                False,
                np.zeros((n_der, T), dtype=float),
                np.zeros((n_der, T), dtype=float),
                np.zeros((n_der, T), dtype=float),
            )

        Sbase = float(self.cfg.Sbase)
        T = int(self.cfg.T)

        f0 = np.asarray(self.cfg.fload_0_orig_kw, dtype=float)  # 7xT
        if f0.shape != (7, T):
            raise ValueError("cfg.fload_0_orig_kw must be 7xT (e.g., 7x24)")

        f0 = float(getattr(self.cfg, "fload_scale", 0.2)) * f0
        f0_21 = np.kron(f0, np.ones((3, 1), dtype=float))  # (21,T)

        if f0_21.shape[0] != n_der:
            if f0_21.shape[0] > n_der:
                f0_21 = f0_21[:n_der, :]
            else:
                reps = int(np.ceil(n_der / f0_21.shape[0]))
                f0_21 = np.tile(f0_21, (reps, 1))[:n_der, :]

        fbase = f0_21 / Sbase
        keep_min = float(getattr(self.cfg, "fload_keep_min_kw", 2.0))
        shed_max_kw = np.maximum(0.0, f0_21 - keep_min)
        shed_max = shed_max_kw / Sbase

        cost_vec = np.asarray(getattr(self.cfg, "fload_cost_vec", [1, 1, 1, 1, 1, 1, 1]), dtype=float).reshape(-1)
        if cost_vec.size != 7:
            raise ValueError("cfg.fload_cost_vec must have length 7")
        cost_21 = np.kron(cost_vec.reshape(7, 1), np.ones((3, 1)))  # (21,1)
        cost_21 = cost_21.reshape(-1)[:n_der]
        fcost = np.repeat(cost_21.reshape(n_der, 1), T, axis=1)
        fcost = float(getattr(self.cfg, "Fload_cost_scale", 1.0)) * fcost

        return True, fbase.astype(float), shed_max.astype(float), fcost.astype(float)

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
        prob["offer_curve_sampling"] = getattr(self.cfg, "offer_curve_sampling", "none")
        prob["p_da_bound_margin_pu"] = getattr(self.cfg, "p_da_bound_margin_pu", 0.05)

        prob["cfg"] = self.cfg
        return prob

    def sample_instances(self, two_ro):
        T = int(getattr(self.cfg, "T", 24))
        if T != 24:
            raise ValueError("offering_network expects T=24")

        Sbase = float(self.cfg.Sbase)

        # price matrix (24,S)
        price_var = getattr(self.cfg, "price_mat_var", None)
        price_matrix = self._load_price_matrix_any(self.cfg.price_mat_path, varname=price_var)
        S = int(price_matrix.shape[1])
        rho = 1.0 / S  # uniform

        lambda_da = (Sbase * np.asarray(price_matrix, dtype=float)) / 1000.0  # (24,S)
        lambda_rt = float(self.cfg.lambda_rt_value) * np.ones(T, dtype=float)

        # network data
        A, A0, RD, XD = self._load_network()
        nb = int(RD.shape[0])

        # v0_term
        V_min = float(getattr(self.cfg, "V_min", 0.9))
        V_max = float(getattr(self.cfg, "V_max", 1.1))
        v_min = V_min ** 2
        v_max = V_max ** 2

        v0 = np.ones(3, dtype=float) * (1.0 ** 2)
        At_inv = np.linalg.pinv(A.T)
        v0_term_1 = -(At_inv @ (A0 @ v0)).reshape(-1)  # (nb,)
        v0_term = np.repeat(v0_term_1.reshape(nb, 1), T, axis=1)

        # loads
        bus_names = self._load_bus_phase_names()
        load_names, load_kW, load_kVar = self._load_loads()
        p_load_bus, q_load_bus = self._build_bus_load_profiles(bus_names, load_names, load_kW, load_kVar, RD)

        # DER mapping
        der_bus = self._der_bus_indices(bus_names)
        n_der = int(der_bus.shape[0])

        # PV bounds
        pv_min_pu, pv_max_pu = self._pv_bounds_pu(n_der)

        # flexible load (optional)
        enable_fload, fbase, shed_max, fcost = self._build_fload_arrays(n_der)

        # ESS per DER
        Emax = float(getattr(self.cfg, "E_ess_per_unit_kwh", 14.5)) / Sbase
        Pmax = float(getattr(self.cfg, "P_ess_per_unit_kw", 11.3)) / Sbase
        eta = float(getattr(self.cfg, "eta", 0.95))
        soc0 = float(getattr(self.cfg, "soc0_frac", 0.5)) * Emax * np.ones(n_der, dtype=float)

        ESS_cost = float(getattr(self.cfg, "ESS_cost", 0.0))
        PV_cost = float(getattr(self.cfg, "PV_cost", 0.0))

        rd_row_abs_sum = np.sum(np.abs(RD), axis=1)
        xd_row_abs_sum = np.sum(np.abs(XD), axis=1)
        net_feats = {
            "rd_abs_sum_max": float(np.max(rd_row_abs_sum)),
            "rd_abs_sum_mean": float(np.mean(rd_row_abs_sum)),
            "xd_abs_sum_max": float(np.max(xd_row_abs_sum)),
            "xd_abs_sum_mean": float(np.mean(xd_row_abs_sum)),
        }

        instances = []
        for s in range(S):
            inst = {
                "T": T,
                "scenario_id": int(s),
                "rho": float(rho),
                "lambda_da": lambda_da[:, s].reshape(-1),
                "lambda_rt": lambda_rt.reshape(-1),

                "nb": int(nb),
                "n_der": int(n_der),
                "der_bus": der_bus.astype(int).reshape(-1),

                "A": np.asarray(A, dtype=float),
                "A0": np.asarray(A0, dtype=float),
                "RD": np.asarray(RD, dtype=float),
                "XD": np.asarray(XD, dtype=float),
                "v0_term": np.asarray(v0_term, dtype=float),

                "v_min": float(v_min),
                "v_max": float(v_max),

                "p_load_bus": np.asarray(p_load_bus, dtype=float),
                "q_load_bus": np.asarray(q_load_bus, dtype=float),

                "p_pv_min": np.asarray(pv_min_pu, dtype=float),
                "p_pv_max": np.asarray(pv_max_pu, dtype=float),

                "enable_fload": bool(enable_fload),
                "fload_base": np.asarray(fbase, dtype=float),
                "fload_shed_max": np.asarray(shed_max, dtype=float),
                "fload_cost": np.asarray(fcost, dtype=float),

                "P_ess_max": (Pmax * np.ones(n_der, dtype=float)).reshape(-1),
                "E_ess_max": (Emax * np.ones(n_der, dtype=float)).reshape(-1),
                "eta": float(eta),
                "soc0": soc0.reshape(-1),

                "ESS_cost": float(ESS_cost),
                "PV_cost": float(PV_cost),
                "Gamma": float(getattr(self.cfg, "Gamma", 12.0)),

                **net_feats,
            }
            instances.append(inst)

        return instances

    # ----------------------------
    # Sampling x / xi
    # ----------------------------
    def _x_bounds_from_inst(self, inst):
        nb = int(inst["nb"])
        n_der = int(inst["n_der"])
        T = int(inst["T"])

        p_load = np.asarray(inst["p_load_bus"], dtype=float)  # (nb,T)

        enable_fload = bool(inst.get("enable_fload", False))
        fbase = np.asarray(inst.get("fload_base", np.zeros((n_der, T))), dtype=float)
        shed_max = np.asarray(inst.get("fload_shed_max", np.zeros((n_der, T))), dtype=float)

        pv_max = np.asarray(inst["p_pv_max"], dtype=float)  # (n_der,T)
        Pmax = np.asarray(inst["P_ess_max"], dtype=float).reshape(-1)  # (n_der,)

        load_sum = np.sum(p_load, axis=0)  # (T,)
        if enable_fload:
            fbase_sum = np.sum(fbase, axis=0)
            shed_sum = np.sum(shed_max, axis=0)
        else:
            fbase_sum = np.zeros(T)
            shed_sum = np.zeros(T)

        pv_sum = np.sum(pv_max, axis=0)
        Psum = float(np.sum(Pmax))

        ub = -load_sum - fbase_sum + shed_sum + pv_sum + Psum
        lb = -load_sum - fbase_sum - Psum

        m = float(getattr(self.cfg, "p_da_bound_margin_pu", 0.05))
        lb = lb - m
        ub = ub + m
        width = np.maximum(ub - lb, 1e-8)
        return lb, ub, width

    def _sample_offer_curve_monotone(self, lambda_da: np.ndarray, inst0):
        T, S = lambda_da.shape
        lb, ub, _ = self._x_bounds_from_inst(inst0)
        p_DA = np.zeros((T, S), dtype=float)
        for t in range(T):
            order = np.argsort(lambda_da[t, :], kind="stable")
            q_sorted = np.sort(np.random.uniform(lb[t], ub[t], size=S))
            p_DA[t, order] = q_sorted
        return p_DA

    def sample_x(self, instance):
        T = int(instance["T"])
        lb, ub, _ = self._x_bounds_from_inst(instance)
        x = np.random.uniform(lb, ub, size=T).astype(float)
        return x

    def sample_xi(self, instance, x):
        T = int(instance["T"])
        pv_min = np.asarray(instance["p_pv_min"], dtype=float)  # (n_der,T)
        pv_max = np.asarray(instance["p_pv_max"], dtype=float)
        Gamma = float(instance["Gamma"])

        dsum = 0.5 * np.sum(pv_max - pv_min, axis=0)  # (T,)
        active = dsum > 1e-12

        delta = np.random.uniform(-1.0, 1.0, size=T)
        target = np.random.uniform(0.0, Gamma)
        denom = np.sum(np.abs(delta[active]))
        if denom < 1e-12:
            delta[:] = 0.0
        else:
            delta = delta * (target / denom)

        delta = np.clip(delta, -1.0, 1.0)
        xi = 0.5 * (delta + 1.0)
        xi[~active] = 0.5
        return xi.astype(float)

    def sample_procs(self, two_ro):
        procs_to_run = []
        instances = self.sample_instances(two_ro)

        # optional subsample scenarios
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

        inst0 = instances[0]

        for _ in range(int(self.cfg.n_samples_fs)):
            if offer_sampling == "monotone_by_price":
                p_DA_mat = self._sample_offer_curve_monotone(lambda_da_mat, inst0)  # (24,S_sel)
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

    # ----------------------------
    # solve wrapper
    # ----------------------------
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
