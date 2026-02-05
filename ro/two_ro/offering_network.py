# # /content/drive/MyDrive/Neur2RO/ro/two_ro/offering_network.py
# import gurobipy as gp
# import numpy as np

# from .two_ro import TwoStageRO


# class OfferingNetwork(TwoStageRO):
#     """
#     Two-stage offering strategy with linear distribution-network constraints.

#     First-stage x (length T):
#         x[t] = p_DA[t]  (positive means sell, negative means buy)

#     Uncertainty xi (length T):
#         pv_upper[:,t] = pv_min[:,t] + xi[t]*(pv_max[:,t]-pv_min[:,t])
#         Budget: sum_{t in active} |2xi[t]-1| <= Gamma

#     Second-stage variables (per DER and/or per bus):
#         - PV dispatch p_pv[d,t]
#         - ESS charge/discharge/soc at DER nodes
#         - optional flexible-load shedding p_shed[d,t]
#         - network net load p_bus[b,t], q_bus[b,t] (q fixed)
#         - voltage squared v_bus[b,t] with linear equation:
#             v = v0_term - 2*RD*p - 2*XD*q
#         - real-time deviation trades p_rt_buy[t], p_rt_sell[t]
#         - market-physical coupling:
#             x[t] + buy[t] - sell[t] = - sum_b p_bus[b,t]
#     """

#     def __init__(self):
#         pass

#     @staticmethod
#     def _pv_upper_from_xi(xi: np.ndarray, pv_min: np.ndarray, pv_max: np.ndarray) -> np.ndarray:
#         xi = np.asarray(xi, dtype=float).reshape(-1)
#         pv_min = np.asarray(pv_min, dtype=float)
#         pv_max = np.asarray(pv_max, dtype=float)
#         if xi.ndim != 1:
#             xi = xi.reshape(-1)
#         if pv_min.shape[1] != xi.shape[0]:
#             raise ValueError(f"xi length {xi.shape[0]} != pv_min.shape[1] {pv_min.shape[1]}")
#         xi_clip = np.clip(xi, 0.0, 1.0)
#         return pv_min + (pv_max - pv_min) * xi_clip.reshape(1, -1)

#     def get_first_stage_obj(self, x, instance):
#         x = np.asarray(x, dtype=float).reshape(-1)
#         lam_da = np.asarray(instance["lambda_da"], dtype=float).reshape(-1)
#         if x.shape[0] != lam_da.shape[0]:
#             raise ValueError(f"x length {x.shape[0]} != lambda_da length {lam_da.shape[0]}")
#         rho = float(instance.get("rho", 1.0))
#         return -rho * float(np.dot(lam_da, x))

#     def solve_second_stage(self, x, xi, instance, gap=0.02, time_limit=600, threads=1, verbose=1):
#         # ---- unpack ----
#         T = int(instance.get("T", 24))
#         x = np.asarray(x, dtype=float).reshape(-1)
#         if x.shape[0] != T:
#             raise ValueError(f"x must be length T={T}, got {x.shape[0]}")

#         lam_rt = np.asarray(instance["lambda_rt"], dtype=float).reshape(-1)
#         lam_da = np.asarray(instance["lambda_da"], dtype=float).reshape(-1)
#         if lam_rt.shape[0] != T or lam_da.shape[0] != T:
#             raise ValueError("lambda_rt / lambda_da must be length T")

#         nb = int(instance["nb"])
#         n_der = int(instance["n_der"])
#         der_bus = np.asarray(instance["der_bus"], dtype=int).reshape(-1)
#         if der_bus.shape[0] != n_der:
#             raise ValueError("der_bus length mismatch")

#         # network data
#         RD = np.asarray(instance["RD"], dtype=float)
#         XD = np.asarray(instance["XD"], dtype=float)
#         if RD.shape != (nb, nb) or XD.shape != (nb, nb):
#             raise ValueError(f"RD/XD must be nb x nb, got {RD.shape}/{XD.shape}")

#         v0_term = np.asarray(instance["v0_term"], dtype=float)  # (nb,T)
#         if v0_term.shape != (nb, T):
#             raise ValueError(f"v0_term must be (nb,T), got {v0_term.shape}")

#         p_load = np.asarray(instance["p_load_bus"], dtype=float)  # (nb,T)
#         q_load = np.asarray(instance["q_load_bus"], dtype=float)  # (nb,T)
#         if p_load.shape != (nb, T) or q_load.shape != (nb, T):
#             raise ValueError("p_load_bus/q_load_bus shape mismatch")

#         # DER-side profiles
#         pv_min = np.asarray(instance["p_pv_min"], dtype=float)  # (n_der,T)
#         pv_max = np.asarray(instance["p_pv_max"], dtype=float)  # (n_der,T)
#         if pv_min.shape != (n_der, T) or pv_max.shape != (n_der, T):
#             raise ValueError("p_pv_min/p_pv_max shape mismatch")

#         enable_fload = bool(instance.get("enable_fload", False))
#         fbase = np.asarray(instance.get("fload_base", np.zeros((n_der, T))), dtype=float)  # (n_der,T)
#         shed_max = np.asarray(instance.get("fload_shed_max", np.zeros((n_der, T))), dtype=float)
#         fcost = np.asarray(instance.get("fload_cost", np.zeros((n_der, T))), dtype=float)
#         if enable_fload:
#             if fbase.shape != (n_der, T) or shed_max.shape != (n_der, T) or fcost.shape != (n_der, T):
#                 raise ValueError("fload arrays shape mismatch")

#         # ESS parameters (per DER)
#         P_ess_max = np.asarray(instance["P_ess_max"], dtype=float).reshape(-1)  # (n_der,)
#         E_ess_max = np.asarray(instance["E_ess_max"], dtype=float).reshape(-1)  # (n_der,)
#         if P_ess_max.shape[0] != n_der or E_ess_max.shape[0] != n_der:
#             raise ValueError("P_ess_max/E_ess_max length mismatch")

#         eta = float(instance["eta"])
#         soc0 = np.asarray(instance["soc0"], dtype=float).reshape(-1)  # (n_der,)
#         if soc0.shape[0] != n_der:
#             raise ValueError("soc0 length mismatch")

#         # voltage bounds
#         v_min = float(instance["v_min"])
#         v_max = float(instance["v_max"])

#         ESS_cost = float(instance.get("ESS_cost", 0.0))
#         PV_cost = float(instance.get("PV_cost", 0.0))

#         # ---- derived PV upper ----
#         pv_upper = self._pv_upper_from_xi(xi, pv_min, pv_max)

#         # ---- first-stage cost for report ----
#         fs_obj = self.get_first_stage_obj(x, instance)

#         # ---- build model ----
#         m = gp.Model("offering_network_recourselp")
#         m.Params.OutputFlag = 1 if verbose else 0
#         m.Params.Threads = int(threads)
#         m.Params.TimeLimit = float(time_limit)
#         m.Params.MIPGap = float(gap)

#         # RT trades
#         p_rt_b = m.addVars(T, lb=0.0, name="p_rt_buy")
#         p_rt_s = m.addVars(T, lb=0.0, name="p_rt_sell")

#         # DER vars
#         p_pv = m.addVars(n_der, T, lb=0.0, name="p_pv")
#         for d in range(n_der):
#             for t in range(T):
#                 p_pv[d, t].UB = float(pv_upper[d, t])

#         # flexible load shedding
#         if enable_fload:
#             p_shed = m.addVars(n_der, T, lb=0.0, name="p_shed")
#             for d in range(n_der):
#                 for t in range(T):
#                     p_shed[d, t].UB = float(max(0.0, shed_max[d, t]))
#         else:
#             p_shed = None

#         # ESS
#         p_ch = m.addVars(n_der, T, lb=0.0, name="p_ch")
#         p_dis = m.addVars(n_der, T, lb=0.0, name="p_dis")
#         soc = m.addVars(n_der, T, lb=0.0, name="soc")
#         b_ess = m.addVars(n_der, T, vtype=gp.GRB.BINARY, name="b_ess")

#         for d in range(n_der):
#             for t in range(T):
#                 p_ch[d, t].UB = float(P_ess_max[d])
#                 p_dis[d, t].UB = float(P_ess_max[d])
#                 soc[d, t].UB = float(E_ess_max[d])
#                 # big-M logic (charge vs discharge)
#                 m.addConstr(p_ch[d, t] <= float(P_ess_max[d]) * b_ess[d, t], name=f"ch_bin_{d}_{t}")
#                 m.addConstr(p_dis[d, t] <= float(P_ess_max[d]) * (1.0 - b_ess[d, t]), name=f"dis_bin_{d}_{t}")

#         # SOC dynamics
#         for d in range(n_der):
#             for t in range(T):
#                 if t == 0:
#                     m.addConstr(
#                         soc[d, t] == float(soc0[d]) + eta * p_ch[d, t] - (1.0 / eta) * p_dis[d, t],
#                         name=f"soc_dyn_{d}_{t}",
#                     )
#                 else:
#                     m.addConstr(
#                         soc[d, t] == soc[d, t - 1] + eta * p_ch[d, t] - (1.0 / eta) * p_dis[d, t],
#                         name=f"soc_dyn_{d}_{t}",
#                     )

#         # bus vars
#         p_bus = m.addVars(nb, T, name="p_bus")   # net active load (positive load)
#         v_bus = m.addVars(nb, T, lb=v_min, ub=v_max, name="v_bus")

#         # fix non-DER bus p = load
#         all_buses = np.arange(nb, dtype=int)
#         non_der = np.setdiff1d(all_buses, der_bus)

#         for b in non_der:
#             for t in range(T):
#                 p_bus[b, t].LB = float(p_load[b, t])
#                 p_bus[b, t].UB = float(p_load[b, t])

#         # DER bus p equation: p = p_load + (fbase - shed) - pv - dis + ch
#         for k, b in enumerate(der_bus):
#             for t in range(T):
#                 expr = float(p_load[b, t]) - p_pv[k, t] - p_dis[k, t] + p_ch[k, t]
#                 if enable_fload:
#                     expr = expr + float(fbase[k, t]) - p_shed[k, t]
#                 m.addConstr(p_bus[b, t] == expr, name=f"p_der_{k}_{t}")

#         # q fixed -> fold into voltage RHS (v0_term - 2*XD*q)
#         rhs_v = v0_term - 2.0 * (XD @ q_load)  # (nb,T)

#         # precompute RD sparsity
#         RD_nz = []
#         thr = 1e-12
#         for i in range(nb):
#             js = np.where(np.abs(RD[i, :]) > thr)[0]
#             RD_nz.append([(int(j), float(RD[i, j])) for j in js])

#         # voltage equation: v = rhs_v - 2*RD*p
#         for i in range(nb):
#             for t in range(T):
#                 lin = gp.LinExpr()
#                 for (j, coeff) in RD_nz[i]:
#                     lin.add(p_bus[j, t], -2.0 * coeff)
#                 m.addConstr(v_bus[i, t] == float(rhs_v[i, t]) + lin, name=f"v_eq_{i}_{t}")

#         # market-physical coupling: x + buy - sell = - sum_b p_bus
#         for t in range(T):
#             m.addConstr(
#                 x[t] + p_rt_b[t] - p_rt_s[t] == -gp.quicksum(p_bus[b, t] for b in range(nb)),
#                 name=f"market_couple_{t}",
#             )

#         # objective (second-stage cost)
#         obj = gp.quicksum(lam_rt[t] * (p_rt_b[t] + p_rt_s[t]) for t in range(T))
#         obj += ESS_cost * gp.quicksum(p_dis[d, t] for d in range(n_der) for t in range(T))
#         obj += PV_cost * gp.quicksum(p_pv[d, t] for d in range(n_der) for t in range(T))
#         if enable_fload:
#             obj += gp.quicksum(float(fcost[d, t]) * p_shed[d, t] for d in range(n_der) for t in range(T))
#         m.setObjective(obj, gp.GRB.MINIMIZE)

#         m.optimize()

#         if m.Status not in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL):
#             raise RuntimeError(f"Second-stage solve failed. Gurobi status={m.Status}")

#         ss_obj = float(m.ObjVal)

#         # attach for inspection
#         m._p_pv = p_pv
#         m._p_ch = p_ch
#         m._p_dis = p_dis
#         m._soc = soc
#         m._p_rt_b = p_rt_b
#         m._p_rt_s = p_rt_s
#         m._p_bus = p_bus
#         m._v_bus = v_bus
#         m._p_shed = p_shed

#         return fs_obj, ss_obj, m




# /content/drive/MyDrive/Neur2RO/ro/two_ro/offering_network.py
import gurobipy as gp
import numpy as np
from typing import Dict, Any, Tuple, Optional

from .two_ro import TwoStageRO

# ============================================================
# Speed-ups:
#  1) Per-process Gurobi Env singleton (avoids repeated WLS handshake where possible)
#  2) Per-instance MODEL TEMPLATE cache: build constraints once, then only update
#       - PV upper bounds (UB)
#       - market coupling RHS (depends on x)
#     and re-optimize
#  3) Optional: relax ESS binary (LP instead of MILP) via instance['use_ess_binary']=False
# ============================================================

_GUROBI_ENV: Optional[gp.Env] = None
_MODEL_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}


def _get_gurobi_env() -> Optional[gp.Env]:
    """
    Create a per-process Env once, so repeated model creation does not repeatedly
    re-initialize licensing (WLS). If Env creation fails, return None and fall back
    to default gp.Model().
    """
    global _GUROBI_ENV
    if _GUROBI_ENV is not None:
        return _GUROBI_ENV

    try:
        env = gp.Env(empty=True)
        # Do NOT set WLS params here (secrets). Assume user already configured env/license.
        env.start()
        _GUROBI_ENV = env
        return _GUROBI_ENV
    except Exception:
        # Fallback to default global environment behavior
        _GUROBI_ENV = None
        return None


def _cache_key_from_instance(instance: Dict[str, Any]) -> Tuple[Any, ...]:
    """
    Cache key for template model. Keep it stable & cheap.
    scenario_id is usually unique per instance; include dimensions and flags.
    """
    return (
        instance.get("scenario_id", None),
        int(instance.get("nb", -1)),
        int(instance.get("n_der", -1)),
        bool(instance.get("enable_fload", False)),
        bool(instance.get("use_ess_binary", True)),
        # If you may have multiple different networks sharing the same scenario_id,
        # add another stable identifier here (e.g., instance.get("net_id")).
    )


class OfferingNetwork(TwoStageRO):
    """
    Two-stage offering strategy with linear distribution-network constraints.

    First-stage x (length T):
        x[t] = p_DA[t]  (positive means sell, negative means buy)

    Uncertainty xi (length T):
        pv_upper[:,t] = pv_min[:,t] + xi[t]*(pv_max[:,t]-pv_min[:,t])
        Budget: sum_{t in active} |2xi[t]-1| <= Gamma

    Second-stage variables (per DER and/or per bus):
        - PV dispatch p_pv[d,t]
        - ESS charge/discharge/soc at DER nodes
        - optional flexible-load shedding p_shed[d,t]
        - network net load p_bus[b,t], q_bus[b,t] (q fixed)
        - voltage squared v_bus[b,t] with linear equation:
            v = v0_term - 2*RD*p - 2*XD*q
        - real-time deviation trades p_rt_buy[t], p_rt_sell[t]
        - market-physical coupling:
            x[t] + buy[t] - sell[t] = - sum_b p_bus[b,t]
    """

    def __init__(self):
        pass

    @staticmethod
    def _pv_upper_from_xi(xi: np.ndarray, pv_min: np.ndarray, pv_max: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=float).reshape(-1)
        pv_min = np.asarray(pv_min, dtype=float)
        pv_max = np.asarray(pv_max, dtype=float)
        if pv_min.shape[1] != xi.shape[0]:
            raise ValueError(f"xi length {xi.shape[0]} != pv_min.shape[1] {pv_min.shape[1]}")
        xi_clip = np.clip(xi, 0.0, 1.0)
        return pv_min + (pv_max - pv_min) * xi_clip.reshape(1, -1)

    def get_first_stage_obj(self, x, instance):
        x = np.asarray(x, dtype=float).reshape(-1)
        lam_da = np.asarray(instance["lambda_da"], dtype=float).reshape(-1)
        if x.shape[0] != lam_da.shape[0]:
            raise ValueError(f"x length {x.shape[0]} != lambda_da length {lam_da.shape[0]}")
        rho = float(instance.get("rho", 1.0))
        return -rho * float(np.dot(lam_da, x))

    # ----------------------------
    # Template builder (cached)
    # ----------------------------
    @staticmethod
    def _build_or_get_template(
        instance: Dict[str, Any],
        gap: float,
        time_limit: float,
        threads: int,
        verbose: int,
    ) -> Dict[str, Any]:
        """
        Build a per-instance template model once and cache it in-process.
        Later solves only update:
          - p_pv UB (depends on xi)
          - market coupling RHS (depends on x)
        """
        use_cache = bool(instance.get("cache_model_template", True))
        key = _cache_key_from_instance(instance)

        if use_cache and key in _MODEL_CACHE:
            tmpl = _MODEL_CACHE[key]
            # Update parameters that may change between calls
            m = tmpl["model"]
            m.Params.OutputFlag = 1 if verbose else 0
            m.Params.Threads = int(threads)
            m.Params.TimeLimit = float(time_limit)
            m.Params.MIPGap = float(gap)
            return tmpl

        # ---- unpack fixed data ----
        T = int(instance.get("T", 24))
        nb = int(instance["nb"])
        n_der = int(instance["n_der"])
        der_bus = np.asarray(instance["der_bus"], dtype=int).reshape(-1)

        RD = np.asarray(instance["RD"], dtype=float)
        XD = np.asarray(instance["XD"], dtype=float)
        v0_term = np.asarray(instance["v0_term"], dtype=float)  # (nb,T)
        p_load = np.asarray(instance["p_load_bus"], dtype=float)  # (nb,T)
        q_load = np.asarray(instance["q_load_bus"], dtype=float)  # (nb,T)

        v_min = float(instance["v_min"])
        v_max = float(instance["v_max"])

        lam_rt = np.asarray(instance["lambda_rt"], dtype=float).reshape(-1)
        if lam_rt.shape[0] != T:
            raise ValueError("lambda_rt must be length T")

        enable_fload = bool(instance.get("enable_fload", False))
        fbase = np.asarray(instance.get("fload_base", np.zeros((n_der, T))), dtype=float)
        shed_max = np.asarray(instance.get("fload_shed_max", np.zeros((n_der, T))), dtype=float)
        fcost = np.asarray(instance.get("fload_cost", np.zeros((n_der, T))), dtype=float)

        P_ess_max = np.asarray(instance["P_ess_max"], dtype=float).reshape(-1)  # (n_der,)
        E_ess_max = np.asarray(instance["E_ess_max"], dtype=float).reshape(-1)  # (n_der,)
        eta = float(instance["eta"])
        soc0 = np.asarray(instance["soc0"], dtype=float).reshape(-1)  # (n_der,)

        ESS_cost = float(instance.get("ESS_cost", 0.0))
        PV_cost = float(instance.get("PV_cost", 0.0))

        use_ess_binary = bool(instance.get("use_ess_binary", True))

        # ---- build model (with optional per-process Env) ----
        env = _get_gurobi_env()
        if env is not None:
            m = gp.Model("offering_network_template", env=env)
        else:
            m = gp.Model("offering_network_template")

        # ---- Parameters ----
        m.Params.OutputFlag = 1 if verbose else 0
        m.Params.Threads = int(threads)
        m.Params.TimeLimit = float(time_limit)
        m.Params.MIPGap = float(gap)

        # Additional speed-oriented defaults (safe, can be overridden if needed)
        # (Keep conservative; you can comment these out if undesired)
        m.Params.Presolve = 2
        m.Params.MIPFocus = 1
        m.Params.Heuristics = 0.05

        # ---- Variables ----
        # RT trades
        p_rt_b = m.addVars(T, lb=0.0, name="p_rt_buy")
        p_rt_s = m.addVars(T, lb=0.0, name="p_rt_sell")

        # DER PV (UB updated per solve)
        p_pv = m.addVars(n_der, T, lb=0.0, name="p_pv")

        # flexible load shedding
        if enable_fload:
            p_shed = m.addVars(n_der, T, lb=0.0, name="p_shed")
            # fixed UB (can still be tightened per solve if you want)
            for d in range(n_der):
                for t in range(T):
                    p_shed[d, t].UB = float(max(0.0, shed_max[d, t]))
        else:
            p_shed = None

        # ESS
        p_ch = m.addVars(n_der, T, lb=0.0, name="p_ch")
        p_dis = m.addVars(n_der, T, lb=0.0, name="p_dis")
        soc = m.addVars(n_der, T, lb=0.0, name="soc")

        if use_ess_binary:
            b_ess = m.addVars(n_der, T, vtype=gp.GRB.BINARY, name="b_ess")
        else:
            b_ess = None

        for d in range(n_der):
            for t in range(T):
                p_ch[d, t].UB = float(P_ess_max[d])
                p_dis[d, t].UB = float(P_ess_max[d])
                soc[d, t].UB = float(E_ess_max[d])

                if use_ess_binary:
                    # big-M logic (charge vs discharge)
                    m.addConstr(
                        p_ch[d, t] <= float(P_ess_max[d]) * b_ess[d, t],
                        name=f"ch_bin_{d}_{t}",
                    )
                    m.addConstr(
                        p_dis[d, t] <= float(P_ess_max[d]) * (1.0 - b_ess[d, t]),
                        name=f"dis_bin_{d}_{t}",
                    )
                else:
                    # LP relaxation: prevent extreme simultaneous charge/discharge
                    # (still allows both >0, but bounded by sum <= Pmax)
                    m.addConstr(
                        p_ch[d, t] + p_dis[d, t] <= float(P_ess_max[d]),
                        name=f"chdis_sum_{d}_{t}",
                    )

        # SOC dynamics
        for d in range(n_der):
            for t in range(T):
                if t == 0:
                    m.addConstr(
                        soc[d, t]
                        == float(soc0[d]) + eta * p_ch[d, t] - (1.0 / eta) * p_dis[d, t],
                        name=f"soc_dyn_{d}_{t}",
                    )
                else:
                    m.addConstr(
                        soc[d, t]
                        == soc[d, t - 1] + eta * p_ch[d, t] - (1.0 / eta) * p_dis[d, t],
                        name=f"soc_dyn_{d}_{t}",
                    )

        # bus vars
        p_bus = m.addVars(nb, T, name="p_bus")  # net active load (positive load)
        v_bus = m.addVars(nb, T, lb=v_min, ub=v_max, name="v_bus")

        # Fix non-DER bus p = load via variable bounds (fast)
        all_buses = np.arange(nb, dtype=int)
        non_der = np.setdiff1d(all_buses, der_bus)

        for b in non_der:
            for t in range(T):
                val = float(p_load[b, t])
                p_bus[b, t].LB = val
                p_bus[b, t].UB = val

        # DER bus p equation: p = p_load + (fbase - shed) - pv - dis + ch
        for k, b in enumerate(der_bus):
            for t in range(T):
                lhs = p_bus[b, t]
                rhs = float(p_load[b, t]) - p_pv[k, t] - p_dis[k, t] + p_ch[k, t]
                if enable_fload:
                    rhs = rhs + float(fbase[k, t]) - p_shed[k, t]
                m.addConstr(lhs == rhs, name=f"p_der_{k}_{t}")

        # q fixed -> fold into voltage RHS (v0_term - 2*XD*q)
        rhs_v = v0_term - 2.0 * (XD @ q_load)  # (nb,T)

        # Precompute RD sparsity once
        thr = 1e-12
        RD_nz = []
        for i in range(nb):
            js = np.where(np.abs(RD[i, :]) > thr)[0]
            RD_nz.append([(int(j), float(RD[i, j])) for j in js])

        # Voltage equation: v = rhs_v - 2*RD*p
        for i in range(nb):
            nz = RD_nz[i]
            for t in range(T):
                lin = gp.LinExpr()
                for (j, coeff) in nz:
                    lin.add(p_bus[j, t], -2.0 * coeff)
                m.addConstr(v_bus[i, t] == float(rhs_v[i, t]) + lin, name=f"v_eq_{i}_{t}")

        # Market-physical coupling: x + buy - sell = - sum_b p_bus
        # Build as: buy - sell + sum_b p_bus = RHS_t, where RHS_t will be set to -x[t] each solve.
        market_constr = []
        for t in range(T):
            lhs = p_rt_b[t] - p_rt_s[t] + gp.quicksum(p_bus[b, t] for b in range(nb))
            c = m.addConstr(lhs == 0.0, name=f"market_couple_{t}")
            market_constr.append(c)

        # Objective (second-stage cost)
        obj = gp.quicksum(float(lam_rt[t]) * (p_rt_b[t] + p_rt_s[t]) for t in range(T))
        obj += float(ESS_cost) * gp.quicksum(p_dis[d, t] for d in range(n_der) for t in range(T))
        obj += float(PV_cost) * gp.quicksum(p_pv[d, t] for d in range(n_der) for t in range(T))
        if enable_fload:
            obj += gp.quicksum(float(fcost[d, t]) * p_shed[d, t] for d in range(n_der) for t in range(T))
        m.setObjective(obj, gp.GRB.MINIMIZE)

        tmpl = {
            "model": m,
            "T": T,
            "nb": nb,
            "n_der": n_der,
            "enable_fload": enable_fload,
            "use_ess_binary": use_ess_binary,
            "p_rt_b": p_rt_b,
            "p_rt_s": p_rt_s,
            "p_pv": p_pv,
            "p_shed": p_shed,
            "p_ch": p_ch,
            "p_dis": p_dis,
            "soc": soc,
            "b_ess": b_ess,
            "p_bus": p_bus,
            "v_bus": v_bus,
            "market_constr": market_constr,
        }

        if use_cache:
            _MODEL_CACHE[key] = tmpl
        return tmpl

    # ----------------------------
    # Second-stage solve (fast path)
    # ----------------------------
    def solve_second_stage(self, x, xi, instance, gap=0.02, time_limit=600, threads=1, verbose=1):
        # ---- unpack basic ----
        T = int(instance.get("T", 24))
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != T:
            raise ValueError(f"x must be length T={T}, got {x.shape[0]}")

        lam_rt = np.asarray(instance["lambda_rt"], dtype=float).reshape(-1)
        lam_da = np.asarray(instance["lambda_da"], dtype=float).reshape(-1)
        if lam_rt.shape[0] != T or lam_da.shape[0] != T:
            raise ValueError("lambda_rt / lambda_da must be length T")

        nb = int(instance["nb"])
        n_der = int(instance["n_der"])
        der_bus = np.asarray(instance["der_bus"], dtype=int).reshape(-1)
        if der_bus.shape[0] != n_der:
            raise ValueError("der_bus length mismatch")

        pv_min = np.asarray(instance["p_pv_min"], dtype=float)  # (n_der,T)
        pv_max = np.asarray(instance["p_pv_max"], dtype=float)  # (n_der,T)
        if pv_min.shape != (n_der, T) or pv_max.shape != (n_der, T):
            raise ValueError("p_pv_min/p_pv_max shape mismatch")

        # ---- derived PV upper ----
        pv_upper = self._pv_upper_from_xi(xi, pv_min, pv_max)

        # ---- first-stage cost for report ----
        fs_obj = self.get_first_stage_obj(x, instance)

        # ---- get/build cached template ----
        tmpl = self._build_or_get_template(instance, gap=gap, time_limit=time_limit, threads=threads, verbose=verbose)
        m = tmpl["model"]

        p_pv = tmpl["p_pv"]
        market_constr = tmpl["market_constr"]

        # ---- update PV upper bounds (depends on xi) ----
        # (n_der*T is small; this is cheap)
        for d in range(n_der):
            for t in range(T):
                p_pv[d, t].UB = float(pv_upper[d, t])

        # ---- update market RHS (depends on x) ----
        # constraint is: buy - sell + sum(p_bus) == RHS_t, set RHS_t = -x[t]
        for t in range(T):
            market_constr[t].RHS = -float(x[t])

        # ---- solve ----
        m.optimize()

        if m.Status not in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL):
            raise RuntimeError(f"Second-stage solve failed. Gurobi status={m.Status}")

        ss_obj = float(m.ObjVal)

        # Attach handles for debugging/inspection (note: template model is reused)
        m._p_pv = tmpl["p_pv"]
        m._p_ch = tmpl["p_ch"]
        m._p_dis = tmpl["p_dis"]
        m._soc = tmpl["soc"]
        m._p_rt_b = tmpl["p_rt_b"]
        m._p_rt_s = tmpl["p_rt_s"]
        m._p_bus = tmpl["p_bus"]
        m._v_bus = tmpl["v_bus"]
        m._p_shed = tmpl["p_shed"]

        return fs_obj, ss_obj, m







