# ro/two_ro/offering_no_network.py:
import gurobipy as gp
import numpy as np

from .two_ro import TwoStageRO


class OfferingNoNetwork(TwoStageRO):
    """
    Two-stage offering strategy (no power flow constraints), consistent with the Matlab SP 'cons_lower + obj_lower'.

    First-stage x:
        x[t] = p_DA[t]  (24-dim vector for a fixed DA price scenario s)

    Uncertainty xi (recommended):
        xi[t] in [0,1]
        pv_upper[t] = pv_min[t] + xi[t]*(pv_max[t]-pv_min[t])

    Second-stage (recourse) variables:
        p_pv[t], p_ch[t], p_dis[t], soc[t], p_rt_buy[t], p_rt_sell[t]
    """

    def __init__(self):
        pass

    @staticmethod
    def _pv_upper_from_xi(xi: np.ndarray, pv_min: np.ndarray, pv_max: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=float).reshape(-1)
        pv_min = np.asarray(pv_min, dtype=float).reshape(-1)
        pv_max = np.asarray(pv_max, dtype=float).reshape(-1)

        if xi.shape[0] != pv_min.shape[0]:
            raise ValueError(f"xi length {xi.shape[0]} != pv_min length {pv_min.shape[0]}")
        xi_clip = np.clip(xi, 0.0, 1.0)
        return pv_min + xi_clip * (pv_max - pv_min)

    def get_first_stage_obj(self, x, instance):
        """First-stage objective (sign consistent with Matlab MP: - DA revenue)."""
        x = np.asarray(x, dtype=float).reshape(-1)
        lam_da = np.asarray(instance["lambda_da"], dtype=float).reshape(-1)
        if x.shape[0] != lam_da.shape[0]:
            raise ValueError(f"x length {x.shape[0]} != lambda_da length {lam_da.shape[0]}")
        rho = float(instance.get("rho", 1.0))
        return -rho * float(np.dot(lam_da, x))

    def solve_second_stage(self, x, xi, instance, gap=0.02, time_limit=600, threads=1, verbose=1):
        """
        Solve recourse LP for given (x, xi, instance).

        Returns:
            fs_obj: first-stage objective value (computed, not optimized here)
            ss_obj: second-stage objective value
            model:  solved gurobi model (with variables attached for debugging)
        """
        # ---- unpack & validate ----
        T = int(instance.get("T", 24))
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != T:
            raise ValueError(f"x must be length T={T}, got {x.shape[0]}")

        lam_rt = np.asarray(instance["lambda_rt"], dtype=float).reshape(-1)
        lam_da = np.asarray(instance["lambda_da"], dtype=float).reshape(-1)
        if lam_rt.shape[0] != T or lam_da.shape[0] != T:
            raise ValueError("lambda_rt / lambda_da must be length T")

        p_load = np.asarray(instance["p_load"], dtype=float).reshape(-1)
        if p_load.shape[0] != T:
            raise ValueError("p_load must be length T")

        pv_min = np.asarray(instance["p_pv_min"], dtype=float).reshape(-1)
        pv_max = np.asarray(instance["p_pv_max"], dtype=float).reshape(-1)
        if pv_min.shape[0] != T or pv_max.shape[0] != T:
            raise ValueError("p_pv_min / p_pv_max must be length T")

        pv_upper = self._pv_upper_from_xi(xi, pv_min, pv_max)

        P_ess_max = float(instance["P_ess_max"])
        E_ess_max = float(instance["E_ess_max"])
        eta = float(instance["eta"])

        ESS_cost = float(instance.get("ESS_cost", 0.0))
        PV_cost = float(instance.get("PV_cost", 0.0))

        # SOC initial (Matlab: 0.5*E_ess_0)
        soc0 = float(instance.get("soc0", 0.5 * E_ess_max))

        # ---- first-stage cost (reported only) ----
        fs_obj = self.get_first_stage_obj(x, instance)

        # ---- build LP ----
        m = gp.Model("offering_recourselp")
        m.Params.OutputFlag = 1 if verbose else 0
        m.Params.Threads = int(threads)
        m.Params.TimeLimit = float(time_limit)

        # variables
        p_pv = m.addVars(T, lb=0.0, name="p_pv")
        p_ch = m.addVars(T, lb=0.0, ub=P_ess_max, name="p_ch")
        p_dis = m.addVars(T, lb=0.0, ub=P_ess_max, name="p_dis")
        soc = m.addVars(T, lb=0.0, ub=E_ess_max, name="soc")

        p_rt_b = m.addVars(T, lb=0.0, name="p_rt_buy")
        p_rt_s = m.addVars(T, lb=0.0, name="p_rt_sell")

        # PV upper bounds
        for t in range(T):
            p_pv[t].UB = float(pv_upper[t])

        # SOC dynamics
        for t in range(T):
            if t == 0:
                m.addConstr(
                    soc[t] == soc0 + eta * p_ch[t] - (1.0 / eta) * p_dis[t],
                    name=f"soc_dyn_{t}",
                )
            else:
                m.addConstr(
                    soc[t] == soc[t - 1] + eta * p_ch[t] - (1.0 / eta) * p_dis[t],
                    name=f"soc_dyn_{t}",
                )

        # power balance
        # p_DA + buy - sell + load = pv + dis - ch
        for t in range(T):
            m.addConstr(
                x[t] + p_rt_b[t] - p_rt_s[t] + p_load[t]
                == p_pv[t] + p_dis[t] - p_ch[t],
                name=f"balance_{t}",
            )

        # objective (second-stage cost)
        obj = gp.quicksum(lam_rt[t] * (p_rt_b[t] + p_rt_s[t]) for t in range(T))
        obj += ESS_cost * gp.quicksum(p_dis[t] for t in range(T))
        obj += PV_cost * gp.quicksum(p_pv[t] for t in range(T))
        m.setObjective(obj, gp.GRB.MINIMIZE)

        m.optimize()

        if m.Status not in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL):
            # keep consistent with existing pipeline
            raise RuntimeError(f"Second-stage solve failed. Gurobi status={m.Status}")

        ss_obj = float(m.ObjVal)

        # attach for downstream inspection (optional)
        m._p_pv = p_pv
        m._p_ch = p_ch
        m._p_dis = p_dis
        m._soc = soc
        m._p_rt_b = p_rt_b
        m._p_rt_s = p_rt_s

        return fs_obj, ss_obj, m





