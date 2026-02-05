# /content/drive/MyDrive/Neur2RO/ro/approximator/offering_network.py
import copy
import numpy as np
import gurobipy as gp
from gurobi_ml import add_predictor_constr
import torch

from .approximator import Approximator


class OfferingNetworkApproximator(Approximator):
    """
    Neur2RO approximator for offering_network.

    Master:
      min_{p_DA,z} fs(p_DA) + z
      s.t. z >= NN(p_DA, xi_k) for scenarios xi_k

    Adversarial:
      max_{xi in U} NN(p_DA_fixed, xi)
      U: sum_{t in active} |2xi[t]-1| <= Gamma, xi in [0,1]
    """

    def __init__(self, args, cfg, net, inst_params):
        self.cfg = cfg
        self.inst = self.get_instance(inst_params)
        self.T = int(self.inst.get("T", 24))

        self.feat_scaler = net.feat_scaler
        self.label_scaler = net.label_scaler.get(self.T, None)

        self.net = self.initialize_nn(net)

        self.x_features, self.xi_features = self.get_inst_nn_features()

        self.initialize_main_model(args)
        self.initialize_adversarial_model(args)

        self.has_feas_adv_model = False

    # ----------------------------
    # bounds + active mask
    # ----------------------------
    def _x_bounds(self):
        T = self.T
        p_load = np.asarray(self.inst["p_load_bus"], dtype=float)
        load_sum = np.sum(p_load, axis=0)

        enable_fload = bool(self.inst.get("enable_fload", False))
        if enable_fload:
            fbase_sum = np.sum(np.asarray(self.inst["fload_base"], dtype=float), axis=0)
            shed_sum = np.sum(np.asarray(self.inst["fload_shed_max"], dtype=float), axis=0)
        else:
            fbase_sum = np.zeros(T)
            shed_sum = np.zeros(T)

        pv_max = np.asarray(self.inst["p_pv_max"], dtype=float)
        pv_sum = np.sum(pv_max, axis=0)

        Psum = float(np.sum(np.asarray(self.inst["P_ess_max"], dtype=float).reshape(-1)))

        ub = -load_sum - fbase_sum + shed_sum + pv_sum + Psum
        lb = -load_sum - fbase_sum - Psum

        m = float(getattr(self.cfg, "p_da_bound_margin_pu", getattr(self.cfg, "x_margin", 0.0)))
        lb = lb - m
        ub = ub + m
        width = np.maximum(ub - lb, 1e-8)
        return lb, ub, width

    def _normalize_x(self, x_phys):
        x = np.asarray(x_phys, dtype=float).reshape(-1)
        lb, _, width = self._x_bounds()
        x01 = (x - lb) / width
        return np.clip(x01, 0.0, 1.0)

    def _active_mask(self, tol=1e-12):
        pv_min = np.asarray(self.inst["p_pv_min"], dtype=float)
        pv_max = np.asarray(self.inst["p_pv_max"], dtype=float)
        dsum = 0.5 * np.sum(pv_max - pv_min, axis=0)
        return dsum > tol

    # ----------------------------
    # NN init / features
    # ----------------------------
    def initialize_nn(self, net_):
        net = copy.deepcopy(net_)
        net = net.eval()
        net = net.cpu()
        net.x_embed_net = net.get_grb_compatible_nn(net.x_embed_layers)
        net.x_post_agg_net = net.get_grb_compatible_nn(net.x_post_agg_layers)
        net.xi_embed_net = net.get_grb_compatible_nn(net.xi_embed_layers)
        net.xi_post_agg_net = net.get_grb_compatible_nn(net.xi_post_agg_layers)
        net.value_net = net.get_grb_compatible_nn(net.value_layers)
        return net

    def get_inst_nn_features(self):
        T = self.T
        x01 = np.zeros(T, dtype=float)
        xi = np.zeros(T, dtype=float)

        # build same inst feature list as OfferingNetworkDataPreprocessor
        lam_da = np.asarray(self.inst["lambda_da"], dtype=float).reshape(-1)
        lam_rt = np.asarray(self.inst["lambda_rt"], dtype=float).reshape(-1)
        p_load = np.asarray(self.inst["p_load_bus"], dtype=float)
        q_load = np.asarray(self.inst["q_load_bus"], dtype=float)
        sum_p_load = np.sum(p_load, axis=0)
        sum_q_load = np.sum(q_load, axis=0)

        pv_min = np.asarray(self.inst["p_pv_min"], dtype=float)
        pv_max = np.asarray(self.inst["p_pv_max"], dtype=float)
        sum_pv_min = np.sum(pv_min, axis=0)
        sum_pv_max = np.sum(pv_max, axis=0)

        enable_fload = bool(self.inst.get("enable_fload", False))
        if enable_fload:
            fbase_sum = np.sum(np.asarray(self.inst["fload_base"], dtype=float), axis=0)
            shed_sum = np.sum(np.asarray(self.inst["fload_shed_max"], dtype=float), axis=0)
        else:
            fbase_sum = np.zeros(T)
            shed_sum = np.zeros(T)

        Psum = float(np.sum(np.asarray(self.inst["P_ess_max"], dtype=float).reshape(-1)))
        Esum = float(np.sum(np.asarray(self.inst["E_ess_max"], dtype=float).reshape(-1)))
        eta = float(self.inst["eta"])
        soc0 = np.asarray(self.inst["soc0"], dtype=float).reshape(-1)
        soc0_mean = float(np.mean(soc0)) if soc0.size > 0 else 0.0

        ESS_cost = float(self.inst.get("ESS_cost", 0.0))
        PV_cost = float(self.inst.get("PV_cost", 0.0))
        Gamma = float(self.inst.get("Gamma", 0.0))
        rho = float(self.inst.get("rho", 1.0))

        rd_abs_sum_max = float(self.inst.get("rd_abs_sum_max", 0.0))
        rd_abs_sum_mean = float(self.inst.get("rd_abs_sum_mean", 0.0))
        xd_abs_sum_max = float(self.inst.get("xd_abs_sum_max", 0.0))
        xd_abs_sum_mean = float(self.inst.get("xd_abs_sum_mean", 0.0))

        inst_feats = []
        for t in range(T):
            inst_feats.append([
                lam_da[t],
                lam_rt[t],
                sum_p_load[t],
                sum_q_load[t],
                sum_pv_min[t],
                sum_pv_max[t],
                fbase_sum[t],
                shed_sum[t],
                Psum,
                Esum,
                eta,
                soc0_mean,
                ESS_cost,
                PV_cost,
                Gamma,
                rho,
                rd_abs_sum_max,
                rd_abs_sum_mean,
                xd_abs_sum_max,
                xd_abs_sum_mean,
            ])
        inst_feats = np.asarray(inst_feats, dtype=float)

        x_feats = np.concatenate([x01.reshape(T, 1), inst_feats], axis=1)
        xi_feats = np.concatenate([xi.reshape(T, 1), inst_feats], axis=1)

        fmin, fmax = self.feat_scaler
        denom = (fmax - fmin)
        denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
        x_feats = (x_feats - fmin) / denom
        xi_feats = (xi_feats - fmin) / denom

        x_features = self.to_tensor(x_feats[None, :, :])
        xi_features = self.to_tensor(xi_feats[None, :, :])
        return x_features, xi_features

    def get_x_embed(self, x_phys):
        x01 = self._normalize_x(x_phys)
        x_tensor = self.x_features.clone()
        x_tensor[:, :, 0] = self.to_tensor(x01)
        with torch.no_grad():
            x_embed = self.net.x_embed_net(x_tensor)
            x_embed = self.net.agg_tensor(x_embed, None)
            x_embed = self.net.x_post_agg_net(x_embed)
        return x_embed.detach().cpu().numpy()[0]

    def get_xi_embed(self, xi):
        xi = np.asarray(xi, dtype=float).reshape(-1)
        xi_tensor = self.xi_features.clone()
        xi_tensor[:, :, 0] = self.to_tensor(xi)
        with torch.no_grad():
            xi_embed = self.net.xi_embed_net(xi_tensor)
            xi_embed = self.net.agg_tensor(xi_embed, None)
            xi_embed = self.net.xi_post_agg_net(xi_embed)
        return xi_embed.detach().cpu().numpy()[0]

    # ----------------------------
    # instance loader: delegate to DM-generated instances (recommended)
    # ----------------------------
    def get_instance(self, inst_params):
        # In this implementation we assume evaluation uses pickled instances,
        # thus approximator should be constructed with inst_params containing 'instance' directly
        if "instance" in inst_params and inst_params["instance"] is not None:
            return inst_params["instance"]

        # fallback: build via DM if user only passes scenario_id
        from ro.dm.offering_network import OfferingNetworkDataManager
        dm = OfferingNetworkDataManager(self.cfg, "offering_network")
        from ro.two_ro import factory_two_ro
        two_ro = factory_two_ro("offering_network")
        insts = dm.sample_instances(two_ro)
        sid = int(inst_params.get("scenario_id", 0))
        return insts[sid]

    # ----------------------------
    # Gurobi fixed inst vars (scaled)
    # ----------------------------
    def init_grb_inst_variables(self, m):
        T = self.T
        feat_min = np.asarray(self.feat_scaler[0][1:], dtype=float)
        feat_max = np.asarray(self.feat_scaler[1][1:], dtype=float)
        denom = (feat_max - feat_min)
        denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)

        # rebuild inst_feats unscaled (same list as above)
        lam_da = np.asarray(self.inst["lambda_da"], dtype=float).reshape(-1)
        lam_rt = np.asarray(self.inst["lambda_rt"], dtype=float).reshape(-1)
        p_load = np.asarray(self.inst["p_load_bus"], dtype=float)
        q_load = np.asarray(self.inst["q_load_bus"], dtype=float)
        sum_p_load = np.sum(p_load, axis=0)
        sum_q_load = np.sum(q_load, axis=0)

        pv_min = np.asarray(self.inst["p_pv_min"], dtype=float)
        pv_max = np.asarray(self.inst["p_pv_max"], dtype=float)
        sum_pv_min = np.sum(pv_min, axis=0)
        sum_pv_max = np.sum(pv_max, axis=0)

        enable_fload = bool(self.inst.get("enable_fload", False))
        if enable_fload:
            fbase_sum = np.sum(np.asarray(self.inst["fload_base"], dtype=float), axis=0)
            shed_sum = np.sum(np.asarray(self.inst["fload_shed_max"], dtype=float), axis=0)
        else:
            fbase_sum = np.zeros(T)
            shed_sum = np.zeros(T)

        Psum = float(np.sum(np.asarray(self.inst["P_ess_max"], dtype=float).reshape(-1)))
        Esum = float(np.sum(np.asarray(self.inst["E_ess_max"], dtype=float).reshape(-1)))
        eta = float(self.inst["eta"])
        soc0 = np.asarray(self.inst["soc0"], dtype=float).reshape(-1)
        soc0_mean = float(np.mean(soc0)) if soc0.size > 0 else 0.0

        ESS_cost = float(self.inst.get("ESS_cost", 0.0))
        PV_cost = float(self.inst.get("PV_cost", 0.0))
        Gamma = float(self.inst.get("Gamma", 0.0))
        rho = float(self.inst.get("rho", 1.0))

        rd_abs_sum_max = float(self.inst.get("rd_abs_sum_max", 0.0))
        rd_abs_sum_mean = float(self.inst.get("rd_abs_sum_mean", 0.0))
        xd_abs_sum_max = float(self.inst.get("xd_abs_sum_max", 0.0))
        xd_abs_sum_mean = float(self.inst.get("xd_abs_sum_mean", 0.0))

        inst_vars = []
        for t in range(T):
            inst_vals = np.array([
                lam_da[t],
                lam_rt[t],
                sum_p_load[t],
                sum_q_load[t],
                sum_pv_min[t],
                sum_pv_max[t],
                fbase_sum[t],
                shed_sum[t],
                Psum,
                Esum,
                eta,
                soc0_mean,
                ESS_cost,
                PV_cost,
                Gamma,
                rho,
                rd_abs_sum_max,
                rd_abs_sum_mean,
                xd_abs_sum_max,
                xd_abs_sum_mean,
            ], dtype=float)

            inst_vals_sc = (inst_vals - feat_min) / denom

            vars_t = []
            for j, val in enumerate(inst_vals_sc):
                v = m.addVar(name=f"inst_{t}_{j}", vtype="C")
                v.lb = float(val)
                v.ub = float(val)
                vars_t.append(v)
            inst_vars.append(vars_t)

        return inst_vars

    # ----------------------------
    # Master model
    # ----------------------------
    def initialize_main_model(self, args):
        m = gp.Model("offering_network_master")
        m.setParam("OutputFlag", int(getattr(args, "verbose", 0)))
        m.setParam("MIPGap", float(getattr(args, "mp_gap", 0.01)))
        m.setParam("TimeLimit", float(getattr(args, "mp_time", 60.0)))
        m.setParam("MIPFocus", int(getattr(args, "mp_focus", 0)))
        m._inc_time = float(getattr(args, "mp_inc_time", 0.0))

        m._pred_out = {"obj": [], "feas": []}

        T = self.T
        rho = float(self.inst.get("rho", 1.0))
        lam_da = np.asarray(self.inst["lambda_da"], dtype=float).reshape(-1)

        lb, ub, width = self._x_bounds()
        p_DA = m.addVars(T, name="p_DA", vtype="C")
        for t in range(T):
            p_DA[t].lb = float(lb[t])
            p_DA[t].ub = float(ub[t])
            p_DA[t].obj = float(-rho * lam_da[t])

        x01 = m.addVars(T, name="x01", vtype="C", lb=0.0, ub=1.0)
        for t in range(T):
            m.addConstr(p_DA[t] == float(lb[t]) + float(width[t]) * x01[t], name=f"x_norm_link_{t}")

        m._x = p_DA
        m._x01 = x01

        m._inst_vars = self.init_grb_inst_variables(m)

        x_gp_input_vars = []
        for t in range(T):
            x_gp_input_vars.append([m._x01[t]] + m._inst_vars[t])

        x_embed_var = self.embed_setbased_model(
            m=m,
            gp_input_vars=x_gp_input_vars,
            set_net=self.net.x_embed_net,
            agg_dim=self.net.x_embed_dims[-1],
            post_agg_net=self.net.x_post_agg_net,
            post_agg_dim=self.net.x_post_agg_dims[-1],
            agg_type=self.net.agg_type,
            name="x_embed",
        )
        m._x_embed = x_embed_var

        z = m.addVar(name="z_wc", vtype="C", lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, obj=1.0)
        m._z = z

        self.main_model = m

    def embed_value_network(self, xi_embed, n_iterations, scen_type):
        xi_embed = np.asarray(xi_embed, dtype=float).reshape(-1)
        dim = int(xi_embed.shape[0])

        xi_embed_var = self.main_model.addVars(dim, name=f"xi_embed_{scen_type}_{n_iterations}", lb=-gp.GRB.INFINITY, obj=0.0)
        for i in range(dim):
            xi_embed_var[i].lb = float(xi_embed[i])
            xi_embed_var[i].ub = float(xi_embed[i])

        pred_in = self.main_model._x_embed.select() + xi_embed_var.select()
        pred_out = self.main_model.addVar(name=f"pred_{scen_type}_{n_iterations}", lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, obj=0.0)
        _ = add_predictor_constr(self.main_model, self.net.value_net, pred_in, pred_out, name=f"pred_constr_{scen_type}_{n_iterations}")
        return pred_out

    def add_worst_case_scenario_to_main(self, xi, n_iterations, scen_type):
        xi_embed = self.get_xi_embed(xi)
        pred = self.embed_value_network(xi_embed, n_iterations, scen_type=scen_type)
        self.main_model.addConstr(self.main_model._z >= pred, name=f"z_ge_pred_{scen_type}_{n_iterations}")
        self.main_model._pred_out[scen_type].append(pred)

    def change_worst_case_scen(self, xi_to_add, scen_id_vars, xi_vals, n_iterations):
        return scen_id_vars

    # ----------------------------
    # Adversarial model
    # ----------------------------
    def embed_net_adversarial(self, m):
        x_embed_dim = int(self.net.x_post_agg_dims[-1])
        x_embed_var = m.addVars(x_embed_dim, name="x_embed", vtype="C", lb=-gp.GRB.INFINITY)
        m._x_embed = x_embed_var

        inst_vars = self.init_grb_inst_variables(m)

        xi_var = m.addVars(self.T, name="xi", vtype="C", lb=0.0, ub=1.0)
        m._xi = xi_var

        gp_input_vars = []
        for t in range(self.T):
            gp_input_vars.append([xi_var[t]] + inst_vars[t])

        xi_embed_var = self.embed_setbased_model(
            m=m,
            gp_input_vars=gp_input_vars,
            set_net=self.net.xi_embed_net,
            agg_dim=self.net.xi_embed_dims[-1],
            post_agg_net=self.net.xi_post_agg_net,
            post_agg_dim=self.net.xi_post_agg_dims[-1],
            agg_type=self.net.agg_type,
            name="xi_embed",
        )
        m._xi_embed = xi_embed_var

        pred_in = x_embed_var.select() + xi_embed_var.select()
        pred_out = m.addVar(name="pred", lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, obj=1.0)
        pred_constr = add_predictor_constr(m, self.net.value_net, pred_in, pred_out, name="pred_constr_adv")

        m._pred_out_var = pred_out
        m._pred_out_constr = pred_constr
        return m

    def _get_adv_obj(self, args):
        m = gp.Model("offering_network_adv")
        m.setParam("OutputFlag", int(getattr(args, "verbose", 0)))
        m.setParam("MIPGap", float(getattr(args, "adversarial_gap", 0.01)))
        m.setParam("TimeLimit", float(getattr(args, "adversarial_time", 60.0)))
        m.setParam("MIPFocus", int(getattr(args, "adversarial_focus", 0)))
        m._inc_time = float(getattr(args, "adversarial_inc_time", 0.0))

        m = self.embed_net_adversarial(m)

        Gamma = float(self.inst.get("Gamma", 0.0))
        active = self._active_mask()

        u = m.addVars(self.T, name="u_abs", vtype="C", lb=0.0, ub=1.0)
        m._u = u

        for t in range(self.T):
            if not bool(active[t]):
                m._xi[t].lb = 0.5
                m._xi[t].ub = 0.5
                u[t].lb = 0.0
                u[t].ub = 0.0
                continue

            m.addConstr(u[t] >= 2.0 * m._xi[t] - 1.0, name=f"abs_pos_{t}")
            m.addConstr(u[t] >= -2.0 * m._xi[t] + 1.0, name=f"abs_neg_{t}")

        m.addConstr(gp.quicksum(u[t] for t in range(self.T)) <= Gamma, name="xi_budget_abs")
        m.setObjective(m._pred_out_var, sense=gp.GRB.MAXIMIZE)
        return m

    def initialize_adversarial_model(self, args):
        self.adv_model = {"obj": self._get_adv_obj(args)}

    def set_first_stage_in_adversarial_model(self, x_phys):
        x_embed = self.get_x_embed(x_phys)
        for i, v in enumerate(self.adv_model["obj"]._x_embed.select()):
            v.lb = float(x_embed[i])
            v.ub = float(x_embed[i])

    # ----------------------------
    # random xi sampler
    # ----------------------------
    def _sample_random_xi(self, rng=None, x=None):
        if rng is None:
            rng = np.random.default_rng()

        T = self.T
        Gamma = float(self.inst.get("Gamma", 0.0))
        active = self._active_mask()

        delta = rng.uniform(-1.0, 1.0, size=T)
        denom = np.sum(np.abs(delta[active]))
        target = rng.uniform(0.0, Gamma)
        if denom < 1e-12:
            delta[:] = 0.0
        else:
            delta = delta * (target / denom)
        delta = np.clip(delta, -1.0, 1.0)

        xi = 0.5 * (delta + 1.0)
        xi[~active] = 0.5
        return xi

    def clamp_xi(self, xi):
        xi.clamp_(0, 1)

    def check_xi(self, xi, x=None):
        xi_ = np.asarray(xi.detach().cpu().numpy(), dtype=float).reshape(-1)
        active = self._active_mask()
        Gamma = float(self.inst.get("Gamma", 0.0))
        cons = float(np.sum(np.abs(2.0 * xi_[active] - 1.0)))
        return cons > Gamma + 1e-8
