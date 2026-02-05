# /content/drive/MyDrive/Neur2RO/ro/data_preprocessor/offering_network.py
import numpy as np
import torch
from torch.utils.data import TensorDataset

from .data_preprocessor import DataPreprocessor


class OfferingNetworkDataPreprocessor(DataPreprocessor):
    """
    Set-encoder dataset for offering_network.

    Items in the set: hours t=1..24 (pad_size=24).
    Feature columns:
      col0: normalized x(t) or xi(t) in [0,1]
      remaining: per-hour instance summary features + a few constant network summary stats
    """

    def __init__(self, cfg, model_type, predict_feas, device):
        super(OfferingNetworkDataPreprocessor, self).__init__(cfg, model_type, predict_feas, device)
        self.pad_size = int(getattr(self.cfg, "T", 24))

    def _x_bounds(self, inst):
        # same logic as DM: compute using aggregate feasible net injection range
        T = int(inst["T"])
        p_load = np.asarray(inst["p_load_bus"], dtype=float)  # (nb,T)
        load_sum = np.sum(p_load, axis=0)

        enable_fload = bool(inst.get("enable_fload", False))
        if enable_fload:
            fbase_sum = np.sum(np.asarray(inst["fload_base"], dtype=float), axis=0)
            shed_sum = np.sum(np.asarray(inst["fload_shed_max"], dtype=float), axis=0)
        else:
            fbase_sum = np.zeros(T)
            shed_sum = np.zeros(T)

        pv_max = np.asarray(inst["p_pv_max"], dtype=float)  # (n_der,T)
        pv_sum = np.sum(pv_max, axis=0)

        Pmax = np.asarray(inst["P_ess_max"], dtype=float).reshape(-1)
        Psum = float(np.sum(Pmax))

        ub = -load_sum - fbase_sum + shed_sum + pv_sum + Psum
        lb = -load_sum - fbase_sum - Psum

        m = float(getattr(self.cfg, "p_da_bound_margin_pu", getattr(self.cfg, "x_margin", 0.0)))
        lb = lb - m
        ub = ub + m
        width = np.maximum(ub - lb, 1e-8)
        return lb, ub, width

    def _normalize_x(self, x, inst):
        x = np.asarray(x, dtype=float).reshape(-1)
        lb, _, width = self._x_bounds(inst)
        x01 = (x - lb) / width
        return np.clip(x01, 0.0, 1.0)

    def _build_inst_feats(self, inst):
        """
        Per-hour features (excluding col0):
          [lambda_da, lambda_rt,
           sum_p_load, sum_q_load,
           sum_pv_min, sum_pv_max,
           sum_fload_base, sum_fload_shed_max,
           Psum, Esum, eta, soc0_mean,
           ESS_cost, PV_cost, Gamma, rho,
           rd_abs_sum_max, rd_abs_sum_mean, xd_abs_sum_max, xd_abs_sum_mean]
        """
        T = int(inst["T"])
        lam_da = np.asarray(inst["lambda_da"], dtype=float).reshape(-1)
        lam_rt = np.asarray(inst["lambda_rt"], dtype=float).reshape(-1)

        p_load = np.asarray(inst["p_load_bus"], dtype=float)
        q_load = np.asarray(inst["q_load_bus"], dtype=float)
        sum_p_load = np.sum(p_load, axis=0)
        sum_q_load = np.sum(q_load, axis=0)

        pv_min = np.asarray(inst["p_pv_min"], dtype=float)
        pv_max = np.asarray(inst["p_pv_max"], dtype=float)
        sum_pv_min = np.sum(pv_min, axis=0)
        sum_pv_max = np.sum(pv_max, axis=0)

        enable_fload = bool(inst.get("enable_fload", False))
        if enable_fload:
            fbase = np.asarray(inst["fload_base"], dtype=float)
            shed_max = np.asarray(inst["fload_shed_max"], dtype=float)
            sum_fbase = np.sum(fbase, axis=0)
            sum_shed = np.sum(shed_max, axis=0)
        else:
            sum_fbase = np.zeros(T)
            sum_shed = np.zeros(T)

        Psum = float(np.sum(np.asarray(inst["P_ess_max"], dtype=float).reshape(-1)))
        Esum = float(np.sum(np.asarray(inst["E_ess_max"], dtype=float).reshape(-1)))
        eta = float(inst["eta"])
        soc0 = np.asarray(inst["soc0"], dtype=float).reshape(-1)
        soc0_mean = float(np.mean(soc0)) if soc0.size > 0 else 0.0

        ESS_cost = float(inst.get("ESS_cost", 0.0))
        PV_cost = float(inst.get("PV_cost", 0.0))
        Gamma = float(inst.get("Gamma", 0.0))
        rho = float(inst.get("rho", 1.0))

        rd_abs_sum_max = float(inst.get("rd_abs_sum_max", 0.0))
        rd_abs_sum_mean = float(inst.get("rd_abs_sum_mean", 0.0))
        xd_abs_sum_max = float(inst.get("xd_abs_sum_max", 0.0))
        xd_abs_sum_mean = float(inst.get("xd_abs_sum_mean", 0.0))

        feats = []
        for t in range(T):
            feats.append([
                lam_da[t],
                lam_rt[t],
                sum_p_load[t],
                sum_q_load[t],
                sum_pv_min[t],
                sum_pv_max[t],
                sum_fbase[t],
                sum_shed[t],
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
        return np.asarray(feats, dtype=float)

    # ----------------------------
    # required API
    # ----------------------------
    def get_set_encoder_dataset(self, dataset):
        x_features = []
        xi_features = []
        T_features = []
        labels = []

        for sample in dataset:
            x = sample["x"]
            xi = sample["xi"]
            inst = sample["instance"]
            T = int(inst["T"])

            x01 = self._normalize_x(x, inst)
            xi01 = np.clip(np.asarray(xi, dtype=float).reshape(-1), 0.0, 1.0)

            inst_feats = self._build_inst_feats(inst)  # (T, feat_inst_dim)

            x_feats = np.concatenate([x01.reshape(T, 1), inst_feats], axis=1)
            xi_feats = np.concatenate([xi01.reshape(T, 1), inst_feats], axis=1)

            if self.feat_scaler is not None:
                fmin, fmax = self.feat_scaler
                denom = (fmax - fmin)
                denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
                x_feats = (x_feats - fmin) / denom
                xi_feats = (xi_feats - fmin) / denom

            x_feats = self.pad_features(np.asarray(x_feats, dtype=float), pad_dim=self.pad_size)
            xi_feats = self.pad_features(np.asarray(xi_feats, dtype=float), pad_dim=self.pad_size)

            label = float(sample["ss_obj"])
            if self.label_scaler is not None:
                min_y, max_y = self.label_scaler.get(T, (None, None))
                if min_y is not None and max_y is not None and abs(max_y - min_y) > 1e-12:
                    label = (label - min_y) / (max_y - min_y)
                else:
                    label = 0.0

            x_features.append(x_feats)
            xi_features.append(xi_feats)
            T_features.append(T)
            labels.append(label)

        x_features = np.asarray(x_features, dtype=float)
        xi_features = np.asarray(xi_features, dtype=float)
        T_features = np.asarray(T_features, dtype=float)
        labels = np.asarray(labels, dtype=float)

        x_features = self.to_tensor(x_features).to(self.device)
        xi_features = self.to_tensor(xi_features).to(self.device)
        T_features = self.to_tensor(T_features).to(self.device)
        labels = self.to_tensor(labels).to(self.device)

        return TensorDataset(x_features, xi_features, T_features, T_features, labels)

    def init_label_scaler(self, dataset):
        Ts = [int(s["instance"]["T"]) for s in dataset]
        if len(Ts) == 0:
            self.label_scaler = {}
            return
        T0 = int(Ts[0])
        ys = np.array([float(s["ss_obj"]) for s in dataset], dtype=float)
        self.label_scaler = {T0: (float(np.min(ys)), float(np.max(ys)))}

    def init_feature_scaler(self, dataset):
        all_rows = []
        for sample in dataset:
            inst = sample["instance"]
            T = int(inst["T"])
            x01 = self._normalize_x(sample["x"], inst)
            inst_feats = self._build_inst_feats(inst)
            x_feats = np.concatenate([x01.reshape(T, 1), inst_feats], axis=1)
            all_rows.append(np.asarray(x_feats, dtype=float))

        X = np.concatenate(all_rows, axis=0)
        feat_min = np.min(X, axis=0).reshape(-1)
        feat_max = np.max(X, axis=0).reshape(-1)

        feat_min[0] = 0.0
        feat_max[0] = 1.0
        self.feat_scaler = (feat_min, feat_max)
