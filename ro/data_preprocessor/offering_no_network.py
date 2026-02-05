# ro/data_preprocessor/offering_no_network.py

import time
import numpy as np

import torch
from torch.utils.data import TensorDataset

from .data_preprocessor import DataPreprocessor


class OfferingNoNetworkDataPreprocessor(DataPreprocessor):
    """
    Set-encoder dataset for offering_no_network.

    Treat each hour t as one “item” in the set (T=24).
    We build two set inputs:
      - x_features: per-hour features with first entry = normalized p_DA(t)
      - xi_features: per-hour features with first entry = xi(t) in [0,1]

    Label: sample['ss_obj'] (same as KP)
    """

    def __init__(self, cfg, model_type, predict_feas, device):
        super(OfferingNoNetworkDataPreprocessor, self).__init__(cfg, model_type, predict_feas, device)
        self.pad_size = int(getattr(self.cfg, "T", 24))

    # ----------------------------
    # internal helpers
    # ----------------------------
    def _get_x_bounds(self, inst):
        """
        Compute per-hour bounds for p_DA(t) consistent with DM sampling.
        lb(t) = -(load(t) + P_ess_max) - margin
        ub(t) =  (pv_max(t) + P_ess_max - load(t)) + margin
        margin uses cfg.p_da_bound_margin_pu if provided, else cfg.x_margin (as absolute pu).
        """
        p_load = np.asarray(inst["p_load"], dtype=float).reshape(-1)
        pv_max = np.asarray(inst["p_pv_max"], dtype=float).reshape(-1)
        Pmax = float(inst["P_ess_max"])

        lb = -(p_load + Pmax)
        ub = (pv_max + Pmax - p_load)

        m = float(getattr(self.cfg, "p_da_bound_margin_pu", getattr(self.cfg, "x_margin", 0.0)))
        lb = lb - m
        ub = ub + m

        width = np.maximum(ub - lb, 1e-8)
        return lb, ub, width

    def _normalize_x(self, x, inst):
        """
        Normalize x(t)=p_DA(t) to [0,1] using per-hour bounds.
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        lb, ub, width = self._get_x_bounds(inst)
        x01 = (x - lb) / width
        return np.clip(x01, 0.0, 1.0)

    def _build_inst_feats(self, inst):
        """
        Per-hour features (excluding the first entry which is x01 or xi):
          [lambda_da(t), lambda_rt(t), p_load(t), pv_min(t), pv_max(t),
           P_ess_max, E_ess_max, eta, soc0, ESS_cost, PV_cost, Gamma, rho]
        """
        T = int(inst["T"])
        lam_da = np.asarray(inst["lambda_da"], dtype=float).reshape(-1)
        lam_rt = np.asarray(inst["lambda_rt"], dtype=float).reshape(-1)
        p_load = np.asarray(inst["p_load"], dtype=float).reshape(-1)
        pv_min = np.asarray(inst["p_pv_min"], dtype=float).reshape(-1)
        pv_max = np.asarray(inst["p_pv_max"], dtype=float).reshape(-1)

        Pmax = float(inst["P_ess_max"])
        Emax = float(inst["E_ess_max"])
        eta = float(inst["eta"])
        soc0 = float(inst["soc0"])
        ess_c = float(inst.get("ESS_cost", 0.0))
        pv_c = float(inst.get("PV_cost", 0.0))
        Gamma = float(inst.get("Gamma", 0.0))
        rho = float(inst.get("rho", 1.0))

        inst_feats = []
        for t in range(T):
            inst_feats.append([
                lam_da[t],
                lam_rt[t],
                p_load[t],
                pv_min[t],
                pv_max[t],
                Pmax,
                Emax,
                eta,
                soc0,
                ess_c,
                pv_c,
                Gamma,
                rho,
            ])
        return np.asarray(inst_feats, dtype=float)

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

            # normalize decision x to [0,1]
            x01 = self._normalize_x(x, inst)
            xi01 = np.clip(np.asarray(xi, dtype=float).reshape(-1), 0.0, 1.0)

            inst_feats = self._build_inst_feats(inst)  # (T, feat_inst_dim)

            # build (T, 1+feat_inst_dim)
            x_feats = np.concatenate([x01.reshape(T, 1), inst_feats], axis=1)
            xi_feats = np.concatenate([xi01.reshape(T, 1), inst_feats], axis=1)

            # scale features (min-max)
            if self.feat_scaler is not None:
                fmin, fmax = self.feat_scaler
                denom = (fmax - fmin)
                denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
                x_feats = (x_feats - fmin) / denom
                xi_feats = (xi_feats - fmin) / denom

            # pad to pad_size (usually 24)
            x_feats = self.pad_features(np.asarray(x_feats, dtype=float), pad_dim=self.pad_size)
            xi_feats = self.pad_features(np.asarray(xi_feats, dtype=float), pad_dim=self.pad_size)

            label = float(sample["ss_obj"])

            # scale label (min-max over T)
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

        # keep the same signature pattern as KP: (x_set, xi_set, n_x, n_x, y)
        tensor_dataset = TensorDataset(x_features, xi_features, T_features, T_features, labels)
        return tensor_dataset

    def init_label_scaler(self, dataset):
        """
        Min/max scaling for labels. Here T is fixed (24), so store one entry keyed by T.
        """
        Ts = [int(s["instance"]["T"]) for s in dataset]
        if len(Ts) == 0:
            self.label_scaler = {}
            return

        T0 = int(Ts[0])
        ys = np.array([float(s["ss_obj"]) for s in dataset], dtype=float)
        self.label_scaler = {T0: (float(np.min(ys)), float(np.max(ys)))}

    def init_feature_scaler(self, dataset):
        """
        Compute global min/max for feature columns (using x_features construction).
        Force column 0 min/max to (0,1) because both x01 and xi are already in [0,1].
        """
        all_rows = []

        for sample in dataset:
            inst = sample["instance"]
            T = int(inst["T"])

            x01 = self._normalize_x(sample["x"], inst)
            inst_feats = self._build_inst_feats(inst)
            x_feats = np.concatenate([x01.reshape(T, 1), inst_feats], axis=1)
            all_rows.append(np.asarray(x_feats, dtype=float))

        X = np.concatenate(all_rows, axis=0)  # (N*T, feat_dim)
        feat_min = np.min(X, axis=0).reshape(-1)
        feat_max = np.max(X, axis=0).reshape(-1)

        # enforce normalized decision/xi column
        feat_min[0] = 0.0
        feat_max[0] = 1.0

        self.feat_scaler = (feat_min, feat_max)
