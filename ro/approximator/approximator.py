# /content/drive/MyDrive/Neur2RO/ro/approximator/approximator.py
"""
Base approximator class (Neur2RO).

Key change (to fix offering_network evaluation):
- do_forward_pass is NO LONGER an abstractmethod.
- A robust default do_forward_pass implementation is provided.
  This prevents subclasses (e.g., OfferingNetworkApproximator) from being
  treated as abstract classes if they do not override do_forward_pass.

This file also provides shared utilities used by multiple approximators:
- to_tensor
- embed_setbased_model (build Gurobi+gurobi-ml constraints for set-encoder blocks)
"""

from __future__ import annotations

import abc
from typing import Any, List, Sequence, Optional, Tuple

import numpy as np
import torch

import gurobipy as gp
from gurobi_ml import add_predictor_constr


class Approximator(abc.ABC):
    """
    Base class for Neur2RO approximators.

    Subclasses typically implement:
      - get_instance()
      - initialize_main_model()
      - initialize_adversarial_model()
      - (optionally) do_forward_pass()

    IMPORTANT:
      do_forward_pass is intentionally NOT abstract here. A default implementation is provided,
      which enables instantiation even if a subclass does not define do_forward_pass.
    """

    def __init__(self, *args, **kwargs):
        # Most subclasses fully manage init; keep base init lightweight.
        pass

    # ----------------------------
    # Torch helpers
    # ----------------------------
    @staticmethod
    def to_tensor(x: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Convert numpy/list/scalar to a CPU torch tensor."""
        if isinstance(x, torch.Tensor):
            return x.detach().to(dtype=dtype, device="cpu")
        return torch.tensor(x, dtype=dtype, device="cpu")

    # ----------------------------
    # Default forward pass (NOT abstract)
    # ----------------------------
    def _inverse_scale_label(self, y_scaled: float) -> float:
        """
        Try to invert label scaling if self.label_scaler exists.

        Supported label_scaler formats:
          1) None -> return y_scaled
          2) (ymin, ymax) tuple/list/np.ndarray -> y = y_scaled*(ymax-ymin)+ymin
          3) sklearn-like scaler with inverse_transform
          4) dict keyed by size (e.g., T) whose values are one of the above
        """
        ls = getattr(self, "label_scaler", None)

        # dict keyed by problem size
        if isinstance(ls, dict):
            key = getattr(self, "T", None)
            if key in ls:
                ls = ls[key]
            elif len(ls) > 0:
                ls = next(iter(ls.values()))
            else:
                ls = None

        if ls is None:
            return float(y_scaled)

        # (ymin, ymax)
        if isinstance(ls, (tuple, list)) and len(ls) == 2:
            ymin, ymax = ls
            ymin = float(np.asarray(ymin).reshape(-1)[0])
            ymax = float(np.asarray(ymax).reshape(-1)[0])
            return float(y_scaled) * (ymax - ymin) + ymin

        # numpy array with shape (2,) possibly
        if isinstance(ls, np.ndarray) and ls.size == 2:
            ymin = float(ls.reshape(-1)[0])
            ymax = float(ls.reshape(-1)[1])
            return float(y_scaled) * (ymax - ymin) + ymin

        # sklearn-like
        if hasattr(ls, "inverse_transform"):
            try:
                yy = ls.inverse_transform(np.array([[float(y_scaled)]]))
                return float(np.asarray(yy).reshape(-1)[0])
            except Exception:
                return float(y_scaled)

        return float(y_scaled)

    def do_forward_pass(self, x: np.ndarray, xi: np.ndarray) -> float:
        """
        Default NN forward pass for objective prediction.
        Subclasses may override.

        Strategy:
          - If self.x_features/self.xi_features exist (torch tensors shaped [1,T,F]),
            clone them and overwrite feature column 0 with normalized x / xi.
          - Else, fall back to using x and xi as (T,1) features.

        Normalization:
          - If self has method _normalize_x, use it to convert physical x -> [0,1] (x01).
          - Otherwise, assume x is already in [0,1] and clip.
        """
        if not hasattr(self, "net"):
            raise RuntimeError("Approximator.do_forward_pass: self.net not found.")

        x = np.asarray(x, dtype=float).reshape(-1)
        xi = np.asarray(xi, dtype=float).reshape(-1)

        # normalize x if subclass provides _normalize_x()
        if hasattr(self, "_normalize_x") and callable(getattr(self, "_normalize_x")):
            x01 = np.asarray(self._normalize_x(x), dtype=float).reshape(-1)
        else:
            x01 = np.clip(x, 0.0, 1.0)

        xi01 = np.clip(xi, 0.0, 1.0)

        # Case A: we have cached scaled feature tensors
        if hasattr(self, "x_features") and hasattr(self, "xi_features"):
            xt = getattr(self, "x_features").clone()
            xit = getattr(self, "xi_features").clone()

            # expected shape: (1, T, F)
            if xt.ndim != 3 or xit.ndim != 3:
                raise RuntimeError("x_features/xi_features must be 3D tensors shaped (1,T,F).")

            T = xt.shape[1]
            if x01.shape[0] != T or xi01.shape[0] != T:
                raise ValueError(f"Length mismatch: x={x01.shape[0]}, xi={xi01.shape[0]}, T={T}")

            xt[:, :, 0] = self.to_tensor(x01).reshape(1, T)
            xit[:, :, 0] = self.to_tensor(xi01).reshape(1, T)

        # Case B: fallback simple features
        else:
            xt = self.to_tensor(x01.reshape(1, -1, 1))
            xit = self.to_tensor(xi01.reshape(1, -1, 1))

        # forward
        with torch.no_grad():
            y = self.net(xt, xit)

        y = float(np.asarray(y.detach().cpu().numpy()).reshape(-1)[0])
        return self._inverse_scale_label(y)

    # ----------------------------
    # Gurobi embedding helper
    # ----------------------------
    @staticmethod
    def _as_list_vars(vs: Any) -> List[gp.Var]:
        """Convert tupledict/select/list into a flat python list of gp.Var."""
        if isinstance(vs, list):
            return vs
        if hasattr(vs, "select"):
            return list(vs.select())
        if isinstance(vs, gp.tupledict):
            return [vs[k] for k in sorted(vs.keys())]
        return list(vs)

    def embed_setbased_model(
        self,
        m: gp.Model,
        gp_input_vars: Sequence[Sequence[gp.Var]],
        set_net: Any,
        agg_dim: int,
        post_agg_net: Any,
        post_agg_dim: int,
        agg_type: str = "sum",
        name: str = "embed",
    ) -> gp.tupledict:
        """
        Build Gurobi constraints for:
          element-wise embedding -> aggregation -> post-aggregation net

        gp_input_vars:
          list over elements i=0..N-1; each item is a list of input vars for that element.

        Returns:
          tupledict of post-aggregation output vars of length post_agg_dim.
        """
        N = len(gp_input_vars)
        if N <= 0:
            raise ValueError("embed_setbased_model: gp_input_vars is empty.")
        if agg_dim <= 0 or post_agg_dim <= 0:
            raise ValueError("embed_setbased_model: agg_dim/post_agg_dim must be positive.")
        agg_type = str(agg_type).lower().strip()
        if agg_type not in ("sum", "mean"):
            raise ValueError(f"embed_setbased_model: unsupported agg_type={agg_type}")

        # element embedding outputs: emb[i,k]
        emb = m.addVars(N, agg_dim, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"{name}_elem")
        for i in range(N):
            in_vars = list(gp_input_vars[i])
            out_vars = [emb[i, k] for k in range(agg_dim)]
            _ = add_predictor_constr(m, set_net, in_vars, out_vars, name=f"{name}_set_{i}")

        # aggregate: agg[k] = sum_i emb[i,k] (or mean)
        agg = m.addVars(agg_dim, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"{name}_agg")
        for k in range(agg_dim):
            if agg_type == "sum":
                m.addConstr(agg[k] == gp.quicksum(emb[i, k] for i in range(N)), name=f"{name}_agg_sum_{k}")
            else:
                m.addConstr(agg[k] == (1.0 / float(N)) * gp.quicksum(emb[i, k] for i in range(N)), name=f"{name}_agg_mean_{k}")

        # post-agg net: post_out = post_agg_net(agg)
        post_out = m.addVars(post_agg_dim, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"{name}_post")
        _ = add_predictor_constr(m, post_agg_net, [agg[k] for k in range(agg_dim)], [post_out[j] for j in range(post_agg_dim)], name=f"{name}_post_net")

        return post_out

    # ----------------------------
    # Required subclass hooks
    # ----------------------------
    @abc.abstractmethod
    def get_instance(self, inst_params: dict) -> dict:
        """Return an instance dict."""
        raise NotImplementedError

    @abc.abstractmethod
    def initialize_main_model(self, args) -> None:
        """Build/attach self.main_model."""
        raise NotImplementedError

    @abc.abstractmethod
    def initialize_adversarial_model(self, args) -> None:
        """Build/attach self.adv_model."""
        raise NotImplementedError






# from abc import ABC, abstractmethod

# import torch

# import gurobipy as gp
# from gurobi_ml import add_predictor_constr

# class Approximator(ABC):

#     @abstractmethod
#     def initialize_main_model(self):
#         """ Gets the gurobi model for the main problem. """
#         pass

#     @abstractmethod
#     def initialize_adversarial_model(self):
#         """ Gets the gurobi model(s) for the adversarial problem. """
#         pass

#     @abstractmethod
#     def initialize_nn(self, net_):
#         pass

#     @abstractmethod
#     def init_grb_inst_variables(self, m):
#         """ Initialize gurobi variables for problem features (i.e., input to NN).  """
#         pass

#     @abstractmethod
#     def add_worst_case_scenario_to_main(self, xi, n_iterations, scen_type):
#         """ Adds worst-case scenario(s) to main problem. """
#         pass

#     @abstractmethod
#     def change_worst_case_scen(self, xi_to_add, scen_id_vars, xi_vals, n_iterations):
#         """ Changes worst-case scenario constraints in main problem. """
#         pass

#     @abstractmethod
#     def get_x_embed(self, x):
#         """ Gets scenario embedding for a particular x input. """
#         pass 

#     @abstractmethod
#     def get_xi_embed(self, xi):
#         """ Gets scenario embedding for a particular scenario input. """
#         pass

#     @abstractmethod
#     def get_instance(self, cfg, inst_params):
#         """  Gets instances based on parameters.  """
#         pass

#     @abstractmethod
#     def do_forward_pass(x, xi, scale=True):
#         pass

#     @abstractmethod
#     def get_inst_nn_features(self):
#         """ Gets features from x (input later), xi (scenario sampling), and inst. """
#         pass

#     @abstractmethod
#     def embed_net_adversarial(self, m):
#         """ Embeds adversarial network.  """
#         pass

#     @abstractmethod
#     def embed_value_network(self, xi_embed, n_iterations, scen_type):
#         """ Embeds value network in main problem.  """
#         pass

#     def to_tensor(self, x):
#         """ Converts numpy/list to a tensor. """
#         return torch.Tensor(x).float()

#     def embed_setbased_model(self, m, gp_input_vars, set_net, agg_dim, post_agg_net, post_agg_dim, agg_type, name):
#         """ Embeds set-based predictive models. """
#         # add predictive constraints for each element of the set
#         n_items_in_set = len(gp_input_vars)
#         set_outputs = []
#         for gp_input_var in gp_input_vars:
#             pre_sum_vars = m.addVars(agg_dim, vtype="C", lb=-gp.GRB.INFINITY)
#             pred_constr = add_predictor_constr(m, set_net, gp_input_var, pre_sum_vars)
#             set_outputs.append(pre_sum_vars)

#         # initaialize gurobi variables for post-summation
#         post_agg_vars = m.addVars(agg_dim, vtype="C", lb=-gp.GRB.INFINITY)

#         # add constraints to set: "post-agg variables == AGG(pre-agg variables)""
#         for i in range(agg_dim):
#             agg_sum = 0
#             for j in range(n_items_in_set):
#                 # print(set_outputs[j])
#                 # agg_sum += set_outputs[j].values()[i]
#                 agg_sum += set_outputs[j][i]
#             if agg_type == "sum":
#                 k = m.addConstr(agg_sum == post_agg_vars[i])
#             elif agg_type == "mean":
#                 k = m.addConstr(agg_sum / n_items_in_set == post_agg_vars[i])

#         # post-agg net variable
#         gp_embed_vars = m.addVars(post_agg_dim, vtype="C", lb=-gp.GRB.INFINITY, name=name)
#         pred_constr = add_predictor_constr(m, post_agg_net, post_agg_vars, gp_embed_vars)

#         return gp_embed_vars
