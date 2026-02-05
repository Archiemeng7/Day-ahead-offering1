# # ro/scripts/dat_scripts/offering_no_network_gen_random_search_table.py
# import argparse
# import hashlib
# import importlib.util
# from pathlib import Path

# import numpy as np

# import ro.params as params
# from ro.utils import factory_get_path


# # ----------------------------
# # Problem registry
# # ----------------------------
# problem_types = {
#     "offering_no_network": ["offering_no_network"],
#     "kp": ["kp"],
# }


# # ----------------------------
# # Samplers
# # ----------------------------
# class ContinuousValueSampler:
#     """Sample uniformly in [lb, ub], optionally with prob_zero."""
#     def __init__(self, lb, ub, prob_zero=0.0, round_digits=6):
#         self.lb = float(lb)
#         self.ub = float(ub)
#         self.prob_zero = float(prob_zero)
#         self.round_digits = int(round_digits)

#     def sample(self):
#         if np.random.rand() < self.prob_zero:
#             return 0.0
#         return float(np.round(np.random.uniform(self.lb, self.ub), self.round_digits))


# class LogUniformSampler:
#     """Sample log-uniform in [lb, ub] (lb>0)."""
#     def __init__(self, lb, ub, prob_zero=0.0, round_digits=8):
#         self.lb = float(lb)
#         self.ub = float(ub)
#         self.prob_zero = float(prob_zero)
#         self.round_digits = int(round_digits)
#         assert self.lb > 0 and self.ub > 0 and self.ub >= self.lb

#     def sample(self):
#         if np.random.rand() < self.prob_zero:
#             return 0.0
#         x = np.exp(np.random.uniform(np.log(self.lb), np.log(self.ub)))
#         return float(np.round(x, self.round_digits))


# class DiscreteSampler:
#     """Sample uniformly from choices."""
#     def __init__(self, choices):
#         self.choices = list(choices)

#     def sample(self):
#         idx = np.random.choice(len(self.choices))
#         return self.choices[idx]


# def _module_exists(modname: str) -> bool:
#     return importlib.util.find_spec(modname) is not None


# def detect_train_module(model_type: str) -> str:
#     """
#     Try to locate an existing training entry module.
#     You can extend this list if your repo uses different names.
#     """
#     candidates = []
#     mt = model_type.lower()

#     # common patterns seen in Neur2RO-style repos
#     if mt == "set_encoder":
#         candidates = [
#             "ro.scripts.train_set_encoder",
#             "ro.scripts.03_train_set_encoder",
#             "ro.scripts.03_train_model",
#             "ro.scripts.train_model",
#         ]
#     else:
#         candidates = [
#             "ro.scripts.03_train_model",
#             "ro.scripts.train_model",
#         ]

#     for m in candidates:
#         if _module_exists(m):
#             return m

#     # fallback (still writes commands; user may adjust)
#     return candidates[0] if candidates else "ro.scripts.03_train_model"


# # ----------------------------
# # Config spaces
# # ----------------------------
# def get_offering_set_encoder_config(problem: str, model_type: str):
#     """
#     Keep hyperparam set conservative (only pass args that are commonly accepted).
#     If your training script supports more architecture args, you can add them here.
#     """
#     # learning rate + regularization usually best as log-uniform
#     config = {
#         "batch_size": DiscreteSampler([32, 64, 128, 256]),
#         "lr": LogUniformSampler(1e-5, 1e-2),
#         "wt_lasso": LogUniformSampler(1e-6, 1e-2, prob_zero=0.30),
#         "wt_ridge": LogUniformSampler(1e-6, 1e-2, prob_zero=0.30),
#         "dropout": ContinuousValueSampler(0.0, 0.5),
#         "optimizer": DiscreteSampler(["Adam", "Adagrad", "RMSprop"]),
#         # "n_epochs": DiscreteSampler([500, 1000, 2000]),
#         "n_epochs": DiscreteSampler([100, 200, 400]),
#         "loss_fn": DiscreteSampler(["MSELoss"]),
#     }
#     return config


# def get_config(problem: str, model_type: str):
#     if "offering_no_network" in problem:
#         if model_type == "set_encoder":
#             return get_offering_set_encoder_config(problem, model_type)
#         else:
#             raise Exception(f"Config not defined for offering_no_network with model_type [{model_type}]")

#     if "kp" in problem:
#         # keep for completeness if you reuse this script
#         raise Exception("Use kp_gen_random_search_table.py for KP.")

#     raise Exception(f"Config not defined for problem [{problem}]")


# def sample_config(problem: str, model_type: str, config: dict, train_module: str):
#     """
#     Build a single training command.
#     NOTE: This only GENERATES commands. It does not run training.
#     """
#     cmd = f"python -m {train_module} --problem {problem} --model_type {model_type}"
#     for param_name, sampler in config.items():
#         val = sampler.sample()
#         if isinstance(val, list):
#             val_str = " ".join(map(str, val))
#             cmd += f" --{param_name} {val_str}"
#         else:
#             cmd += f" --{param_name} {val}"
#     return cmd


# def _infer_problem_dir(problem: str) -> Path:
#     cfg = getattr(params, problem)
#     get_path = factory_get_path(problem)
#     # use the shared convention: test_instances/ directory exists under <data_path>/<problem>/
#     inst_dir = Path(get_path(cfg.data_path, cfg, "test_instances/"))
#     return inst_dir.parent


# def main(args):
#     cmds = []

#     for problem in args.problems:
#         if problem not in problem_types:
#             raise ValueError(f"Unknown problem key: {problem}. Available: {list(problem_types.keys())}")

#         train_module = detect_train_module(args.model_type)
#         cfg = getattr(params, problem)

#         # ensure random_search dir exists
#         prob_dir = _infer_problem_dir(problem)
#         rs_dir = prob_dir / "random_search"
#         rs_dir.mkdir(parents=True, exist_ok=True)

#         for ptypes in problem_types[problem]:
#             config = get_config(ptypes, args.model_type)

#             for i in range(args.n_configs):
#                 # stable seed per (problem, model, i)
#                 tag = f"{ptypes}|{args.model_type}|{args.seed}|{i}"
#                 p_hash = int(hashlib.md5(tag.encode("utf-8")).hexdigest(), 16)
#                 np.random.seed(p_hash % (2**32 - 1))

#                 cmds.append(sample_config(ptypes, args.model_type, config, train_module))

#     # write .dat
#     out_fp = Path(args.file_name)
#     out_fp.parent.mkdir(parents=True, exist_ok=True)

#     with open(out_fp, "w", encoding="utf-8") as f:
#         for k, cmd in enumerate(cmds, start=args.start_idx):
#             f.write(f"{k} {cmd}\n")

#     print(f"Wrote {len(cmds)} commands to: {out_fp}")
#     print("NOTE: You must RUN these commands to generate:")
#     print("  data/offering_no_network/random_search/<model_type>_tr_res__*.pkl and <model_type>__*.pt")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Generate random-search command table for offering_no_network.")
#     parser.add_argument("--problems", type=str, nargs="+", default=["offering_no_network"])
#     parser.add_argument("--model_type", type=str, default="set_encoder")
#     parser.add_argument("--n_configs", type=int, default=50)
#     parser.add_argument("--file_name", type=str, default="data/offering_no_network/random_search/table_set_encoder.dat")
#     parser.add_argument("--start_idx", type=int, default=1)
#     parser.add_argument("--seed", type=int, default=1234)
#     args = parser.parse_args()
#     main(args)







import argparse
import hashlib
import numpy as np


# 只新增 offering_no_network，不改 KP 的脚本/功能
problem_types = {
    "offering_no_network": ["offering_no_network"],
}


class ContinuousValueSampler(object):
    """Uniform sampler in [lb, ub], with optional prob of sampling 0."""
    def __init__(self, lb, ub, prob_zero=0.0, round_digits=8):
        self.lb = lb
        self.ub = ub
        self.prob_zero = prob_zero
        self.round_digits = round_digits

    def sample(self):
        if np.random.rand() < self.prob_zero:
            return 0.0
        return float(np.round(np.random.uniform(self.lb, self.ub), self.round_digits))


class DiscreteSampler(object):
    """Uniform sampler from a discrete list."""
    def __init__(self, choices):
        self.choices = list(choices)

    def sample(self):
        idx = np.random.choice(len(self.choices))
        return self.choices[idx]


def get_offering_set_encoder_config(problem, model_type):
    """
    offering_no_network 的 random search 参数空间：
    只采样“通用训练参数”，避免依赖你本地 set_encoder 的额外网络结构参数名（不同版本可能不一致）。
    """
    LR_LB, LR_UB = 1e-5, 3e-3
    L1_LB, L1_UB = 1e-8, 1e-2
    L2_LB, L2_UB = 1e-8, 1e-2

    config = {
        # 通用训练超参（通常 03_train_model.py 都支持）
        "batch_size": DiscreteSampler([32, 64, 128, 256]),
        "lr": ContinuousValueSampler(LR_LB, LR_UB),
        "wt_lasso": ContinuousValueSampler(L1_LB, L1_UB, prob_zero=0.25),
        "wt_ridge": ContinuousValueSampler(L2_LB, L2_UB, prob_zero=0.25),
        "dropout": ContinuousValueSampler(0.0, 0.5),
        "optimizer": DiscreteSampler(["Adam", "Adagrad", "RMSprop"]),
        "loss_fn": DiscreteSampler(["MSELoss"]),
        "n_epochs": DiscreteSampler([2000]),
        # 可按需加入 early_stop / scheduler 等（前提是 03_train_model.py 解析了这些参数）
    }
    return config


def get_config(problem, model_type):
    if "offering_no_network" in problem:
        if model_type == "set_encoder":
            return get_offering_set_encoder_config(problem, model_type)
        else:
            raise Exception(f"Config not defined for model_type [{model_type}] under offering_no_network.")
    raise Exception(f"Config not defined for problem [{problem}].")


def sample_config_cmd(problem, model_type, config):
    """
    关键：调用 03_train_model，让训练结果落到 data/<problem>/random_search 下，
    从而生成 set_encoder_tr_res__*.pkl / set_encoder__*.pt 供 04_get_best_model 使用。
    """
    cmd = f"python -m ro.scripts.03_train_model --problem {problem} --model_type {model_type}"
    for param_name, sampler in config.items():
        val = sampler.sample()
        if isinstance(val, list):
            cmd += " --{} {}".format(param_name, " ".join(map(str, val)))
        else:
            cmd += f" --{param_name} {val}"
    return cmd


def main(args):
    cmds = []

    for prob_key in args.problems:
        if prob_key not in problem_types:
            raise ValueError(f"Unsupported problem key: {prob_key}. Available: {list(problem_types.keys())}")

        for problem in problem_types[prob_key]:
            for model_type in args.model_type:
                config = get_config(problem, model_type)

                # 稳定随机种子：与 problem + i 绑定
                p_hash = int(hashlib.md5(problem.encode("utf-8")).hexdigest(), 16) % (2**32 - 1)

                for i in range(args.n_configs):
                    np.random.seed((args.seed + i + p_hash) % (2**32 - 1))
                    cmds.append(sample_config_cmd(problem, model_type, config))

    # 写出 dat 文件：每行 "index command..."
    with open(args.file_name, "w") as f:
        for i, cmd in enumerate(cmds):
            f.write(f"{i + args.start_idx} {cmd}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random-search command table for offering_no_network.")
    parser.add_argument("--problems", type=str, nargs="+", default=["offering_no_network"])
    parser.add_argument("--model_type", type=str, nargs="+", default=["set_encoder"])
    parser.add_argument("--n_configs", type=int, default=50)
    parser.add_argument("--file_name", type=str, default="offering_no_network_random_search.dat")
    parser.add_argument("--start_idx", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    main(args)






