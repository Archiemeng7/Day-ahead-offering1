# # ro/utils/offering_no_network.py
# import os

# def get_path(data_path, cfg, which, suffix=""):
#     """
#     Compatible with scripts/00_init_directories.py which calls:
#         get_path(cfg.data_path, cfg, "", suffix="")
#     and with DataManager which calls:
#         get_path(cfg.data_path, cfg, "problem"/"ml_data"/"test_instances/")
#     """

#     # strip leading "./" so split("/") in scripts works as expected
#     if isinstance(data_path, str) and data_path.startswith("./"):
#         data_path = data_path[2:]
#     data_path = data_path.rstrip("/")

#     base = os.path.join(data_path, "offering_no_network")
#     os.makedirs(base, exist_ok=True)

#     # when which is empty, return a 3-segment path "data/offering_no_network/problem.pkl"
#     if which in ("", "problem"):
#         return os.path.join(base, "problem.pkl")

#     if which == "ml_data":
#         # suffix may already include ".pkl"
#         if suffix:
#             if suffix.endswith(".pkl"):
#                 return os.path.join(base, f"ml_data{suffix}")
#             return os.path.join(base, f"ml_data{suffix}.pkl")
#         return os.path.join(base, "ml_data.pkl")

#     if which == "test_instances/":
#         inst_dir = os.path.join(base, "test_instances")
#         os.makedirs(inst_dir, exist_ok=True)
#         return inst_dir + os.sep

#     # directory init scripts may pass other tokens; be permissive
#     return os.path.join(base, "problem.pkl")




# # ro/utils/offering_no_network.py
# import os


# def _num_token(v):
#     """
#     Convert numbers to filename-safe tokens:
#       12.0 -> "12"
#       0.95 -> "0p95"
#       -1.2 -> "m1p2"
#     """
#     try:
#         fv = float(v)
#     except Exception:
#         return str(v)

#     if abs(fv - round(fv)) <= 1e-12:
#         return str(int(round(fv)))

#     s = f"{fv:.12g}"  # compact, stable
#     s = s.replace("-", "m").replace(".", "p")
#     return s


# def _cfg_tag(cfg):
#     """
#     Make a deterministic tag similar in spirit to cb/kp naming,
#     so later training code can reliably locate the right files.
#     """
#     T = getattr(cfg, "T", 24)
#     S = getattr(cfg, "n_scenarios", getattr(cfg, "scenarios", "S"))
#     Gamma = getattr(cfg, "Gamma", "G")
#     rt = getattr(cfg, "lambda_rt_value", getattr(cfg, "rt_price_const", "rt"))

#     nsi = getattr(cfg, "n_samples_inst", "nsi")
#     nsf = getattr(cfg, "n_samples_fs", "nsf")
#     nsu = getattr(cfg, "n_samples_per_fs", "nsu")
#     sd = getattr(cfg, "seed", "sd")

#     ocs = getattr(cfg, "offer_curve_sampling", "ocs")

#     tag = (
#         f"T{_num_token(T)}"
#         f"_S{_num_token(S)}"
#         f"_G{_num_token(Gamma)}"
#         f"_rt{_num_token(rt)}"
#         f"_ocs-{str(ocs)}"
#         f"_nsi{_num_token(nsi)}"
#         f"_nsf{_num_token(nsf)}"
#         f"_nsu{_num_token(nsu)}"
#         f"_sd{_num_token(sd)}"
#     )
#     return tag


# def get_path(data_path, cfg, which, suffix=""):
#     """
#     Compatible with:
#       - scripts/00_init_directories.py: get_path(cfg.data_path, cfg, "", suffix="")
#       - DataManager: get_path(cfg.data_path, cfg, "problem"/"ml_data"/"test_instances/")
#     Produces config-tagged filenames like:
#       data/offering_no_network/problem_T24_S25_G12_rt800_..._sd7.pkl
#     """
#     # keep relative "data/..." so scripts' split("/") logic works
#     if isinstance(data_path, str) and data_path.startswith("./"):
#         data_path = data_path[2:]
#     data_path = str(data_path).rstrip("/")

#     base = os.path.join(data_path, "offering_no_network")
#     os.makedirs(base, exist_ok=True)

#     tag = _cfg_tag(cfg)

#     # 00_init_directories passes which="" just to infer "data/<problem>/..."
#     if which in ("", "problem"):
#         return os.path.join(base, f"problem_{tag}.pkl")

#     if which == "ml_data":
#         # suffix may be "", "_xxx", "_xxx.pkl", "xxx.pkl" etc.
#         suf = str(suffix or "")
#         if suf and not suf.startswith("_"):
#             suf = "_" + suf
#         if suf and not suf.endswith(".pkl"):
#             suf = suf + ".pkl"
#         return os.path.join(base, f"ml_data_{tag}{suf}" if suf else f"ml_data_{tag}.pkl")

#     if which == "test_instances/":
#         inst_dir = os.path.join(base, "test_instances")
#         os.makedirs(inst_dir, exist_ok=True)
#         return inst_dir + os.sep

#     # be permissive for other tokens used by scripts
#     return os.path.join(base, f"problem_{tag}.pkl")






# ro/utils/offering_no_network.py
import os


def _num_token(v):
    """
    Convert numbers to filename-safe tokens:
      12.0 -> "12"
      0.95 -> "0p95"
      -1.2 -> "m1p2"
    """
    try:
        fv = float(v)
    except Exception:
        return str(v)

    if abs(fv - round(fv)) <= 1e-12:
        return str(int(round(fv)))

    s = f"{fv:.12g}"  # compact, stable
    s = s.replace("-", "m").replace(".", "p")
    return s


def _cfg_tag(cfg):
    """
    Make a deterministic tag similar in spirit to cb/kp naming,
    so later training code can reliably locate the right files.
    """
    T = getattr(cfg, "T", 24)
    S = getattr(cfg, "n_scenarios", getattr(cfg, "scenarios", "S"))
    Gamma = getattr(cfg, "Gamma", "G")
    rt = getattr(cfg, "lambda_rt_value", getattr(cfg, "rt_price_const", "rt"))

    nsi = getattr(cfg, "n_samples_inst", "nsi")
    nsf = getattr(cfg, "n_samples_fs", "nsf")
    nsu = getattr(cfg, "n_samples_per_fs", "nsu")
    sd = getattr(cfg, "seed", "sd")

    ocs = getattr(cfg, "offer_curve_sampling", "ocs")

    tag = (
        f"T{_num_token(T)}"
        f"_S{_num_token(S)}"
        f"_G{_num_token(Gamma)}"
        f"_rt{_num_token(rt)}"
        f"_ocs-{str(ocs)}"
        f"_nsi{_num_token(nsi)}"
        f"_nsf{_num_token(nsf)}"
        f"_nsu{_num_token(nsu)}"
        f"_sd{_num_token(sd)}"
    )
    return tag


def get_path(data_path, cfg, which, suffix=""):
    """
    Compatible with:
      - scripts/00_init_directories.py: get_path(cfg.data_path, cfg, "", suffix="")
      - DataManager: get_path(cfg.data_path, cfg, "problem"/"ml_data"/"test_instances/")
      - 03_train_model.py: get_path(cfg.data_path, cfg, "random_search/set_encoder", suffix=".pt")
    Produces config-tagged filenames like:
      data/offering_no_network/problem_T24_S25_G12_rt800_..._sd7.pkl
    """
    # keep relative "data/..." so scripts' split("/") logic works
    if isinstance(data_path, str) and data_path.startswith("./"):
        data_path = data_path[2:]
    data_path = str(data_path).rstrip("/")

    base = os.path.join(data_path, "offering_no_network")
    os.makedirs(base, exist_ok=True)

    tag = _cfg_tag(cfg)

    # 00_init_directories passes which="" just to infer "data/<problem>/..."
    if which in ("", "problem"):
        return os.path.join(base, f"problem_{tag}.pkl")

    if which == "ml_data":
        # suffix may be "", "_xxx", "_xxx.pkl", "xxx.pkl" etc.
        suf = str(suffix or "")
        if suf and not suf.startswith("_"):
            suf = "_" + suf
        if suf and not suf.endswith(".pkl"):
            suf = suf + ".pkl"
        return os.path.join(base, f"ml_data_{tag}{suf}" if suf else f"ml_data_{tag}.pkl")

    if which == "test_instances/":
        inst_dir = os.path.join(base, "test_instances")
        os.makedirs(inst_dir, exist_ok=True)
        return inst_dir + os.sep

    # -------------------------------------------------------------------------
    # Handle random_search and other subdirectory patterns
    # Pattern: "random_search/set_encoder" or "random_search/set_encoder_tr_res"
    # -------------------------------------------------------------------------
    if which.startswith("random_search/"):
        rs_dir = os.path.join(base, "random_search")
        os.makedirs(rs_dir, exist_ok=True)
        
        # Extract the filename part after "random_search/"
        filename_part = which[len("random_search/"):]
        
        # Build filename with tag
        # e.g., "set_encoder" -> "set_encoder_{tag}.pt" (if suffix=".pt")
        # e.g., "set_encoder_tr_res" -> "set_encoder_tr_res_{tag}.pkl"
        suf = str(suffix or "")
        if not suf:
            suf = ".pkl"  # default extension
        
        full_filename = f"{filename_part}_{tag}{suf}"
        return os.path.join(rs_dir, full_filename)

    # Handle other known directory patterns
    dir_keys = {
        "random_search": "random_search",
        "ml_ccg_results": "ml_ccg_results",
        "ml_ccg_pga_results": "ml_ccg_pga_results",
        "eval_results": "eval_results",
        "eval_results_pga": "eval_results_pga",
        "eval_instances": "eval_instances",
        "baseline_results": "baseline_results",
    }
    
    # Check if which is a directory key (with or without trailing slash)
    which_clean = which.rstrip("/")
    if which_clean in dir_keys:
        d = os.path.join(base, dir_keys[which_clean])
        os.makedirs(d, exist_ok=True)
        if suffix:
            return os.path.join(d, suffix)
        return d + os.sep

    # be permissive for other tokens used by scripts
    return os.path.join(base, f"problem_{tag}.pkl")



    