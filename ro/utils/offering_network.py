# /content/drive/MyDrive/Neur2RO/ro/utils/offering_network.py
import os


def _num_token(v):
    try:
        fv = float(v)
    except Exception:
        return str(v)

    if abs(fv - round(fv)) <= 1e-12:
        return str(int(round(fv)))

    s = f"{fv:.12g}"
    s = s.replace("-", "m").replace(".", "p")
    return s


def _cfg_tag(cfg):
    T = getattr(cfg, "T", 24)
    S = getattr(cfg, "n_scenarios", getattr(cfg, "scenarios", "S"))
    Gamma = getattr(cfg, "Gamma", "G")
    rt = getattr(cfg, "lambda_rt_value", getattr(cfg, "rt_price_const", "rt"))

    nsi = getattr(cfg, "n_samples_inst", "nsi")
    nsf = getattr(cfg, "n_samples_fs", "nsf")
    nsu = getattr(cfg, "n_samples_per_fs", "nsu")
    sd = getattr(cfg, "seed", "sd")

    ocs = getattr(cfg, "offer_curve_sampling", "ocs")
    sbase = getattr(cfg, "Sbase", "Sb")

    tag = (
        f"T{_num_token(T)}"
        f"_S{_num_token(S)}"
        f"_Sb{_num_token(sbase)}"
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
    if isinstance(data_path, str) and data_path.startswith("./"):
        data_path = data_path[2:]
    data_path = str(data_path).rstrip("/")

    base = os.path.join(data_path, "offering_network")
    os.makedirs(base, exist_ok=True)

    tag = _cfg_tag(cfg)

    if which in ("", "problem"):
        return os.path.join(base, f"problem_{tag}.pkl")

    if which == "ml_data":
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

    if which.startswith("random_search/"):
        rs_dir = os.path.join(base, "random_search")
        os.makedirs(rs_dir, exist_ok=True)

        filename_part = which[len("random_search/"):]
        suf = str(suffix or "")
        if not suf:
            suf = ".pkl"
        full_filename = f"{filename_part}_{tag}{suf}"
        return os.path.join(rs_dir, full_filename)

    dir_keys = {
        "random_search": "random_search",
        "ml_ccg_results": "ml_ccg_results",
        "ml_ccg_pga_results": "ml_ccg_pga_results",
        "eval_results": "eval_results",
        "eval_results_pga": "eval_results_pga",
        "eval_instances": "eval_instances",
        "baseline_results": "baseline_results",
    }

    which_clean = which.rstrip("/")
    if which_clean in dir_keys:
        d = os.path.join(base, dir_keys[which_clean])
        os.makedirs(d, exist_ok=True)
        if suffix:
            return os.path.join(d, suffix)
        return d + os.sep

    return os.path.join(base, f"problem_{tag}.pkl")
