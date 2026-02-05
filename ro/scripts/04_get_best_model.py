# import argparse
# import pickle as pkl
# import shutil

# import numpy as np

# import ro.params as params
# from ro.utils import factory_get_path


# def parse_run_name(run_name):
#     """ Maybe useful at some point. """
#     pass


# def get_best_model(args):
#     cfg = getattr(params, args.problem)
#     get_path = factory_get_path(args.problem)
#     results_fp = get_path(cfg.data_path, cfg, ptype=f'random_search/{args.model_type}_tr_res', suffix='.pkl')
#     results_fp_prefix = str(results_fp.stem)
#     model_suffix = '.pt'
#     model_fp = get_path(cfg.data_path, cfg, ptype=f'random_search/{args.model_type}', suffix=model_suffix)

#     # get all tr_res files
#     results_paths = [str(x) for x in model_fp.parent.iterdir()]
#     results_paths = [x for x in model_fp.parent.iterdir() if results_fp_prefix in str(x.stem)]
#     results_paths = [x for x in results_paths if "__" in str(x.stem)]

#     # find_best_result
#     best_criterion, best_results_path = np.infty, None

#     print(f'Checking {len(results_paths)} model files...')
#     for rp in results_paths:
#         rdict = pkl.load(open(rp, 'rb'))
#         if best_criterion > rdict[args.criterion]:
#             best_criterion = rdict[args.criterion]
#             best_results_path = rp

#     # generate best model path and save
#     parts = str(best_results_path.stem).split('_')
#     parts.remove('tr')
#     parts.remove('res')
#     best_model_path = "_".join(parts) + model_suffix
#     best_model_path = best_results_path.parent.joinpath(best_model_path)

#     best_model_path = str(best_model_path)
#     best_results_path = str(best_results_path)

#     model_fp = str(model_fp)
#     results_fp = str(results_fp)

#     results_fp = results_fp.replace('random_search/', '')
#     model_fp = model_fp.replace('random_search/', '')

#     print(f'Best model: {best_results_path}')
#     print(f'Best {args.criterion}: {best_criterion}')
#     print(f"Saving to:", {model_fp})

#     shutil.copy(best_results_path, results_fp)
#     shutil.copy(best_model_path, model_fp)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--problem', type=str, default='kp')
#     parser.add_argument('--model_type', type=str, default='set_encoder')
#     parser.add_argument('--criterion', type=str, default='val_mae')

#     args = parser.parse_args()

#     get_best_model(args)



# ro/scripts/04_get_best_model.py
#
# Robust "get best model" script that works across problems with different get_path() signatures.
# Key changes vs. original:
# 1) Do NOT call get_path(..., ptype=..., suffix=...) (not supported by many problems).
# 2) Infer <problem_dir> via get_path(...,"test_instances/") and then locate random_search under it.
# 3) Search result files using multiple filename patterns (because different runs may name them differently).
# 4) If no result files exist, DO NOT raise; print diagnostics and exit cleanly (so command does not error).

import argparse
import pickle as pkl
import shutil
from pathlib import Path

import numpy as np

import ro.params as params
from ro.utils import factory_get_path


# def _get_problem_dir(get_path_fn, cfg) -> Path:
#     """
#     Infer the problem directory from get_path():
#       <data_path>/<problem_name>/test_instances/
#     -> parent = <data_path>/<problem_name>
#     """
#     inst_dir = Path(get_path_fn(cfg.data_path, cfg, "test_instances/"))
#     return inst_dir.parent


def _get_problem_dir(cfg, problem_name: str) -> Path:
    """
    Determine the problem directory without relying on 'test_instances/'.

    Expected structure:
      <data_path>/<problem_name>/
        random_search/
        ...

    So problem_dir = <data_path>/<problem_name>
    """
    return Path(cfg.data_path) / problem_name



def _collect_candidates(rs_dir: Path, model_type: str) -> list[Path]:
    """
    Collect training-result pkls using multiple naming conventions.
    """
    patterns = [
        f"{model_type}_tr_res__*.pkl",   # most common in Neur2RO random search
        f"{model_type}_tr_res_*.pkl",    # variant with single underscore
        f"{model_type}_tr_res*.pkl",     # any prefix match
        f"*{model_type}*tr*res*.pkl",    # very loose fallback
    ]
    seen = set()
    out = []
    for pat in patterns:
        for p in rs_dir.glob(pat):
            if p.is_file() and str(p) not in seen:
                seen.add(str(p))
                out.append(p)
    return sorted(out)


def _infer_model_path(best_results_path: Path, model_type: str) -> Path | None:
    """
    Try to infer the corresponding .pt weights file from a results .pkl path.
    """
    stem = best_results_path.stem

    # common:  <model_type>_tr_res__XYZ.pkl  -> <model_type>__XYZ.pt
    if f"{model_type}_tr_res__" in stem:
        expected_stem = stem.replace(f"{model_type}_tr_res__", f"{model_type}__", 1)
        cand = best_results_path.with_name(expected_stem + ".pt")
        if cand.exists():
            return cand

    # common: <model_type>_tr_res_... -> <model_type>_...
    if f"{model_type}_tr_res_" in stem:
        expected_stem = stem.replace(f"{model_type}_tr_res_", f"{model_type}_", 1)
        cand = best_results_path.with_name(expected_stem + ".pt")
        if cand.exists():
            return cand

    # loose: if results has "__XYZ" then try <model_type>__XYZ.pt
    if "__" in stem:
        suffix_after = stem.split("__", 1)[1]
        cand = best_results_path.parent / f"{model_type}__{suffix_after}.pt"
        if cand.exists():
            return cand

    # last resort: pick any .pt that shares the longest common prefix with results stem
    pts = list(best_results_path.parent.glob(f"{model_type}*.pt"))
    if not pts:
        return None
    pts = sorted(pts, key=lambda p: len(_common_prefix(p.stem, stem)), reverse=True)
    return pts[0] if pts else None


def _common_prefix(a: str, b: str) -> str:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return a[:i]


def get_best_model(args):
    cfg = getattr(params, args.problem)
    get_path = factory_get_path(args.problem)

    # prob_dir = _get_problem_dir(get_path, cfg)
    # rs_dir = prob_dir / "random_search"
   
    prob_dir = _get_problem_dir(cfg, args.problem)
    rs_dir = prob_dir / "random_search"



    if not rs_dir.exists():
        print(f"[WARN] random_search directory not found: {rs_dir}")
        print(f"       Run: python -m ro.scripts.00_init_directories --problem {args.problem}")
        return  # exit cleanly

    candidates = _collect_candidates(rs_dir, args.model_type)

    print(f"Checking {len(candidates)} model result files in {rs_dir} ...")

    if len(candidates) == 0:
        # Diagnostics but no exception (so command does not error)
        pkls = sorted(rs_dir.glob("*.pkl"))
        pts = sorted(rs_dir.glob("*.pt"))
        print("[WARN] No training result files found for model_type =", args.model_type)
        print("       Looked for patterns like:")
        print(f"         - {args.model_type}_tr_res__*.pkl")
        print(f"         - {args.model_type}_tr_res*.pkl")
        print("       Existing files under random_search (first 30):")
        for p in (pkls + pts)[:30]:
            print("        ", p.name)
        print("       If you haven't run random search for this problem yet, run the training/search script first.")
        return  # exit cleanly

    best_criterion = np.inf
    best_results_path = None
    best_dict = None

    for rp in candidates:
        try:
            rdict = pkl.load(open(rp, "rb"))
        except Exception as e:
            print(f"  Skipping unreadable: {rp.name} ({e})")
            continue

        if args.criterion not in rdict:
            # do not fail; just skip
            continue

        val = rdict[args.criterion]
        try:
            val_f = float(val)
        except Exception:
            continue

        if val_f < best_criterion:
            best_criterion = val_f
            best_results_path = rp
            best_dict = rdict

    if best_results_path is None:
        # No file had the criterion; pick the first readable one
        for rp in candidates:
            try:
                rdict = pkl.load(open(rp, "rb"))
                best_results_path = rp
                best_dict = rdict
                best_criterion = np.nan
                print(f"[WARN] None of the result files contains criterion '{args.criterion}'.")
                print(f"       Falling back to first readable file: {rp.name}")
                break
            except Exception:
                continue

    if best_results_path is None:
        print("[WARN] All candidate files were unreadable. Nothing copied.")
        return

    best_model_path = _infer_model_path(best_results_path, args.model_type)

    dst_results = prob_dir / f"{args.model_type}_tr_res.pkl"
    dst_model = prob_dir / f"{args.model_type}.pt"

    print(f"Best results: {best_results_path.name}")
    if np.isfinite(best_criterion):
        print(f"Best {args.criterion}: {best_criterion}")
    else:
        print(f"Best {args.criterion}: (not available)")

    print(f"Copy results -> {dst_results}")
    shutil.copy(best_results_path, dst_results)

    if best_model_path is None or not best_model_path.exists():
        print("[WARN] Could not locate corresponding .pt weights file. Results copied only.")
        return

    print(f"Copy model   -> {dst_model}")
    shutil.copy(best_model_path, dst_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="kp")
    parser.add_argument("--model_type", type=str, default="set_encoder")
    parser.add_argument("--criterion", type=str, default="val_mae")
    args = parser.parse_args()
    get_best_model(args)

