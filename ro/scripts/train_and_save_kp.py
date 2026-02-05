# ro/scripts/train_and_save_kp.py
import argparse
import os
import subprocess
import sys
from pathlib import Path
import glob
import time

def run_cmd(args_list):
    print(">>", " ".join(args_list))
    p = subprocess.run(args_list, stdout=sys.stdout, stderr=sys.stderr)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(args_list)}")

def ensure_dirs(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def dataset_exists():
    return len(glob.glob("data/kp/ml_data_*.pkl")) > 0

def list_candidates_models():
    cands = []
    cands += glob.glob("data/kp/models/*.pt") + glob.glob("data/kp/models/*.pth") + glob.glob("data/kp/models/*.ckpt")
    cands += glob.glob("data/kp/random_search/**/*.pt", recursive=True)
    cands += glob.glob("data/kp/random_search/**/*.pth", recursive=True)
    cands += glob.glob("data/kp/random_search/**/*.ckpt", recursive=True)
    return sorted(set(cands))

def pick_best_by_csv():
    import pandas as pd
    csvs = glob.glob("data/kp/random_search/*.csv")
    if not csvs:
        return None, None
    best_path, best_score, mode = None, None, None
    for csv in csvs:
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if 'val_loss' in df.columns:
            idx = df['val_loss'].astype(float).idxmin()
            score = float(df.loc[idx, 'val_loss'])
            model_path = df.loc[idx].get('model_path') or df.loc[idx].get('checkpoint')
            if isinstance(model_path, str) and (best_score is None or score < best_score):
                best_score, best_path, mode = score, model_path, 'min'
        elif 'val_metric' in df.columns:
            idx = df['val_metric'].astype(float).idxmax()
            score = float(df.loc[idx, 'val_metric'])
            model_path = df.loc[idx].get('model_path') or df.loc[idx].get('checkpoint')
            if isinstance(model_path, str) and (best_score is None or score > best_score):
                best_score, best_path, mode = score, model_path, 'max'
    return best_path, mode

def pick_best_model():
    best_from_csv, mode = pick_best_by_csv()
    if best_from_csv and os.path.isfile(best_from_csv):
        print(f"[best] chosen by CSV ({'min' if mode=='min' else 'max'}): {best_from_csv}")
        return best_from_csv
    cands = list_candidates_models()
    if not cands:
        return None
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    print(f"[best] fallback newest: {cands[0]}")
    return cands[0]

def copy_file(src, dst):
    ensure_dirs(os.path.dirname(dst))
    import shutil
    shutil.copy2(src, dst)
    print(f"[save] copied: {src} -> {dst}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="kp")
    ap.add_argument("--skip_data", type=int, default=0)
    ap.add_argument("--n_procs", type=int, default=1)
    ap.add_argument("--debug", type=int, default=1)
    ap.add_argument("--x_embed_dims", nargs="+", default=["32","16"])
    ap.add_argument("--x_post_agg_dims", nargs="+", default=["64","8"])
    ap.add_argument("--xi_embed_dims", nargs="+", default=["32","16"])
    ap.add_argument("--xi_post_agg_dims", nargs="+", default=["64","8"])
    ap.add_argument("--value_dims", type=int, default=8)
    ap.add_argument("--n_epochs", type=int, default=20)
    ap.add_argument("--save_dir", default="data/kp/models")
    ap.add_argument("--best_name", default="best_kp.pt")
    ap.add_argument("--gdrive_dir", default=None)
    args = ap.parse_args()

    ensure_dirs("data"); ensure_dirs(f"data/{args.problem}"); ensure_dirs(args.save_dir)
    run_cmd([sys.executable, "-m", "ro.scripts.00_init_directories", "--problem", args.problem])
    run_cmd([sys.executable, "-m", "ro.scripts.01_init_problem", "--problem", args.problem])

    need_data = (not args.skip_data) or (not dataset_exists())
    if need_data:
        run_cmd([sys.executable, "-m", "ro.scripts.02_generate_dataset",
                 "--problem", args.problem, "--n_procs", str(args.n_procs), "--debug", str(args.debug)])
    else:
        print("[info] skip dataset (ml_data exists)")

    train_cmd = [sys.executable, "-m", "ro.scripts.03_train_model",
                 "--problem", args.problem,
                 "--value_dims", str(args.value_dims),
                 "--n_epochs", str(args.n_epochs)]
    train_cmd += ["--x_embed_dims"] + [*args.x_embed_dims]
    train_cmd += ["--x_post_agg_dims"] + [*args.x_post_agg_dims]
    train_cmd += ["--xi_embed_dims"] + [*args.xi_embed_dims]
    train_cmd += ["--xi_post_agg_dims"] + [*args.xi_post_agg_dims]
    run_cmd(train_cmd)

    best_path = pick_best_model()
    if not best_path or not os.path.isfile(best_path):
        print("[warn] no candidate model; try 04_get_best_model ...")
        try:
            run_cmd([sys.executable, "-m", "ro.scripts.04_get_best_model", "--problem", args.problem])
            best_path = pick_best_model()
        except Exception as e:
            print(f"[warn] 04_get_best_model failed: {e}")

    if not best_path or not os.path.isfile(best_path):
        raise RuntimeError("no model file found; check training outputs and directories")

    fixed_path = str(Path(args.save_dir) / args.best_name)
    copy_file(best_path, fixed_path)

    if args.gdrive_dir:
        Path(args.gdrive_dir).mkdir(parents=True, exist_ok=True)
        copy_file(fixed_path, str(Path(args.gdrive_dir) / args.best_name))

    meta = Path(args.save_dir) / (Path(args.best_name).stem + "_meta.txt")
    with open(meta, "w") as f:
        f.write(f"time={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"problem={args.problem}\n")
        f.write(f"x_embed_dims={' '.join(args.x_embed_dims)}\n")
        f.write(f"x_post_agg_dims={' '.join(args.x_post_agg_dims)}\n")
        f.write(f"xi_embed_dims={' '.join(args.xi_embed_dims)}\n")
        f.write(f"xi_post_agg_dims={' '.join(args.xi_post_agg_dims)}\n")
        f.write(f"value_dims={args.value_dims}\n")
        f.write(f"n_epochs={args.n_epochs}\n")
        f.write(f"fixed_path={fixed_path}\n")
        f.write(f"source={best_path}\n")
    print(f"[done] fixed best model -> {fixed_path}\n[meta] {meta}")

if __name__ == "__main__":
    main()
