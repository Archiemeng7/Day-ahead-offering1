from argparse import ArgumentParser

from ro.dm import factory_dm
from ro.utils import DataManagerModes as Modes


def main(args):
    dm = factory_dm(args.problem)

    # 统一把命令行传入的后缀传给 DataManager
    if args.by_inst:
        dm.generate_dataset_by_inst(args.n_procs, args.debug, name_suffix=args.ml_suffix)
    else:
        dm.generate_dataset(args.n_procs, args.debug, name_suffix=args.ml_suffix)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--problem', type=str, default="kp",
                        help="Problem from ro/params.py.")
    parser.add_argument('--n_procs', type=int, default=1,
                        help="Number of processes for multiprocessing.")
    parser.add_argument('--debug', type=int, default=0,
                        help="If 1, run single-process for easier debugging (and may early-exit).")
    parser.add_argument('--by_inst', type=int, default=0,
                        help="If 1, generate dataset by instance split; else pooled.")
    # ★ 新增：数据集文件名后缀
    parser.add_argument('--ml_suffix', type=str, default="",
                        help="Optional suffix for the saved ML dataset filename, e.g. _v1 or _20251013.")

    args = parser.parse_args()
    main(args)
