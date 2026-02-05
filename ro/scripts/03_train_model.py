# /content/drive/MyDrive/Neur2RO/ro/scripts/03_train_model.py
# CLEANED + MODIFIED:
#   1) add offering_network support in initialize_ml_model
#   2) add --ml_suffix to load ml_data with suffix
#   3) remove duplicated code blocks to avoid running twice

# general
import os
import time
import copy
import argparse
import numpy as np
import pickle as pkl

# (kept for compatibility with original script imports)
import gurobipy as gp
from gurobi_ml import add_predictor_constr
import matplotlib.pyplot as plt

# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# sklearn
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE

# ro
import ro.params as ro_params
from ro.utils import factory_get_path
from ro.models.set_encoder import SetEncoder
from ro.data_preprocessor import factory_dp


#--------------------------------------#
#           General Functions          #
#--------------------------------------#
def list_to_str(x):
    x = list(map(lambda y: str(y), x))
    return "-".join(x)


def mix_instances(cfg, dataset):
    """Mixes training/test instances and splits on decisions (Neur2RO default)."""
    combined_data = dataset["tr_data"] + dataset["val_data"]
    perm = np.random.permutation(len(combined_data))
    split_idx = int(cfg.tr_split * len(combined_data))
    tr_idx = perm[:split_idx].tolist()
    val_idx = perm[split_idx:].tolist()

    combined_data_arr = np.array(combined_data, dtype=object)
    dataset["tr_data"] = combined_data_arr[tr_idx].tolist()
    dataset["val_data"] = combined_data_arr[val_idx].tolist()
    return dataset


#------------------------------------------------#
#                  ML Model                      #
#------------------------------------------------#
def initialize_ml_model(args, dataset):
    """
    dataset is a torch TensorDataset after preprocessing:
      dataset[:] -> (x_feat_tensor, xi_feat_tensor, label_tensor)
    """
    x_n_features = dataset[:][0].shape[-1]

    if "kp" in args.problem:
        xi_n_features = dataset[:][1].shape[-1]
        output_dim = 1

    elif "cb" in args.problem:
        xi_n_features = dataset[:][1].shape[-1]
        output_dim = 2 if args.predict_feas else 1

    elif "offering_no_network" in args.problem:
        xi_n_features = dataset[:][1].shape[-1]
        output_dim = 1

    elif "offering_network" in args.problem:
        xi_n_features = dataset[:][1].shape[-1]
        output_dim = 1

    else:
        raise ValueError(
            f"Unknown/unsupported problem='{args.problem}' in initialize_ml_model(). "
            f"Please add a branch that sets xi_n_features and output_dim."
        )

    net = SetEncoder(
        x_input_dim=x_n_features,
        x_embed_dims=args.x_embed_dims,
        x_post_agg_dims=args.x_post_agg_dims,
        xi_input_dim=xi_n_features,
        xi_embed_dims=args.xi_embed_dims,
        xi_post_agg_dims=args.xi_post_agg_dims,
        value_net_dims=args.value_dims,
        output_dim=output_dim,
        agg_type=args.agg_type,
        bias_embed=args.bias_embed,
        bias_value=args.bias_value,
        dropout=args.dropout,
    )
    return net


#------------------------------------------------#
#               Evaluation Functions             #
#------------------------------------------------#
def eval_net(net, loader):
    preds, labels = [], []
    for batch in loader:
        feats = batch[:][:-1]
        label = batch[:][-1].detach().cpu().numpy().reshape(-1)
        pred = net(*feats).detach().cpu().numpy().reshape(-1)
        preds += list(pred)
        labels += list(label)

    return {"mae": MAE(preds, labels), "mse": MSE(preds, labels)}


#------------------------------------------------#
#                  Plotting                      #
#------------------------------------------------#
def plot_label_dist(args, tr_dataset, val_dataset, fp_dist):
    tr_labels = tr_dataset[:][-1].detach().cpu().numpy().reshape(-1)
    val_labels = val_dataset[:][-1].detach().cpu().numpy().reshape(-1)

    mybins = np.arange(-0.1, 1.1, 0.05)
    plt.hist(tr_labels, bins=mybins, alpha=0.40, label="tr", density=True)
    plt.hist(val_labels, bins=mybins, alpha=0.40, label="val", density=True)
    plt.xlabel("Labels")
    plt.ylabel("Density")
    plt.title(f"Label distribution for {args.problem}")
    plt.legend()
    plt.savefig(fp_dist)


#------------------------------------------------#
#             Parameters for Model               #
#------------------------------------------------#
def get_model_id_str(args):
    fp_id = f"__bs-{args.batch_size}_"
    fp_id += f"lr-{args.lr}_"
    fp_id += f"l1-{args.wt_lasso}_"
    fp_id += f"l2-{args.wt_ridge}_"
    fp_id += f"ep-{args.n_epochs}_"
    fp_id += f"loss_fn-{args.loss_fn}_"
    fp_id += f"d-{args.dropout}_"
    fp_id += f"o-{args.optimizer}_"
    fp_id += f"a-{args.agg_type}_"
    fp_id += f"be-{args.bias_embed}_"
    fp_id += f"bv-{args.bias_value}_"
    fp_id += f"x-ed-{list_to_str(args.x_embed_dims)}_"
    fp_id += f"x-pad-{list_to_str(args.x_post_agg_dims)}_"
    fp_id += f"xi-ed-{list_to_str(args.xi_embed_dims)}_"
    fp_id += f"xi-pad-{list_to_str(args.xi_post_agg_dims)}_"
    fp_id += f"vd-{list_to_str(args.value_dims)}__"
    return fp_id


def get_model_param_dict(args):
    return {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "wt_lasso": args.wt_lasso,
        "wt_ridge": args.wt_ridge,
        "n_epochs": args.n_epochs,
        "loss_fn": args.loss_fn,
        "dropout": args.dropout,
        "optimizer": args.optimizer,
        "agg_type": args.agg_type,
        "value_dims": args.value_dims,
        "bias_embed": args.bias_embed,
        "bias_value": args.bias_value,
        "ml_suffix": args.ml_suffix,
        "x_embed_dims": args.x_embed_dims,
        "x_post_agg_dims": args.x_post_agg_dims,
        "xi_embed_dims": args.xi_embed_dims,
        "xi_post_agg_dims": args.xi_post_agg_dims,
    }


#------------------------------------------------#
#                     Main                       #
#------------------------------------------------#
def _normalize_ml_suffix(s: str) -> str:
    """
    Allow:
      --ml_suffix ""            -> ".pkl" (default)
      --ml_suffix ".pkl"        -> ".pkl"
      --ml_suffix "_XXXX.pkl"   -> "_XXXX.pkl"
      --ml_suffix "XXXX.pkl"    -> "_XXXX.pkl"
    """
    if s is None:
        return ".pkl"
    s = str(s).strip()
    if s == "":
        return ".pkl"
    if s == ".pkl":
        return ".pkl"
    if not s.startswith("_") and not s.startswith("."):
        s = "_" + s
    return s


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print(f"Getting instance/path info for {args.problem} ...")

    cfg = getattr(ro_params, args.problem)
    get_path = factory_get_path(args.problem)

    ml_suffix = _normalize_ml_suffix(args.ml_suffix)
    fp_data = get_path(cfg.data_path, cfg, "ml_data", suffix=ml_suffix)

    fp_dist = get_path("dists/", cfg, "dist", suffix=".png")
    fp_model = get_path(cfg.data_path, cfg, f"random_search/{args.model_type}", suffix=".pt")
    fp_res = get_path(cfg.data_path, cfg, f"random_search/{args.model_type}_tr_res")

    print("Loading data for machine learning ...")
    print("  ml_data file:", fp_data)

    if not os.path.exists(fp_data):
        raise FileNotFoundError(
            f"Cannot find ml_data file: {fp_data}\n"
            f"Hint:\n"
            f"  1) Run: python -m ro.scripts.02_generate_dataset --problem {args.problem}\n"
            f"  2) Or pass the correct suffix: --ml_suffix _XXXX.pkl\n"
        )

    with open(fp_data, "rb") as pf:
        dataset = pkl.load(pf)

    if args.debug:
        dataset["tr_data"] = dataset["tr_data"][:2500]
        dataset["val_data"] = dataset["val_data"][:500]

    if args.mix_instances:
        print("Mixing instances and splitting train/validation by decision/uncertainty ...")
        dataset = mix_instances(cfg, dataset)

    print(f"Loading {args.problem} data preprocessor...")
    data_preprocessor = factory_dp(cfg, args.model_type, args.predict_feas, args.problem, device)

    if args.scale_labels:
        print("Getting label scaler ...")
        data_preprocessor.init_label_scaler(dataset["tr_data"])

    if args.scale_feats:
        print("Getting feature scaler ...")
        data_preprocessor.init_feature_scaler(dataset["tr_data"])

    print("Getting pytorch dataset ...")
    tr_dataset, val_dataset = data_preprocessor.preprocess_data(dataset["tr_data"], dataset["val_data"])

    if args.plot_label_dist:
        print("Plotting label distribution ...")
        plot_label_dist(args, tr_dataset, val_dataset, fp_dist)

    loader = {
        "tr": DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(val_dataset, batch_size=args.batch_size_eval, shuffle=False),
    }

    print(f"Initializing {args.model_type} ...")
    net = initialize_ml_model(args, tr_dataset)
    net.to(device)
    net.set_scalers(data_preprocessor.label_scaler, data_preprocessor.feat_scaler)

    Optimizer = getattr(torch.optim, args.optimizer)
    opt = Optimizer(net.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss(reduction="none")

    print(f"Training {args.model_type} ...")

    best_eval_metric = 1e10
    best_net = None

    tr_dataset_size = len(dataset["tr_data"])
    val_results, tr_results, val_maes, losses = [], [], [], []
    train_time = time.time()

    for ep in range(args.n_epochs):
        loss_total = 0.0
        net.train()

        for batch in loader["tr"]:
            features = batch[:][:-1]
            labels = batch[:][-1]

            y_pred = net(*features)
            y_pred = torch.squeeze(y_pred)

            loss = loss_fn(y_pred, labels)

            opt.zero_grad()
            loss.mean().backward()
            opt.step()

            loss_total += loss.sum().item()

        loss_epoch = loss_total / max(tr_dataset_size, 1)
        losses.append(loss_epoch)

        if ((ep + 1) % args.eval_freq) == 0:
            with torch.no_grad():
                net.eval()
                tr_res = eval_net(net, loader["tr"])
                val_res = eval_net(net, loader["val"])

                tr_mae = tr_res["mae"]
                val_mae = val_res["mae"]

                print(f"    Epoch {ep+1}/{args.n_epochs}: [eval on train and val]")
                print(f"        tr Loss (total):     {loss_total:.6f}")
                print(f"        tr Loss (mean):      {loss_epoch:.6f}")
                print(f"        tr mae:              {tr_mae:.6f}")
                print(f"        val mae:             {val_mae:.6f}")
                print(f"        best val mae:        {best_eval_metric:.6f}")

                tr_results.append(tr_res)
                val_results.append(val_res)
                val_maes.append(val_mae)

            if val_mae < best_eval_metric:
                best_eval_metric = val_mae
                best_net = copy.deepcopy(net)
                print("      new best model!")

    if best_net is None:
        best_net = net

    best_net.eval()
    train_time = time.time() - train_time

    print("Saving training results ...")

    results = {
        "val_mae": best_eval_metric,
        "train_time": train_time,
        "training_stats": {"val_maes": val_maes, "val_results": val_results, "tr_results": tr_results, "losses": losses},
        "params": get_model_param_dict(args),
    }

    fp_id = get_model_id_str(args)

    fp_res = str(fp_res).replace(".pkl", f"{fp_id}.pkl")
    os.makedirs(os.path.dirname(fp_res), exist_ok=True)
    with open(fp_res, "wb") as p:
        pkl.dump(results, p)
    print("  Saved training results to:", fp_res)

    print("Saving model ...")
    fp_model = str(fp_model).replace(".pt", f"{fp_id}.pt")
    os.makedirs(os.path.dirname(fp_model), exist_ok=True)
    torch.save(best_net, fp_model)
    print("  Saved model to:", fp_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML model for any problem. Default network is SetEncoder")

    parser.add_argument("--problem", type=str, default="kp")
    parser.add_argument("--model_type", type=str, default="set_encoder")

    parser.add_argument(
        "--ml_suffix",
        type=str,
        default="",
        help="Suffix for ml_data file. Examples: '' (default), '.pkl', or '_20260122T121314.pkl'",
    )

    parser.add_argument("--plot_label_dist", type=int, default=0)
    parser.add_argument("--mix_instances", type=int, default=0)
    parser.add_argument("--predict_feas", type=int, default=0)

    parser.add_argument("--scale_labels", type=int, default=1)
    parser.add_argument("--scale_feats", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--batch_size_eval", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--loss_fn", type=str, default="MSELoss")
    parser.add_argument("--wt_lasso", type=float, default=0.0)
    parser.add_argument("--wt_ridge", type=float, default=0.0)
    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument("--n_epochs", type=int, default=250)

    parser.add_argument("--value_dims", nargs="+", type=int, default=[4])
    parser.add_argument("--x_embed_dims", nargs="+", type=int, default=[16, 8])
    parser.add_argument("--x_post_agg_dims", nargs="+", type=int, default=[64, 2])
    parser.add_argument("--xi_embed_dims", nargs="+", type=int, default=[16, 8])
    parser.add_argument("--xi_post_agg_dims", nargs="+", type=int, default=[64, 2])

    parser.add_argument("--agg_type", type=str, default="sum")
    parser.add_argument("--bias_embed", type=int, default=0)
    parser.add_argument("--bias_value", type=int, default=1)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--debug", type=int, default=0)

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    import sys
    sys.exit(0)



#------------------------------------------------#
#                  ML Model                      #
#------------------------------------------------#

# def initialize_ml_model(args, dataset):
#     """ Initializes ML model.  
#         Note that this will depend on the problem/types of uncertainty.  
#     """

#     # get input/output dimension depending on the problem
#     single_uncertainty = True
#     x_n_features = dataset[:][0].shape[-1]

#     # knapsack problem
#     if "kp" in args.problem:
#         xi_n_features = dataset[:][1].shape[-1]
#         output_dim = 1
        
#     # capital budgeting problem
#     elif "cb" in args.problem:
#         xi_n_features = dataset[:][1].shape[-1]
#         if args.predict_feas:
#             output_dim = 2
#         else:
#             output_dim = 1

#     net = SetEncoder(
#         # dimensions for x embedding network
#         x_input_dim = x_n_features,
#         x_embed_dims = args.x_embed_dims,
#         x_post_agg_dims = args.x_post_agg_dims,

#         # dimensions for xi embedding network
#         xi_input_dim = xi_n_features,
#         xi_embed_dims = args.xi_embed_dims,
#         xi_post_agg_dims = args.xi_post_agg_dims,

#         # dimensions of value net
#         value_net_dims = args.value_dims,

#         # output dimension
#         output_dim = output_dim,

#         agg_type = args.agg_type,
#         bias_embed = args.bias_embed,
#         bias_value = args.bias_value,
#         dropout = args.dropout)

#     return net
def initialize_ml_model(args, dataset):
    """Initializes ML model depending on the problem."""

    # dataset is a torch TensorDataset after preprocessing:
    # dataset[:] -> tuple(tensor_x, tensor_xi, tensor_label) for KP/CB/offering_no_network
    x_n_features = dataset[:][0].shape[-1]

    # ----------------------------
    # Problem-specific dimensions
    # ----------------------------
    if "kp" in args.problem:
        xi_n_features = dataset[:][1].shape[-1]
        output_dim = 1

    elif "cb" in args.problem:
        xi_n_features = dataset[:][1].shape[-1]
        output_dim = 2 if args.predict_feas else 1

    elif "offering_no_network" in args.problem:
        # x: p_DA (24,)
        # xi: PV uncertainty vector (24,) or equivalent
        xi_n_features = dataset[:][1].shape[-1]
        output_dim = 1

    else:
        raise ValueError(
            f"Unknown/unsupported problem='{args.problem}' in initialize_ml_model(). "
            f"Please add a branch that sets xi_n_features and output_dim."
        )

    # ----------------------------
    # Build model
    # ----------------------------
    net = SetEncoder(
        # dimensions for x embedding network
        x_input_dim=x_n_features,
        x_embed_dims=args.x_embed_dims,
        x_post_agg_dims=args.x_post_agg_dims,

        # dimensions for xi embedding network
        xi_input_dim=xi_n_features,
        xi_embed_dims=args.xi_embed_dims,
        xi_post_agg_dims=args.xi_post_agg_dims,

        # dimensions of value net
        value_net_dims=args.value_dims,

        # output dimension
        output_dim=output_dim,

        agg_type=args.agg_type,
        bias_embed=args.bias_embed,
        bias_value=args.bias_value,
        dropout=args.dropout,
    )

    return net




#------------------------------------------------#
#               Evaluation Functions             #
#------------------------------------------------#

def eval_net(net, loader):
    """ Get eval stats for problems that only require objective prediction. """
    preds = []
    labels = []

    for batch in loader:

        # get features/labels
        feats = batch[:][:-1]
        label = batch[:][-1].detach().cpu().numpy().reshape(-1)

        # get predictions
        pred = net(*feats).detach().cpu().numpy().reshape(-1)

        preds += list(pred)
        labels += list(label)

    # evaluates
    mae = MAE(preds, labels)
    mse = MSE(preds, labels)

    res = {
        'mae' : mae,
        'mse' : mse,
        # todo: add more metrics to track here if needed
    }

    return res


def eval_net_feas(net, loader):
    """ Get eval stats for problems that only require objective + feasibility prediction. """
    # to be implemented for problems wherein feasibility must be predicted.
    raise Exception("eval_net_feas has not been implemented!")




#------------------------------------------------#
#                  Plotting                      #
#------------------------------------------------#

def plot_label_dist(args, tr_dataset, val_dataset, fp_dist, bins=25):
    """ Plots label distribution. """
    tr_labels = tr_dataset[:][-1].detach().cpu().numpy().reshape(-1)
    val_labels = val_dataset[:][-1].detach().cpu().numpy().reshape(-1)

    mybins = np.arange(-0.1, 1.1, 0.05)

    plt.hist(tr_labels, bins=mybins, alpha=0.40 , label = 'tr', density=True)  
    plt.hist(val_labels, bins=mybins, alpha=0.40 , label = 'val', density=True)

    plt.xlabel("Labels")
    plt.ylabel("Density")

    plt.title(f"Label distribution for {args.problem}")

    plt.legend()

    plt.savefig(fp_dist)




#------------------------------------------------#
#             Paremeters for Model               #
#------------------------------------------------#

def get_model_id_str(args):
    """ Gets identifier for model based on args.  
        This will be used to save the correct file path.
    """
    # get model id based on params
    fp_id = f"__bs-{args.batch_size}_" 
    fp_id += f"lr-{args.lr}_" 
    fp_id += f"l1-{args.wt_lasso}_" 
    fp_id += f"l2-{args.wt_ridge}_" 
    fp_id += f"ep-{args.n_epochs}_" 
    fp_id += f"loss_fn-{args.loss_fn}_" 
    fp_id += f"d-{args.dropout}_" 
    fp_id += f"o-{args.optimizer}_" 
    fp_id += f"a-{args.agg_type}_" 
    fp_id += f"be-{args.bias_embed}_" 
    fp_id += f"bv-{args.bias_value}_" 
    fp_id += f"x-ed-{list_to_str(args.x_embed_dims)}_" 
    fp_id += f"x-pad-{list_to_str(args.x_post_agg_dims)}_" 
    fp_id += f"xi-pad-{list_to_str(args.xi_post_agg_dims)}_" 
    fp_id += f"vd-{list_to_str(args.value_dims)}__" 

    return fp_id


def get_model_param_dict(args):
    """ Gets parameter dict for model.  """
    # get model id based on params
    param_dict = {
        'batch_size' : args.batch_size,
        'lr' : args.lr,
        'wt_lasso' : args.wt_lasso,
        'wt_ridge' : args.wt_ridge,
        'n_epochs' : args.n_epochs,
        'loss_fn' : args.loss_fn,
        'dropout' : args.dropout,
        'optimizer' : args.optimizer,
        'agg_type' : args.agg_type,
        'value_dims' : args.value_dims,
        'bias_embed' : args.bias_embed,
        'bias_value' : args.bias_value,
    }

    param_dict['x_embed_dims'] = args.x_embed_dims
    param_dict['x_post_agg_dims'] = args.x_post_agg_dims
    param_dict['xi_embed_dims'] = args.xi_embed_dims
    param_dict['xi_post_agg_dims'] = args.xi_post_agg_dims

    return param_dict


#------------------------------------------------#
#                     Main                       #
#------------------------------------------------#

def main(args):
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    print(f"Getting instance/path info for {args.problem} ...")
    
    # Get cfg, paths, functions
    cfg = getattr(ro_params, args.problem)

    get_path = factory_get_path(args.problem)

    # get paths
    fp_data = get_path(cfg.data_path, cfg, "ml_data")
    fp_dist = get_path('dists/', cfg, "dist", suffix=".png")
    fp_model = get_path(cfg.data_path, cfg, f"random_search/{args.model_type}", suffix=".pt")
    fp_res = get_path(cfg.data_path, cfg, f"random_search/{args.model_type}_tr_res")

    # load data
    print("Loading data for machine learning ... ")
    with open(fp_data, 'rb') as pf:
        dataset = pkl.load(pf)

    # optional debugging by reducing the dataset size.
    # generally this will be useful debugging new models or problems to
    # ensure that everything loads properly.
    if args.debug:
        dataset['tr_data'] = dataset['tr_data'][:2500]
        dataset['val_data'] = dataset['val_data'][:500]

    # mix instances in train/validation, then split by training over decisions/uncertainty.
    # this was done in Neur2RO, but in general may not be ideal as one wants to generalize over
    # instances.  Note that this non-mixing had not been extensively tested.
    if args.mix_instances:
        print("Mixing instances and splitting train/validation by decision/uncertainty, i.e., Neur2RO default ... ")
        dataset = mix_instances(cfg, dataset)

    # initialize data preprocessor
    data_preprocessor = factory_dp(cfg, args.model_type, args.predict_feas, args.problem, device)

    # initialize label scalers
    if args.scale_labels:
        print("Getting label scaler ... ")
        data_preprocessor.init_label_scaler(dataset['tr_data'])

    # initialize feature scalers
    if args.scale_feats:
        print("Getting feature scaler ... ")
        data_preprocessor.init_feature_scaler(dataset['tr_data'])

    # get pytorch datasets
    print("Getting pytorch dataset ...")
    tr_dataset, val_dataset = data_preprocessor.preprocess_data(dataset['tr_data'], dataset['val_data'])

    # plot distribution of normalized labels
    if args.plot_label_dist:
        print("Plotting label distribution ...")
        plot_label_dist(args, tr_dataset, val_dataset, fp_dist)

    # build dataloader
    loader = {
        'tr': DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=args.batch_size_eval, shuffle=False)
    }

    # initialize model
    print(f"Initializing {args.model_type} ...")
    net = initialize_ml_model(args, tr_dataset)
    net.to(device)
    net.set_scalers(data_preprocessor.label_scaler, data_preprocessor.feat_scaler)

    # set optimizer
    Optimizer = getattr(torch.optim, args.optimizer)
    opt = Optimizer(net.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss(reduction='none')

    # train
    print(f"Training {args.model_type} ...")

    best_eval_metric = 1e10
    best_net = None

    tr_dataset_size = len(dataset['tr_data'])

    val_results = []
    tr_results = []
    val_maes = []
    losses = []

    train_time = time.time()

    for ep in range(args.n_epochs):
        loss_total = 0

        # batch updates
        net.train()

        for batch in loader['tr']:
            features = batch[:][:-1]
            labels = batch[:][-1]

            y_pred = net(*features)
            y_pred = torch.squeeze(y_pred)
            
            # if args.use_exact_cons:
            loss = loss_fn(y_pred, labels)

            # do not compute loss for np.nan values if prediciting feas + obj
            # to be implemented...
            # else:
            #     y_ = torch.clone(y)
            #     nan_idx = torch.isnan(y_)
            #     y_[nan_idx] = y_pred[nan_idx]
            #     loss = loss_fn(y_pred, y_)

            # loss = loss_fn(y_pred, y)
            opt.zero_grad()
            loss.mean().backward()
            opt.step()

            loss_total += loss.sum().item()
     
        loss_epoch = loss_total / tr_dataset_size

        losses.append(loss_epoch)     

        # evaluate on train/validation sets
        if ((ep+1) % args.eval_freq) == 0:
            with torch.no_grad():

                net.eval()

                if not args.predict_feas:
                    tr_res = eval_net(net, loader['tr'])
                    tr_mae = tr_res['mae']

                    val_res = eval_net(net, loader['val'])
                    val_mae = val_res['mae']

                    # print("    Epoch {}/{}: tr Loss {:.6f}, val mae {:.6f}, best mae {:.6f}".format(ep+1, args.n_epochs, loss_epoch, val_mae, best_eval_metric))
                    print("    Epoch {}/{}: [eval on train and val]".format(ep+1, args.n_epochs))
                    print("        tr Loss (total):     {:.6f}".format(loss_total))
                    print("        tr Loss (mean):      {:.6f}".format(loss_epoch))
                    print("        tr mae:              {:.6f}".format(tr_mae))
                    print("        val mae:             {:.6f}".format(val_mae))
                    print("        best val mae:        {:.6f}".format(best_eval_metric))

                else:
                    pass 
                    # to be implemented ...
                    # val_res = eval_net_feas(net, loader['val'])
     
                tr_results.append(tr_res)
                val_results.append(val_res)
                val_maes.append(val_mae)

            if val_mae < best_eval_metric:
                best_eval_metric = val_mae
                best_net = copy.deepcopy(net)
                print('      new best model! ')

    best_net.eval()
    train_time = time.time() - train_time

    # save results
    print("Saving training results ...")
    
    results = {
        'val_mae' : best_eval_metric,
        'train_time' : train_time,
        'training_stats' : {
            'val_maes' : val_maes,
            'val_results' : val_results,
            'tr_results' : tr_results, 
            'losses' : losses,
        },
        'params' : get_model_param_dict(args),
    }

    # get string for model
    fp_id = get_model_id_str(args)

    # save training/validation results!
    fp_res = str(fp_res).replace(".pkl", f"{fp_id}.pkl")
    with open(fp_res, 'wb') as p:
        pkl.dump(results, p)

    print('  Saved training results to:', fp_res)

    # save model
    print("Saving model ...")

    fp_model = str(fp_model).replace(".pt", f"{fp_id}.pt")
    torch.save(best_net, fp_model)
    
    print('  Saved model to:', fp_model)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ML model for any problem problem.  Default network is SetNetwork')

    parser.add_argument('--problem', type=str, default="kp")
    parser.add_argument('--model_type', type=str, default='set_encoder')

    # Distribution visualization
    parser.add_argument('--plot_label_dist', type=int, default=0, help='Plots label distribution')

    # Type of data to train on
    parser.add_argument('--mix_instances', type=int, default=0, help='Indictor to mix instances and validate over decision/uncertainty.  Default for Neur2RO is 1.')

    # Problem specific arguments
    parser.add_argument('--predict_feas', type=int, default=0, help='Feasibility prediction.  Only nessecary for problems without recourse.  Optional for cb ')

    # Scaling arguments
    parser.add_argument('--scale_labels', type=int, default=1, help='Boolean to scale labels')
    parser.add_argument('--scale_feats', type=int, default=1, help='Boolean to scale features')

    # General NN parameters
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--batch_size_eval', type=int, default=256, help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate.')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--loss_fn', type=str, default='MSELoss')
    parser.add_argument('--wt_lasso', type=float, default=0)
    parser.add_argument('--wt_ridge', type=float, default=0)
    parser.add_argument('--eval_freq', type=int, default=10, help='Frequency to evaluate model.')
    parser.add_argument('--n_epochs', type=int, default=250, help='Number of training epochs.')

    # SetEncoder parameters
    parser.add_argument('--value_dims', nargs="+", type=int, default=[4], help='List of hidden dims in tiny network.')
    parser.add_argument('--x_embed_dims', nargs="+", type=int, default=[16, 8], help='List of hidden dims in x embedding network.')
    parser.add_argument('--x_post_agg_dims', nargs="+", type=int, default=[64, 2], help='List of hidden dims in x embedding network.')
    
    # KP/CB 
    parser.add_argument('--xi_embed_dims', nargs="+", type=int, default=[16, 8], help='List of hidden dims in xi embedding network.')
    parser.add_argument('--xi_post_agg_dims', nargs="+", type=int, default=[64, 2], help='List of hidden dims in xi embedding network.')
    
    # aggregation/bias
    parser.add_argument('--agg_type', type=str, default="sum", help='Type of aggregation (sum, mean, etc).')
    parser.add_argument('--bias_embed', type=int, default=0, help='Bias in embedding network(s) (default true only for now!).')
    parser.add_argument('--bias_value', type=int, default=1, help='Bias in value network  (default true only for now!).')

    # device
    parser.add_argument('--device', type=str, default="cpu", help='Device.')

    # Random seed
    parser.add_argument('--seed', type=int, default=12345, help='Seed.')

    # Debug on subset of data, does not save anything
    parser.add_argument('--debug', type=int, default=0, help='Indicator to debug on subset of data.  Note that model/results will note be saved.')

    args = parser.parse_args()

    main(args)



















