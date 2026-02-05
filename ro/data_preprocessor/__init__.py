# import ro.params as params


# def factory_dp(cfg, model_type, predict_feas, problem, device):
#     cfg = getattr(params, problem)

#     if "kp" in problem:
#         print("Loading Knapsack data preprocessor...")
#         from .kp import KnapsackDataPreprocessor
#         return KnapsackDataPreprocessor(cfg, model_type, predict_feas, device)

#     elif "cb" in problem:
#         print("Loading Captial Budgeting data preprocessor...")
#         from .cb import CapitalBudgetingDataPreprocessor
#         return CapitalBudgetingDataPreprocessor(cfg, model_type, predict_feas, device)

#     else:
#         raise ValueError("Invalid problem type!")

# ro/data_preprocessor/__init__.py
# (replace file content with this version)

import ro.params as params


def factory_dp(cfg, model_type, predict_feas, problem, device):
    cfg = getattr(params, problem)

    if "kp" in problem:
        print("Loading Knapsack data preprocessor...")
        from .kp import KnapsackDataPreprocessor
        return KnapsackDataPreprocessor(cfg, model_type, predict_feas, device)

    elif "cb" in problem:
        print("Loading Captial Budgeting data preprocessor...")
        from .cb import CapitalBudgetingDataPreprocessor
        return CapitalBudgetingDataPreprocessor(cfg, model_type, predict_feas, device)

    elif "offering_no_network" in problem:
        print("Loading offering_no_network data preprocessor...")
        from .offering_no_network import OfferingNoNetworkDataPreprocessor
        return OfferingNoNetworkDataPreprocessor(cfg, model_type, predict_feas, device)


    elif "offering_network" in problem:
        print("Loading offering_network data preprocessor...")
        from .offering_network import OfferingNetworkDataPreprocessor
        return OfferingNetworkDataPreprocessor(cfg, model_type, predict_feas, device)


    else:
        raise ValueError("Invalid problem type!")
