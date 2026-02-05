import ro.params as params


def factory_dm(problem):
    cfg = getattr(params, problem)

    if "kp" in problem:
        print("Loading Knapsack data manager...")
        from .kp import KnapsackDataManager
        return KnapsackDataManager(cfg, problem)

    elif "offering_no_network" in problem:
        print("Loading offering_no_network data manager...")
        from .offering_no_network import OfferingNoNetworkDataManager
        return OfferingNoNetworkDataManager(cfg, problem)

    elif "offering_network" in problem:
        print("Loading offering_network data manager...")
        from .offering_network import OfferingNetworkDataManager
        return OfferingNetworkDataManager(cfg, problem)


    elif "cb" in problem:
        print("Loading Capital Budgeting data manager...")
        from .cb import CapitalBudgetingDataManager
        return CapitalBudgetingDataManager(cfg, problem)

    else:
        raise ValueError("Invalid problem type!")
