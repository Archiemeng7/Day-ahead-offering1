import ro.params as params


def factory_approximator(args, cfg, net, inst_params):
        
    if "cb" in args.problem:
        from .cb import CapitalBudgetingApproximator
        return CapitalBudgetingApproximator(args, cfg, net, inst_params)

    elif "kp" in args.problem:
        from .kp import KnapsackApproximator
        return KnapsackApproximator(args, cfg, net, inst_params)

    elif "offering_no_network" in args.problem:
        from .offering_no_network import OfferingNoNetworkApproximator
        return OfferingNoNetworkApproximator(args, cfg, net, inst_params)

    elif "offering_network" in args.problem:
        from .offering_network import OfferingNetworkApproximator
        return OfferingNetworkApproximator(args, cfg, net, inst_params)



    else:
        raise ValueError("Invalid problem type!")
