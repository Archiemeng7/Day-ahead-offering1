import ro.params as params


def factory_two_ro(problem):
    """ Initializes TwoRO class.  """
    if "kp" in problem:
        print("Loading Knapsack data manager...")
        from .kp import Knapsack
        return Knapsack()

    elif "offering_no_network" in problem:
        print("Loading offering_no_network data manager...")
        from .offering_no_network import OfferingNoNetwork
        return OfferingNoNetwork()

    elif "offering_network" in problem:
        print("Loading offering_network data manager...")
        from .offering_network import OfferingNetwork
        return OfferingNetwork()

    elif "cb" in problem:
        print("Loading Capital Budgeting data manager...")
        from .cb import CapitalBudgeting
        return CapitalBudgeting()

    # add new problems here
    
    else:
        raise ValueError("Invalid problem type!")
