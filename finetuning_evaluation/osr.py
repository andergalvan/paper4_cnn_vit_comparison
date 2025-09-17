from openmax import *


def calculate_weibulls(net, train_loader, device, num_classes, weibull_tail, weibull_distance):
    
    """ Calculate the Weibull distributions for OpenMax. """

    _, mavs, dists = compute_train_score_and_mavs_and_dists(num_classes, train_loader, device, net)
    categories = list(range(0, num_classes))
    weibull_models = fit_weibull(mavs, dists, categories, weibull_tail, weibull_distance)
    
    return weibull_models
