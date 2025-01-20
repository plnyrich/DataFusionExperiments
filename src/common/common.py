import math


def fixMetric(metric):
    if metric == 1:
        # metric = 1 - math.pow(10, -8)
        metric -= 0.001
    elif metric == 0:
        # metric = math.pow(10, -8)
         metric += 0.001
    return metric
