import torch
import numpy as np
import math
def calculate_class_weights(num_class_trials, beta):
    effective_num = 1.0 - torch.pow(beta, num_class_trials)
    weights = (1.0 - beta) / effective_num

    weights = weights / torch.sum(weights) * len(num_class_trials)
    return weights

def apply_class_balanced_loss(cl_loss, labels, num_class_trials, beta):
    class_weights = calculate_class_weights(num_class_trials, beta)

    weights = class_weights[labels]

    weighted_loss = cl_loss.squeeze() * weights
    return weighted_loss.mean()
