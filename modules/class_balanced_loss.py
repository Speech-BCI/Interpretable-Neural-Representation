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

def scaling_neg_dist(neg_dist, scale_factor = 0.7):
    batch_size, num_samples = neg_dist.shape
    sorted_indices = torch.argsort(neg_dist, dim=1, descending=False)  # Sort in descending order
    ranks = torch.argsort(sorted_indices, dim=1).float()
    x = np.linspace(0, math.e, num_samples)
    y = (1.0 - scale_factor) * np.exp(-x) + scale_factor
    y_scaler = torch.from_numpy(y).to(neg_dist.device)
    scaled_neg_dist = neg_dist * y_scaler[ranks.long()]
    return scaled_neg_dist
