import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm


def enable_dropout(model):
    """
    Go through all modules and activate dropout modules
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

def prepare_model_for_uncertainty_estimation(model): 
    """
    Prepare the model for uncertainty estimation by enabling dropout with evaluation mode
    """
    # Ensure the model is in evaluation mode
    model.eval() 
    enable_dropout(model)
    return model


def predictive_entropy(output):
    """
    Calculates the predictive entropy for a batch of samples.
    Args:
        output: Tensor of shape (num_samples, batch_size, num_classes) containing 
                probabilities for each class.
    Returns:
        Tensor of shape (batch_size,) containing the predictive entropy for each 
        sample in the batch.
    """
    eps = 1e-8  # For numerical stability, avoid log(0)
    mean_probs = output.mean(dim=0)  # Average probabilities over samples
    entropy = -torch.sum(mean_probs * torch.log(mean_probs + eps), dim=-1)
    return entropy


def mutual_information(output):
    """
    Calculates the mutual information for a batch of samples.
    Args:
        output: Tensor of shape (num_samples, batch_size, num_classes) containing 
                probabilities for each class.
    Returns:
        Tensor of shape (batch_size,) containing the mutual information for each 
        sample in the batch.
    """
    eps = 1e-8
    mean_probs = output.mean(dim=0)
    entropy_of_mean = - \
        torch.sum(mean_probs * torch.log(mean_probs + eps), dim=-1)
    expected_entropy = - \
        torch.mean(torch.sum(output * torch.log(output + eps), dim=-1), dim=0)
    mutual_info = entropy_of_mean - expected_entropy
    return mutual_info

def variation_ratio(output):
    """
    Calculates the variation ratio for a batch of samples.
    Args:
        output: Tensor of shape (num_samples, batch_size, num_classes) containing 
                probabilities for each class.
    Returns:
        Tensor of shape (batch_size,) containing the variation ratio for each 
        sample in the batch.
    """
    output = output.cpu()
    predicted_classes = torch.argmax(output, dim=-1)  # [num_samples, batch_size]
    most_frequent_class = torch.mode(
        predicted_classes, dim=0).values  # [batch_size]
    est_variation_ratio = 1 - (torch.sum(predicted_classes ==
                               most_frequent_class, dim=0) / output.shape[0])  # [batch_size]
    return est_variation_ratio