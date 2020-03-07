import torch
import torch.nn.functional as F

def cross_entropy2d(predicts, targets):
    """
    Hint: look at torch.nn.NLLLoss.  Considering weighting
    classes for better accuracy.
    """
    return torch.nn.NLLLoss(F.log_softmax(predicts), targets)

def cross_entropy1d(predicts, targets):
	new_targets = targets.max(dim=0)
    return cross_entropy2d(predicts, targets)