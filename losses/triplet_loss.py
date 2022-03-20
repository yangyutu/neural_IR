import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum

class TripletDistanceMetric(Enum):
    """
    The metric for triplet loss
    """
    
    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)

def euclidean_distance(x, y):
    return F.pairwise_distance(x, y, p=2)

def get_distance_metric(distance_metric):

    if distance_metric == 'euclidean':
        return euclidean_distance
    else:
        raise ValueError(f'{distance_metric} is not supported!')

class TripletLoss(nn.Module):
    def __init__(self, distance_metric='euclidean', margin=0.0):
        super().__init__()

        self.distance_metric = get_distance_metric(distance_metric)
        self.margin = margin

    def forward(self, triplet_reps, label=None):

        rep_anchor, rep_pos, rep_neg = triplet_reps

        distance_pos = self.distance_metric(rep_anchor, rep_pos)
        distance_neg = self.distance_metric(rep_anchor, rep_neg)

        losses = F.relu(distance_pos - distance_neg + self.margin)

        return losses.mean()
