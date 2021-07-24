from data_provider import TempuckeyVideoSentencePairsDataset as TempuckeyDataset
from data_provider import Normalize_VideoSentencePair
from utils import sys_utils as utils

from numpy import linalg
import numpy as np
import argparse
import pdb
import torch 
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_metrics(mat):
    """
    Input Parameter:
        mat: N x N matrix of Video-to-Text (or Text-to-Video) distance in the embedding space
    Output:
        [recall@1, recall@5, recall@10, MeanRank, MedianRank]
    """
    N = mat.shape[0]

    # obtain the rank of i_th text embedding among all embeddings
    # that is the rank of the corresponding text for the given video
    ranks = np.array([np.where(np.argsort(mat[i]) == i)[0][0] for i in range(N)])
        
    recall_1  = 100.0 * len(np.where(ranks < 1)[0])/N
    recall_5  = 100.0 * len(np.where(ranks < 5)[0])/N
    recall_10 = 100.0 * len(np.where(ranks < 10)[0])/N
    mean_rank = ranks.mean() + 1
    med_rank = np.floor(np.median(ranks)) + 1
    
    metrics = [recall_1, recall_5, recall_10, mean_rank, med_rank]
    metrics = [round(m,2) for m in metrics]
    
    return metrics, ranks


def normalize_metrics(metrics_experiment, n_samples_experiment, n_samples_baseline = 1000):
    # the resulting normalized metrics compare against a baseline with 1000 samples
    # R@1, R@5, R@10, mean_rank, median_rank
    # [0.1, 0.5, 1, 500, 500]
    
    metrics_experiment = np.array(metrics_experiment)
    
    recall_at_k_normed = metrics_experiment[:3]*n_samples_experiment/n_samples_baseline
    ranks_normed = metrics_experiment[3:5]*n_samples_baseline/n_samples_experiment
    
    metrics_normed = [round(r, 2) for r in recall_at_k_normed]
    metrics_normed.extend([int(r) for r in ranks_normed])
    
    return metrics_normed


def calc_cosine_distance(embs_mode1, embs_mode2):      
    return np.dot(embs_mode1, embs_mode2.T)


def calc_l2_distance(embs_mode1, embs_mode2):
    return np.linalg.norm(embs_mode1[:, None, :] - embs_mode2[None, :, :], axis=-1)


  