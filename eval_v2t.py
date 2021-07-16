from data_provider import TempuckeyVideoSentencePairsDataset as TempuckeyDataset
from data_provider import Normalize_VideoSentencePair
from layers.v2t import V2T
import utilities as utils

from numpy import linalg
import numpy as np
import argparse
import pdb
import torch 
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def encode_data_v2t(data_loader, model_v2t):
    '''
    Input Parameters:
        split_path: data split path for data loader
        model_v2t: a video encoder/decoder model
    Returns
        list of ids, encoded video embeddeings, encoded text embeddings
    '''
   
    ids_list = []
    pred_t_list = []
    orig_t_list = []
    
    model_v2t.eval()
    
    for sample in data_loader:
        i = sample['id']
        v = sample['video']
        t = sample['sent']

        ids_list.extend(i)
        orig_t_list.extend(t.numpy())

        with torch.no_grad():
            pred_t = model_v2t(v)
    
        pred_t_list.extend(pred_t.cpu().numpy())
    
    pred_t_list = torch.from_numpy(np.array(pred_t_list)).squeeze()
    orig_t_list = torch.tensor(orig_t_list)
    assert len(ids_list) == len(pred_t_list)

    return ids_list, pred_t_list, orig_t_list    

def encode_data(data_loader, model_v, model_t):
    '''
    Input Parameters:
        split_path: data split path for data loader
        model_v: a video encoder/decoder model
        model_t: a text  encoder/decoder model
    Returns
        list of ids, encoded video embeddeings, encoded text embeddings
    '''
   
    ids = []
    embeddings_t = []
    embeddings_v = []
    
    model_v.eval()
    model_t.eval()
    
    for sample in data_loader:
        i = sample['id']
        v = sample['video']
        t = sample['sent']

        with torch.no_grad():
            embs_t = model_t.encoder(t)
            embs_v = model_v.encoder(v)
    
        ids.extend(i)
        embeddings_t.extend(embs_t.numpy())
        embeddings_v.extend(embs_v.numpy())
    
    embeddings_v = torch.from_numpy(np.array(embeddings_v)).squeeze()
    embeddings_t = torch.from_numpy(np.array(embeddings_t)).squeeze()
    assert len(ids) == len(embeddings_t) and len(ids) == len(embeddings_v)

    return ids, embeddings_v, embeddings_t    
    

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
    metrics = [int(m) for m in metrics]
    
    return metrics, ranks


def calc_cosine_distance(embs_mode1, embs_mode2):      
    return np.dot(embs_mode1, embs_mode2.T)


def calc_l2_distance(embs_mode1, embs_mode2):
    return np.linalg.norm(embs_mode1[:, None, :] - embs_mode2[None, :, :], axis=-1)


def load_model(model_path, n_feats):
    model = V2T(n_feats)
    model_file = open(model_path, 'rb')
    model_sd = torch.load(model_file)
    model.load_state_dict(model_sd)
    model.to(device)
    
    return model
    

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
    
if __name__ == '__main__':
    ### python -W ignore eval.py --experiment_name=experiment_0.0001_1.0_0.1_0.0001_32_20_1x512_1x2048_5bb195b6a8c54698aeb198df073e2535 --split_path='valid.split.pkl'
    
    parser = argparse.ArgumentParser ()
      
    # batch_size
    parser.add_argument('--batch_size', type = int, default = 64, help = 'batch size')

    # feat sequence length
    parser.add_argument('--t_feat_len', type = int, default = 1, help = 'length of feat vector')
    parser.add_argument('--v_feat_len', type = int, default = 1, help = 'length of feat vector')

    # num feats
    parser.add_argument('--t_num_feats', type = int, default = 512, help = 'number of feats in each vector')
    parser.add_argument('--v_num_feats', type = int, default = 2048, help = 'number of feats in each vector')
    
    # data path params
    parser.add_argument('--data_dir', default = '/usr/local/data02/zahra/datasets/Tempuckey/sentence_segments')
    parser.add_argument('--video_feats_dir', default = 'feats/video/r2plus1d_resnet50_kinetics400')
    parser.add_argument('--text_feats_path', default = 'feats/text/universal/sentence_feats.pkl')
    parser.add_argument('--split_path', default = 'valid.split.pkl')
    
    # repo path params
    parser.add_argument('--repo_dir', default = '/usr/local/extstore01/zahra/Video-Text-Retrieval_OOD')
    parser.add_argument('--output_dir', default = '/usr/local/extstore01/zahra/Video-Text-Retrieval_OOD/output')
    parser.add_argument('--experiment_name', default = 'experiment_0.01_10.0_0.1_0.0004_15_1x512_1x2048_16f00bd96e25498bbc63d09cc4d64067', help = 'trained model name')

    args = parser.parse_args()
    
    T = args.v_feat_len
    L = args.t_feat_len
      
    batch_size = args.batch_size
    
    experiment_name = args.experiment_name
    
    dl_params = {'batch_size': batch_size,
        'shuffle': False,
        'num_workers': 1}
        
    data_dir = args.data_dir
    v_feats_dir = f'{data_dir}/{args.video_feats_dir}'
    t_feats_path = f'{data_dir}/{args.text_feats_path}'
    split_path = f'{data_dir}/{args.split_path}'
    
    repo_dir = args.repo_dir
    output_path = args.output_dir
    
    experiment_names = [experiment_name]     
# 'experiment_0.005053182667988411_1.3094966900369656_0.1_0.007414787983815832_16_2000_1x512_1x2048_01ee777a7fb7403eb16d8ec2fdfbbfa9']
    
    # evaluate model using different noise thresholds
    for relevance_score in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]:
        for experiment_name in experiment_names:
            print(f'----------------------\n\n{experiment_name},\nrelevance {relevance_score}')
            model_v_path = f'{repo_dir}/output/experiments/{experiment_name}/model_v_{experiment_name}.sd'

            model_v2t = load_model(model_v_path, args.v_num_feats)

            data_ids = utils.load_picklefile(split_path)
            dataset = TempuckeyDataset(v_feats_dir, t_feats_path, data_ids, video_feat_seq_len=1, sent_feat_seq_len=1, transform=None, relevance_score=relevance_score)
            data_loader = torch.utils.data.DataLoader(dataset, **dl_params)

            ids, pred_t, orig_t = encode_data_v2t(data_loader, model_v2t)

            dist_matrix_v2t = calc_l2_distance(orig_t, pred_t)
            metrics_v2t, ranks_v2t = get_metrics(dist_matrix_v2t)

            # print(metrics_v2t)
            print(normalize_metrics(metrics_v2t, n_samples_experiment = dataset.__len__()))

    #import pickle as pkl
    #sents = pkl.load(open('/usr/local/data02/zahra/datasets/Tempuckey/sentence_segments/sentences.pkl','rb'))
    #pdb.set_trace()
    