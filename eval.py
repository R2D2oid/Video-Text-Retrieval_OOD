from data_provider import TempuckeyVideoSentencePairsDataset as TempuckeyDataset
from data_provider import Normalize_VideoSentencePair
from layers.AE import AE
import utilities as utils

from numpy import linalg
import numpy as np
import argparse
import pdb
import torch 
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def encode_data(split_path, model_v, mode_t):
    '''
    Input Parameters:
        split_path: data split path for data loader
        model_v: a video encoder/decoder model
        model_t: a text  encoder/decoder model
    Returns
        list of ids, encoded video embeddeings, encoded text embeddings
    '''
    data_ids = utils.load_picklefile(split_path)
    dataset = TempuckeyDataset(v_feats_dir, t_feats_path, data_ids, video_feat_seq_len=1, sent_feat_seq_len=1, transform=[Normalize_VideoSentencePair()])
    data_loader = torch.utils.data.DataLoader(dataset, **dl_params)
    
    ids = []
    embeddings_t = []
    embeddings_v = []
    
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
    
    
def calc_cosine_distance(embeddings_v, embeddings_t):
    return np.dot(embeddings_t, embeddings_v.T)


def load_model(model_path, n_feats):
    model = AE(n_feats)
    model_file = open(model_path, 'rb')
    model_sd = torch.load(model_file)
    model.load_state_dict(model_sd)
    model.to(device)
    
    return model
    
    
if __name__ == '__main__':
    ### python eval.py --experiment_name=experiment_0.01_10.0_0.1_0.0004_15_1x512_1x2048_7c1b5c8129b949209352c35b20983037

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
    parser.add_argument('--train_split_path', default = 'train.split.pkl')
    parser.add_argument('--valid_split_path', default = 'valid.split.pkl')
    parser.add_argument('--test_split_path', default = 'test.split.pkl')
    
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
    train_split_path = f'{data_dir}/{args.train_split_path}'
    v_feats_dir = f'{data_dir}/{args.video_feats_dir}'
    t_feats_path = f'{data_dir}/{args.text_feats_path}'
    train_split_path = f'{data_dir}/{args.train_split_path}'
    test_split_path = f'{data_dir}/{args.test_split_path}'
    
    repo_dir = args.repo_dir
    output_path = args.output_dir
    model_v_path = f'{repo_dir}/output/experiments/{experiment_name}/model_v_{experiment_name}.sd'
    model_t_path = f'{repo_dir}/output/experiments/{experiment_name}/model_t_{experiment_name}.sd'

    model_v = load_model(model_v_path, args.v_num_feats)
    model_t = load_model(model_t_path, args.t_num_feats)
    
    ids, embs_v, embs_t = encode_data(test_split_path, model_v, model_t)
    dist_matrix = calc_cosine_distance(embs_v, embs_t)
    