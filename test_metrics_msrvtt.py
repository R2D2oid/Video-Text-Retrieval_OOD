import torch
import argparse
import numpy as np

from models.VT import VT
from msrvtt_dataset import MSRVTTDataset as MSRVTT
from msrvtt_dataset import Standardize_VideoSentencePair, ToTensor_VideoSentencePair
from eval import get_metrics, calc_l2_distance, compute_metrics
from utils.train_utils import validation_metrics, normalize_metrics, get_dataloader_msrvtt

'''
python test_metrics_msrvtt.py \
    --batch-size 128 \
    --exp-name experiment_loss_cross_correlation_lr_9.5e-05_wdecay_0.007323_bsz_128_1628109801
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser ()
        
    # num feats
    parser.add_argument('--t_num_feats', type = int, default = 512, help = 'number of feats in each vector')
    parser.add_argument('--v_num_feats', type = int, default = 2048, help = 'number of feats in each vector')
    parser.add_argument('--batch-size', type = int, default = 128, help = 'number of feats in each vector')

    # io params
    parser.add_argument('--repo_dir', default = '/usr/local/extstore01/zahra/datasets/MSRVTT')
    parser.add_argument('--video_feats_dir', default = 'feats/video/r2plus1d_Test')
    parser.add_argument('--text_feats_path', default = 'feats/text/msrvtt_captions_universal_test.pkl')
    parser.add_argument('--test_split_path', default = 'Test_videoid_sentid.txt')
    parser.add_argument('--output-path', default = '/usr/local/extstore01/zahra/VTR_OOD/output_msrvtt')
    parser.add_argument('--exp-name', type=str, help='trained model name')
    
    parser.add_argument('--projector', default='1024-1024-1024', type=str, metavar='MLP', help='projector MLP')
    parser.add_argument('--loss-criterion', default='cross_correlation', type=str, help='loss function')
    
    args = parser.parse_args()
    
    n_feats_t = args.t_num_feats
    n_feats_v = args.v_num_feats
    
    repo_dir = args.repo_dir
    test_split_path = f'{repo_dir}/{args.test_split_path}'
    v_feats_dir = f'{repo_dir}/{args.video_feats_dir}'
    t_feats_path = f'{repo_dir}/{args.text_feats_path}'
    
    batch_size = args.batch_size
    exp_name = args.exp_name
    output_path = args.output_path
    model_path = f'{output_path}/experiments/{exp_name}/model_vt.sd'
        
    dl_params = {'batch_size': batch_size,
                 'shuffle': False,
                 'num_workers': 1}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = VT(args)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    dataloader = get_dataloader_msrvtt(test_split_path, v_feats_dir, t_feats_path, dl_params)

    embeddings_v = []
    embeddings_t = []
    
    for y1,y2 in dataloader:
        y1 = y1.cuda()
        y2 = y2.cuda()
        v_rep,t_rep = model.get_v_and_t_representation(y1,y2)
        v_rep = v_rep.detach()
        t_rep = t_rep.detach()

        embeddings_v.extend(v_rep)
        embeddings_t.extend(t_rep)      

    embeddings_v = torch.stack(embeddings_v, dim=0).cpu().numpy()
    embeddings_t = torch.stack(embeddings_t, dim=0).cpu().numpy()

    # remove duplicate videos
    n_caption = 20
    idx = range(0, embeddings_v.shape[0], n_caption)
    
    # keep only one video/sentence correpondence for simplicity (just for now)
    embeddings_v = embeddings_v[idx,:]
    embeddings_t = embeddings_t[idx,:]
    n_caption = 1
    
    # dist = calc_l2_distance(embeddings_v, embeddings_t)
    dist = np.dot(embeddings_v, embeddings_t.T)
    metrics, ranks = get_metrics(dist)
    print(metrics)
    
    
    
