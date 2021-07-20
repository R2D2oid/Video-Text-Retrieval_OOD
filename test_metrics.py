from model_utils import validation_metrics, normalize_metrics, get_data_loader
from layers.v2t import V2T
import torch
import argparse

def get_results(model, split_path, v_feats_dir, t_feats_path, relevance_score, dl_params):
    dataloader = get_data_loader(split_path, v_feats_dir, t_feats_path, relevance_score, dl_params)
    metrics, _, _ = validation_metrics(dataloader, model)

    return metrics

# python test_metrics.py --test_split_path test.clean.split.pkl
if __name__ == '__main__':
    parser = argparse.ArgumentParser ()
        
    # num feats
    parser.add_argument('--t_num_feats', type = int, default = 512, help = 'number of feats in each vector')
    parser.add_argument('--v_num_feats', type = int, default = 2048, help = 'number of feats in each vector')
    
    # io params
    parser.add_argument('--repo_dir', default = '/usr/local/data02/zahra/datasets/Tempuckey/sentence_segments')
    parser.add_argument('--video_feats_dir', default = 'feats/video/r2plus1d_resnet50_kinetics400')
    parser.add_argument('--text_feats_path', default = 'feats/text/universal/sentence_feats.pkl')
    parser.add_argument('--test_split_path', default = 'test.split.pkl')
    parser.add_argument('--output_path', default = '/usr/local/extstore01/zahra/Video-Text-Retrieval_OOD/output')
    parser.add_argument('--model_name', default = 'experiment_shuffle_yes_loss_mse_lr_0.000956_lr_step_306_gamma_0.9_wdecay_0.000164_bsz_128_epochs_500_relevance_0.3_1x512_1x2048_f9f7b433923b4d97b0c6e75062bf6931')
    
    args = parser.parse_args()
        
    batch_size = 128
    relevance_score = 0.0
    shuffle = False
    
    dl_params = {'batch_size': batch_size,
                 'shuffle': shuffle,
                 'num_workers': 1}
    
    n_feats_t = args.t_num_feats
    n_feats_v = args.v_num_feats
        
    repo_dir = args.repo_dir
    test_split_path = f'{repo_dir}/{args.test_split_path}'
    output_path = args.output_path
    v_feats_dir = f'{repo_dir}/{args.video_feats_dir}'
    t_feats_path = f'{repo_dir}/{args.text_feats_path}'
    
    model_path = f'output/experiments/{args.model_name}/model_v_{args.model_name}.sd'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ### instantiate t2v model
    model_v = V2T(n_feats_v)
    model_v.load_state_dict(torch.load(model_path))
    model_v.to(device)
    model_v.eval()
    
    output = get_results(model_v, test_split_path, v_feats_dir, t_feats_path, relevance_score, dl_params)
    print(output)


    
    
    
    
