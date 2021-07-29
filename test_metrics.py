import torch
import argparse
from utils.train_utils import validation_metrics, normalize_metrics, get_dataloader
from eval import get_metrics, calc_l2_distance
from models.VT import VT
                                                 
# python test_metrics.py --test_split_path test.clean.split.pkl --batch-size 128
if __name__ == '__main__':
    parser = argparse.ArgumentParser ()
        
    # num feats
    parser.add_argument('--t_num_feats', type = int, default = 512, help = 'number of feats in each vector')
    parser.add_argument('--v_num_feats', type = int, default = 2048, help = 'number of feats in each vector')
    
    parser.add_argument('--batch-size', type = int, default = 128, help = 'number of feats in each vector')

    # io params
    parser.add_argument('--repo_dir', default = '/usr/local/data02/zahra/datasets/Tempuckey/sentence_segments')
    parser.add_argument('--video_feats_dir', default = 'feats/video/r2plus1d_resnet50_kinetics400')
    parser.add_argument('--text_feats_path', default = 'feats/text/universal/sentence_feats.pkl')
    parser.add_argument('--test_split_path', default = 'test.split.pkl')
    parser.add_argument('--output_path', default = '/usr/local/extstore01/zahra/VTR_OOD/output')
    
    parser.add_argument('--projector', default='1024-1024-1024', type=str, metavar='MLP', help='projector MLP')

    args = parser.parse_args()
        
    experiment_name = 'experiment_shuffle_no_loss_None_lr_0.000694_lr_step_140_gamma_0.8_wdecay_0.000614_bsz_32_epochs_20_relevance_0.59_1x512_1x2048_81c7da41544446279c801985a7b20ef8'
    model_path = f'/usr/local/extstore01/zahra/VTR_OOD/output/{experiment_name}/model_vt.sd'
    
    batch_size = args.batch_size
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
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VT(args)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    dataloader = get_dataloader(test_split_path, v_feats_dir, t_feats_path, relevance_score, dl_params)

    embeddings_v = []
    embeddings_t = []
    
    for y1,y2 in dataloader:
        y1 = y1.cuda()
        y2 = y2.cuda()
        v_rep,t_rep = model.get_v_and_t_representation(y1,y2)
        v_rep = v_rep.detach().cpu()
        t_rep = t_rep.detach().cpu()

        embeddings_v.extend(v_rep)
        embeddings_t.extend(t_rep)
        
        break

    embeddings_v = torch.stack(embeddings_v, dim=0).numpy()
    embeddings_t = torch.stack(embeddings_t, dim=0).numpy()
    dist_matrix = calc_l2_distance(embeddings_v, embeddings_t)
    metrics, ranks = get_metrics(dist_matrix)
    metrics_norm = normalize_metrics(metrics, n_smples_experiment=dataloader.dataset.__len__(), n_smples_baseline = 1000)
    
    print(metrics_norm)

    
    
    
    
