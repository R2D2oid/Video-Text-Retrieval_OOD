import utilities as utils
from data_provider import TempuckeyVideoSentencePairsDataset as TempuckeyDataset
from eval_v2t import calc_l2_distance, get_metrics, encode_data_v2t
import torch
import numpy as np

def normalize_metrics(metrics_experiment, n_smples_experiment, n_smples_baseline = 1000):
    metrics_experiment = np.array(metrics_experiment)
    
    recall_at_k_normed = metrics_experiment[:3]*n_smples_experiment/n_smples_baseline
    ranks_normed = metrics_experiment[3:5]*n_smples_baseline/n_smples_experiment
    
    metrics_normed = [round(r, 2) for r in recall_at_k_normed]
    metrics_normed.extend([int(r) for r in ranks_normed])
    return metrics_normed

########################################

def validation_metrics(data_loader, model):
    ids, pred_v, orig_v = encode_data_v2t(data_loader, model)
    dist_matrix = calc_l2_distance(orig_v, pred_v)
    metrics, ranks = get_metrics(dist_matrix)
    metrics_norm = normalize_metrics(metrics, n_smples_experiment=data_loader.dataset.__len__(), n_smples_baseline = 1000)
    
    return metrics_norm, ranks, dist_matrix

########################################
def save_experiment(model, valid_loss, train_loss, exp_dir, exp_name):
    # save models
    torch.save(model.state_dict(), f'{exp_dir}/model_v_{exp_name}.sd')
    
    # save valid losses
    utils.dump_picklefile(valid_loss, f'{exp_dir}/losses_validation_{exp_name}.pkl')
    
########################################

def log_experiment_info(lr, lr_step_size, weight_decay, lr_gamma, n_epochs, n_feats_t, n_feats_v, T, L, batch_size, relevance_score, shuffle, loss_criterion, write_it=True):
    import uuid
    random_hash = uuid.uuid4().hex

    shuffle_flag = 'yes' if shuffle else 'no'
    exp_name = f'experiment_t2v_shuffle_{shuffle_flag}_loss_{loss_criterion}_lr_{round(lr,6)}_lr_step_{round(lr_step_size,6)}_gamma_{round(lr_gamma,6)}_wdecay_{round(weight_decay,6)}_bsz_{batch_size}_epochs_{n_epochs}_relevance_{round(relevance_score,2)}_{L}x{n_feats_t}_{T}x{n_feats_v}_{random_hash}'
    exp_dir = f'{output_path}/experiments/{exp_name}'
    
    if write_it:
        utils.create_dir_if_not_exist(exp_dir)

        info_path = f'{exp_dir}/experiment_info.txt'
        info = get_experiment_info(lr, lr_step_size, weight_decay, lr_gamma, n_epochs, n_feats_t, n_feats_v, T, L, batch_size)
        utils.dump_textfile(info, info_path)
    
    return exp_dir, exp_name

########################################

def get_experiment_info(lr, lr_step_size, weight_decay, lr_gamma, n_epochs, n_feats_t, n_feats_v, T, L, batch_size):
    
    info = []
    
    for item,val in args.__dict__.items():
        info.append(f'{item}: {val}')
    info.append(f'lr: {lr}')
    info.append(f'lr_step_size: {lr_step_size}')
    info.append(f'weight_decay: {weight_decay}')
    info.append(f'batch_size: {batch_size}')

    return info

########################################

def get_data_loader(split_path, v_feats_dir, t_feats_path, relevance_score, dl_params):
    ids = utils.load_picklefile(split_path)
    dataset = TempuckeyDataset(v_feats_dir, t_feats_path, ids, video_feat_seq_len=1, sent_feat_seq_len=1, transform=None, relevance_score=relevance_score)
    data_loader = torch.utils.data.DataLoader(dataset, **dl_params)
    
    return data_loader
