import argparse
import math
import torch 
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
import logging
from datetime import datetime as dt
import utilities as utils
from layers.t2v import T2V
from data_provider import TempuckeyVideoSentencePairsDataset as TempuckeyDataset
from data_provider import Normalize_VideoSentencePair
from eval_t2v import calc_l2_distance, get_metrics, encode_data_t2v
from layers.contrastive_loss import ContrastiveLoss

# torch.set_default_tensor_type(torch.cuda.FloatTensor)

# init tensorboard
writer = SummaryWriter('runs/')
torch.manual_seed(42)

# init logging
logfile = 'logs/logfile_{}.log'.format(dt.now())
logformat = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
loglevel = 30 ## levels: NOTSET = 0 | DEBUG = 10 | INFO = 20 | WARNING = 30 | ERROR = 40 | CRITICAL = 50
logging.basicConfig (
    filename = logfile.format (dt.now().date()),
    level = loglevel,
    format = logformat)

logging.getLogger ().addHandler (logging.StreamHandler())
logger = logging.getLogger()
    
################################################

def optimize_t2v_model(lr, lr_step_size, weight_decay, batch_size_exp, relevance_score):
  
    # use batch_size provided by bayes_opt as 2**int(value)
    batch_size = int(np.power(2,int(batch_size_exp)))
    
    dl_params = {'batch_size': batch_size,
                 'shuffle': shuffle,
                 'num_workers': 1}
    
    lr_step_size = int(lr_step_size)
    
    # display experiment info
    exp_info = get_experiment_info(lr, lr_step_size, weight_decay, lr_gamma, n_epochs, n_feats_t, n_feats_v, T, L, batch_size)
    logger.info(exp_info)
    
    # get data loaders for train and valid sets
    dataloader_train = get_data_loader(train_split_path, v_feats_dir, t_feats_path, relevance_score, dl_params)
    dataloader_valid = get_data_loader(valid_split_path, v_feats_dir, t_feats_path, relevance_score, dl_params)

    # get experiment name 
    _, exp_name = log_experiment_info(lr, lr_step_size, weight_decay, lr_gamma, n_epochs, n_feats_t, n_feats_v, T, L, batch_size, relevance_score, shuffle, loss_criterion, write_it=False)
    
    # train 
    torch.set_grad_enabled(True)
    model, train_loss = train_model(dataloader_train, dataloader_valid, lr, lr_step_size, weight_decay, lr_gamma, n_epochs, n_feats_t, n_feats_v, T, L, dl_params, exp_name)
    
    # loss is nan
    if model is None:
        logger.warning('NaN encountered in loss... Moving on to the next iteration of bayes_opt!')
        return -10
    
    # calculate loss on validation
    valid_loss = evaluate_validation(dataloader_valid, model)
       
    # log experiment meta data 
    exp_dir, exp_name = log_experiment_info(lr, lr_step_size, weight_decay, lr_gamma, n_epochs, n_feats_t, n_feats_v, T, L, batch_size, relevance_score, shuffle, loss_criterion, write_it=True)
    
    # save trained model, training losses, and validation losses
    save_experiment(model, None, train_loss, exp_dir, exp_name)
    logger.warning(f'saved model_t2v and train/valid loss to {exp_dir}')
    
    metrics_train, _, _ = validation_metrics(dataloader_train, model)
    recall_at_1_train = metrics_train[0]

    #metrics_valid, ranks_valid, dist_matrix = validation_metrics(dataloader_valid, model)
    #recall_at_1_valid = metrics_valid[0]
    
    logger.warning(f'loss train: {train_loss}')
    logger.warning(f'recall_at_1 train: {recall_at_1_train}')
    
    return recall_at_1_train

########################################

def normalize_metrics(metrics_experiment, n_smples_experiment, n_smples_baseline = 1000):
    metrics_experiment = np.array(metrics_experiment)
    
    recall_at_k_normed = metrics_experiment[:3]*n_smples_experiment/n_smples_baseline
    ranks_normed = metrics_experiment[3:5]*n_smples_baseline/n_smples_experiment
    
    metrics_normed = [round(r, 2) for r in recall_at_k_normed]
    metrics_normed.extend([int(r) for r in ranks_normed])
    return metrics_normed

########################################

def validation_metrics(data_loader, model):
    ids, pred_v, orig_v = encode_data_t2v(data_loader, model)
    dist_matrix = calc_l2_distance(orig_v, pred_v)
    metrics, ranks = get_metrics(dist_matrix)
    metrics_norm = normalize_metrics(metrics, n_smples_experiment=data_loader.__len__(), n_smples_baseline = 1000)
    
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

################################

def evaluate_validation(data_loader, model):
    criterion, target_tensor = instantiate_loss_criterion(loss_criterion, temperature=temperature)

    model.eval()

    # run trained model on validation
    for sample in data_loader:
        v = sample['video']
        t = sample['sent']
        
        v = torch.tensor(v).float().cuda()
        t = torch.tensor(t).float().cuda()

        with torch.no_grad():
            # forward
            pred_v = model(t)

            if loss_criterion == 'contrastive':
                pred_v = pred_v.unsqueeze(dim=1)
                v = v.unsqueeze(dim=1)
                vv = torch.cat([v,pred_v], dim=1) 
                loss = criterion(vv)
            elif loss_criterion == 'cosine':
                loss = criterion(pred_v, v, target_tensor)
            else: # mse
                loss = criterion(pred_v, v)
    return loss 
    
########################################

def instantiate_loss_criterion(loss_criterion, temperature):
    
    target_tensor = None
    
    if loss_criterion == 'cosine':
        # target_tensor = torch.Tensor(1) # use 1 to train for bringing together corresponding (positive) vectors
        # target_tensor = torch.Tensor(-1) # use -1 to train for pushing apart dissimilar (negative) vectors
        criterion = nn.CosineEmbeddingLoss()
        # the cosine embedding loss takes a target y=1 for training positive (similar) vectors and y=-1 for training dissimilar (negative) vectors
        target_tensor = torch.Tensor(1)
    elif loss_criterion=='contrastive':
        #temp = 1
        #criterion = ContrastiveLoss(temperature=temp)
        criterion = ContrastiveLoss(temperature=temperature, contrast_mode='all', base_temperature=temperature)
    else:
        criterion = nn.MSELoss()
        
    return criterion, target_tensor

########################################

def get_data_loader(split_path, v_feats_dir, t_feats_path, relevance_score, dl_params):
    ids = utils.load_picklefile(split_path)
    dataset = TempuckeyDataset(v_feats_dir, t_feats_path, ids, video_feat_seq_len=T, sent_feat_seq_len=L, transform=None, relevance_score=relevance_score)
    data_loader = torch.utils.data.DataLoader(dataset, **dl_params)
    
    return data_loader

########################################

def train_model(data_loader_train, data_loader_valid, lr, lr_step_size, weight_decay, lr_gamma, n_epochs, n_feats_t, n_feats_v, T, L, dl_params, exp_name):   
             
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_samples = data_loader_train.__len__()
    
    ### instantiate t2v model
    model_t2v = T2V(n_feats_t)

    model_t2v.to(device)
    
    # Adam optimizer
    optimizer_t2v = torch.optim.Adam(model_t2v.parameters(), lr = lr, weight_decay = weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_t2v, step_size = lr_step_size, gamma = lr_gamma)
    
    losses_avg = []
    flag = True
    temperature_init = 10.0
    counter = 0
    ### train the model
    for epoch in range(n_epochs):
        model_t2v.train()

        global temperature 
        temperature = max(temperature_init-epoch, 1.0)
        logger.info(f'Contrastive Loss Temperature is {temperature} at epoch {epoch}')
        
        criterion, target_tensor = instantiate_loss_criterion(loss_criterion, temperature = temperature)
        
        losses = []
        counter = 1    
        for sample in data_loader_train:
            v = sample['video']
            t = sample['sent']

            v = torch.tensor(v).float().cuda()
            t = torch.tensor(t).float().cuda()
            
            # forward
            pred_v = model_t2v(t)
            
            if loss_criterion == 'contrastive':
                pred_v = pred_v.unsqueeze(dim=1)
                v = v.unsqueeze(dim=1)
                vv = torch.cat([v,pred_v], dim=1) 
                loss = criterion(vv)
            elif loss_criterion == 'cosine':
                loss = criterion(pred_v, v, target_tensor)
            else: # mse
                loss = criterion(pred_v, v)
                
            # backprop and optimize
            optimizer_t2v.zero_grad()
            loss.backward()
            optimizer_t2v.step()
            
            logger.debug('Epoch[{}/{}], Step[{}/{}] Loss: {}\n'.format(epoch + 1, n_epochs, counter, num_samples, loss.item()))

            counter = counter + 1 
            
            losses.append(loss.item())
            
        import math
        if math.isnan(loss):
            return None, None
        
        lr_value = lr_scheduler.optimizer.param_groups[0]['lr']
        writer.add_scalar(f'{exp_name}/train/lr', lr_value, epoch)
        lr_scheduler.step()
        
        loss_avg = np.array(losses).mean()
        losses_avg.append(loss_avg)
        logger.info(f'epoch[{epoch + 1}/{n_epochs}]')
           
        train_loss = loss_avg
        #valid_loss = evaluate_validation(data_loader_valid, model_t2v)
        #logger.warning(f'loss valid/train: {valid_loss}/{train_loss}')
        logger.info(f'    loss train: {train_loss}')

#         t2v_metrics_valid, _, _ = validation_metrics(data_loader_valid, model_t2v)
#         recall_at_1_valid, recall_at_5_valid, recall_at_10_valid, _, _ = t2v_metrics_valid
#         logger.warning(f'    recall_at_1/5/10 valid: {recall_at_1_valid}/{recall_at_5_valid}/{recall_at_10_valid}')

#         t2v_metrics_train, _, _ = validation_metrics(data_loader_train, model_t2v)
#         recall_at_1_train, recall_at_5_train, recall_at_10_train, _, _ = t2v_metrics_train

#         logger.warning(f'    recall_at_1/5/10 train: {recall_at_1_train}/{recall_at_5_train}/{recall_at_10_train}')

        if flag == True:
            writer.add_graph(model_t2v, t)
            flag = False
                
        # write train and valid loss to tensorboard
        writer.add_scalar(f'{exp_name}/loss/train', loss, epoch)
        
    writer.flush()
    return model_t2v, loss_avg

########################################

if __name__ == '__main__':
    ### python -W ignore train_t2v.py --n_epochs 15 --t_num_feats 512 --v_num_feats 2048 
    # python -W ignore train_t2v.py --n_epochs 500 --t_num_feats 512 --v_num_feats 2048 --batch_size_exp_min 7 --batch_size_exp_max 7 --lr_min 0.0001 --lr_max 0.001 --weight_decay_min 0.00001 --weight_decay_max 0.001 --lr_step_size_min 50 --lr_step_size_max 400 --lr_gamma 0.9 --relevance_score_min 0.3 --relevance_score_max 0.31 --shuffle --loss_criterion contrastive

    parser = argparse.ArgumentParser ()
    parser.add_argument('--n_epochs', type = int, default = 20, help = 'number of iterations')
    parser.add_argument('--n_train_samples', type = int, default = None, help = 'number of training samples')
        
    # loss criterion
    parser.add_argument('--loss_criterion', default = 'mse') # MSELoss
    
    # lr step size
    parser.add_argument('--lr_step_size_min', type = int, default = 50, help = 'lr schedule: step size lower bound')
    parser.add_argument('--lr_step_size_max', type = int, default = 400, help = 'lr schedule: step size upper bound')
    
    # lr gamma
    parser.add_argument('--lr_gamma', type = float, default = 0.8, help = 'lr schedule: gamma')
    
    # lr
    parser.add_argument('--lr_min', type = float, default = 0.00001, help = 'learning rate lower bound')
    parser.add_argument('--lr_max', type = float, default = 0.001, help = 'learning rate upper bound')
    
    # weight decay
    parser.add_argument('--weight_decay_min', type = float, default = 0.00001, help = 'weight decay lower bound')
    parser.add_argument('--weight_decay_max', type = float, default = 0.001, help = 'weight decay upper bound')
    
    # batch size
    parser.add_argument('--batch_size_exp_min', type = int, default = 5, help = 'batch size exponent lower bound; batch_size=2**n')
    parser.add_argument('--batch_size_exp_max', type = int, default = 7, help = 'batch size exponent upper bound; batch_size=2**n')
    
    # num feats
    parser.add_argument('--t_num_feats', type = int, default = 512, help = 'number of feats in each vector')
    parser.add_argument('--v_num_feats', type = int, default = 2048, help = 'number of feats in each vector')

    # feat sequence length
    parser.add_argument('--t_feat_len', type = int, default = 1, help = 'length of feat vector')
    parser.add_argument('--v_feat_len', type = int, default = 1, help = 'length of feat vector')
    
    # bayesian optimization parameters
    parser.add_argument('--bayes_n_iter', type = int, default = 5, help = 'bayesian optimization num iterations')
    parser.add_argument('--bayes_init_points', type = int, default = 5, help = 'bayesian optimization init points')
    
    # io params
    parser.add_argument('--repo_dir', default = '/usr/local/data02/zahra/datasets/Tempuckey/sentence_segments')
    parser.add_argument('--video_feats_dir', default = 'feats/video/r2plus1d_resnet50_kinetics400')
    parser.add_argument('--text_feats_path', default = 'feats/text/universal/sentence_feats.pkl')
    parser.add_argument('--train_split_path', default = 'train_valid.split.pkl')    
    parser.add_argument('--valid_split_path', default = 'valid.split.pkl')
    parser.add_argument('--output_path', default = '/usr/local/extstore01/zahra/Video-Text-Retrieval_OOD/output')

    parser.add_argument('--relevance_score_min', type = float, default = 0.05, help = 'relevance score in range (0.0, 1.0)')
    parser.add_argument('--relevance_score_max', type = float, default = 0.7, help = 'relevance score in range (0.0, 1.0)')
    
    parser.add_argument('--shuffle', dest='shuffle', action='store_true')
    
    args = parser.parse_args()
    
    logger.info(args)

    relevance_score_min = args.relevance_score_min
    relevance_score_max = args.relevance_score_max
    
    lr_min = args.lr_min
    lr_max = args.lr_max
    
    lr_step_size_min = args.lr_step_size_min
    lr_step_size_max = args.lr_step_size_max
    
    lr_gamma = args.lr_gamma
    
    weight_decay_min = args.weight_decay_min
    weight_decay_max = args.weight_decay_max
    
    batch_size_exp_min = args.batch_size_exp_min
    batch_size_exp_max = args.batch_size_exp_max
    
    shuffle = args.shuffle
    
    n_epochs = args.n_epochs
    n_train_samples = args.n_train_samples
    
    n_feats_t = args.t_num_feats
    n_feats_v = args.v_num_feats
    T = args.v_feat_len
    L = args.t_feat_len
        
    repo_dir = args.repo_dir
    train_split_path = f'{repo_dir}/{args.train_split_path}'
    valid_split_path = f'{repo_dir}/{args.valid_split_path}'
    output_path = args.output_path
    v_feats_dir = f'{repo_dir}/{args.video_feats_dir}'
    t_feats_path = f'{repo_dir}/{args.text_feats_path}'
    
    ## bayes opt
    bayes_init_points = args.bayes_init_points
    bayes_n_iter = args.bayes_n_iter
       
    loss_criterion = args.loss_criterion
    
    # bounds of parameter space
    pbounds = {'lr': (lr_min, lr_max), 
               'lr_step_size': (lr_step_size_min, lr_step_size_max), 
               'weight_decay':(weight_decay_min, weight_decay_max), 
               'batch_size_exp': (batch_size_exp_min, batch_size_exp_max), 
               'relevance_score': (relevance_score_min,relevance_score_max)
              }

    optimizer = BayesianOptimization(
        f=optimize_t2v_model,
        pbounds=pbounds,
        random_state=42,
    )

    optimizer.maximize(
        init_points=bayes_init_points,
        n_iter=bayes_n_iter,
    )

