import argparse
import math
import logging
import time
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from bayes_opt import BayesianOptimization
from datetime import datetime as dt
from torchvision import transforms

from models.VT import VT 
from msrvtt_dataset import MSRVTTDataset as MSRVTT
from msrvtt_dataset import Standardize_VideoSentencePair, ToTensor_VideoSentencePair
from utils.train_utils import get_experiment_info, log_experiment_info_msrvtt, save_experiment, get_dataloader_msrvtt
from utils.sys_utils import create_dir_if_not_exist

# init tensorboard
writer = SummaryWriter('runs/')
torch.manual_seed(42)

# init logging
logfile = 'logs/logfile_{}.log'.format(dt.now())
logformat = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
loglevel = 20 ## levels: NOTSET = 0 | DEBUG = 10 | INFO = 20 | WARNING = 30 | ERROR = 40 | CRITICAL = 50
logging.basicConfig (
    filename = logfile.format (dt.now().date()),
    level = loglevel,
    format = logformat)

logging.getLogger ().addHandler (logging.StreamHandler())
logger = logging.getLogger()
    

def optimize_model(lr, lr_step_size, weight_decay, batch_size_exp):
  
    global batch_size
    # use batch_size provided by bayes_opt as 2**int(value)
    batch_size = int(np.power(2,int(batch_size_exp)))
    
    dl_params = {'batch_size': batch_size,
                 'shuffle': shuffle,
                 'num_workers': 1}
    
    lr_step_size = int(lr_step_size)
    
    model_vt = VT(args)

    if init_pretrained:
        pretrained_ssl_experiment_name = 'experiment_shuffle_yes_loss_None_lr_0.000956_lr_step_306_gamma_0.9_wdecay_0.000603_bsz_128_epochs_1000_1x512_1x2048_f53a80d0575a473885e22942c4353eaf'

        model_v_path = f'output_msrvtt/experiments/{pretrained_ssl_experiment_name}/model_v2t.sd'
        model_t_path = f'output_msrvtt/experiments/{pretrained_ssl_experiment_name}/model_t2v.sd'
        
        model_v2r_file = open(model_v_path, 'rb')
        model_t2r_file = open(model_t_path, 'rb')

        model_v2r_sd = torch.load(model_v2r_file)
        model_t2r_sd = torch.load(model_t2r_file)

        model_vt.v2r.load_state_dict(model_v2r_sd)
        model_vt.t2r.load_state_dict(model_t2r_sd)
    
    # display experiment info
    exp_info = get_experiment_info(lr, lr_step_size, weight_decay, lr_gamma, n_epochs, n_feats_t, n_feats_v, T, L, batch_size)
    logger.info(exp_info)
    
    # get data loaders for train and valid sets
    dataloader_trainval = get_dataloader_msrvtt(trainval_split_path, v_feats_dir_trainval, t_feats_path_trainval, dl_params)
    #dataloader_test = get_dataloader_msrvtt(test_split_path, v_feats_dir_trainval, t_feats_path_test, dl_params)
    
    # get experiment name 
    _, exp_name = log_experiment_info_msrvtt(output_path, lr, lr_step_size, weight_decay, lr_gamma, n_epochs, n_feats_t, n_feats_v, T, L, batch_size, shuffle, loss_criterion, write_it=False)
    
    # train 
    torch.set_grad_enabled(True)
    model, train_loss = train_model(dataloader_trainval, lr, lr_step_size, weight_decay, lr_gamma, n_epochs, n_feats_t, n_feats_v, T, L, dl_params, exp_name)
            
    # calculate loss on validation
    #valid_loss = evaluate_validation(dataloader_test, model)
    
    # log experiment meta data 
    exp_dir, exp_name = log_experiment_info_msrvtt(output_path, lr, lr_step_size, weight_decay, lr_gamma, n_epochs, n_feats_t, n_feats_v, T, L, batch_size, shuffle, loss_criterion = None, write_it=True)
    
    # save trained model, training losses, and validation losses
    save_experiment(model, None, train_loss, exp_dir, exp_name)
    logger.warning(f'saved model and train/valid loss to {exp_dir}')

    logger.warning(f'loss train: {train_loss}')
    
    output_path_ = f'{output_path}/experiments/{exp_name}'
    create_dir_if_not_exist(output_path_)
    model.save(output_path_)
    return train_loss


def evaluate_validation(dataloader, model):
    model.eval()

    total_loss = 0
    
    # run trained model on validation
    for (v,t) in dataloader:
        v = v.cuda()
        t = t.cuda()
        with torch.no_grad():
            # forwards to obtain diagonal items on cross-correlation
            loss = model(v,t)
        total_loss+=loss.item()
        
    return total_loss/len(dataloader)
    

def train_model(data_loader_train, lr, lr_step_size, weight_decay, lr_gamma, n_epochs, n_feats_t, n_feats_v, T, L, dl_params, exp_name):   
             
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_samples = data_loader_train.dataset.__len__()
    flag = True 
    
    ### instantiate model
    model = VT(args)
    model.to(device)
    
    loader = data_loader_train
    scaler = torch.cuda.amp.GradScaler()
    
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    # Stepwise LR
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = lr_step_size, gamma = lr_gamma)
    # CosineAnnealing LR
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs*len(loader), eta_min=0, last_epoch=-1)
    
    avg_loss = []
    
    start_time = time.time()
    for epoch in range(n_epochs):
        total_loss = 0
        for step, (y1, y2) in enumerate(loader, start=epoch * len(loader)):
            y1 = y1.cuda()
            y2 = y2.cuda()
            
            optimizer.zero_grad()
            loss = model.forward(y1, y2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            lr_value = lr_scheduler.optimizer.param_groups[0]['lr']
            writer.add_scalar(f'{exp_name}/train/lr', lr_value, epoch)
            lr_scheduler.step()

            total_loss+=loss.item()

#             if flag == True:
#                 writer.add_graph(model, (y1,y2))
#                 flag = False
        
        # write train loss to tensorboard
        avg_loss.append(total_loss/len(loader))
        writer.add_scalar(f'{exp_name}/loss/train', avg_loss[-1], epoch)
        logger.info(f'epoch[{epoch + 1}/{n_epochs}]\n\t loss train: {avg_loss[-1]}')
        
    avg_loss = np.array(avg_loss)
    
    writer.flush()
    return model, avg_loss.mean()

# python -W ignore train_vt_msrvtt.py --n_epochs 10 --t_num_feats 512 --v_num_feats 2048 --batch_size_exp_min 7 --batch_size_exp_max 7 --lr_min 0.0001 --lr_max 0.001 --weight_decay_min 0.00001 --weight_decay_max 0.001 --lr_step_size_min 50 --lr_step_size_max 400 --lr_gamma 0.9 --shuffle --init-pretrained
if __name__ == '__main__':
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
    parser.add_argument('--bayes_n_iter', type = int, default = 1, help = 'bayesian optimization num iterations')
    parser.add_argument('--bayes_init_points', type = int, default = 1, help = 'bayesian optimization init points')
    
    # io params
    parser.add_argument('--repo_dir', default = '/usr/local/extstore01/zahra/datasets/MSRVTT')
    
    parser.add_argument('--video_feats_dir_trainval', default = 'feats/video/r2plus1d_TrainVal')
    parser.add_argument('--text_feats_path_trainval', default = 'feats/text/msrvtt_captions_universal_trainval.pkl')
    
    parser.add_argument('--video_feats_dir_test', default = 'feats/video/r2plus1d_Test')
    parser.add_argument('--text_feats_path_test', default = 'feats/text/msrvtt_captions_universal_test.pkl')
    
    parser.add_argument('--trainval_split_path', default = 'TrainVal_videoid_sentid.txt')    
    parser.add_argument('--test_split_path', default = 'Test_videoid_sentid.txt')    

    parser.add_argument('--output_path', default = '/usr/local/extstore01/zahra/VTR_OOD/output_msrvtt')
    
    parser.add_argument('--projector', default='1024-1024-1024', type=str, metavar='MLP', help='projector MLP')
    
    parser.add_argument('--lambd', default=0.0051, type=float, metavar='L', help='weight on off-diagonal terms')
    
    parser.add_argument('--shuffle', dest='shuffle', action='store_true')
    
    parser.add_argument('--print-freq', default=1, type=int, metavar='N', help='print frequency')
    
    parser.add_argument('--init-pretrained', action='store_true')
    
    
    args = parser.parse_args()
    
    logger.info(args)
    
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
    trainval_split_path = f'{repo_dir}/{args.trainval_split_path}'
    test_split_path = f'{repo_dir}/{args.test_split_path}'
    output_path = args.output_path
    v_feats_dir_trainval = f'{repo_dir}/{args.video_feats_dir_trainval}'
    t_feats_path_trainval = f'{repo_dir}/{args.text_feats_path_trainval}'
    
    v_feats_dir_test = f'{repo_dir}/{args.video_feats_dir_test}'
    t_feats_path_test = f'{repo_dir}/{args.text_feats_path_test}'
    
    init_pretrained = args.init_pretrained
    
    ## bayes opt
    bayes_init_points = args.bayes_init_points
    bayes_n_iter = args.bayes_n_iter
       
    loss_criterion = args.loss_criterion
    
    # bounds of parameter space
    pbounds = {'lr': (lr_min, lr_max), 
               'lr_step_size': (lr_step_size_min, lr_step_size_max), 
               'weight_decay':(weight_decay_min, weight_decay_max), 
               'batch_size_exp': (batch_size_exp_min, batch_size_exp_max)
              }

    optimizer = BayesianOptimization(
        f=optimize_model,
        pbounds=pbounds,
        random_state=42,
    )

    optimizer.maximize(
        init_points=bayes_init_points,
        n_iter=bayes_n_iter,
    )

