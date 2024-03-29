import argparse
import math
import logging
import time
import sys
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
from utils.train_utils import get_experiment_info, log_experiment_info_msrvtt, save_experiment, get_dataloader
from utils.sys_utils import create_dir_if_not_exist

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
    

def optimize_model(lr, weight_decay, batch_size_exp):
  
    global batch_size
    # use batch_size provided by bayes_opt as 2**int(value)
    batch_size = int(np.power(2,int(batch_size_exp)))
    
    dl_params = {'batch_size': batch_size,
                 'shuffle': shuffle,
                 'num_workers': 1}
    
    # display experiment info
    exp_info = get_experiment_info(lr, weight_decay, n_epochs, n_feats_t, n_feats_v, batch_size)
    logger.info(exp_info)
    
    # get data loaders for train and valid sets
    dataset_trainval = MSRVTT(vid_feats_dir=v_feats_dir, txt_feats_path=t_feats_path, ids_path=trainval_split_path, transform=None)
    dataset_stats = dataset_trainval.get_dataset_mean_std()
    standardize = Standardize_VideoSentencePair(dataset_stats)
    trnsfrm = transforms.Compose([standardize, ToTensor_VideoSentencePair()])
    dataset_trainval = MSRVTT(vid_feats_dir=v_feats_dir, txt_feats_path=t_feats_path, ids_path=trainval_split_path, transform=trnsfrm)
    
    dataset_valid = MSRVTT(vid_feats_dir=v_feats_dir_val, txt_feats_path=t_feats_path_val, ids_path=trainval_split_path_val, transform=trnsfrm)

    dataloader_trainval = torch.utils.data.DataLoader(dataset_trainval, **dl_params)
    dataloader_valid = torch.utils.data.DataLoader(dataset_valid, **dl_params)
    
    # get experiment name 
    _, exp_name = log_experiment_info_msrvtt(output_path, lr, weight_decay, n_epochs, n_feats_t, n_feats_v, batch_size, loss_criterion, write_it=False)
    
    # init tensorboard
    global writer
    writer = SummaryWriter(f'runs/{exp_name}')

    # train 
    torch.set_grad_enabled(True)
    model, train_loss = train_model(dataloader_trainval, dataloader_valid, lr, weight_decay, n_epochs, n_feats_t, n_feats_v, dl_params, exp_name)
            
    # log experiment meta data 
    exp_dir, exp_name = log_experiment_info_msrvtt(output_path, lr, weight_decay, n_epochs, n_feats_t, n_feats_v, batch_size, loss_criterion, write_it=True)
    
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
    

def early_stop(model, best, current_loss, max_target_loss, stop_counter, exp_dir, exp_name):
    stop = False
    is_best = current_loss < best
    best = min(current_loss, best)

    if is_best:
        save_experiment(model, None, best, exp_dir, exp_name)
        logger.info(f'saved BEST model and train/valid loss to {exp_dir}')
        logger.info(f'loss train:{best}')
        output_path_ = f'{output_path}/experiments/{exp_name}'
        create_dir_if_not_exist(output_path_)
        model.save(output_path_)
        stop_counter = 0
    elif not is_best and best<max_target_loss:
        stop_counter += 1
        if stop_counter > patience:
            logger.info('Early stopping')
            stop = True

    return best, stop_counter, stop

def train_model(data_loader_train, data_loader_val, lr, weight_decay, n_epochs, n_feats_t, n_feats_v, dl_params, exp_name):   
             
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_samples = data_loader_train.dataset.__len__()
    flag = True 
    
    ### instantiate model
    model = VT(args)
    if args.init_model_name is not None:
            model.load_state_dict(torch.load(init_model_path))
    model.to(device)
    
    loader = data_loader_train
    scaler = torch.cuda.amp.GradScaler()
    
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)

    # CosineAnnealing LR
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs*len(loader), eta_min=0, last_epoch=-1)
    
    avg_loss = []
    
    start_time = time.time()
    
    best_loss = max_target_loss # only consider loss candidates less than "max_target_loss" 
    stop_counter = 0
    
    # log experiment meta data 
    exp_dir, exp_name = log_experiment_info_msrvtt(output_path, lr, weight_decay, n_epochs, n_feats_t, n_feats_v, dl_params['batch_size'], loss_criterion, write_it=True)
    
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

            if flag == True:
                writer.add_graph(model, (y1,y2))
                flag = False
        
        loss_train = total_loss/len(loader)
        loss_val = evaluate_validation(data_loader_val, model)
        
        
        # write train/val loss to tensorboard
        writer.add_scalar(f'{exp_name}/loss/train', loss_train, epoch)
        writer.add_scalar(f'{exp_name}/loss/val', loss_val, epoch)
        
        logger.info(f'epoch[{epoch+1}/{n_epochs}]\n Validation Perfomance:{loss_val}\n Current Perfomance:{loss_train}\n Best Perfomance:{best_loss}\n')

        # early stopping
        current_loss = loss_train
        best_loss, stop_counter, stop = early_stop(model, 
                                                   best_loss, 
                                                   current_loss, 
                                                   max_target_loss, 
                                                   stop_counter, 
                                                   exp_dir, 
                                                   exp_name)
        if stop: break
       
        all_weights = model.get_weights()
        all_activations = model.get_all_activations(y1,y2)
        
        for k,v in all_weights.items():
            writer.add_histogram(f'weights {k}', v, epoch)
            
        for k,v in all_activations.items():
            writer.add_histogram(f'activations {k}', v, epoch)
            
        avg_loss.append(loss_train)

    avg_loss = np.array(avg_loss)
    
    writer.flush()
    return model, avg_loss.mean()

if __name__ == '__main__':
    '''
    python -W ignore train_msrvtt.py \
                        --n_epochs 500 \
                        --t_num_feats 512 \
                        --v_num_feats 2048 \
                        --loss_criterion cross_correlation \
                        --batch_size_exp_min 7 \
                        --batch_size_exp_max 10 \
                        --lr_min 0.000000001 \
                        --lr_max 0.00001 \
                        --weight_decay_min 0.0001 \
                        --weight_decay_max 0.1 \
                        --shuffle \
                        --patience 50 \
                        --max-target-loss 1000 \
                        --init-model-name experiment_loss_cross_correlation_lr_2.1e-05_wdecay_0.018264_bsz_512_1628163958
    '''

    parser = argparse.ArgumentParser ()
    parser.add_argument('--n_epochs', type = int, default = 20, help = 'number of iterations')
    parser.add_argument('--n_train_samples', type = int, default = None, help = 'number of training samples')
        
    # loss criterion
    parser.add_argument('--loss_criterion', default = 'cross_correlation') # 'mse', 'cross_correlation', 'contrastive', 'cosine'
    
    # lr
    parser.add_argument('--lr_min', type = float, default = 0.00001, help = 'learning rate lower bound')
    parser.add_argument('--lr_max', type = float, default = 0.01, help = 'learning rate upper bound')
    
    # weight decay
    parser.add_argument('--weight_decay_min', type = float, default = 0.00001, help = 'weight decay lower bound')
    parser.add_argument('--weight_decay_max', type = float, default = 0.001, help = 'weight decay upper bound')
    
    # batch size
    parser.add_argument('--batch_size_exp_min', type = int, default = 5, help = 'batch size exponent min; batch_size=2**n')
    parser.add_argument('--batch_size_exp_max', type = int, default = 7, help = 'batch size exponent max; batch_size=2**n')
    
    # num feats
    parser.add_argument('--t_num_feats', type = int, default = 512, help = 'number of feats in each vector')
    parser.add_argument('--v_num_feats', type = int, default = 2048, help = 'number of feats in each vector')
    
    # bayesian optimization parameters
    parser.add_argument('--bayes_n_iter', type = int, default = 10, help = 'bayesian optimization num iterations')
    parser.add_argument('--bayes_init_points', type = int, default = 10, help = 'bayesian optimization init points')
    
    # io params
    parser.add_argument('--repo_dir', default = '/usr/local/extstore01/zahra/datasets/MSRVTT')
    parser.add_argument('--video_feats_dir', default = 'feats/video/r2plus1d_TrainVal')
    parser.add_argument('--text_feats_path', default = 'feats/text/msrvtt_captions_universal_trainval.pkl')
    parser.add_argument('--trainval_split_path', default = 'TrainVal_videoid_sentid.txt')    
    parser.add_argument('--output_path', default = '/usr/local/extstore01/zahra/VTR_OOD/output_msrvtt')

    parser.add_argument('--video_feats_dir_val', default = 'feats/video/r2plus1d_Test')
    parser.add_argument('--text_feats_path_val', default = 'feats/text/msrvtt_captions_universal_test.pkl')
    parser.add_argument('--trainval_split_path_val', default = 'Test_videoid_sentid.txt')
    
    parser.add_argument('--projector', default='1024-1024-1024', type=str, help='projector MLP')
    parser.add_argument('--lambd', default=0.0051, type=float, help='weight on off-diagonal terms')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--patience', type = int, default = 10, help = 'early stopping: patience counter')
    parser.add_argument('--max-target-loss', type = int, default = 1000, help = 'early stopping: maximum loss to settle for')

    parser.add_argument('--init-model-name', type = str) 
    # 'experiment_loss_cross_correlation_lr_2.1e-05_wdecay_0.018264_bsz_512_1628163958'

    args = parser.parse_args()
    
    logger.info(args)
    
    lr_min = args.lr_min
    lr_max = args.lr_max
    
    patience = args.patience
    max_target_loss = args.max_target_loss
        
    weight_decay_min = args.weight_decay_min
    weight_decay_max = args.weight_decay_max
    
    batch_size_exp_min = args.batch_size_exp_min
    batch_size_exp_max = args.batch_size_exp_max
    
    shuffle = args.shuffle
    
    n_epochs = args.n_epochs
    n_train_samples = args.n_train_samples
    
    n_feats_t = args.t_num_feats
    n_feats_v = args.v_num_feats
        
    repo_dir = args.repo_dir
    trainval_split_path = f'{repo_dir}/{args.trainval_split_path}'
    output_path = args.output_path
    v_feats_dir = f'{repo_dir}/{args.video_feats_dir}'
    t_feats_path = f'{repo_dir}/{args.text_feats_path}'

    trainval_split_path_val = f'{repo_dir}/{args.trainval_split_path_val}'
    v_feats_dir_val = f'{repo_dir}/{args.video_feats_dir_val}'
    t_feats_path_val = f'{repo_dir}/{args.text_feats_path_val}'
    
    ## bayes opt
    bayes_init_points = args.bayes_init_points
    bayes_n_iter = args.bayes_n_iter
       
    loss_criterion = args.loss_criterion
    
    init_model_path = f'output_msrvtt/experiments/{args.init_model_name}/model_vt.sd'

    # bounds of parameter space
    pbounds = {'lr': (lr_min, lr_max), 
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

