import argparse
import math
import torch 
import torch.nn as nn
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
import logging
from datetime import datetime as dt
import utilities as utils
from preprocessing import load_video_text_features
from layers.AEwithAttention import AEwithAttention

torch.set_default_tensor_type(torch.cuda.FloatTensor)

# init logging
logfile = 'logs/logfile_{}.log'.format(dt.now())
logformat = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
loglevel = 10 ## levels: NOTSET = 0 | DEBUG = 10 | INFO = 20 | WARNING = 30 | ERROR = 40 | CRITICAL = 50
logging.basicConfig (
    filename = logfile.format (dt.now().date()),
    level = loglevel,
    format = logformat)

logging.getLogger ().addHandler (logging.StreamHandler())
logger = logging.getLogger()
    
def evaluate_model(lr, lr_step_size, weight_decay):

    # load train data
    vids, caps = load_video_text_features(v_feats_dir, t_feats_path, n_feats_t, n_feats_v, T, L, train_split_path, n_max=None)
    
    # train model
    model_v, model_t, train_losses, train_losses_avg = train_model(vids, caps, lr, lr_step_size, weight_decay, lr_gamma, n_epochs, n_filt, n_feats_t, n_feats_v, T, L)
    
    # load valid data
    valid_split_path = '/usr/local/extstore01/zahra/Video-Text-Retrieval_OOD/output/valid.split.pkl'
    vids, caps = load_video_text_features(v_feats_dir, t_feats_path, n_feats_t, n_feats_v, T, L, valid_split_path)
    
    # calculate loss on validationn
    valid_loss, valid_losses, valid_losses_avg = evaluate_validation(model_v, model_t, vids, caps)
    
    # log experiment meta data 
    exp_dir, exp_name = log_experiment_info(lr, lr_step_size, weight_decay, lr_gamma, n_epochs, n_filt, n_feats_t, n_feats_v, T, L)
    
    # save trained model, training losses, and validation losses
    save_experiment(model_v, model_t, train_losses, train_losses_avg, valid_losses, valid_losses_avg, exp_dir, exp_name)
    logger.info(f'saved model_t, model_v, training/validation loss to {exp_dir}')

    # validation loss returned for bayesian parameter optimization 
    return valid_loss

########################################

def save_experiment(model_v, model_t, train_losses, train_losses_avg, valid_losses, valid_losses_avg, exp_dir, exp_name):
    # save models
    torch.save(model_v.state_dict(), f'{exp_dir}/model_v_{exp_name}.sd')
    torch.save(model_t.state_dict(), f'{exp_dir}/model_t_{exp_name}.sd')
    
    # save train losses
    utils.dump_picklefile(train_losses, f'{exp_dir}/losses_train_{exp_name}.pkl')
    utils.dump_picklefile(train_losses_avg, f'{exp_dir}/losses_train_avg_{exp_name}.pkl')
    
    # save valid losses
    utils.dump_picklefile(valid_losses_avg, f'{exp_dir}/losses_validation_avg_{exp_name}.pkl')
    utils.dump_picklefile(valid_losses, f'{exp_dir}/losses_validation_{exp_name}.pkl')
    
    return True

########################################
    
def forward_multimodal(model_v, model_t, criterion, v, t, coefs = (1, 1, 1, 1)):
    # model_v and model_t are the corresponding models for video and captions respectively

    v = torch.tensor(v).float()
    t = torch.tensor(t).float()

    dims_v = v.shape
    dims_t = t.shape

    # recons loss
    v_reconst = model_v(v).reshape(dims_v[0], dims_v[1])
    t_reconst = model_t(t).reshape(dims_t[0], dims_t[1])
    loss_recons_v = criterion(v_reconst, v)
    loss_recons_t = criterion(t_reconst, t)

    # joint loss
    loss_joint = criterion(model_v.encoder(v), model_t.encoder(t))

    # cross loss
    loss_cross_t = criterion(model_t.decoder(model_v.encoder(v)).reshape(dims_t[0], dims_t[1]), t)
    loss_cross_v = criterion(model_v.decoder(model_t.encoder(t)).reshape(dims_v[0], dims_v[1]), v)
    
    # cycle loss
    loss_cycle_t = criterion(model_t.decoder(model_v.encoder(model_v.decoder(model_t.encoder(t)))).reshape(dims_t[0], dims_t[1]), t)
    loss_cycle_v = criterion(model_v.decoder(model_t.encoder(model_t.decoder(model_v.encoder(v)))).reshape(dims_v[0], dims_v[1]), v)

    # set coef hyperparams 
    a0, a1, a2, a3 = coefs
    
    # total loss
    loss_total = a0 * (loss_recons_t + loss_recons_v) + (a1 * loss_joint) + a2 * (loss_cross_t + loss_cross_v) + a3 * (loss_cycle_t + loss_cycle_v)

    return (loss_recons_t, loss_recons_v, loss_joint, loss_cross_t, loss_cross_v, loss_cycle_t, loss_cycle_v, loss_total)

################################

def average_loss(losses):
    losses = pd.DataFrame(losses[1:], columns = losses[0])
    return losses.mean()

################################

def evaluate_validation(model_v, model_t, vids, caps):
    
    losses = [['recons_t', 'recons_v', 'joint', 'cross_t', 'cross_v', 'cycle_t', 'cycle_v', 'total']]
    criterion = nn.MSELoss()

    for v,t in zip(vids,caps):
        loss = forward_multimodal(model_v, model_t, criterion, v, t)
        losses.append([l.item() for l in loss])

    losses_avg = average_loss(losses)
    loss = losses_avg['total']
    
    logger.info(f'validation loss: {loss}')
        
    if loss is math.nan:
        print('oops! nan again!')
        loss = 1000
        
    return -loss, losses, losses_avg

########################################

def log_experiment_info(lr, lr_step_size, weight_decay, lr_gamma, n_epochs, n_filt, n_feats_t, n_feats_v, T, L):
    
    exp_name = f'experiment_{lr}_{lr_step_size}_{lr_gamma}_{weight_decay}_{n_epochs}_{n_filt}_{L}x{n_feats_t}_{T}x{n_feats_v}'
    exp_dir = f'{output_path}/experiments/{exp_name}'
    utils.create_dir_if_not_exist(exp_dir)
        
    exp_info_path = f'{exp_dir}/experiment_info.txt'
    exp_info = [f'lr: {lr}',
                f'lr_step_size: {lr_step_size}',
                f'lr_gamma: {lr_gamma}',
                f'lr_weight_decay: {weight_decay}',
                f'n_epochs: {n_epochs}',
                f'n_filt: {n_filt}',
                f'n_feats_t: {n_feats_t}',
                f'n_feats_v: {n_feats_t}',
                f'T: {T}',
                f'L: {L}']
    utils.dump_textfile(exp_info, exp_info_path)
    
    return exp_dir, exp_name

########################################

def train_model(vids, caps, lr, lr_step_size, weight_decay, lr_gamma, n_epochs, n_filt, n_feats_t, n_feats_v, T, L):   
             
    ### create AE model for video and text encoding  
    model_v = AEwithAttention(n_feats_v, T, n_filt)
    model_t = AEwithAttention(n_feats_t, L, n_filt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_v.to(device)
    model_t.to(device)
    
    # Adam optimizer
    optimizer_v = torch.optim.Adam(model_v.parameters(), lr = lr, weight_decay = weight_decay)
    optimizer_t = torch.optim.Adam(model_t.parameters(), lr = lr, weight_decay = weight_decay)

    optimizer_E_v = torch.optim.Adam(model_v.encoder_.parameters(), lr = lr, weight_decay = weight_decay)
    optimizer_E_t = torch.optim.Adam(model_t.encoder_.parameters(), lr = lr, weight_decay = weight_decay)

    optimizer_G_v = torch.optim.Adam(model_v.decoder_.parameters(), lr = lr, weight_decay = weight_decay)
    optimizer_G_t = torch.optim.Adam(model_t.decoder_.parameters(), lr = lr, weight_decay = weight_decay)


    torch.optim.lr_scheduler.StepLR(optimizer_v, step_size = lr_step_size, gamma = lr_gamma)
    torch.optim.lr_scheduler.StepLR(optimizer_t, step_size = lr_step_size, gamma = lr_gamma)

    torch.optim.lr_scheduler.StepLR(optimizer_E_v, step_size = lr_step_size, gamma = lr_gamma)
    torch.optim.lr_scheduler.StepLR(optimizer_E_t, step_size = lr_step_size, gamma = lr_gamma)

    torch.optim.lr_scheduler.StepLR(optimizer_G_v, step_size = lr_step_size, gamma = lr_gamma)
    torch.optim.lr_scheduler.StepLR(optimizer_G_t, step_size = lr_step_size, gamma = lr_gamma)

    criterion = nn.MSELoss()

    losses_avg = []
    ### train the model
    for epoch in range(n_epochs):
        counter = 1
        losses = [['recons_t', 'recons_v', 'joint', 'cross_t', 'cross_v', 'cycle_t', 'cycle_v', 'total']]
        for v,t in zip(vids,caps):
            loss = forward_multimodal(model_v, model_t, criterion, v, t)
            losses.append([l.item() for l in loss])

            # Backprop and optimize
            optimizer_v.zero_grad()
            optimizer_t.zero_grad()
            optimizer_E_v.zero_grad()
            optimizer_E_t.zero_grad()
            optimizer_G_v.zero_grad()
            optimizer_G_t.zero_grad()

            loss[-1].backward()

            optimizer_v.step()
            optimizer_t.step()
            optimizer_E_v.step()
            optimizer_E_t.step()
            optimizer_G_v.step()
            optimizer_G_t.step()

            logger.info('Epoch[{}/{}], Step[{}/{}] Loss: {}\n'.format(epoch + 1, n_epochs, counter, len(vids), loss[-1].item()))

        counter = counter + 1 
        losses_avg.append(average_loss(losses)) 
        logger.info(f'Epoch[{epoch + 1}/{n_epochs}], Loss: {losses_avg[-1]}')
        
    return model_v, model_t, losses, losses_avg

########################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser ()
    parser.add_argument('--n_epochs', dest = 'n_epochs', type = int, default = '10', help = 'number of iterations')
    parser.add_argument('--n_filters', dest = 'n_filters', type = int, default = '10', help = 'number of filters')
    parser.add_argument('--lr_step_size', dest = 'lr_step_size', type = int, default = 2, help = 'lr schedule: step size')
    parser.add_argument('--lr_gamma', dest = 'lr_gamma', type = float, default = 0.1, help = 'lr schedule: gamma')
    parser.add_argument('--lr', dest = 'lr', type = float, default = 0.001, help = 'learning rate')
    parser.add_argument('--weight_decay', dest = 'weight_decay', type = float, default = 0.01, help = 'weight decay')
    
    parser.add_argument('--num_feats', dest = 'num_feats', type = int, default = 512, help = 'number of feats in each vector')
    parser.add_argument('--t_feat_len', dest = 't_feat_len', type = int, default = 1, help = 'length of feat vector')
    parser.add_argument('--v_feat_len', dest = 'v_feat_len', type = int, default = 5, help = 'length of feat vector')
    
    parser.add_argument('--bayes_n_iter', dest = 'bayes_n_iter', type = int, default = 10, help = 'bayesian optimization num iterations')
    parser.add_argument('--bayes_init_points', dest = 'bayes_init_points', type = int, default = 5, help = 'bayesian optimization init points')
    
    parser.add_argument('--video_feats_dir', dest = 'video_feats_dir', default = '/usr/local/extstore01/zahra/Video-Text-Retrieval_OOD/output/sentence_segments_feats/video/c3d')
    parser.add_argument('--text_feats_path', dest = 'text_feats_path', default = '/usr/local/extstore01/zahra/Video-Text-Retrieval_OOD/output/sentence_segments_feats/text/fasttext/sentence_feats.pkl')
    parser.add_argument('--train_split_path', dest = 'train_split_path', default = '/usr/local/extstore01/zahra/Video-Text-Retrieval_OOD/output/train.split.pkl')
    parser.add_argument('--output_path', dest = 'output_path', default = '/usr/local/extstore01/zahra/Video-Text-Retrieval_OOD/output')

      
    ### get args
    args = parser.parse_args()
    
    logger.info('\n\n**************************\n\nStarting a new run with bayes optimizer\n***************************')
    
    lr = args.lr
    lr_step_size = args.lr_step_size
    lr_gamma = args.lr_gamma
    weight_decay = args.weight_decay
    n_epochs = args.n_epochs
    n_filt = args.n_filters
    v_feats_dir = args.video_feats_dir
    t_feats_path = args.text_feats_path
    n_feats_t = args.num_feats
    n_feats_v = args.num_feats
    T = args.v_feat_len
    L = args.t_feat_len
    train_split_path = args.train_split_path
    output_path = args.output_path
    
    bayes_init_points = args.bayes_init_points
    bayes_n_iter = args.bayes_n_iter
    
    # bounds of parameter space
    pbounds = {'lr': (0.000001, 0.01), 'lr_step_size': (1, 10), 'weight_decay':(0.001,0.1)}

    optimizer = BayesianOptimization(
        f=evaluate_model,
        pbounds=pbounds,
        random_state=42,
    )

    optimizer.maximize(
        init_points=bayes_init_points,
        n_iter=bayes_n_iter,
    )
    