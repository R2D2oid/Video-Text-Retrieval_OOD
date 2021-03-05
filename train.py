import argparse
import torch 
import torch.nn as nn
import numpy as np
from bayes_opt import BayesianOptimization
import logging
from datetime import datetime as dt
import utilities as utils
from preprocessing import load_video_text_features
from layers.AEwithAttention import AEwithAttention

# init logging
logfile = 'logs/logfile_{}.log'.format(dt.now().date())
logformat = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
loglevel = 10 ## levels: NOTSET = 0 | DEBUG = 10 | INFO = 20 | WARNING = 30 | ERROR = 40 | CRITICAL = 50
logging.basicConfig (
    filename = logfile.format (dt.now().date()),
    level = loglevel,
    format = logformat)

logging.getLogger ().addHandler (logging.StreamHandler())
logger = logging.getLogger()
    
def evaluate_model(lr, lr_step_size, weight_decay):
    # train model
    model_v, model_t = train_model(lr, lr_step_size, weight_decay, lr_gamma, n_epochs, n_filt, v_feats_dir, t_feats_path, n_feats_t, n_feats_v, T, L, train_split_path, output_path)
    
    # calculate loss on validation set
    valid_split_path = '/usr/local/extstore01/zahra/Video-Text-Retrieval_OOD/output/valid.split.pkl'
    vids, caps = load_video_text_features(v_feats_dir, t_feats_path, n_feats_t, n_feats_v, T, L, valid_split_path)
    loss = evaluate_validation(model_v, model_t, vids, caps, exp_dir)
    
    return loss


########################################
    
def evaluate_validation(model_v, model_t, vids, caps, exp_dir):
    criterion = nn.MSELoss()
    
    losses = init_losses()
    
    for v,t in zip(vids,caps):
        # Forward pass        
        v = torch.tensor(v).float()
        t = torch.tensor(t).float()

        # Compute recons loss 
        dims_v = v.shape
        dims_t = t.shape

        v_reconst = model_v(v).reshape(dims_v[0], dims_v[1])
        t_reconst = model_t(t).reshape(dims_t[0], dims_t[1])

        loss_recons_v = criterion(v_reconst, v)
        loss_recons_t = criterion(t_reconst, t)

        losses['recons1'].append(loss_recons_v)
        losses['recons2'].append(loss_recons_t)

        loss_recons = loss_recons_v + loss_recons_t
        # the following losses require paired video/caption data (v and t)
        # model_v and model_t are the corresponding models for video and captions respectively

        # Compute joint loss
        loss_joint = criterion(model_v.encoder(v), model_t.encoder(t))

        losses['joint'].append(loss_joint)

        # Compute cross loss
        loss_cross1 = criterion(model_t.decoder(model_v.encoder(v)).reshape(dims_t[0], dims_t[1]), t)
        loss_cross2 = criterion(model_v.decoder(model_t.encoder(t)).reshape(dims_v[0], dims_v[1]), v)
        loss_cross = loss_cross1 + loss_cross2

        losses['cross1'].append(loss_cross1)
        losses['cross2'].append(loss_cross2)

        # Compute cycle loss
        loss_cycle1 = criterion(model_t.decoder(model_v.encoder(model_v.decoder(model_t.encoder(t)))).reshape(dims_t[0], dims_t[1]), t)
        loss_cycle2 = criterion(model_v.decoder(model_t.encoder(model_t.decoder(model_v.encoder(v)))).reshape(dims_v[0], dims_v[1]), v)
        loss_cycle = loss_cycle1 + loss_cycle2

        losses['cycle1'].append(loss_cycle1)
        losses['cycle2'].append(loss_cycle2)

        # set hyperparams 
        a1, a2, a3 = 1, 1, 1

        # Compute total loss
        loss = loss_recons + a1 * loss_joint + a2 * loss_cross + a3 * loss_cycle

        losses['all'].append(loss)
    
    utils.dump_picklefile(losses, f'{exp_dir}/losses_validation.pkl')

    try:
        loss = np.array([losses['all'][i].item() for i in range(len(losses['all']))])
        loss = np.mean(loss)
    except Exception as e:
        #pdb.set_trace()
        loss = 1000
    return -loss


########################################

def init_losses():
    losses = {}
    loss_types = ['recons1', 'recons2', 'joint', 'cross1', 'cross2', 'cycle1', 'cycle2', 'all']
    for t in loss_types: losses[t] = []
    return losses


########################################

def log_experiment_info(lr, lr_step_size, weight_decay, lr_gamma, n_epochs, n_filt, n_feats_t, n_feats_v, T, L):
    
    exp_name = f'experiment_{lr}_{lr_step_size}_{lr_gamma}_{weight_decay}_{n_epochs}_{n_filt}_{L}x{n_feats_t}_{T}x{n_feats_v}'
    exp_dir = f'{output_path}/experiments/{exp_name}'
    utils.create_dir_if_not_exist(exp_dir)
        
    exp_info_path = f'{exp_dir}/experiment_info.txt'
    exp_info = [f'lr: {lr}\n',
                f'lr_step_size: {lr_step_size}\n',
                f'lr_gamma: {lr_gamma}\n',
                f'lr_weight_decay: {weight_decay}\n',
                f'n_epochs: {n_epochs}\n',
                f'n_filt: {n_filt}\n',
                f'n_feats_t: {n_feats_t}\n',
                f'n_feats_v: {n_feats_t}\n',
                f'T: {n_feats_t}\n',
                f'L: {n_feats_t}\n']
    utils.dump_textfile(exp_info, exp_info_path)
    
    return exp_dir

########################################

def train_model(lr, lr_step_size, weight_decay, lr_gamma, n_epochs, n_filt, v_feats_dir, t_feats_path, n_feats_t, n_feats_v, T, L, train_split_path, output_path):   
        
    exp_dir = log_experiment_info(lr, lr_step_size, weight_decay, lr_gamma, n_epochs, n_filt, n_feats_t, n_feats_v, T, L)
    
    vids, caps = load_video_text_features(v_feats_dir, t_feats_path, n_feats_t, n_feats_v, T, L, train_split_path)
    
    ### create AE model for video and text encoding  
    model_v = AEwithAttention(n_feats_v, T, n_filt)
    model_t = AEwithAttention(n_feats_t, L, n_filt)
    
    criterion = nn.MSELoss()

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

    
    ### train the model
    # training
    losses = init_losses()
    
    for epoch in range(n_epochs):
        counter = 1
        for v,t in zip(vids,caps):
            # Forward pass        
            v = torch.tensor(v).float()
            t = torch.tensor(t).float()

            # Compute recons loss 
            dims_v = v.shape
            dims_t = t.shape

            v_reconst = model_v(v).reshape(dims_v[0], dims_v[1])
            t_reconst = model_t(t).reshape(dims_t[0], dims_t[1])

            loss_recons_v = criterion(v_reconst, v)
            loss_recons_t = criterion(t_reconst, t)

            losses['recons1'].append(loss_recons_v)
            losses['recons2'].append(loss_recons_t)
            
            loss_recons = loss_recons_v + loss_recons_t
            # the following losses require paired video/caption data (v and t)
            # model_v and model_t are the corresponding models for video and captions respectively

            # Compute joint loss
            loss_joint = criterion(model_v.encoder(v), model_t.encoder(t))
            
            losses['joint'].append(loss_joint)

            # Compute cross loss
            loss_cross1 = criterion(model_t.decoder(model_v.encoder(v)).reshape(dims_t[0], dims_t[1]), t)
            loss_cross2 = criterion(model_v.decoder(model_t.encoder(t)).reshape(dims_v[0], dims_v[1]), v)
            loss_cross = loss_cross1 + loss_cross2

            losses['cross1'].append(loss_cross1)
            losses['cross2'].append(loss_cross2)

            # Compute cycle loss
            loss_cycle1 = criterion(model_t.decoder(model_v.encoder(model_v.decoder(model_t.encoder(t)))).reshape(dims_t[0], dims_t[1]), t)
            loss_cycle2 = criterion(model_v.decoder(model_t.encoder(model_t.decoder(model_v.encoder(v)))).reshape(dims_v[0], dims_v[1]), v)
            loss_cycle = loss_cycle1 + loss_cycle2

            losses['cycle1'].append(loss_cycle1)
            losses['cycle2'].append(loss_cycle2)
                
            # set hyperparams 
            a1, a2, a3 = 1, 1, 1

            # Compute total loss
            loss = loss_recons + a1 * loss_joint + a2 * loss_cross + a3 * loss_cycle

            losses['all'].append(loss)

            # Backprop and optimize
            optimizer_v.zero_grad()
            optimizer_t.zero_grad()
            optimizer_E_v.zero_grad()
            optimizer_E_t.zero_grad()
            optimizer_G_v.zero_grad()
            optimizer_G_t.zero_grad()

            loss.backward()

            optimizer_v.step()
            optimizer_t.step()
            optimizer_E_v.step()
            optimizer_E_t.step()
            optimizer_G_v.step()
            optimizer_G_t.step()

            logger.info('Epoch[{}/{}], Step[{}/{}] Loss: {}\n'.format(epoch + 1, n_epochs, counter, len(vids), loss.item()))

            counter = counter + 1    
    
    # save experiment configs and results
    torch.save(model_v.state_dict(), f'{exp_dir}/model_v.sd')
    torch.save(model_t.state_dict(), f'{exp_dir}/model_t.sd')
    utils.dump_picklefile(losses, f'{exp_dir}/losses_training.pkl')
    
    logger.log(f'saved model_t, model_v, losses_training to {exp_dir}')
    
    
    return model_v, model_t

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
    
    # bounds of parameter space
    pbounds = {'lr': (0.000001, 0.01), 'lr_step_size': (1, 10), 'weight_decay':(0.001,0.1)}

    optimizer = BayesianOptimization(
        f=evaluate_model,
        pbounds=pbounds,
        random_state=42,
    )

    optimizer.maximize(
        init_points=4,
        n_iter=10,
    )
    


