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
from layers.AE import AE
from data_provider import TempuckeyVideoSentencePairsDataset as TempuckeyDataset
from data_provider import Normalize_VideoSentencePair
from eval import encode_data, calc_l2_distance, calc_cosine_distance, get_metrics , normalize_metrics
from layers.loss import TripletLoss

# torch.set_default_tensor_type(torch.cuda.FloatTensor)

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
    
################################################

def optimize_vtr_model(lr, lr_step_size, weight_decay, batch_size_exp, activated_losses, relevance_score):
  
    # use batch_size provided by bayes_opt as 2**int(value)
    batch_size = int(np.power(2,int(batch_size_exp)))
    
    # use activated_losses sample provided by bayes_opt to activate the corresponding loss portions
    if activate_all_losses:
        activated_losses = 127 # that is 1111111 which sets all 7 losses to True
    activated_losses = bin(int(activated_losses))[2:].zfill(7) # get rid of the 0b starting characters in the binary
    activated_losses = tuple(l=='1' for l in list(activated_losses))
    
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
    model_v, model_t, train_losses, train_losses_avg = train_model(dataloader_train, dataloader_valid, lr, lr_step_size, weight_decay, lr_gamma, n_epochs, n_feats_t, n_feats_v, T, L, dl_params, coefs = loss_coefs, active_losses = activated_losses)
    
    # loss is nan
    if model_v is None:
        logger.warning('NaN encountered in loss... Moving on to the next iteration of bayes_opt!')
        return -10.0

    # calculate loss on validation
    valid_loss, valid_losses, valid_losses_avg = evaluate_validation(dataloader_valid, model_v, model_t, coefs=loss_coefs, active_losses=activated_losses)
    
    # log experiment meta data 
    exp_dir, exp_name = log_experiment_info(lr, lr_step_size, weight_decay, lr_gamma, n_epochs, n_feats_t, n_feats_v, T, L, batch_size, relevance_score, shuffle, loss_criterion, write_it=True)
    
    # save trained model, training losses, and validation losses
    save_experiment(model_v, model_t, train_losses, train_losses_avg, valid_losses, valid_losses_avg, exp_dir, exp_name)
    logger.info(f'saved model_t, model_v, training/validation loss to {exp_dir}')

    train_joint_loss = train_losses[-1][0]
    logger.warning(f'loss train: {train_joint_loss}')
    
    metrics_train, _, _ = validation_metrics(dataloader_train, model_v, model_t)
    recall_at_1_train = metrics_train[0]
    logger.warning(f'recall_at_1 train: {recall_at_1_train}')
    
#     #metrics_valid, ranks_valid, dist_matrix_v2t = validation_metrics(dataloader_valid, model_v, model_t)
#     metrics_valid, _, _ = validation_metrics(dataloader_valid, model_v, model_t)
#     recall_at_1_valid = metrics_valid[0]
    
#     writer.add_scalar(f'recall_at_1/train',recall_at_1_train)
    
    #metrics = validation_metrics(dataloader_valid, model_v, model_t)
    #recall_at_1 = v2t_metrics[0]
    
#     import pdb
#     pdb.set_trace()
    
    return recall_at_1_train
#     return -train_joint_loss

########################################

def validation_metrics(data_loader, model_v, model_t):
    _, embs_v, embs_t = encode_data(data_loader, model_v, model_t)
    
    dist_matrix = calc_l2_distance(embs_v, embs_t)
    #dist_matrix = calc_cosine_distance(embs_v, embs_t)
    metrics, ranks = get_metrics(dist_matrix)
    metrics = normalize_metrics(metrics, n_samples_experiment=data_loader.__len__(), n_samples_baseline = 1000)

    return metrics, ranks, dist_matrix

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
    
########################################

def log_experiment_info(lr, lr_step_size, weight_decay, lr_gamma, n_epochs, n_feats_t, n_feats_v, T, L, batch_size, relevance_score, shuffle, loss_criterion, write_it=True):
    import uuid
    random_hash = uuid.uuid4().hex

    shuffle_flag = 'yes' if shuffle else 'no'
    exp_name = f'experiment_shuffle_{shuffle_flag}_loss_{loss_criterion}_lr_{round(lr,6)}_lr_step_{round(lr_step_size,6)}_gamma_{round(lr_gamma,6)}_wdecay_{round(weight_decay,6)}_bsz_{batch_size}_epochs_{n_epochs}_relevance_{round(relevance_score,2)}_{L}x{n_feats_t}_{T}x{n_feats_v}_{random_hash}'
    exp_dir = f'{output_path}/experiments/{exp_name}'
    
    if write_it:
        utils.create_dir_if_not_exist(exp_dir)

        info_path = f'{exp_dir}/experiment_info.txt'
        info = get_experiment_info(lr, lr_step_size, weight_decay, lr_gamma, n_epochs, n_feats_t, n_feats_v, T, L, batch_size)
        utils.dump_textfile(info, info_path)
    
    return exp_dir, exp_name

# def log_experiment_info(lr, lr_step_size, weight_decay, lr_gamma, n_epochs, n_feats_t, n_feats_v, T, L, batch_size):
#     import uuid
#     random_hash = uuid.uuid4().hex

#     exp_name = f'experiment_{lr}_{lr_step_size}_{lr_gamma}_{weight_decay}_{batch_size}_{n_epochs}_{L}x{n_feats_t}_{T}x{n_feats_v}_{random_hash}'
#     exp_dir = f'{output_path}/experiments/{exp_name}'
#     utils.create_dir_if_not_exist(exp_dir)
        
#     info_path = f'{exp_dir}/experiment_info.txt'
#     info = get_experiment_info(lr, lr_step_size, weight_decay, lr_gamma, n_epochs, n_feats_t, n_feats_v, T, L, batch_size)
#     utils.dump_textfile(info, info_path)
    
#     return exp_dir, exp_name

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
    
def forward_multimodal(model_v, model_t, criterion, v, t, coefs=None, active_losses=None, target=None):
    '''
    Input: 
    model_v: video AE model
    model_t: text  AE model
    criterion: loss function, default MSE 
    v: video item   (video as a T x n_feats feature vector)
    t: text item (sentence as a L x n_feats feature vector)
    coefs: coefficients of loss components when computing total loss (i.e. reconstruction, joint, cross, cycle loss components)
    active losses: enables training the models based on a subset of loss components
    '''
    joint_active,reconst_v_active,reconst_t_active,cross_v_active,cross_t_active,cycle_v_active,cycle_t_active = active_losses
    # model_v and model_t are the corresponding models for video and captions respectively
    v = torch.tensor(v).float()
    t = torch.tensor(t).float()

    dims_v = v.shape
    dims_t = t.shape

    # recons loss
    loss_recons_v = 0
    if reconst_v_active:
        logger.debug('active loss: reconst v')
        v_reconst = model_v(v)
        loss_recons_v = criterion(v_reconst, v) if loss_criterion!='cosine' else criterion(v_reconst, v, target)
        
    loss_recons_t = 0
    if reconst_t_active:
        logger.debug('active loss: reconst t')
        t_reconst = model_t(t)
        loss_recons_t = criterion(t_reconst, t) if loss_criterion!='cosine' else criterion(t_reconst, t, target)

    # joint loss
    loss_joint = 0
    if joint_active:
        logger.debug('active loss: joint')
        loss_joint = criterion(model_v.encoder(v), model_t.encoder(t)) if loss_criterion!='cosine' else criterion(model_v.encoder(v), model_t.encoder(t), target)

    # cross loss
    loss_cross_t = 0
    if cross_t_active:
        logger.debug('active loss: cross t')
        loss_cross_t = criterion(model_t.decoder(model_v.encoder(v)), t) if loss_criterion!='cosine' else criterion(model_t.decoder(model_v.encoder(v)), t, target)
        
    loss_cross_v = 0
    if cross_v_active:
        logger.debug('active loss: cross v')
        loss_cross_v = criterion(model_v.decoder(model_t.encoder(t)), v) if loss_criterion!='cosine' else criterion(model_v.decoder(model_t.encoder(t)), v, target)
    
    # cycle loss
    loss_cycle_t = 0
    if cycle_t_active:
        logger.debug('active loss: cycle t')
        loss_cycle_t = criterion(model_t.decoder(model_v.encoder(model_v.decoder(model_t.encoder(t)))), t) if loss_criterion!='cosine' else criterion(model_t.decoder(model_v.encoder(model_v.decoder(model_t.encoder(t)))), t, target)
        
    loss_cycle_v = 0
    if cycle_v_active:
        logger.debug('active loss: cycle v')
        loss_cycle_v = criterion(model_v.decoder(model_t.encoder(model_t.decoder(model_v.encoder(v)))), v) if loss_criterion!='cosine' else criterion(model_v.decoder(model_t.encoder(model_t.decoder(model_v.encoder(v)))), v, target)

    # set coef hyperparams 
    a0, a1, a2, a3 = coefs
    
    # total loss
    loss_total = a0 * (loss_recons_t + loss_recons_v) + (a1 * loss_joint) + a2 * (loss_cross_t + loss_cross_v) + a3 * (loss_cycle_t + loss_cycle_v)

    return (loss_joint, loss_recons_v, loss_recons_t, loss_cross_v, loss_cross_t, loss_cycle_v, loss_cycle_t, loss_total)

################################

def average(losses):
    losses = pd.DataFrame(losses[1:], columns = losses[0])
    return losses.mean()

################################

def evaluate_validation(data_loader, model_v, model_t, coefs, active_losses):
    losses = [['joint', 'recons_v', 'recons_t', 'cross_v', 'cross_t', 'cycle_v', 'cycle_t', 'total']]
    criterion, target_tensor = instantiate_loss_criterion(loss_criterion)

    model_v.eval()
    model_t.eval()
    
    # run trained model on validation
    for sample in data_loader:
        v = sample['video']
        t = sample['sent']
        
        v = torch.tensor(v).float().cuda()
        t = torch.tensor(t).float().cuda()

        with torch.no_grad():
            loss = forward_multimodal(model_v, model_t, criterion, v, t, coefs, active_losses, target = target_tensor)
            losses.append([l.item() if isinstance(l,torch.Tensor) else l for l in loss])
            vv = model_v.encoder(v)
            if bool(torch.all(vv[0].eq(vv[1]))):
                print('Mode Collapse! :(')
                return None,None,None

    losses_avg = average(losses)
    loss = -losses_avg['total']
            
    return loss, losses, losses_avg
 
    
########################################

def instantiate_loss_criterion(loss_criterion):
    target_tensor = None
    
    if loss_criterion == 'cosine':
        # target_tensor = torch.Tensor(1) # use 1 to train for bringing together corresponding (positive) vectors
        # target_tensor = torch.Tensor(-1) # use -1 to train for pushing apart dissimilar (negative) vectors
        criterion = nn.CosineEmbeddingLoss()
        # the cosine embedding loss takes a target y=1 for training positive (similar) vectors and y=-1 for training dissimilar (negative) vectors
        target_tensor = torch.Tensor(1)
    elif loss_criterion == 'triplet':
        criterion = TripletLoss() 
#             margin=opt.margin,
#             measure=opt.measure, 
#             max_violation=opt.max_violation,
#             cost_style=opt.cost_style, 
#             direction=opt.direction)
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

def train_model(data_loader_train, data_loader_valid, lr, lr_step_size, weight_decay, lr_gamma, n_epochs, n_feats_t, n_feats_v, T, L, dl_params, coefs, active_losses):   
             
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_samples = data_loader_train.__len__()
    
    ### create AE model for video and text encoding
    model_v = AE(n_feats_v)
    model_t = AE(n_feats_t)

    if load_existing_model:
        model_v_file = open(model_v_path, 'rb')
        model_t_file = open(model_t_path, 'rb')

        model_v_sd = torch.load(model_v_file)
        model_t_sd = torch.load(model_t_file)

        model_v.load_state_dict(model_v_sd)
        model_t.load_state_dict(model_t_sd)

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
    
    losses_avg = []

    criterion, target_tensor = instantiate_loss_criterion(loss_criterion)

    flag = True

    ### train the model
    for epoch in range(n_epochs):
        model_v.train()
        model_t.train()
        
        counter = 1    
        
        losses = [['joint', 'recons_v', 'recons_t', 'cross_v', 'cross_t', 'cycle_v', 'cycle_t', 'total']]
        
        for sample in data_loader_train:
            v = sample['video']
            t = sample['sent']

            v = torch.tensor(v).float().cuda()
            t = torch.tensor(t).float().cuda()
            
            if flag == True:
                writer.add_graph(model_v, v)
                flag = False
            
            loss = forward_multimodal(model_v, model_t, criterion, v, t, coefs, active_losses, target = target_tensor)
            losses.append([l.item() if isinstance(l,torch.Tensor) else l for l in loss])

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

            logger.debug('Epoch[{}/{}], Step[{}/{}] Loss: {}\n'.format(epoch + 1, n_epochs, counter, num_samples, loss[-1].item()))

            counter = counter + 1 

        losses_avg.append(average(losses)) 
        logger.info(f'Finshed Epoch[{epoch + 1}/{n_epochs}]\nAverage Loss Summary:\n{losses_avg[-1]}\n')
        
        # calculate loss on validation
        _, _, valid_losses_avg = evaluate_validation(data_loader_valid, 
                            model_v,
                            model_t, 
                            coefs = loss_coefs, 
                            active_losses = active_losses)

        if valid_losses_avg is None:
            # Mode collapse! Move on from this round of training!
            return None, None, None, None
        
        metrics_train, _, _ = validation_metrics(data_loader_train, model_v, model_t)
        recall_at_1_train,recall_at_5_train,recall_at_10_train = metrics_train[:3]

        metrics_valid, _, _ = validation_metrics(data_loader_valid, model_v, model_t)
        recall_at_1_valid,recall_at_5_valid,recall_at_10_valid = metrics_valid[:3]

        writer.add_scalar(f'recall_at_1/train',recall_at_1_train, epoch)
        writer.add_scalar(f'recall_at_5/train',recall_at_5_train, epoch)
        writer.add_scalar(f'recall_at_10/train',recall_at_10_train, epoch)

        writer.add_scalar(f'recall_at_1/valid',recall_at_1_valid, epoch)
        writer.add_scalar(f'recall_at_5/valid',recall_at_5_valid, epoch)
        writer.add_scalar(f'recall_at_10/valid',recall_at_10_valid, epoch)
    
        # write train and valid loss to tensorboard
        for loss_idx, loss_type in zip(range(len(losses[0])),losses[0]):
            l_train = losses_avg[-1][loss_idx]
            l_valid = valid_losses_avg[loss_idx]
            if l_train > 0.0:
                writer.add_scalar(f'{loss_type}/train', l_train, epoch)
                writer.add_scalar(f'{loss_type}/valid', l_valid, epoch)
                       
    writer.flush()
    return model_v, model_t, losses, losses_avg

########################################

if __name__ == '__main__':
    ### python -W ignore train_dual_ae.py --n_epochs 15 --t_num_feats 512 --v_num_feats 2048 --batch_size_exp_min 5 --batch_size_exp_max 8
    ### best result experiment: 
    #     |  28       |  1.08     |  5.987    |  0.0001   |  1.0      |  0.0001   |


    parser = argparse.ArgumentParser ()
    parser.add_argument('--n_epochs', type = int, default = 20, help = 'number of iterations')
    parser.add_argument('--n_train_samples', type = int, default = None, help = 'number of training samples')
    
    parser.add_argument('--init_model_path', default = None, help = 'if None, a new model will be instantiated. If path is provided, additional training will be done on the existing model.')

    # active losses
    parser.add_argument('--activate_all_losses', action='store_true', help = 'enables training using all the losses below')

    parser.add_argument('--activate_reconst_t', action='store_true', help = 'enables training using text reconst loss')
    parser.add_argument('--activate_reconst_v', action='store_true', help = 'enables training using video reconst loss')
    
    parser.add_argument('--activate_cross_t', action='store_true', help = 'enables training using text cross loss')
    parser.add_argument('--activate_cross_v', action='store_true', help = 'enables training using video cross loss')

    parser.add_argument('--activate_cycle_t', action='store_true', help = 'enables training using text cycle loss')
    parser.add_argument('--activate_cycle_v', action='store_true', help = 'enables training using video cycle loss')
    
    parser.add_argument('--activate_joint', action='store_true', help = 'enables training using joint loss')
    
    parser.add_argument('--activated_losses_binary_min', type=int, default = 127, help = 'its binary indicates which losses to activate')
    parser.add_argument('--activated_losses_binary_max', type=int, default =127,  help = 'its binary indicates which losses to activate')
    
    # loss criterion
    parser.add_argument('--loss_criterion', default = 'mse') # MSELoss
    
    # lr step size
    parser.add_argument('--lr_step_size_min', type = int, default = 1, help = 'lr schedule: step size lower bound')
    parser.add_argument('--lr_step_size_max', type = int, default = 10, help = 'lr schedule: step size upper bound')
    
    # lr gamma
    parser.add_argument('--lr_gamma', type = float, default = 0.1, help = 'lr schedule: gamma')
    
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
    parser.add_argument('--bayes_n_iter', type = int, default = 0, help = 'bayesian optimization num iterations')
    parser.add_argument('--bayes_init_points', type = int, default = 1, help = 'bayesian optimization init points')
    
    # io params
    parser.add_argument('--repo_dir', default = '/usr/local/data02/zahra/datasets/Tempuckey/sentence_segments')
    parser.add_argument('--video_feats_dir', default = 'feats/video/r2plus1d_resnet50_kinetics400')
    parser.add_argument('--text_feats_path', default = 'feats/text/universal/sentence_feats.pkl')
    parser.add_argument('--train_split_path', default = 'train.split.pkl')    
    parser.add_argument('--valid_split_path', default = 'valid.split.pkl')
    parser.add_argument('--output_path', default = '/usr/local/extstore01/zahra/Video-Text-Retrieval_OOD/output')

    parser.add_argument('--relevance_score_min', type = float, default = 0.01, help = 'relevance score in range (0.0, 1.0)')
    parser.add_argument('--relevance_score_max', type = float, default = 0.01, help = 'relevance score in range (0.0, 1.0)')
    
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
    
    if args.init_model_path is not None:
        model_name = args.init_model_path.split('/')[-1]
        model_t_path = f'{repo_dir}/{args.init_model_path}/model_t_{model_name}.sd'
        model_v_path = f'{repo_dir}/{args.init_model_path}/model_v_{model_name}.sd'
        load_existing_model = True
    else:
        load_existing_model = False
        
    bayes_init_points = args.bayes_init_points
    bayes_n_iter = args.bayes_n_iter
    
    loss_coefs = (1,1,1,1)
    
    loss_criterion = args.loss_criterion

    activate_all_losses = args.activate_all_losses
    activated_losses_binary_min = args.activated_losses_binary_min
    activated_losses_binary_max = args.activated_losses_binary_max
        
#     # joint_active,reconst_v_active,reconst_t_active,cross_v_active,cross_t_active,cycle_v_active,cycle_t_active
#     if args.activate_all_losses:
#         activated_losses = (True, True, True, True, True, True, True)
#     else:
#         activated_losses = (args.activate_joint, args.activate_reconst_v, args.activate_reconst_t, args.activate_cross_v, args.activate_cross_t, args.activate_cycle_v, args.activate_cycle_t)
    
    # bounds of parameter space
    pbounds = {'lr': (lr_min, lr_max), 
               'lr_step_size': (lr_step_size_min, lr_step_size_max), 
               'weight_decay':(weight_decay_min, weight_decay_max), 
               'batch_size_exp': (batch_size_exp_min, batch_size_exp_max), 
               'activated_losses': (activated_losses_binary_min,activated_losses_binary_max),
               'relevance_score': (relevance_score_min,relevance_score_max)
              }

    optimizer = BayesianOptimization(
        f=optimize_vtr_model,
        pbounds=pbounds,
        random_state=42,
    )

    optimizer.maximize(
        init_points=bayes_init_points,
        n_iter=bayes_n_iter,
    )

