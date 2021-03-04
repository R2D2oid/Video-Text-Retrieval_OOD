import argparse
import torch 
import torch.nn as nn
import numpy as np

import utilities as utils
from preprocessing import load_video_text_features
from layers.AEwithAttention import AEwithAttention

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
    parser.add_argument('--output_path', dest = 'output_path', default = '/usr/local/extstore01/zahra/Video-Text-Retrieval_OOD/output/')

  
    ### get args
    args = parser.parse_args()

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
    loss_1_recons1 = []
    loss_1_recons2 = []
    loss_2_joint = []
    loss_3_cross1 = []
    loss_3_cross2 = []
    loss_4_cycle1 = []
    loss_4_cycle2 = []
    loss_all = []

    for epoch in range(n_epochs):
        counter = 1
        for v,t in zip(vids,caps):
            # Forward pass        
            v = torch.tensor(v).float()
            t = torch.tensor(t).float()

            # Compute recons loss 
            dims_v = v.shape
            dims_t = t.shape

            v_reconst = model_v(v)
            v_reconst = v_reconst.reshape(dims_v[0], dims_v[1])

            t_reconst = model_t(t)
            t_reconst = t_reconst.reshape(dims_t[0], dims_t[1])

            loss_recons_v = criterion(v_reconst, v)
            loss_recons_t = criterion(t_reconst, t)

            loss_1_recons1.append(loss_recons_v)
            loss_1_recons2.append(loss_recons_t)

            loss_recons = loss_recons_v + loss_recons_t
            # the following losses require paired video/caption data (v and t)
            # model_v and model_t are the corresponding models for video and captions respectively

            # Compute joint loss
            loss_joint = criterion(model_v.encoder(v), model_t.encoder(t))

            loss_2_joint.append(loss_joint)

            # Compute cross loss
            loss_cross1 = criterion(model_t.decoder(model_v.encoder(v)).reshape(dims_t[0], dims_t[1]), t)
            loss_cross2 = criterion(model_v.decoder(model_t.encoder(t)).reshape(dims_v[0], dims_v[1]), v)
            loss_cross = loss_cross1 + loss_cross2

            loss_3_cross1.append(loss_cross1)
            loss_3_cross2.append(loss_cross2)

            # Compute cycle loss
            loss_cycle1 = criterion(model_t.decoder(model_v.encoder(model_v.decoder(model_t.encoder(t)))).reshape(dims_t[0], dims_t[1]), t)
            loss_cycle2 = criterion(model_v.decoder(model_t.encoder(model_t.decoder(model_v.encoder(v)))).reshape(dims_v[0], dims_v[1]), v)
            loss_cycle = loss_cycle1 + loss_cycle2

            loss_4_cycle1.append(loss_cycle1)
            loss_4_cycle2.append(loss_cycle2)

            # set hyperparams 
            a1, a2, a3 = 1, 1, 1

            # Compute total loss
            loss = loss_recons + a1 * loss_joint + a2 * loss_cross + a3 * loss_cycle

            loss_all.append(loss)
            # loss = loss_recons

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

            print ('Epoch[{}/{}], Step[{}/{}] Loss: {}\n'.format(epoch + 1, n_epochs, counter, len(vids), loss.item()))

            counter = counter + 1    

    losses = {}
    losses['loss_1_recons1'] = loss_1_recons1
    losses['loss_1_recons2'] = loss_1_recons2
    losses['loss_2_joint'] = loss_2_joint
    losses['loss_3_cross1'] = loss_3_cross1
    losses['loss_3_cross2'] = loss_3_cross2
    losses['loss_4_cycle1'] = loss_4_cycle1
    losses['loss_4_cycle2'] = loss_4_cycle2
    losses['loss_all'] = loss_all
    
    # save experiment configs and results
    torch.save(model_v.state_dict(), f'{output_path}/model_v.sd')
    torch.save(model_t.state_dict(), f'{output_path}/model_t.sd')
    utils.dump_picklefile(losses, f'{output_path}/losses.pkl')


