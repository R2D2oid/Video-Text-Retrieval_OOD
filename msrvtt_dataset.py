import torch
import pickle as pkl
from torch.utils.data import Dataset
import utils.sys_utils as utils
import numpy as np

class MSRVTTDataset(Dataset):
    '''
    Loads video and sentence features for all sentence/video segment pair from MSRVTT Video Caption Pairs Dataset.
    
    Example Usage: 
        import torch
        from msrvtt_dataset import MSRVTTDataset as MSRVTT
        from utils.train_utils import get_dataloader

        repo_dir = '/usr/local/extstore01/zahra/datasets/MSRVTT/'
        video_feats_dir = f'{repo_dir}/feats/video/r2plus1d_TrainVal'
        text_feats_path = f'{repo_dir}/feats/text/msrvtt_captions_np.pkl'
        ids_path = f'{repo_dir}/TrainVal_videoid_sentid.txt'

        dl_params = {'batch_size': 64,
                     'shuffle': True,
                     'num_workers': 1}

        dataset = MSRVTT(vid_feats_dir=video_feats_dir, txt_feats_path=text_feats_path, ids_path=ids_path, transform=None)
        dataloader = torch.utils.data.DataLoader(dataset, **dl_params)
    '''
    def __init__(self, vid_feats_dir, txt_feats_path, ids_path, transform=None):
        '''
        Args:
            vid_feats_dir (string): str path to the video features directory
            txt_feats_path (string): str path to sentence features pickle file
            split_ids: list of video/sentence names in the split (ex. train/test/valid)
            video_feat_seq_len: int length of the video feature vector sequence
            sent_feat_seq_len: int length of the sentence feature vector sequence
            transform (callable, optional): Optional transform to be applied on a sample
        '''        
        # load pre-computed video features and text features
        vids = utils.load_video_feats(vid_feats_dir) 
        sens = utils.load_picklefile(txt_feats_path)
        vidid_sentid = utils.load_textfile(ids_path)  
        
        self.s2v_id = {}
        self.sen_id = []
        for item in vidid_sentid:
            vidid, senid = item.split('_')
            self.s2v_id[senid] = vidid
            self.sen_id.append(senid) 

        self.videos = vids
        self.transform = transform
        self.sents = {}
        for feat,senid in zip(sens,self.sen_id):
            self.sents[senid] = feat
        
        
    def __len__(self):
        return len(self.sen_id)

    def __getitem__(self, idx):
        '''
        Input:
            idx: integer index
        Output:
            sample: dict containing a video feature vector and a sentence feature vector
        '''
        sen_id = self.sen_id[idx] 
        vid_id = self.s2v_id[sen_id]
        
        v = self.videos[vid_id].squeeze()
        t = self.sents[sen_id].squeeze()
        
        v = torch.tensor(v).float()
        t = torch.tensor(t).float()
        
        # sample = {'id': vid_id+'_'+sen_id, 'video': torch.tensor(vid_feat).float(), 'sent': torch.tensor(snt_feat).float()}

        if self.transform is not None:
            (v,t) = self.transform((v,t))
            
        v = v.squeeze()
        t = t.squeeze()
        
        return (v,t)
    
    def get_dataset_mean_std(self):
        '''
        Computes mean and standard deviation for video and sentence features, separately.
        '''
        # videos
        feats = np.array([f for f in self.videos.values()])

        v_mean = feats.mean(axis=0)
        v_std  = feats.std(axis=0)

        # sentences
        feats = np.array([f for f in self.sents.values()])

        t_mean = feats.mean(axis=0)
        t_std  = feats.std(axis=0)

        return {'videos': (v_mean, v_std), 'sents': (t_mean, t_std)}

    
class Standardize_VideoSentencePair(object):
    '''
    Standardizes the input sample using the dataset mean and std
    '''
    
    def __init__(self, dataset_stats):
        self.v_mean, self.v_std = dataset_stats['videos']
        self.t_mean, self.t_std = dataset_stats['sents']
        
    def __call__(self, sample):
                
        v,t = sample
        
        v = (v - self.v_mean)/self.v_std
        t = (t - self.t_mean)/self.t_std

        return (v,t)
    
    
class ToTensor_VideoSentencePair(object):
    '''
    Converts video sentence pair sample to tensor
    '''
    
    def __call__(self, sample):
        v,t = sample
        
        v = torch.tensor(v)
        t = torch.tensor(t)
        
        return (v,t)