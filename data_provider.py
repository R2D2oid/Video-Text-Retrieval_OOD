import torch
import pickle as pkl
from torch.utils.data import Dataset
import utilities as utils
import numpy as np

class TempuckeyVideoSentencePairsDataset(Dataset):
    '''
    Loads video C3D and sentence features for all sentence/video segment pair from Tempuckey Video Sentence Pairs Dataset
    '''
    def __init__(self, vid_feats_dir, txt_feats_path, split_ids, video_feat_seq_len, sent_feat_seq_len, transform=None):
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
        videos = utils.load_video_feats(vid_feats_dir) 
        sents  = utils.load_picklefile(txt_feats_path)
        
        # remove video ids with an invalid feature vector
        videos, sents = remove_invalid_video_sentence_features(videos, sents)
  
        self.videos = videos
        self.sents  = sents
        self.transform = transform()
        self.split_ids = split_ids # a list of id names associated with each video/sentence pair 

        self.video_feat_seq_len = video_feat_seq_len
        self.sent_feat_seq_len = sent_feat_seq_len
        
    def __len__(self):
        return len(self.split_ids)

    def __getitem__(self, idx):
        '''
        Input:
            idx: integer index
        Output:
            sample: dict containing a video feature vector and a sentence feature vector
        '''
        # idx_name is the name of the video/sentence
        # ex. 312_TRIPPING_2017-11-16-phi-wpg-home_00_17_33.052000_to_00_17_39.125000.mp4_00:17:46.532000_00:17:52.171000
        idx_name = self.split_ids[idx] 
        
        vid_feat = self.videos[idx_name]
        snt_feat = self.sents[idx_name].reshape(1,-1)
        
        vid_feat = unify_embedding_length(vid_feat, self.video_feat_seq_len)
        snt_feat = unify_embedding_length(snt_feat, self.sent_feat_seq_len)
        
        self.dataset_stats = self.get_dataset_mean_std()
        
        sample = {'video': vid_feat, 'sent': snt_feat}

        if self.transform:
            sample = self.transform(sample, self.dataset_stats)
            
        return sample

    def get_dataset_mean_std(self):
        '''
        Computes mean and standard deviation for video and sentence features, separately.
        '''
        # videos
        feat_vecs = []
        for k,v in self.videos.items():
            for vec in v:
                feat_vecs.append(vec)
        feat_vecs = np.array(feat_vecs)
        v_mean = feat_vecs.mean(axis=0)
        v_std = feat_vecs.std(axis=0)

        # sentences
        feat_vecs = []
        for k,v in self.sents.items():
            feat_vecs.append(v)
        feat_vecs = np.array(feat_vecs)
        t_mean = feat_vecs.mean(axis=0)
        t_std = feat_vecs.std(axis=0)

        return {'videos': (v_mean, v_std), 'sents': (t_mean, t_std)}

    
class Normalize_VideoSentencePair(object):
    '''
    Normalizes the input sample using the dataset mean and std
    '''
    
    def __call__(self, sample, dataset_stats):
        
        v_mean, v_std = dataset_stats['videos']
        t_mean, t_std = dataset_stats['sents']

        sample['video'] = (sample['video'] - v_mean)/v_std
        sample['sent'] = (sample['sent'] - t_mean)/t_std

        return {'video': sample['video'], 'sent': sample['sent']}
    

def unify_embedding_length(emb, target_len):
    '''
    Unify feat size to ensure all embeddings are n_feats x T
        if embedding is smaller, then augment it with zeros at the end
        if embedding is larger, crop the extra rows
    '''
    emb_len, num_feats = emb.shape
    if emb_len < target_len:
        len_diff = target_len - emb_len
        zero_padding = np.zeros([len_diff, num_feats])
        return torch.tensor(np.vstack((emb, zero_padding))).float()
    else:
        return torch.tensor(emb[0:target_len]).float()
    
def remove_invalid_video_sentence_features(vids, caps):
    '''
    Remove invalid feature vector entries, including the empty vectors
    Verifies that each video has a corresponding caption 
    '''
    # remove video-captions that have an empty c3d features
    empty_ids = [k for k,v in vids.items() if len(v)==0]
    for e in empty_ids:
        del vids[e]

    # if there are any differences between txt and vids remove those txt entries
    caps_k = set(caps.keys())
    vids_k = set(vids.keys())

    diff = caps_k - caps_k.intersection(vids_k)
    for k in diff:
        del caps[k]
    
    assert len(vids)==len(caps), 'vids and caps should correspond!'
    
    return vids, caps