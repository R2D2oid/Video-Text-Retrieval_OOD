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
            vid_feats_dir (string): Path to the video features directory
            txt_feats_path (string): Path to sentence features pickle file
            transform (callable, optional): Optional transform to be applied on a sample
        '''        
        # load pre-computed video features and text features
        videos = utils.load_video_feats(vid_feats_dir) 
        sents  = utils.load_picklefile(txt_feats_path)
        
        # remove video ids with an invalid feature vector
        videos, sents = remove_invalid_video_sentence_features(videos, sents)
  
        self.videos = videos
        self.sents  = sents
        self.split_ids = split_ids # a list of id names associated with each video/sentence pair 

        self.video_feat_seq_len = video_feat_seq_len
        self.sent_feat_seq_len = sent_feat_seq_len
        
    def __len__(self):
        return len(self.split_ids)

    def __getitem__(self, idx):
        # idx is an integer index
        # idx_name is the name of the video/sentence 
        # example: 312_TRIPPING_2017-11-16-phi-wpg-home_00_17_33.052000_to_00_17_39.125000.mp4_00:17:46.532000_00:17:52.171000
        idx_name = self.split_ids[idx]
        
        vid_feat = self.videos[idx_name][:self.video_feat_seq_len]
        snt_feat = self.sents[idx_name].reshape(1,-1)[:self.sent_feat_seq_len]
        
#         vid_feat = unify_embedding_length(vid_feat, self.video_feat_seq_len)
#         snt_feat = unify_embedding_length(snt_feat, self.sent_feat_seq_len)    
        
        sample = {'video': vid_feat, 'sent': snt_feat}
        return sample

    def get_dataset_mean_std(self):
        # compute mean and std for video feature vectors
        feat_vecs = []
        for k,v in self.videos.items():
            for vec in v:
                feat_vecs.append(vec)
        feat_vecs = np.array(feat_vecs)
        v_mean = feat_vecs.mean(axis=0)
        v_std = feat_vecs.std(axis=0)

        # compute mean and std for text feature vectors
        feat_vecs = []
        for k,v in self.sents.items():
            feat_vecs.append(v)
        feat_vecs = np.array(feat_vecs)
        t_mean = feat_vecs.mean(axis=0)
        t_std = feat_vecs.std(axis=0)

        return {'videos': (v_mean, v_std), 'sents': (t_mean, t_std)}

        
# unify feat size to ensure all embeddings are n_feats x T
# if embedding is smaller augment it with zeros at the end
# if embedding is larger crop the extra rows
def unify_embedding_length(emb, target_len):
    emb_len = len(emb)
    print('emb: ',emb_len)
    print('target: ', target_len)
    if emb_len < target_len:
        len_diff = target_len - emb_len
        zero_padding = np.zeros([len_diff])
        return torch.tensor(np.hstack((emb, zero_padding)))
    else:
        return torch.tensor(emb[0:target_len])
    
def remove_invalid_video_sentence_features(vids, caps):
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