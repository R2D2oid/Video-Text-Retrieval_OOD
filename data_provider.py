import torch
import pickle as pkl
from torch.utils.data import Dataset
import utilities as utils

class TempuckeyVideoSentencePairsDataset(Dataset):
    '''
    Loads video C3D and sentence features for all sentence/video segment pair from Tempuckey Video Sentence Pairs Dataset
    '''
    def __init__(self, vid_feats_dir, txt_feats_path, split_ids, transform=None):
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

        self.transform = transform

    def __len__(self):
        return len(self.split_ids)

    def __getitem__(self, idx):
        # idx is an integer index
        # idx_name is the name of the video/sentence 
        # example: 312_TRIPPING_2017-11-16-phi-wpg-home_00_17_33.052000_to_00_17_39.125000.mp4_00:17:46.532000_00:17:52.171000
        idx_name = self.split_ids[idx]
        sample = {'video': self.videos[idx_name][0], 'sent': self.sents[idx_name]}
        return sample
    

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