import torch
import pickle as pkl
from torch.utils.data import Dataset
import utils.sys_utils as utils
import numpy as np

class TempuckeyVideoSentencePairsDataset(Dataset):
    '''
    Loads video C3D and sentence features for all sentence/video segment pair from Tempuckey Video Sentence Pairs Dataset.
    
    Example Usage: 
        from data_provider import TempuckeyVideoSentencePairsDataset as TempuckeyDataset
        from data_provider import Normalize_VideoSentencePair, ToTensor_VideoSentencePair

        my_transforms = [Normalize_VideoSentencePair(), ToTensor_VideoSentencePair()]
        dataset_train = TempuckeyDataset(v_feats_dir, t_feats_path, ids_train, video_feat_seq_len=2, sent_feat_seq_len=2, transform=my_transforms)
    '''
    def __init__(self, vid_feats_dir, txt_feats_path, split_ids, video_feat_seq_len, sent_feat_seq_len, transform=None, relevance_score=0.0):
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
        self.transform = transform
        self.split_ids = split_ids # a list of id names associated with each video/sentence pair 

        self.video_feat_seq_len = video_feat_seq_len
        self.sent_feat_seq_len = sent_feat_seq_len
        
        self.relevance_score = relevance_score
               
        if self.relevance_score > 0.0:
            self.split_ids = filter_dataset_by_relevance_score(self.split_ids, self.relevance_score)
            
        print(f'len dataset: {len(self.split_ids)}')
        
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
        
        vid_feat = self.videos[idx_name].squeeze()
        snt_feat = self.sents[idx_name].squeeze()

#         vid_feat = unify_embedding_length(vid_feat, self.video_feat_seq_len)
#         snt_feat = unify_embedding_length(snt_feat, self.sent_feat_seq_len)
        
        sample = {'id': idx_name, 'video': torch.tensor(vid_feat).float(), 'sent': torch.tensor(snt_feat).float()}

#         sample = {'id': idx_name, 'video': vid_feat, 'sent': snt_feat}
#         self.dataset_stats = self.get_dataset_mean_std()
#         for trnsfrm in self.transform:
#             sample = trnsfrm(sample, dataset_stats=self.dataset_stats)

        sample = {'id': idx_name ,'video': torch.tensor(sample['video']).float(), 'sent': torch.tensor(sample['sent']).float()}
        
#         return sample
        
        v = sample['video']
        t = sample['sent']

#         v = torch.tensor(v).float()
#         t = torch.tensor(t).float()
        
        return (v,t)

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


def filter_dataset_by_relevance_score(split_ids, min_score):
    included_ids = []
    sents = utils.load_picklefile('/usr/local/data02/zahra/datasets/Tempuckey/sentence_segments/sentences.pkl')
    scores = utils.load_picklefile('/usr/local/data02/zahra/datasets/Tempuckey/sentence_segments/sentence2relevancescore_mapping.pkl')
    for id_ in split_ids:
        s = sents[id_]['sentence']             
        if scores[s] > min_score:
            included_ids.append(id_)

    return included_ids

class Normalize_VideoSentencePair(object):
    '''
    Normalizes the input sample using the dataset mean and std
    '''
    
    def __call__(self, sample, **args):
    
        if 'dataset_stats' not in args.keys():
            raise exception('Missing argument: dataset_stats dict, format: {"videos":(mean,std), "sents":(mean,std)}')
            
        dataset_stats = args['dataset_stats']
        v_mean, v_std = dataset_stats['videos']
        t_mean, t_std = dataset_stats['sents']

        sample['video'] = (sample['video'] - v_mean)/v_std
        sample['sent'] = (sample['sent'] - t_mean)/t_std

        return {'video': sample['video'], 'sent': sample['sent']}
    
    
class ToTensor_VideoSentencePair(object):
    '''
    Converts video sentence pair sample to tensor
    '''
    
    def __call__(self, sample, **args):
        return {'video': torch.tensor(sample['video']), 'sent': torch.tensor(sample['sent'])}
    
# def unify_embedding_length(emb, target_len):
#     '''
#     Unify feat size to ensure all embeddings are n_feats x T
#         if embedding is smaller, then augment it with zeros at the end
#         if embedding is larger, crop the extra rows
#     '''
#     emb_len, num_feats = emb.shape
#     if emb_len < target_len:
#         len_diff = target_len - emb_len
#         zero_padding = np.zeros([len_diff, num_feats])
#         return np.vstack((emb, zero_padding))
#     else:
#         return np.array(emb[0:target_len])
    
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