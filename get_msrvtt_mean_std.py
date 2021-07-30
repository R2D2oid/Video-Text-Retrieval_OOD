import argparse
import torch 
from torchvision import transforms
from msrvtt_dataset import MSRVTTDataset as MSRVTT
from msrvtt_dataset import Standardize_VideoSentencePair, ToTensor_VideoSentencePair
from utils.train_utils import get_dataloader_msrvtt
from utils.sys_utils import dump_picklefile
    
# stores training data stats to MSRVTT folder for future use
# python get_msrvtt_mean_std.py 
if __name__ == '__main__':
    parser = argparse.ArgumentParser ()
    parser.add_argument('--repo_dir', default = '/usr/local/extstore01/zahra/datasets/MSRVTT')
    parser.add_argument('--video_feats_dir', default = 'feats/video/r2plus1d_TrainVal')
    parser.add_argument('--text_feats_path', default = 'feats/text/msrvtt_captions_universal_trainval.pkl')    
    parser.add_argument('--split_path', default = 'TrainVal_videoid_sentid.txt')    
    parser.add_argument('--output_path', default = '/usr/local/extstore01/zahra/datasets/MSRVTT/dataset_stats.pkl')

    args = parser.parse_args()
        
    repo_dir = args.repo_dir
    split_path = f'{repo_dir}/{args.split_path}'
    output_path = args.output_path
    t_feats_path = f'{repo_dir}/{args.text_feats_path}'
    v_feats_dir = f'{repo_dir}/{args.video_feats_dir}'
    
    dataset = MSRVTT(vid_feats_dir=v_feats_dir, txt_feats_path=t_feats_path, ids_path=split_path, transform=None)
    dataset_stats = dataset.get_dataset_mean_std()
    
    dump_picklefile(dataset_stats, output_path)
    
