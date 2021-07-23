import pickle as pkl
import numpy as np
from os import listdir, path, makedirs
from os.path import isfile, join
import subprocess
import csv
import time 
import datetime

def get_filenames(dir_):
    return listdir(dir_)

def get_filepaths(dir_):
    return ['{}/{}'.format(dir_, f) for f in get_filenames(dir_)]

def get_subfolders_path(path_):
    '''
        Given a path returns all the folders under that path
        Input: 
            folderpath
        Output: 
            a list of subfolders
    '''
    return [join(path_, f) for f in listdir(path_) if not isfile(join(path_, f))]

def create_dir_if_not_exist(path_):
    if not path.exists(path_):
        makedirs(path_)

def load_picklefile(path_):
    with open(path_, 'rb') as f:
        return pkl.load(f)

def dump_picklefile(object_, path_):
    with open(path_, 'wb') as f:
        pkl.dump(object_, f, protocol = 3)
    return True

def load_textfile(path_):
    with open(path_, 'r') as f:
        lines = f.readlines()
    return [l.strip() for l in lines]

def load_csvfile(path_, delim = ','):
    lines = []
    with open(path_) as csvfile:
        csvfile = csv.reader(csvfile, delimiter=delim)
        for row in csvfile:
            lines.append(row)
    return lines

def dump_textfile(data, path_):
    with open(path_, 'w') as f:
        for item in data:
            if item is not None:
                f.write(item+'\n')
    return True

def load_precomputed_embeddings(dir_, low_memory_mode = False):
    filepaths = get_filepaths(dir_)
    precomputed_embeddings = []
    for path_ in filepaths:
        print('loading {}'.format(path_))
        sentence_embeddings = load_picklefile(path_)
        precomputed_embeddings.extend(sentence_embeddings)

        if low_memory_mode and len(precomputed_embeddings) > 100000:
            print('low memory mode is active.')
            print('a smaller subset of the embeddings are returned inconsideration for memory usage')
            return precomputed_embeddings
    return precomputed_embeddings

def sentence_contains_term(sentence, terms):
    for t in terms:
        if t in sentence:
            return 1
    return 0

def replace_words_in_sentence(s, words, replacement_token = 'UNK', emoticons = False):
    if emoticons:
        for w in words:
            s = s.replace(w, replacement_token)
        return s
    else:
        for w in words:
            s = s.replace(' '+w+' ', ' {} '.format(replacement_token))
            if s.startswith(w+' '):
                idx = len(w) + 1
                s = '{} {}'.format(replacement_token, s[idx:])
        return s

def csv_to_dict(csv_):
    '''
    first column is used as the key
    the remaining columns are used as the value
    '''
    dct = {}
    for row in csv_:
        dct[row[0]] = row[1:]
    return dct

def run_cmd(cmd):
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output, error = process.communicate()
    return output,error
  
# def ffmpeg_cut(src, dest, start_time, duration_time):  
    # cmd = 'ffmpeg -ss {} -i \"{}\" -c copy -t {} \"{}\"'.format(start_time, src, duration_time, dest)
def ffmpeg_cut(src, dest, start_time, end_time):  
    cmd = 'ffmpeg -i \"{}\" -ss {} -to {} -c copy  \"{}\"'.format(src, start_time, end_time, dest)
    print(cmd)
    return run_cmd(cmd)

def ffmpeg_cut_re_encode(src, dest, start_time, end_time):  
    # ffmpeg -i inputVideo.mp4 -ss 00:03 -to 00:08 -c:v libx264 outputVideo_trimmed_opseek_encode.mp4
    cmd = 'ffmpeg -ss {} -i \"{}\" -to {} -c:v libx264  \"{}\"'.format(start_time, src, end_time, dest)
    print(cmd)
    return run_cmd(cmd)

def seconds_to_time(seconds): 
    return time.strftime('%H:%M:%S.%M', time.gmtime(seconds))
      
def seconds_to_dt_time(seconds):
    tt = seconds_to_time(seconds)
    hh,mm,ss = tt.split(':')
    ss,_ = ss.split('.')

    return datetime.time(hour=int(hh),minute=int(mm),second=int(ss))

def load_video_feats(vid_feat_dir):
    feat_type = vid_feat_dir.split('/')[-1]
    vid_feats = {}
    if feat_type == 'c3d' or feat_type.startswith('r2plus1d'):
        fnames = get_filenames(vid_feat_dir)
        for fname in fnames:
            id_ = fname[:-8]
            if feat_type.startswith(''):
                id_ = fname[30:-13]
            with open(f'{vid_feat_dir}/{fname}','rb') as f:
                vid_feat = np.load(f)
                vid_feats[id_] = vid_feat
    elif feat_type == 'resnet18':
        vid_dirs = get_subfolders_path(vid_feat_dir)
        for v in vid_dirs:
            id_ = v.split('/')[-1][:-4]
            fnames = get_filenames(v)
            vid_feat = []
            for f in fnames:
                with open(f'{v}/{f}','rb') as f:
                    feat = np.load(f)
                    vid_feat.append(feat)
            vid_feats[id_] = np.array(vid_feat)
    else:
        raise ValueError(f'Unknown {feat_type}!')
    return vid_feats
