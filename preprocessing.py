import torch 
import numpy as np
import utilities as utils

def preprocess_embeddings(embeddings, num_feats, T):
    target_len = T * num_feats
    processed_embeddings = []
    count = 0
    for emb in embeddings:
        emb = emb.reshape(-1)
        processed_emb = unify_embedding_length(emb, target_len)
        processed_emb = processed_emb.reshape(-1, num_feats)
        processed_embeddings.append(processed_emb)
        count = count + 1
        
    return processed_embeddings

# unify feat size to ensure all embeddings are 1024xT
# if embedding is smaller augment it with zeros at the end
# if embedding is larger crop the extra rows
def unify_embedding_length(emb, target_len):
    emb_len = len(emb)
    if emb_len < target_len:
        len_diff = target_len - emb_len
        zero_padding = np.zeros([len_diff])
        return torch.tensor(np.hstack((emb, zero_padding)))
    elif emb_len > target_len:
        return emb[0:target_len]
    else:
        return emb
    
def load_video_text_features(v_feats_dir, t_feats_path, n_feats_t, n_feats_v, T, L, train_split_path):
    # load pre-computed video features and text features
    vids = utils.load_video_feats(v_feats_dir) 
    caps = utils.load_picklefile(t_feats_path)
    
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
        
    ### create dataset based on training ids
    train_ids_ = utils.load_picklefile(train_split_path)
    
    videos = []
    captions = []
    for id_ in train_ids_:
        videos.append(vids[id_])
        captions.append(caps[id_])

    videos = preprocess_embeddings(videos, n_feats_v, T)
    captions = preprocess_embeddings(captions, n_feats_t, L)
    
    return videos, captions