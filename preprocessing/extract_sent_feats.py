import utilities as utils

def load_model(model_name = 'universal'):
    if model_name == 'universal': # universal-sentence-encoder'
        import tensorflow as tf
        import tensorflow_hub as hub
        model = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
    elif model_name == 'fasttext':
        import sister
        model = sister.MeanEmbedding(lang="en")
    else:
        raise ValueError(f'Unknown model name {model_name}!')
    return model

# model_name = 'universal' 
# universal does not work locally. 
# ran it on colab at: https://colab.research.google.com/drive/18PcCZiNsP6P078YlqxWw5mSLKnkzGG_Y#scrollTo=cPMCaxrZwp7t
# resulting feats stored at /usr/local/data02/zahra/datasets/Tempuckey/sentence_segments/feats/text/universal/sentence_feats.pkl
model_name = 'fasttext'

home_dir = '/usr/local/data02/zahra/datasets/Tempuckey/sentence_segments'
path_input = f'{home_dir}/sentences.pkl'
path_output = f'{home_dir}/feats/text/{model_name}'

encoder = load_model(model_name = model_name)
sent_segs = utils.load_picklefile(path_input)

id_to_feat = {}
for sent_id,v in sent_segs.items():
    snt_beg_ts = v['sent_start_time']
    snt_end_ts = v['sent_end_time']
    vid_beg_sec = v['vid_start_time']
    vid_end_sec = v['vid_end_time']
    vid_id = v['video_id'] 
    snt = v['sentence']
    
    if model_name == 'universal':
        snt_emb = encoder([snt])
    else:
        snt_emb = encoder(snt)
    id_to_feat[sent_id] = snt_emb
    
path_ = f'{path_output}/sent_feats.pkl'
utils.dump_picklefile(id_to_feat, path_)
print(f'generated {path_}')