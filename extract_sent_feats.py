import utilities as utils

def load_model(model_name = 'universal-sentence-encoder'):
	if model_name == 'universal-sentence-encoder':
		import tensorflow as tf
		import tensorflow_hub as hub
		model = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
	elif model_name == 'fasttext':
		import sister
		model = sister.MeanEmbedding(lang="en")

	return model

home_dir = '/home/pishu/Desktop/repos'
path_output = f'{home_dir}/VTR_OOD/output/sentence_segments'
path_input = f'{home_dir}/VTR_OOD/output/sentence_segments.pkl'

sent_segs = utils.load_picklefile(path_input)
encoder = load_model(model_name = 'fasttext')

id_to_feat = {}
for sent_id,v in sent_segs.items():
	snt_beg_ts = v['sent_start_time']
	snt_end_ts = v['sent_end_time']
	vid_beg_sec = v['vid_start_time']
	vid_end_sec = v['vid_end_time']
	vid_id = v['video_id'] 
	snt = v['sentence']

	snt_emb = encoder(snt)

	id_to_feat[sent_id] = snt_emb

path_ = f'{home_dir}/VTR_OOD/output/sent_feats.pkl'
utils.dump_picklefile(id_to_feat, path_)
print(f'generated {path_}')
	