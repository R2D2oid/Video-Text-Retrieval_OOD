
#### set up python environment
`virtualenv --system-site-packages -p python3 env_vtr`

`source env_vtr/bin/activate`

`pip install -r requirements.txt`

#### create sentence-clip correspondence dictionary (sentence segments)
`generate_video_text_correpondence.py`

`generate_sentence_segments.py`

#### generate clip segments (video cuts) for each sentence
`generate_video_segments.py`

#### extract video and text features for each clip and sentence
`python3 extract_sent_feats.py`

`python3 extract_c3d_feats.py`

#### train vtr model to learn a joint video-text subspace mapping function
`Train_VTR_AE_with_Attention.ipynb`
