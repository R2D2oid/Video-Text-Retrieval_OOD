
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

In order to evaluate the out of domain sentence detection model on Video Text Retireval (VTR) task, we have created a new presentation (format) of the Tempuckey dataset (in this repo) that generates (cuts) a short video segment per each sentence in the subtitles. This format allows us to create a Video Text Retreival model to retreive the videos using the sentences that the annotators say and vice versa.
The video segments (one per sentence) are stored at `/usr/local/extstore01/zahra/Video-Text-Retrieval_OOD/output/sentence_segments/` and the sentences associated with each video are stored at `/usr/local/extstore01/zahra/Video-Text-Retrieval_OOD/output/sentence_segments.pkl`

*note: some c3d video features are empty due to an issue with the feature extractor that relies on videos with a certain minimum length.
