
#### set up python environment
`virtualenv --system-site-packages -p python3 env_vtr`

`source env_vtr/bin/activate`

`pip install -r requirements.txt`

### Self-supervised representation learning using a variation of the BarlowTwins
```
python -W ignore train_bt.py --n_epochs 100 --t_num_feats 512 --v_num_feats 2048 --batch_size_exp_min 7 --batch_size_exp_max 7 --lr_min 0.0001 --lr_max 0.001 --weight_decay_min 0.00001 --weight_decay_max 0.001 --lr_step_size_min 50 --lr_step_size_max 400 --lr_gamma 0.9 --relevance_score_min 0.00001 --relevance_score_max 0.0001 --shuffle
```

#### Self-supervised representation learning on MSRVTT dataset
```
python -W ignore train_bt_msrvtt.py --n_epochs 10 --t_num_feats 512 --v_num_feats 2048 --batch_size_exp_min 7 --batch_size_exp_max 7 --lr_min 0.0001 --lr_max 0.001 --weight_decay_min 0.00001 --weight_decay_max 0.001 --lr_step_size_min 50 --lr_step_size_max 400 --lr_gamma 0.9 --shuffle
```

### Train video-text retrieval model 
It initilaizes the vtr model using the video and text encoder weights from the SSL training. It loads the previously generated models at `output/experiments/<exp_name>/model_v2r.sd` and `output/experiments/<exp_name>/model_t2r.sd`

```
python train_vt.py
```





##### following is defunct
### Train V2T model
```
python -W ignore train_v2t.py --n_epochs 500 --t_num_feats 512 --v_num_feats 2048 --batch_size_exp_min 7 --batch_size_exp_max 7 --lr_min 0.0001 --lr_max 0.001 --weight_decay_min 0.00001 --weight_decay_max 0.001 --lr_step_size_min 50 --lr_step_size_max 400 --lr_gamma 0.9 --relevance_score_min 0.4 --relevance_score_max 0.40001 --shuffle --loss_criterion mse
```
### Train T2V model
```
python -W ignore train_t2v.py --n_epochs 500 --t_num_feats 512 --v_num_feats 2048 --batch_size_exp_min 7 --batch_size_exp_max 7 --lr_min 0.0001 --lr_max 0.001 --weight_decay_min 0.00001 --weight_decay_max 0.001 --lr_step_size_min 50 --lr_step_size_max 400 --lr_gamma 0.9 --relevance_score_min 0.4 --relevance_score_max 0.40001 --shuffle --loss_criterion mse
```

The resulting models will be stored at `output/experiments`

### Evaluate metrics on test set
```python test_metrics.py --test_split_path test.clean.split.pkl```


Train the model with all the ae loss functions activated:
`python -W ignore train_dual_ae.py --n_epochs 10 --t_num_feats 512 --v_num_feats 2048 --batch_size_exp_min 4 --batch_size_exp_max 8 --lr_min 0.01 --lr_max 0.00001 --weight_decay_min 0.01 --weight_decay_max 0.00001 --lr_step_size_min 1 --lr_step_size_max 10  --activated_losses_binary_min 127 --activated_losses_binary_max 127`

The `train_v2t.py` and `train_t2v.py` use a simple MLP model for video to text and text to video mapping. While the `train_dual_ae.py` trains two autoencoders simultaneously for v2t and t2v.

In order to evaluate the out of domain sentence detection model on Video Text Retireval (VTR) task, we have created a new presentation (format) of the Tempuckey dataset (in this repo) that generates (cuts) a short video segment per each sentence in the subtitles. This format allows us to create a Video Text Retreival model to retreive the videos using the sentences that the annotators say and vice versa.
The video segments (one per sentence) are stored at `/usr/local/data02/zahra/datasets/Tempuckey/sentence_segments/video_segments` and the annotations (inclusing the sentences and their start/end time associated with each video) are stored at `/usr/local/data02/zahra/datasets/Tempuckey/sentence_segments/sentences.pkl`

The video and text features can be found at `/usr/local/data02/zahra/datasets/Tempuckey/sentence_segments/feats`

*note: some c3d video features are empty due to an issue with the feature extractor that relies on videos with a certain minimum length.


### One-time process (no need to run the following again):

#### create sentence-clip correspondence dictionary (sentence segments)
`generate_video_text_correpondence.py`

`generate_sentence_segments.py`

#### generate clip segments (video cuts) for each sentence
`generate_video_segments.py`

#### extract video and text features for each clip and sentence
`python3 extract_sent_feats.py`

`python3 extract_c3d_feats.py`
