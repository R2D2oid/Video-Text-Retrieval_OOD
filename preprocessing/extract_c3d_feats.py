import utilities as utils

fnames = utils.get_filenames('/usr/local/data01/zahra/repos/ig65m-pytorch/zv_data/sentence_segments')
feats_fnames = utils.get_filenames('/usr/local/data01/zahra/repos/ig65m-pytorch/zv_data/features_c3d')

for v in fnames:
    # PWD must be /usr/local/data01/zahra/repos/ig65m-pytorch/zv_data
    feat_fname = f'{v}.npy'
    if feat_fname in feats_fnames:
        print(f'{feat_fname} previously extracted. skipping.')
        continue

    cmd = f'docker run --runtime=nvidia --ipc=host -v $PWD:/zv_data moabitcoin/ig65m-pytorch:latest-gpu extract /zv_data/sentence_segments/{v} /zv_data/features_c3d/{v}.npy'
    utils.run_cmd(cmd)
