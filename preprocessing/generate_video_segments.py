import utilities as utils
import datetime
import pdb

home_dir = '/home/pishu/Desktop/repos'
path_tempuckey = f'{home_dir}/datasets/Tempuckey'
path_output = f'{path_tempuckey}/VTR_OOD_output_segments'
path_input = f'{path_tempuckey}/sentence_segments.pkl'

sent_segs = utils.load_picklefile(path_input)

misalignment_param = 0 # seconds

for sent_id,v in sent_segs.items():
	snt_beg_ts = v['sent_start_time']
	snt_end_ts = v['sent_end_time']
	vid_beg_sec = v['vid_start_time']
	vid_end_sec = v['vid_end_time']
	vid_id = v['video_id'] 
	snt = v['sentence']

	clip_path = f'{path_tempuckey}/videos/{vid_id}'

	vid_beg_ts = utils.seconds_to_dt_time(vid_beg_sec)
	vid_beg_td = datetime.timedelta(hours=vid_beg_ts.hour, minutes=vid_beg_ts.minute, seconds=vid_beg_ts.second)

	vid_end_ts = utils.seconds_to_dt_time(vid_end_sec)
	vid_end_td = datetime.timedelta(hours=vid_end_ts.hour, minutes=vid_end_ts.minute, seconds=vid_end_ts.second)

	snt_beg_td = datetime.timedelta(hours=snt_beg_ts.hour, minutes=snt_beg_ts.minute, seconds=snt_beg_ts.second)
	snt_end_td = datetime.timedelta(hours=snt_end_ts.hour, minutes=snt_end_ts.minute, seconds=snt_end_ts.second)

	duration = snt_end_td - snt_beg_td

	clp_beg_td = snt_beg_td - vid_beg_td
	clp_end_td = snt_end_td - vid_beg_td

	if clp_beg_td.days < 0:
		# sentence starts before the clip!
		continue
	if clp_end_td > vid_end_td:
		clp_end_td = vid_end_td

	src_ = f'{path_tempuckey}/videos/{vid_id}'
	dst_ = f'{path_output}/{sent_id}.mp4'

	try:
		out, err = utils.ffmpeg_cut_re_encode(src_, dst_, clp_beg_td, clp_end_td)
	except Exception as e:
		pdb.set_trace()
