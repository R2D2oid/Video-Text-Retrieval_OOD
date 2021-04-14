import json
import time 
import datetime
import utilities as utils

home_dir = '/home/pishu/Desktop/repos/datasets' # rearranged sentence segments data under Tempuckey on rocket. paths below will need to be updated
json_path = '{}/Tempuckey/Tempuckey_video_text_pair.json'.format(home_dir)
corpus_path = '{}/NHL_ClosedCaption/corpus_with_timestamp'.format(home_dir)
videos_path = '{}/Tempuckey/videos'.format(home_dir)

misalignment_param = 10 # in seconds

with open(json_path,'r') as f:
	content = json.load(f)

ts = content['sentences']
vs = content['videos']

vid_txt_pairs = []
for txt,vid in zip(ts,vs):
	item = {}
	vid_file = vid['video_id']
	cap_file = txt['caption_name']
	beg_time = vid['start_time']
	end_time = vid['end_time']

	cap_path = '{}/{}'.format(corpus_path, cap_file)
	# vid_path = '{}/{}'.format(videos_path, vid_file)

	beg_ts = utils.seconds_to_dt_time(beg_time)
	end_ts = utils.seconds_to_dt_time(end_time)

	beg_td = datetime.timedelta(hours=beg_ts.hour, minutes=beg_ts.minute, seconds=beg_ts.second)
	end_td = datetime.timedelta(hours=end_ts.hour, minutes=end_ts.minute, seconds=end_ts.second)

	m = datetime.timedelta(hours=0, minutes=0, seconds=misalignment_param)

	caps = utils.load_picklefile(cap_path)

	entries = []
	sent_parts = []
	for k,v in caps.items():
		if k[0] is None:
			continue

		sent_beg_td = datetime.timedelta(hours=k[0].hour, minutes=k[0].minute, seconds=k[0].second)
		sent_end_td = datetime.timedelta(hours=k[1].hour, minutes=k[1].minute, seconds=k[1].second)

		if sent_beg_td > beg_td-m and sent_end_td < end_td+m:
			if (len(sent_parts) == 0):
					sent_beg_time = k[0]
			if '.' in v:
				sent_end_time = k[1]
				sent_parts.append(v)
				sentence = ' '.join(sent_parts)
				entry = [sent_beg_time, sent_end_time, sentence]
				entries.append(entry)

				sent_parts = []
				sent_beg_time = None
				sent_end_time = None
			else:
				sent_parts.append(v)
		
	item['video_id'] = vid_file
	item['start_time'] = beg_time
	item['end_time'] = end_time
	item['sentences'] = entries

	vid_txt_pairs.append(item)

path_='{}/Tempuckey/video_text.pkl'.format(home_dir)
utils.dump_picklefile(vid_txt_pairs, path_)

print('generated {}'.format(path_))
