import json
import time 
import datetime
import utilities as utils

home_dir = '/home/pishu/Desktop/repos/datasets'
json_path = '{}/Tempuckey/Tempuckey_video_text_pair.json'.format(home_dir)
corpus_path = '{}/NHL_ClosedCaption/corpus_with_timestamp'.format(home_dir)
videos_path = '{}/Tempuckey/videos'.format(home_dir)

misalignment_param = 5 # in seconds

def seconds_to_time(seconds): 
	return time.strftime('%H:%M:%S.%M', time.gmtime(seconds))
      
def seconds_to_dt_time(seconds):
	tt = seconds_to_time(seconds)
	hh,mm,ss = tt.split(':')
	ss,_ = ss.split('.')

	return datetime.time(hour=int(hh),minute=int(mm),second=int(ss))

with open(json_path,'r') as f:
	content = json.load(f)

ts = content['sentences']
vs = content['videos']

dct = {}
for txt,vid in zip(ts,vs):
	vid_file = vid['video_id']
	cap_file = txt['caption_name']
	beg_time = vid['start_time']
	end_time = vid['end_time']

	cap_path = '{}/{}'.format(corpus_path, cap_file)
	vid_path = '{}/{}'.format(videos_path, vid_file)

	beg_ts = seconds_to_dt_time(beg_time)
	end_ts = seconds_to_dt_time(end_time)

	beg_td = datetime.timedelta(hours=beg_ts.hour, minutes=beg_ts.minute, seconds=beg_ts.second)
	end_td = datetime.timedelta(hours=end_ts.hour, minutes=end_ts.minute, seconds=end_ts.second)

	m = datetime.timedelta(hours=0, minutes=0, seconds=5)

	caps = utils.load_picklefile(cap_path)

	entries = []
	sent_parts = []
	for k,v in caps.items():
		if k[0] is None:
			continue

		sent_beg_td = datetime.timedelta(hours=k[0].hour, minutes=k[0].minute, seconds=k[0].second)
		sent_end_td = datetime.timedelta(hours=k[1].hour, minutes=k[1].minute, seconds=k[1].second)

		if sent_beg_td>=beg_td-m and sent_end_td<=end_td+m:
			if '.' not in v:
				sent_parts.append(v)
				if (len(sent_parts) == 1):
					start_time = k[0]
			else:
				end_time = k[1]
				sent_parts.append(v)
				sentence = ' '.join(sent_parts)
				sent_parts = []
				entry = [start_time, end_time, sentence]
				entries.append(entry)


	dct['video_id'] = vid['video_id']
	dct['start_time'] = vid['start_time']
	dct['end_time'] = vid['end_time']
	dct['sentences'] = entries

utils.dump_picklefile(dct, path_='{}/Tempuckey/video_sentences.dct.pkl'.format(home_dir))

