import cv2

def get_frame_snapshot_at(input_vid, sec):
    '''
    obtains a snapshot frame of the video at the given second
    
    Args:
        input_vid: cv2.VideoCapture of the input video
        sec: int 
    Returns:
        hasFrames: Boolean
        image: array2d 
    '''
    input_vid.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    hasFrames, image = input_vid.read()

    if hasFrames:
        return hasFrames, image
    return False, None

def resize_dims(current_dims, resizing_dims):
    '''
    resizes the image dimensions according to resizing_dims described below.
    Input: 
        current_dims: (w,h)
        resizing_dims: float tuple (target_w, target_h)
                     resize the smaller dim to target_dim[0]; and 
                     resize the larger dim to target_dim[1]; and
                     if target_dim[0] or target_dim[1] is set to -1, the dims change while preserving the aspect ratio;
                     if target_dim = (-1,-1) there is no resizing. the original dims are maintained. 
    Output:
        int tuple (new_w, new_h)
    '''
    #   validate resizing dims.
    if (resizing_dims[0] < 1 and resizing_dims[0] != -1):
        raise ValueError('Invalid resizing dimenension: {}'.format(resizing_dims))
    if (resizing_dims[1] < 1 and resizing_dims[1] != -1):
        raise ValueError('Invalid resizing dimenension: {}'.format(resizing_dims)) 

    w = current_dims[0]
    h = current_dims[1]

    target_w = resizing_dims[0]
    target_h = resizing_dims[1]

    if resizing_dims == (-1,-1):
        return (int(w) , int(h))
    elif target_w != -1 and target_h != -1:
        return (int(target_w) , int(target_h))
    elif target_w != -1:
        return (int(target_w) , int(h * target_w/w))
    else:
        return (int(w * target_h/h) , int(target_h))

def get_video_frames(video_path, output_dir, start_sec = 0, end_sec = 10000, fps = None, resizing_dims = (-1,-1)):
    '''
    splits a given input video break it into image frames and stores them in the output dir
    **note: the fps for the given video can be adjusted using fps parameter. If no fps is specified the default fps is used.
    
    Args:
        video_path: string path of the input video
        output_dir: string directory path where the output frames will be stored
        fps: int frame per second
        resize_dims: resizing frames.
                     resize the smaller dim target_dim[0]; and 
                     resize the larger dim to target_dim[1]; and
                     if target_dim[0] or target_dim[1] is set to -1, the dims change while preserving the aspect ratio;
                     if target_dim = (-1,-1) there is no resizing. the original sims are maintained. 
    Returns:
        None
    '''
    cap = cv2.VideoCapture(video_path)    
    
    if fps == None:
        fps = cap.get(cv2.CAP_PROP_FPS)

    frameRate = 1.0/fps
    
    sec = start_sec
    count = 1
    hasFrames = True

    # make sure the output_dir format is properly set
    if output_dir[-1] != '/':
        output_dir = '{}/'.format(output_dir)

    current_dims = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_dims = resize_dims(current_dims, resizing_dims)

    while hasFrames:
        if sec > end_sec:
            break
        hasFrames, image = get_frame_snapshot_at(cap, sec)
        
        if hasFrames: 
            image = cv2.resize(image, target_dims, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(output_dir + '{:06d}.jpg'.format(count), image) 
            count += 1
            sec += frameRate

    return count-1


def get_video_frames_simp(video_path, output_dir):
    '''
    splits a given input video break it into image frames using videos's default fps and stores them in the output dir
    **note: the fps for the frame extraction can not be adjusted 
    Args:
        video_path: string path of the input video
        output_dir: string directory path where the output frames will be stored
    
    Returns:
        None
    '''
    cap = cv2.VideoCapture(video_path)    
    
    count = 0
    hasFrames = True

    # make sure the output_dir format is properly set
    if output_dir[-1] != '/':
        output_dir = '{}/'.format(output_dir)

    while hasFrames:
        hasFrames, image = cap.read()
        count += 1

        if hasFrames: cv2.imwrite(output_dir + '{:06d}.jpg'.format(count), image) 
    return count-1


def display_video_information(video_path):
    '''
    displays video information including fps, number of frames, duration 
    
    Args:
        video_path: string path to a single video file 

    Returns:
        None 
    '''
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    try:
        duration = num_frames/fps
    except ZeroDivisionError as e:
        # print(e, video_path, '\n')
        duration = 0

    print('fps = {}'.format(fps))
    print('number of frames = {}'.format(num_frames))
    print('duration(sec) = {}'.format(duration))


def get_vide_duration(video_path):
    video = cv2.VideoCapture(video_path)
    duration = video.get(cv2.CAP_PROP_POS_MSEC)
    return duration

def get_video_information(video_path):
    '''
    obtains video information as a list containing [fps, duration(sec), number_of_frames] 
    
    Args:
        video_path: string path to the video

    Returns:
        tuple: (fps as float, duration(sec) as float, number_of_frames as int)
    '''
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 
    try:
        duration = num_frames/fps
    except ZeroDivisionError as e:
        # print(e, video_path, '\n')
        duration = 0
    
    return (fps, duration, num_frames)

