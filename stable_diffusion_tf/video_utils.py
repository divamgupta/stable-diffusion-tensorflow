import numpy as np
import pandas as pd
import cv2
import subprocess
import re
from tensorflow import keras
from skimage.exposure import cumulative_distribution


def load_sd_style_model(style_model_name, generator):
    if style_model_name == "Illustration-Diffusion":
        diffusion_model_weights = keras.utils.get_file(
            origin="https://huggingface.co/ogkalu/Illustration-Diffusion/resolve/main/hollie-mengert.ckpt",
            file_hash="2c4c9a75f6045b861b3f9252f51442dc4880c70fb792b78446940abc232bdbb7",
        )
        generator.load_weights_from_pytorch_ckpt(diffusion_model_weights)
        return generator
    
    elif style_model_name == "Comic-Diffusion":
        diffusion_model_weights = keras.utils.get_file(
            origin="https://huggingface.co/ogkalu/Comic-Diffusion/resolve/main/comic-diffusion.ckpt",
            file_hash="33789685ab6488d34e6310f7e6da5c981194ce59ef4b6890f681d5cc5b9c62cc",
        )
        generator.load_weights_from_pytorch_ckpt(diffusion_model_weights)
        return generator
    
    elif style_model_name == "Superhero-Diffusion":
        diffusion_model_weights = keras.utils.get_file(
            origin="https://huggingface.co/ogkalu/Superhero-Diffusion/resolve/main/superhero-diffusion.ckpt",
            file_hash="cac0a972cfa40cfe44e3c00d3a488dcbe34668bf291dd6245d70266247643a7c",
        )
        generator.load_weights_from_pytorch_ckpt(diffusion_model_weights)
        return generator

    else:
        return generator


# these parsing methods are taking from the Deforum Stable Diffusion Notebook(https://colab.research.google.com/github/deforum/stable-diffusion/blob/main/Deforum_Stable_Diffusion.ipynb) 

def check_is_number(value):
    float_pattern = r'^(?=.)([+-]?([0-9]*)(\.([0-9]+))?)$'
    return re.match(float_pattern, value)


def get_inbetweens(key_frames, max_frames, integer=False, interp_method='Linear'):
    import numexpr
    key_frame_series = pd.Series([np.nan for a in range(max_frames)])
    
    for i in range(0, max_frames):
        if i in key_frames:
            value = key_frames[i]
            value_is_number = check_is_number(value)
            # if it's only a number, leave the rest for the default interpolation
            if value_is_number:
                t = i
                key_frame_series[i] = value
        if not value_is_number:
            t = i
            key_frame_series[i] = numexpr.evaluate(value)
    key_frame_series = key_frame_series.astype(float)
    
    if interp_method == 'Cubic' and len(key_frames.items()) <= 3:
      interp_method = 'Quadratic'    
    if interp_method == 'Quadratic' and len(key_frames.items()) <= 2:
      interp_method = 'Linear'
          
    key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
    key_frame_series[max_frames-1] = key_frame_series[key_frame_series.last_valid_index()]
    key_frame_series = key_frame_series.interpolate(method=interp_method.lower(), limit_direction='both')
    if integer:
        return key_frame_series.astype(int)
    return key_frame_series


def parse_key_frames(string, prompt_parser=None):
    # because math functions (i.e. sin(t)) can utilize brackets 
    # it extracts the value in form of some stuff
    # which has previously been enclosed with brackets and
    # with a comma or end of line existing after the closing one
    pattern = r'((?P<frame>[0-9]+):[\s]*\((?P<param>[\S\s]*?)\)([,][\s]?|[\s]?$))'
    frames = dict()
    for match_object in re.finditer(pattern, string):
        frame = int(match_object.groupdict()['frame'])
        param = match_object.groupdict()['param']
        if prompt_parser:
            frames[frame] = prompt_parser(param)
        else:
            frames[frame] = param
    if frames == {} and len(string) != 0:
        raise RuntimeError('Key Frame string not correctly formatted')
    return frames


def generate_frames_translation(ax_trans, max_num_frames):
    str_for_parse_key_frames = f"0:({ax_trans})"
    key_frames = parse_key_frames(str_for_parse_key_frames)
    frames_translation_values = get_inbetweens(key_frames, max_num_frames)
    return frames_translation_values


def create_prompts_frames_dict(first_prompt, first_frame=0, second_prompt=None, second_frame=None, third_prompt=None, third_frame=None, fourth_prompt=None, fourth_frame=None):
    prompts_frames_dict = dict()
    
    if not first_prompt:
        raise RuntimeError('Please make sure you passed a valid prompt')
    
    if int(first_frame) != 0:
        raise RuntimeError('The number of the first frame must to be 0')

    prompts_frames_dict['prompt1'] = [first_prompt, first_frame]

    if second_prompt is not None and second_frame is not None:
        prompts_frames_dict['prompt2'] = [second_prompt, second_frame]

        if third_prompt is not None and third_frame is not None:
            prompts_frames_dict['prompt3'] = [third_prompt, third_frame]

            if fourth_prompt is not None and fourth_frame is not None:
                prompts_frames_dict['prompt4'] = [fourth_prompt, fourth_frame]    

    return prompts_frames_dict


def hist_matching(c, c_t, im):
    b = np.interp(c, c_t, np.arange(256))   # find closest matches to b_t
    pix_repl = {i:b[i] for i in range(256)} # dictionary to replace the pixels
    mp = np.arange(0,256)
    for (k, v) in pix_repl.items():
        mp[k] = v
    s = im.shape
    im = np.reshape(mp[im.ravel()], im.shape)
    im = np.reshape(im, s)
    return im


def cdf(im):
    c, b = cumulative_distribution(im)
    #print(b)
    for i in range(b[0]):
        c = np.insert(c, 0, 0)
    for i in range(b[-1]+1, 256):
        c = np.append(c, 1)
    return c


def maintain_colors(prev_img, color_match_sample):
    im1 = np.zeros(prev_img.shape)
    for i in range(3):
        c = cdf(prev_img[...,i])
        c_t = cdf(color_match_sample[...,i].astype(np.uint8))
        im1[...,i] = hist_matching(c, c_t, prev_img[...,i])
    return im1[...,:3]


def anim_frame_warp_2d(prev_img_cv2, args, idx):
    angle = args['angle']
    zoom = args['zoom']
    translation_x = args['translation_x'][idx]
    translation_y = args['translation_y'][idx]

    center = (512 // 2, 512 // 2)
    trans_mat = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    rot_mat = cv2.getRotationMatrix2D(center, angle, zoom)
    trans_mat = np.vstack([trans_mat, [0,0,1]])
    rot_mat = np.vstack([rot_mat, [0,0,1]])
    xform = np.matmul(rot_mat, trans_mat)

    return cv2.warpPerspective(
        prev_img_cv2,
        xform,
        (prev_img_cv2.shape[1], prev_img_cv2.shape[0]),
        cv2.BORDER_REPLICATE
    )


def create_prompt_idx_dict(start_idx, end_idx, prompt):
    prompt_dict = dict()
    for i in range(start_idx, end_idx):
        prompt_dict[i] = prompt
    return prompt_dict


def next_seed(args):
    if args['seed_behavior'] == 'iter':
        args['seed'] = args['seed'] + 1
    elif args['seed_behavior'] == 'fix':
        pass


def create_prompt_iprompt_seq(args, prompts_frames_dict):
    prompt_iprompt_seq_lst = []
    promprs_dict_keys_lst = list(prompts_frames_dict.keys())

    if len(promprs_dict_keys_lst) == 1:
        prompt_idx_dict = create_prompt_idx_dict(0, args['maximum_number_of_frames'], prompts_frames_dict['prompt1'][0])
        prompt_iprompt_seq_lst.append(prompt_idx_dict)
    else:
        for ip, p in enumerate(prompts_frames_dict):
            prompt_vals = prompts_frames_dict[p]
            start_idx = int(prompt_vals[1])
            if ip == len(promprs_dict_keys_lst)-1:
                prompt_idx_dict = create_prompt_idx_dict(start_idx, args['maximum_number_of_frames'], prompt_vals[0])
                prompt_iprompt_seq_lst.append(prompt_idx_dict)
            else:
                end_idx = int(prompts_frames_dict[promprs_dict_keys_lst[ip+1]][1])
                prompt_idx_dict = create_prompt_idx_dict(start_idx, end_idx, prompt_vals[0])
                prompt_iprompt_seq_lst.append(prompt_idx_dict)

    return prompt_iprompt_seq_lst


def generate_init_frame(curr_prompt, args, generator):
    img = generator.generate(
        curr_prompt,
        seed=args['seed'],
        num_steps=40,
        unconditional_guidance_scale=7,
        temperature=1,
        batch_size=1,
    )
    next_seed(args)
    return img[0]


def construct_ffmpeg_video_cmd(args, frames_path, mp4_path):
    fps = args['fps']
    max_frames = args['maximum_number_of_frames']

    cmd = [ 
        'ffmpeg',
        '-y',
        '-vcodec', 'png',
        '-r', str(fps),
        '-start_number', str(0),
        '-i', frames_path,
        '-frames:v', str(max_frames),
        '-c:v', 'libx264',
        '-vf',
        f'fps={fps}',
        '-pix_fmt', 'yuv420p',
        '-crf', '17',
        '-preset', 'fast',
        '-pattern_type', 'sequence',
        mp4_path
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print(stderr)
        raise RuntimeError(stderr)


def create_audio(args, sound_path, mp3_path):
    cmd = [ 
        'ffmpeg',
        '-i', sound_path,
        '-t', f"{args['video_length']}",
        mp3_path
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print(stderr)
        raise RuntimeError(stderr)
    else:
        print("audio file created seccefully!")


def construct_ffmpeg_combined_cmd(vid_path, aud_path, combined_path):
    cmd = [
        'ffmpeg',
        '-i', vid_path,
        '-i', aud_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        combined_path
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print(stderr)
        raise RuntimeError(stderr)
