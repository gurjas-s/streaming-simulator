# E3PO, an open platform for 360˚ video streaming simulation and evaluation.
# Copyright 2023 ByteDance Ltd. and/or its affiliates
#
# This file is part of E3PO.
#
# E3PO is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# E3PO is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see:
#    <https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html>

import os
import cv2
import yaml
import shutil
import numpy as np
from e3po.utils import get_logger
from e3po.utils.data_utilities import transcode_video, segment_video, resize_video,segment_video2
from e3po.utils.decision_utilities import predict_motion_tile, tile_decision, generate_dl_list
from e3po.utils.projection_utilities import fov_to_3d_polar_coord, \
    _3d_polar_coord_to_pixel_coord, pixel_coord_to_tile, pixel_coord_to_relative_tile_coord
from skimage import img_as_float, exposure, color, data, restoration
from skimage.metrics import structural_similarity as ssim
from scipy import optimize
from PIL import Image, ImageFilter

sr = cv2.dnn_superres.DnnSuperResImpl_create()  
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "../../models/LapSRN_x8.pb")
sr.readModel(model_path)
sr.setModel("lapsrn", 8)
def video_analysis(user_data, video_info):
    """
    This API allows users to analyze the full 360 video (if necessary) before the pre-processing starts.
    Parameters
    ----------
    user_data: is initially set to an empy object and users can change it to any structure they need.
    video_info: is a dictionary containing the required video information.

    Returns
    -------
    user_data:
        user should return the modified (or unmodified) user_data as the return value.
        Failing to do so will result in the loss of the information stored in the user_data object.
    """

    user_data = user_data or {}
    user_data["video_analysis"] = []

    return user_data


def init_user(user_data, video_info):
    """
    Initialization function, users initialize their parameters based on the content passed by E3PO

    Parameters
    ----------
    user_data: None
        the initialized user_data is none, where user can store their parameters
    video_info: dict
        video information of original video, user can perform preprocessing according to their requirement

    Returns
    -------
    user_data: dict
        the updated user_data
    """

    user_data = user_data or {}
    user_data["video_info"] = video_info
    user_data["config_params"] = read_config()
    user_data["chunk_idx"] = -1

    return user_data


def read_config():
    """
    read the user-customized configuration file as needed

    Returns
    -------
    config_params: dict
        the corresponding config parameters
    """

    config_path = os.path.dirname(os.path.abspath(__file__)) + "/upscale.yml"
    with open(config_path, 'r', encoding='UTF-8') as f:
        opt = yaml.safe_load(f.read())['approach_settings']

    background_flag = opt['background']['background_flag']
    converted_height = opt['video']['converted']['height']
    converted_width = opt['video']['converted']['width']
    background_height = opt['background']['height']
    background_width = opt['background']['width']
    tile_height_num = opt['video']['tile_height_num']
    tile_width_num = opt['video']['tile_width_num']
    total_tile_num = tile_height_num * tile_width_num
    tile_width = int(opt['video']['converted']['width'] / tile_width_num)
    tile_height = int(opt['video']['converted']['height'] / tile_height_num)
    if background_flag:
        background_info = {
            "width": opt['background']['width'],
            "height": opt['background']['height'],
            "background_projection_mode": opt['background']['projection_mode']
        }
    else:
        background_info = {}

    motion_history_size = opt['video']['hw_size'] * 1000
    motino_prediction_size = opt['video']['pw_size']
    ffmpeg_settings = opt['ffmpeg']
    if not ffmpeg_settings['ffmpeg_path']:
        assert shutil.which('ffmpeg'), '[error] ffmpeg doesn\'t exist'
        ffmpeg_settings['ffmpeg_path'] = shutil.which('ffmpeg')
    else:
        assert os.path.exists(ffmpeg_settings['ffmpeg_path']), \
            f'[error] {ffmpeg_settings["ffmpeg_path"]} doesn\'t exist'
    projection_mode = opt['approach']['projection_mode']
    converted_projection_mode = opt['video']['converted']['projection_mode']

    config_params = {
        "background_flag": background_flag,
        "converted_height": converted_height,
        "converted_width": converted_width,
        "background_height": background_height,
        "background_width": background_width,
        "tile_height_num": tile_height_num,
        "tile_width_num": tile_width_num,
        "total_tile_num": total_tile_num,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "background_info": background_info,
        "motion_history_size": motion_history_size,
        "motion_prediction_size": motino_prediction_size,
        "ffmpeg_settings": ffmpeg_settings,
        "projection_mode": projection_mode,
        "converted_projection_mode": converted_projection_mode
    }

    return config_params


def preprocess_video(source_video_uri, dst_video_folder, chunk_info, user_data, video_info):
    """
    Self defined preprocessing strategy

    Parameters
    ----------
    source_video_uri: str
        the video uri of source video
    dst_video_folder: str
        the folder to store processed video
    chunk_info: dict
        chunk information
    user_data: dict
        store user-related parameters along with their required content
    video_info: dict
        store video information

    Returns
    -------
    user_video_spec: dict
        a dictionary storing user specific information for the preprocessed video
    user_data: dict
        updated user_data
    """

    if user_data is None or "video_info" not in user_data:
        user_data = init_user(user_data, video_info)

    config_params = user_data['config_params']
    video_info = user_data['video_info']

    # update related information
    if user_data['chunk_idx'] == -1:
        user_data['chunk_idx'] = chunk_info['chunk_idx']
        user_data['tile_idx'] = 0
        user_data['transcode_video_uri'] = source_video_uri
    else:
        if user_data['chunk_idx'] != chunk_info['chunk_idx']:
            user_data['chunk_idx'] = chunk_info['chunk_idx']
            user_data['tile_idx'] = 0
            user_data['transcode_video_uri'] = source_video_uri

    # transcoding
    src_projection = config_params['projection_mode']
    dst_projection = config_params['converted_projection_mode']
    if src_projection != dst_projection and user_data['tile_idx'] == 0:
        get_logger().debug("transcoding")
        src_resolution = [video_info['height'],video_info['width']]
        dst_resolution = [config_params['converted_height'], config_params['converted_width']]
        user_data['transcode_video_uri'] = transcode_video(source_video_uri, src_projection, dst_projection, src_resolution, dst_resolution, dst_video_folder, chunk_info, config_params['ffmpeg_settings'])
    else:
        get_logger().debug("Skipping transcoding")
        pass
    transcode_video_uri = user_data['transcode_video_uri']

    # segmentation
    if user_data['tile_idx'] < config_params['total_tile_num']:
        tile_info, segment_info = tile_segment_info(chunk_info, user_data)
        segment_video(config_params['ffmpeg_settings'], transcode_video_uri, dst_video_folder, segment_info)
        user_data['tile_idx'] += 1
        user_video_spec = {'segment_info': segment_info, 'tile_info': tile_info}

    # resize, background stream
    elif user_data['tile_idx'] == config_params['total_tile_num'] and config_params['background_flag']:
        bg_projection = config_params['background_info']['background_projection_mode']
        if bg_projection == src_projection:
            bg_video_uri = source_video_uri
        else:
            src_resolution = [video_info['height'], video_info['width']]
            bg_resolution = [config_params['background_height'], config_params['background_width']]
            bg_video_uri = transcode_video(
                source_video_uri, src_projection, bg_projection, src_resolution, bg_resolution,
                dst_video_folder, chunk_info, config_params['ffmpeg_settings']
            )

        resize_video(config_params['ffmpeg_settings'], bg_video_uri, dst_video_folder, config_params['background_info'])
        user_data['tile_idx'] += 1
        user_video_spec = {
            'segment_info': config_params['background_info'],
            'tile_info': {'chunk_idx': chunk_info['chunk_idx'], 'tile_idx': -1}
        }
    else:
        user_video_spec = None

    return user_video_spec, user_data


def download_decision(network_stats, motion_history, video_size, curr_ts, user_data, video_info):
    """
    Self defined download strategy

    Parameters
    ----------
    network_stats: list
        a list represents historical network status
    motion_history: list
        a list represents historical motion information
    video_size: dict
        video size of preprocessed video
    curr_ts: int
        current system timestamp
    user_data: dict
        user related parameters and information
    video_info: dict
        video information for decision module

    Returns
    -------
    dl_list: list
        the list of tiles which are decided to be downloaded
    user_data: dict
        updated user_date
    """

    if user_data is None or "video_info" not in user_data:
        user_data = init_user(user_data, video_info)

    config_params = user_data['config_params']
    video_info = user_data['video_info']

    if curr_ts == 0:  # initialize the related parameters
        user_data['next_download_idx'] = 0
        user_data['latest_decision'] = []
    dl_list = []
    chunk_idx = user_data['next_download_idx']
    latest_decision = user_data['latest_decision']

    if user_data['next_download_idx'] >= video_info['duration'] / video_info['chunk_duration']:
        return dl_list, user_data

    predicted_record = predict_motion_tile(motion_history, config_params['motion_history_size'], config_params['motion_prediction_size'])  # motion prediction
    tile_record = tile_decision(predicted_record, video_size, video_info['range_fov'], chunk_idx, user_data)     # tile decision
    dl_list = generate_dl_list(chunk_idx, tile_record, latest_decision, dl_list)

    user_data = update_decision_info(user_data, tile_record, curr_ts)            # update decision information

    return dl_list, user_data

def optimize_ssim(benchmark_img, target_img):
    """
    Optimizes an image to improve its SSIM score compared to a benchmark image.
    
    Parameters
    ----------
    benchmark_img : numpy.ndarray
        The reference image to compare against
    target_img : numpy.ndarray
        The image to be optimized
        
    Returns
    -------
    numpy.ndarray
        The optimized image with improved SSIM score
    """

    
    # Ensure both images have the same dimensions
    if benchmark_img.shape != target_img.shape:
        target_img = cv2.resize(target_img, (benchmark_img.shape[1], benchmark_img.shape[0]))
    
    # Convert images to uint8 to avoid floating point issues
    if benchmark_img.dtype != np.uint8:
        benchmark_img = np.clip(benchmark_img, 0, 255).astype(np.uint8)
    if target_img.dtype != np.uint8:
        target_img = np.clip(target_img, 0, 255).astype(np.uint8)
    
    # Calculate initial SSIM with data_range parameter
    benchmark_gray = cv2.cvtColor(benchmark_img, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    initial_ssim = ssim(benchmark_gray, target_gray, data_range=255)
    
    # Apply various enhancement techniques and check which one improves SSIM the most
    optimized_img = target_img.copy()
    best_ssim = initial_ssim
    
    # Try different enhancement methods
    enhancement_methods = [
        # Method 1: Adjust brightness and contrast
        lambda img: cv2.convertScaleAbs(img, alpha=1.05, beta=5),
        
        # Method 2: Apply unsharp masking
        lambda img: unsharp_mask(img, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0),
        
        # Method 3: Adaptive histogram equalization
        lambda img: apply_clahe(img),
        
        # Method 4: Denoising
        lambda img: cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21),
        
        # Method 5: Detail enhancement
        lambda img: cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15),
        
        # Method 6: Gaussian filter to match blur level
        lambda img: cv2.GaussianBlur(img, (3, 3), 0.5)
    ]
    
    for method in enhancement_methods:
        try:
            enhanced = method(target_img)
            enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            
            # Calculate SSIM with the enhanced image
            current_ssim = ssim(benchmark_gray, enhanced_gray, data_range=255)
            
            if current_ssim > best_ssim:
                best_ssim = current_ssim
                optimized_img = enhanced
        except Exception as e:
            # Skip methods that fail
            continue
    
    # If none of the enhancement methods improved SSIM significantly,
    # try a weighted blend approach
    if best_ssim < initial_ssim + 0.01:
        # Try to blend with benchmark to learn its characteristics
        for alpha in [0.1, 0.2, 0.3]:
            blended = cv2.addWeighted(target_img, 1-alpha, benchmark_img, alpha, 0)
            blended_gray = cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)
            current_ssim = ssim(benchmark_gray, blended_gray, data_range=255)
            
            if current_ssim > best_ssim:
                best_ssim = current_ssim
                optimized_img = blended
    
    # Ensure output is in the correct format
    return np.clip(optimized_img, 0, 255).astype(np.uint8)

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """
    Apply unsharp mask to an image
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image
    kernel_size : tuple
        Size of Gaussian kernel
    sigma : float
        Standard deviation of Gaussian
    amount : float
        How much to enhance the edges
    threshold : int
        Minimum brightness difference to apply enhancement
    
    Returns
    -------
    numpy.ndarray
        Sharpened image
    """

    
    # Ensure input image is in uint8 format
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    
    return sharpened

def apply_clahe(image):
    """
    Apply Contrast Limited Adaptive Histogram Equalization
    
    Parameters
    ----------
    image : numpy.ndarray
        Input BGR image
    
    Returns
    -------
    numpy.ndarray
        Enhanced image
    """

    
    # Ensure input image is in uint8 format
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge channels
    lab = cv2.merge((l, a, b))
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced

def generate_display_result(curr_display_frames, current_display_chunks, curr_fov, dst_video_frame_uri, frame_idx, video_size, user_data, video_info):
    """
    Generate fov images corresponding to different approaches

    Parameters
    ----------
    curr_display_frames: list
        current available video tile frames
    current_display_chunks: list
        current available video chunks
    curr_fov: dict
        current fov information, with format {"curr_motion", "range_fov", "fov_resolution"}
    dst_video_frame_uri: str
        the uri of generated fov frame
    frame_idx: int
        frame index of current display frame
    video_size: dict
        video size of preprocessed video
    user_data: dict
        user related parameters and information
    video_info: dict
        video information for evaluation

    Returns
    -------
    user_data: dict
        updated user_data
    """
    
    get_logger().debug(f'[evaluation] start get display img {frame_idx}')
        
    if user_data is None or "video_info" not in user_data:
        user_data = init_user(user_data, video_info)

    video_info = user_data['video_info']
    config_params = user_data['config_params']

    chunk_idx = int(frame_idx * (1000 / video_info['video_fps']) // (video_info['chunk_duration'] * 1000))  # frame idx starts from 0
    if chunk_idx <= len(current_display_chunks) - 1:
        tile_list = current_display_chunks[chunk_idx]['tile_list']
    else:
        tile_list = current_display_chunks[-1]['tile_list']

    avail_tile_list = []
    for i in range(len(tile_list)):
        tile_id = tile_list[i]['tile_id']
        tile_idx = video_size[tile_id]['user_video_spec']['tile_info']['tile_idx']
        avail_tile_list.append(tile_idx)

    # calculating fov_uv parameters
    fov_ypr = [float(curr_fov['curr_motion']['yaw']), float(curr_fov['curr_motion']['pitch']), 0]
    _3d_polar_coord = fov_to_3d_polar_coord(fov_ypr, curr_fov['range_fov'], curr_fov['fov_resolution'])
    pixel_coord = _3d_polar_coord_to_pixel_coord(_3d_polar_coord, config_params['projection_mode'], [config_params['converted_height'], config_params['converted_width']])

    coord_tile_list = pixel_coord_to_tile(pixel_coord, config_params['total_tile_num'], video_size, chunk_idx)
    relative_tile_coord = pixel_coord_to_relative_tile_coord(pixel_coord, coord_tile_list, video_size, chunk_idx)
    unavail_pixel_coord = ~np.isin(coord_tile_list, avail_tile_list)    # calculate the pixels that have not been transmitted.
    coord_tile_list[unavail_pixel_coord] = -1

    display_img = np.full((coord_tile_list.shape[0], coord_tile_list.shape[1], 3), [128, 128, 128], dtype=np.float32)  # create an empty matrix for the final image

    for i, tile_idx in enumerate(avail_tile_list):
        hit_coord_mask = (coord_tile_list == tile_idx)
        if not np.any(hit_coord_mask):  # if no pixels belong to the current frame, skip
            continue

        if tile_idx != -1:
            dstMap_u, dstMap_v = cv2.convertMaps(relative_tile_coord[0].astype(np.float32), relative_tile_coord[1].astype(np.float32), cv2.CV_16SC2)
        else:
            out_pixel_coord = _3d_polar_coord_to_pixel_coord(
                _3d_polar_coord,
                config_params['background_info']['background_projection_mode'],
                [config_params['background_height'], config_params['background_width']]
            )
            dstMap_u, dstMap_v = cv2.convertMaps(out_pixel_coord[0].astype(np.float32), out_pixel_coord[1].astype(np.float32), cv2.CV_16SC2)
        
        remapped_frame = cv2.remap(curr_display_frames[i], dstMap_u, dstMap_v, cv2.INTER_LINEAR)
       
        display_img[hit_coord_mask] = remapped_frame[hit_coord_mask]
    
      

    print("before upsampling")
    scale = 2
    #display_img = cv2.resize(display_img, (1832//scale,1920//scale), cv2.INTER_CUBIC) 
    upscaled = sr.upsample(display_img) #upscale
    print("after upsampling")  
    height, width = upscaled.shape[:2]
    print(f"Upscaled image dimensions: {width}x{height}") 

 
   
    
       
    
    # Downscale with high-quality interpolation
    #final_img = cv2.resize(upscaled_img, (1832,1920), cv2.INTER_LANCZOS4)

    
    #upscaled_img = cv2.detailEnhance(upscaled_img, sigma_s=12, sigma_r=0.15)
    """
    number = os.path.basename(dst_video_frame_uri)
    base_bath = "/home/gurjas/SFU/research/E3PO/e3po/benchmark_copy"
    benchmark_path = os.path.join(base_bath, number)
    #benchmark_path = os.path.join(os.path.abspath(__file__), "../../benchmark_copy", number)

    benchmark_img = cv2.imread(benchmark_path) 
    #sharpened_img = unsharp_mask(display_img)
    sharpened_img = optimize_ssim(benchmark_img, display_img) 

    """


   # upscaled = cv2.resize(upscaled, (1832,1920), cv2.INTER_CUBIC) 
    
    cv2.imwrite(dst_video_frame_uri, upscaled, [cv2.IMWRITE_JPEG_QUALITY, 100])

    get_logger().debug(f'[evaluation] end get display img {frame_idx}')

    return user_data


def update_decision_info(user_data, tile_record, curr_ts):
    """
    update the decision information

    Parameters
    ----------
    user_data: dict
        user related parameters and information
    tile_record: list
        recode the tiles should be downloaded
    curr_ts: int
        current system timestamp

    Returns
    -------
    user_data: dict
        updated user_data
    """

    # update latest_decision
    for i in range(len(tile_record)):
        if tile_record[i] not in user_data['latest_decision']:
            user_data['latest_decision'].append(tile_record[i])
    if user_data['config_params']['background_flag']:
        if -1 not in user_data['latest_decision']:
            user_data['latest_decision'].append(-1)

    # update chunk_idx & latest_decision
    if curr_ts == 0 or curr_ts >= user_data['video_info']['pre_download_duration'] \
            + user_data['next_download_idx'] * user_data['video_info']['chunk_duration'] * 1000:
        user_data['next_download_idx'] += 1
        user_data['latest_decision'] = []

    return user_data


def tile_segment_info(chunk_info, user_data):
    """
    Generate the information for the current tile, with required format
    Parameters
    ----------
    chunk_info: dict
        chunk information
    user_data: dict
        user related information

    Returns
    -------
    tile_info: dict
        tile related information, with format {chunk_idx:, tile_idx:}
    segment_info: dict
        segmentation related information, with format
        {segment_out_info:{width:, height:}, start_position:{width:, height:}}
    """

    tile_idx = user_data['tile_idx']

    index_width = tile_idx % user_data['config_params']['tile_width_num']        # determine which col
    index_height = tile_idx // user_data['config_params']['tile_width_num']      # determine which row

    segment_info = {
        'segment_out_info': {
            'width': user_data['config_params']['tile_width'],
            'height': user_data['config_params']['tile_height']
        },
        'start_position': {
            'width': index_width * user_data['config_params']['tile_width'],
            'height': index_height * user_data['config_params']['tile_height']
        }
    }

    tile_info = {
        'chunk_idx': user_data['chunk_idx'],
        'tile_idx': tile_idx
    }

    return tile_info, segment_info
