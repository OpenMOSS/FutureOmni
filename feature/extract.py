from PIL import Image
from PIL.Image import Image as ImageObject
import math
import os
import glob
from typing import Union
import numpy as np
import librosa
import pandas as pd
import torch
from multiprocessing import Pool, cpu_count, current_process, Manager
from functools import partial
import time
from tqdm import tqdm
import json
from utils import trans_seconds, trans_seconds2
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
from utils.constructor import trans2seconds

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_TRAIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
import cv2
import pdb
MAX_RATIO = 200
IMAGE_FACTOR = 28
def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor
def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = -1, max_pixels: int = -1
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(height / beta, factor))
        w_bar = max(factor, floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar
def get_duration(video_path):
    """Return (video_path, duration) pair."""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps > 0:
            duration = frame_count / fps
        else:
            duration = 0.0
    except Exception:
        duration = 0.0
    return video_path, duration


def read_frames(video_path, frame_num=8, end_sec = None, save_dir=None):
    # pdb.set_trace()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    duration = total_frames / fps

    # If timestamps are not provided, generate evenly spaced timestamps
    if end_sec is None:
        timestamps = np.linspace(0, duration, frame_num, endpoint=False)
    else:
        start_frame = 0
        end_frame = int(fps * end_sec)
        if end_frame == 0:
            end_frame += 1
        
        timestamps = [int(start_frame + (end_frame - start_frame) * i / (frame_num - 1)) for i in range(frame_num)]
        
        
    frames = []
    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    for t in timestamps:
        if end_sec is None:
            frame_id = int(round(t * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            
            ret, frame = cap.read()
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ret, frame = cap.read()
        # pdb.set_trace()
        if not ret:
            print(f"Warning: could not read frame at {t:.2f}s")
            continue
        # frames[float(t)] = frame
        frames.append(np.transpose(frame, (2, 0, 1)))

        if save_dir is not None:
            save_path = os.path.join(save_dir, f"frame_{t:.2f}s.jpg")
            cv2.imwrite(save_path, frame)

    cap.release()
    # pdb.set_trace()
    return frames
def get_train_frames(frame_dir, video_name):
    frame_pattern = os.path.join(frame_dir, f"{video_name}_frame_*.jpg")
    frame_files = sorted(glob.glob(frame_pattern))
    return frame_files

def _regularize_videos(videos: list[str], frame_num:int = -1,end_sec = None) -> dict[str, list[list["ImageObject"]]]:
    r"""Regularizes videos to avoid error. Including reading, resizing and converting."""
    results = []
    # pdb.set_trace()
    for video in videos:
        frames: list[ImageObject] = []
        frames = read_frames(video, frame_num= frame_num, end_sec=end_sec)
        results.append(frames)

    return {"videos": results}

def _regularize_audios(
         audios: list[str], sampling_rate: float, duration: float = None, **kwargs
    ) -> dict[str, Union[list["NDArray"], list[float]]]:
        r"""Regularizes audios to avoid error. Including reading and resampling."""
        results, sampling_rates = [], []
        for audio in audios:
            if not isinstance(audio, np.ndarray):
                if duration is not None:
                    audio, sampling_rate = librosa.load(audio, sr=sampling_rate,offset=0, duration=duration)
                else:
                    audio, sampling_rate = librosa.load(audio, sr=sampling_rate)

            results.append(audio)
            sampling_rates.append(sampling_rate)

        return {"audios": results, "sampling_rates": sampling_rates}


def feature_extract_single(video_info, processor_path, feature_dir,  progress_queue=None,video_frames=100,dynamic=False,dataset="worldsense", max_frames=32, model_type="qwen2_5omni"):
    """
    单个视频的特征提取函数（多进程版本）
    
    Args:
        video_info: 包含video_name和duration的字典
        processor_path: processor路径
        feature_dir: 特征保存目录
        frame_dir: 帧目录
        audio_dir: 音频目录（可选）
        progress_queue: 进度队列
    
    Returns:
        (video_name, success, error_message)
    """
    
    process_name = current_process().name
    video_path = video_info['video_name'] # futureomni: [path, qid, source] general:{"video_name":xxx, 'duration':float}
    qid = None
    if dynamic or isinstance(video_path, list):
        video_name = video_path[0].split("/")[-1]
        qid = video_path[1]
        source = video_path[-1]
        feature_dir = f"{feature_dir}/{source}_{max_frames}"
    else:
        feature_dir = f"{feature_dir}/{dataset}_{max_frames}"
        video_name = video_path.split("/")[-1]
        
    
    if not dynamic:
        duration = video_info['duration']
    else:
        duration_seg, duration = video_info['duration'][0], video_info['duration'][1]
    save_video_dir = f"{feature_dir}/video"
    os.makedirs(save_video_dir, exist_ok=True)

    save_audio_dir = f"{feature_dir}/audio"
    os.makedirs(save_audio_dir, exist_ok=True)

    feature_dir = f"{feature_dir}/feature"
    os.makedirs(feature_dir, exist_ok=True)

    if qid is not None:
        feature_path = os.path.join(feature_dir, f"{qid}.pt")
        save_video_path = os.path.join(save_video_dir, f"{qid}.pt")
        save_audio_path = os.path.join(save_audio_dir, f"{qid}.pt")
    else:
        feature_path = os.path.join(feature_dir, f"{video_name}.pt")
        save_video_path = os.path.join(save_video_dir, f"{video_name}.pt")
        save_audio_path = os.path.join(save_audio_dir, f"{video_name}.pt")
    
    if os.path.exists(feature_path):
        if isinstance(video_name, list):
            print(f"[{process_name}] Passing Existed File: {qid}")

        else:
            print(f"[{process_name}] Passing Existed File: {video_name}")
        if progress_queue:
            progress_queue.put(1)
        return (video_name, True, "already_exists")
    
    if model_type == "qwen2_5omni":
        from transformers import Qwen2_5OmniProcessor
        processor = Qwen2_5OmniProcessor.from_pretrained(processor_path)
    elif model_type == "qwen3omni":
        from transformers import Qwen3OmniMoeProcessor
        processor = Qwen3OmniMoeProcessor.from_pretrained(processor_path)
    else:
        raise NotImplementedError()
    feature_extractor = getattr(processor, "feature_extractor", None)
    temporal_patch_size: int = 2
    
    mm_inputs = {}
    
    # process video
    
    if not dynamic:
        assert duration > 0:
        fps = video_frames / duration
    else:    
        fps = max_frames / duration_seg
    
    if isinstance(video_path, list):
        videos = [video_path[0]]
    else:
        videos = [video_path]
    video_processor = getattr(processor, "video_processor")
    if not os.path.exists(save_video_path):
        if dynamic:
            videos_processed = _regularize_videos(
                videos,
                frame_num= max_frames, 
                end_sec= duration_seg
            )["videos"]
        else:
            videos_processed = _regularize_videos(
                videos,
                frame_num= max_frames, 
                end_sec= None
            )["videos"]
            
        videos_tensor = torch.tensor(videos_processed[0])
        nframes, channel , height, width = videos_tensor.shape
        
        assert channel == 3, nframes < height
        assert nframes < width
            
        resized_height, resized_width = smart_resize(
                height,
                width,
                factor=28,
                min_pixels=VIDEO_MIN_PIXELS,
                max_pixels=VIDEO_MAX_PIXELS,
            )
        videos_tensor = transforms.functional.resize(
            videos_tensor,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        

        if not os.path.exists(save_video_path):
            torch.save(videos_tensor, save_video_path)
    else:
        #
        try:
            videos_tensor = torch.load(save_video_path)
        except:
            print(f"wrong format:{save_video_path}")
    mm_inputs.update(video_processor(images=None, videos=[videos_tensor], return_tensors="pt"))
    
    mm_inputs["video_second_per_grid"] = torch.tensor([temporal_patch_size / fps])
    
    if dataset not in ['videomme','mlvu']:
        audio_path = videos[0]
        if not os.path.exists(save_audio_path):
            if os.path.exists(audio_path):
                audios = [audio_path]
                feature_extractor = getattr(processor, "feature_extractor", None)
                if dynamic:
                    audios_processed = _regularize_audios(
                        audios,
                        sampling_rate=getattr(processor, "audio_sampling_rate", 16000),duration= duration_seg
                    )["audios"]
                else:
                    # pdb.set_trace()
                    audios_processed = _regularize_audios(
                        audios,
                        sampling_rate=getattr(processor, "audio_sampling_rate", 16000),duration= None
                    )["audios"]

                if not os.path.exists(save_audio_path):
                    audio_tensor = torch.tensor(audios_processed[0])
                    torch.save(audio_tensor, save_audio_path)
        else:
            try:
                audios_processed = torch.load(save_audio_path)
            except:
                pdb.set_trace()
        audio_features = feature_extractor(
            audios_processed,
            sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
            return_attention_mask=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        mm_inputs.update(audio_features)
        
        mm_inputs["feature_attention_mask"] = mm_inputs.pop("attention_mask", None)
        
        
    # save feature
    torch.save(mm_inputs, feature_path)
    
    
    if progress_queue:
        progress_queue.put(1)
    
    return (video_name, True, "success")
        

def progress_monitor(progress_queue, total_videos):
    """进度监控器"""
    completed = 0
    with tqdm(total=total_videos, desc="特征提取进度", unit="videos") as pbar:
        while completed < total_videos:
            try:
                progress_queue.get(timeout=1)
                completed += 1
                pbar.update(1)
            except:
                continue

def batch_feature_extract(
    video_infos,
    processor_path,
    feature_dir,

    max_workers=None,
    dataset=None,
    chunk_size=1,dynamic = False, model_type="qwen2_5", write=False
):
    """
    批量特征提取（多进程版本）
    
    Args:
        video_infos: 视频信息列表，每个元素包含video_name和duration
        processor_path: processor路径
        feature_dir: 特征保存目录
        frame_dir: 帧目录
        audio_dir: 音频目录（可选）
        max_workers: 最大进程数
        chunk_size: 每个进程处理的批次大小
    
    Returns:
        处理结果统计
    """
    
    if max_workers is None:
        max_workers = min(cpu_count() // 2, 8)  # 保守设置，避免内存不足
    
    # 创建输出目录
    os.makedirs(feature_dir, exist_ok=True)
    
    total_videos = len(video_infos)
    print(f"开始批量特征提取")
    print(f"总视频数: {total_videos}")
    print(f"使用进程数: {max_workers}")
    print(f"处理器路径: {processor_path}")
    print(f"特征保存目录: {feature_dir}")
    print("-" * 60)
    
    # 创建进度队列
    manager = Manager()
    progress_queue = manager.Queue()
    
    # 启动进度监控
    from multiprocessing import Process
    monitor_process = Process(
        target=progress_monitor,
        args=(progress_queue, total_videos)
    )
    monitor_process.start()
    
    # 创建部分函数
    extract_func = partial(
        feature_extract_single,
        processor_path=processor_path,
        feature_dir=feature_dir,
        frame_dir=frame_dir,
        audio_dir=audio_dir,
        progress_queue=progress_queue,
        dataset = dataset,
        dynamic = dynamic,
        model_type=model_type, write=write
    )
    
    start_time = time.time()
    results = []
    
    try:
        with Pool(processes=max_workers) as pool:
            # 使用map进行并行处理
            results = pool.map(extract_func, video_infos, chunksize=chunk_size)
            
    except KeyboardInterrupt:
        print("\n用户中断处理")
        monitor_process.terminate()
        return None
    except Exception as e:
        print(f"处理过程中出错: {e}")
        monitor_process.terminate()
        return None
    
    # 等待进度监控完成
    monitor_process.join()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 统计结果
    successful = sum(1 for _, success, _ in results if success)
    failed = total_videos - successful
    already_exists = sum(1 for _, success, msg in results if success and msg == "already_exists")
    
    print("\n" + "=" * 60)
    print("特征提取完成统计")
    print("=" * 60)
    print(f"总视频数: {total_videos}")
    print(f"成功处理: {successful}")
    print(f"处理失败: {failed}")
    print(f"已存在文件: {already_exists}")
    print(f"新处理文件: {successful - already_exists}")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    print(f"平均每个视频: {elapsed_time/total_videos:.2f} 秒")
    
    # 保存失败列表
    failed_videos = [video_name for video_name, success, _ in results if not success]
    if failed_videos:
        failed_path = os.path.join(feature_dir, "failed_videos.txt")
        with open(failed_path, 'w') as f:
            for video_name in failed_videos:
                f.write(f"{video_name}\n")
        print(f"失败视频列表已保存到: {failed_path}")
    
    return {
        'total': total_videos,
        'successful': successful,
        'failed': failed,
        'already_exists': already_exists,
        'elapsed_time': elapsed_time,
        'results': results
    }

    
def prepare_video_infos_general(video_dir=None, duration_dict=None, dataset="worldsense"):
    """Prepare video information list by filtering valid duration videos.

    This function processes a duration dictionary to generate a standardized list of video information,
    only including videos with positive duration values.

    Args:
        video_dir (str, optional): Path to the directory where videos are stored. Defaults to None.
        duration_dict (str, optional): Path of a dictionary mapping video names to their corresponding durations (in seconds).
            Key: str (video name), Value: float/int (video duration). Defaults to None.
        dataset (str, optional): Name of the target dataset (default: "worldsense").

    Returns:
        list[dict]: A list of dictionaries containing video information. Each dictionary has the following keys:
            - video_name (str): Name of the video file
            - duration (float/int): Valid duration of the video (greater than 0 seconds)
    """
    with open(duration_dict, "r") as fr:
        duration_dict = json.load(fr)
    video_infos = []
    for video_name, duration in duration_dict.items():
        if duration > 0:
            video_infos.append({
                'video_name': video_name,
                'duration': duration
            })
    return video_infos


def prepare_video_infos_omninext(data_file, video_root = None, train = False):
    with open(data_file, "r") as fr:
        data = json.load(fr)
    if video_root is None:
        raise Error("Root is not defined.")
    video_infos = []
    for item in data:
        new_item = {}
        if train:
            train_item = item['videos'] # [path, timeformat, id, duration]
            idx = train_item[-2]
            video_name = train_item[0]
            new_item['video_name'] = [f"{video_root}/{video_name}", idx, "futureomni_train"]
            timestamp = train_item[1]
            st, ed = trans2seconds(timestamp)
            duration = train_item[-1]
            new_item['duration'] = [ed, duration]
        else:
            new_item['video_name'] = [f"{video_root}/{item['original_video']}", item['id'], "futureomni_test"]
            new_item['duration'] = [item['split_point'], item['duration']]
        video_infos.append(new_item)
    return video_infos
import argparse
if __name__ == "__main__":
    # 配置参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--processor_path", default="Qwen2.5-Omni-7B")
    parser.add_argument("--dataset", default="omnivideobench")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--start_id",type=int,default=0)
    parser.add_argument('--debug',action="store_true")
    parser.add_argument("--qid",type=int,default=0)
    parser.add_argument("--model_type",type=str,default="qwen2_5omni")
    parser.add_argument("--video_root",type=str,required=True)
    parser.add_argument("--save_dir",type=str,required=True, help="path to store the extracted feature")
    parser.add_argument('--train',action="store_true")
    parser.add_argument("--duration_dict",type=str, help="needed when dataset is general")
    parser.add_argument("--max_frames",type=int,default=32)

    args = parser.parse_args()
    
    
    processor_path = args.processor_path
    dataset = args.dataset
    
    model_type = args.model_type
    
    
    feature_dir = args.save_dir
    os.makedirs(feature_dir, exist_ok=True)
    
    max_workers = 64  
   
    if "futureomni" in args.dataset:
        video_infos = prepare_video_infos_omninext(args.data_path, video_root=args.video_root, train = args.train)
    
        dynamic = True
    else:
        video_infos = prepare_video_infos_general(video_root=args.video_root, duration_dict=args.duration_dict, dataset=args.dataset)
        dynamic = False
    if args.debug:
        qid_x = None
        
        _ = feature_extract_single(
            video_info= video_infos[qid_x] if qid_x else  video_infos[args.qid],
            processor_path=processor_path,
            feature_dir=feature_dir,
            dynamic= dynamic,
            dataset=dataset, 
            model_type=args.model_type,
            max_frames=args.max_frames
        )
    else:
        
        results = batch_feature_extract(
            video_infos=video_infos,
            processor_path=processor_path,
            feature_dir=feature_dir,
            max_workers=max_workers,
            dataset=dataset,
            chunk_size=1, 
            dynamic=dynamic,
            model_type=args.model_type 
            )
        
        if results:
            print(f"\nFeature Extracted Finished! Stored in: {feature_dir}")
        else:
            print("Feature Extracted Failed!")
        
    

