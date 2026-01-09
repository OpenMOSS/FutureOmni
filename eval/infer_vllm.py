import soundfile as sf
from dataset.load_dataset import load_dataset
from qwen_omni_utils import process_mm_info
import json
import torch

import pdb
from vllm import LLM, SamplingParams

import os
from tqdm import tqdm
import argparse
import pandas as pd
import re
from typing import Optional, List, Union
from constructor import trans_seconds2
from utils import load_dataset
os.environ['VLLM_USE_V1'] = '0'

MAX_MODEL_LEN = 32768

def _prepare_content(video_path: Optional[Union[str, List[str]]], prompt_text: str,audios: Optional[List[str]], MAX_FRAMES:int = 32):
    """Prepare content for video processing"""
    messages = []
    contents = []
    if video_path is not None and isinstance(video_path, str):
        contents = [
            {'type': 'text', 'text': prompt_text},
            {
                'type': 'video', 
                'video': video_path, 
                'min_pixels': 100352, 
                'max_pixels': 602112, 
                'total_pixels': 38535168 if MAX_FRAMES == 64 else 19267584, 
                'nframes': MAX_FRAMES
            }
        ]
    if audios is not None:
        # for audio in audios:
        contents.append({'type': 'audio', 'audio': audios})
    messages.append({
        'role': 'user',
        'content': contents
    })
    
    return messages
def check_exist(idx, save_dir):
    file_path = os.path.join(save_dir, str(idx) + ".json")
    return os.path.exists(file_path)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process videos with Qwen2.5-Omni using vLLM")
    parser.add_argument("--data_file", type=str,default=None, help="processing file path")
    parser.add_argument("--dataset", type=str,default=None, help="futureomni")
    parser.add_argument("--model_path", type=str,default="Qwen2.5-Omni-7B", help="processing file path")
    parser.add_argument("--model_type", type=str,default=None, help="qwen2_5omni")
    parser.add_argument("--batch_size", type=int,default=1)
    parser.add_argument("--gpu_device", type=str, default=None, help="GPU device to use, e.g., '0' or '0,1'")
    parser.add_argument("--max_frames", type=int,default=32)
    parser.add_argument("--root", type=str, requires=True, help="Root path of stored videos")
    parser.add_argument("--feature_dir", type=str, default=None, help="Root path of stored extracted feature, including video, audio and feature")

    args = parser.parse_args()
    if args.gpu_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    MODEL_PATH = args.model_path
    MAX_FRAMES = args.max_frames
    save_dir = f"./results/{args.model_type}/{args.dataset}_{MAX_FRAMES}"
    os.makedirs(save_dir, exist_ok=True)
    if  "qwen2_5omni" in args.model_type and "qwen2_5_vl" not in args.model_type:
        from tranformers import Qwen2_5OmniProcessor
        feat_type = "qwen2_5"
        processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
    elif "qwen3_vl" in args.model_type:
        from transformers import AutoProcessor
        feat_type = "qwen3_vl"
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
    elif "qwen3omni" in args.model_type:
        from transformers import Qwen3OmniMoeProcessor
        feat_type = "qwen3"
        from transformers import Qwen3OmniMoeProcessor
        processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
    elif "qwen2_5_vl" in args.model_type:
        feat_type = "qwen2_5_vl"
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
    MAX_PIXELS = 768 * 28 * 28
    MIN_PIXELS = 128 * 28 * 28
    BATCH_SIZE = args.batch_size
    
    data_file = args.data_file
    dataset = args.dataset
    save_file = f"./results/{dataset}_{args.model_type}_{MAX_FRAMES}.json"
    
    USE_AUDIO_IN_VIDEO = True
    
    if "qwen3_vl" in args.model_type or "qwen2_5_vl" in args.model_type:
        USE_AUDIO_IN_VIDEO = False
    FRAMES_TMPL_AUDIO = """
    These are the frames of a video and the corresponding audio. \
    Select the best answer to the following multiple-choice question based on the video. \
    """
    FRAMES_TMPL_VIDEO = """
    Select the best answer to the following multiple-choice question based on the video. \
    Respond with only the letter of the correct option.\n
    """
    FRAMES_TMPL_AUDIO_SINGLE = """
    These are the frames of a audio. \
    Select the best answer to the following multiple-choice question based on the video. \
    """
    PROMPT_WITH_SIX_OPTION = """
    These are the frames of a video and the corresponding audio.
    Please answer the following multiple-choice question based on the video and audio content.
    Choose the correct option and respond with ONLY the letter (A, B, C, D, E, F) of your choice.
    """
    
    PROMPT_WITHOUT_OPTION = """
    These are the frames of a video and the corresponding audio.
    Please answer the following question based on the video. \
    Respond with only Yes/No.
    """

    items = load_dataset(dataset=args.dataset, file_path=args.data_file, video_root=args.root)
    all_items = []
    
    
    
    limit_mm_per_prompt = {'video': 1, 'audio': 1}
    llm = LLM(
    model=MODEL_PATH, trust_remote_code=True, gpu_memory_utilization=0.95,
            tensor_parallel_size=torch.cuda.device_count(),
            limit_mm_per_prompt=limit_mm_per_prompt,
            max_model_len = 65536  if 'Qwen3-Omni' in args.model_path else 32768,
            seed=1234        
    )
    
    sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=1000,
        )
    for i in tqdm(range(0, len(items), BATCH_SIZE), desc="processing batches..."):
        batch_items = [items[i + j] for j in range(BATCH_SIZE) if i + j < len(items) and not check_exist(i + j, save_dir=save_dir)]
        all_items.extend([items[i + j] for j in range(BATCH_SIZE) if i + j < len(items) and  check_exist(i + j, save_dir=save_dir)])
        if len(batch_items) == 0:
            continue

        processed_inputs = []
        read_pt = []
        feature_dir = args.feature_dir
        
        for idx, item in enumerate(batch_items):
            if feature_dir is not None:
                if dataset == "futureomni":
                    save_feature_path = "/".join([feature_dir, "feature", str(item['id']) + ".pt"])
                else:
                    video_path = item['video']
                    video_name = video_path.split("/")[-1]
                    save_feature_path = "/".join([feature_dir, "feature", f"{video_name}.pt"])
                read_pt.append(save_feature_path)
                        
            question_str = ''
            question = item['question']
            formatted_choices = '\n'.join(item['options'])
            if item['options'] is not None:
                question_str = item['question'] +  '\n'.join(item['options'])
            else:
                question_str = item['question'] +  '\n'
            if dataset not in ['videomme']:
            
                prompt = 'Question: {}\nAnswer: '.format(question_str)
            else:
                prompt = f"Question: {question}\nOptions:\n{formatted_choices}\nThe best answer is:"
            
            video_path = item['video']
            
            prompt_text = ""
            alphs = ['A','B','C','D','E','F']
            width = len(item['options'])
            option_text = ", ".join([alpha for alpha in alphs[:width]])
            if dataset in ['videomme','mlvu']:
                prompt_text = FRAMES_TMPL_VIDEO +  prompt 
            elif dataset in ['futureomni']:
                prompt_text = PROMPT_WITH_SIX_OPTION + prompt
            else:
                prompt_text = FRAMES_TMPL_AUDIO + SUFFIX + prompt 
            
            
            messages = _prepare_content(video_path, prompt_text, audios = None , MAX_FRAMES=MAX_FRAMES)
            text = processor.apply_chat_template(
                            [messages], tokenize=False, add_generation_prompt=True
                            )
            # print(f"text:{text}")
            video_name = video_path.split("/")[-1]
            
            # print(f"save_video_path:{save_video_path}")
            if feature_dir is not None:
                if dataset == "futureomni":
                    save_video_path =  "/".join([feature_dir, "video", str(item['id']) + ".pt"])
                    save_audio_path =  "/".join([feature_dir, "audio", str(item['id']) + ".pt"])
                else:
                    save_video_path =  "/".join([feature_dir, "video", video_name + ".pt"])
                    save_audio_path = "/".join([feature_dir, "audio",  video_name + ".pt"])
                
            if os.path.exists(save_video_path):
                videos = [torch.load(save_video_path)]
            
            if os.path.exists(save_audio_path):
                audios = [torch.load(save_audio_path)]
        
            if USE_AUDIO_IN_VIDEO:
            # Prepare vLLM inputs
                inputs_vllm = {
                    "prompt": text[0],
                    "multi_modal_data": {
                        "video": videos, 
                        "audio": audios
                    },
                    "mm_processor_kwargs": {"use_audio_in_video": USE_AUDIO_IN_VIDEO},
                }
            elif "qwen3_vl" in args.model_type:
                inputs_vllm = {
                    "prompt": text[0],
                    "multi_modal_data": {
                        "video": videos,
                    } 
                    }
            elif "qwen2_5_vl" in args.model_type:
                inputs_vllm = {
                    "prompt": text[0],
                    "multi_modal_data": {
                        "video": videos,
                    } 
                    }
            processed_inputs.append(inputs_vllm)
        outputs = llm.generate(processed_inputs, sampling_params=sampling_params, read_pt  =read_pt)
        result = [outputs[i].outputs[0].text for i in range(len(outputs))]
        
        for j in range(BATCH_SIZE):
            if i + j < len(items):
                # if result[j] != 'fail':
                items[i + j]['pred'] = result[j]
                # if "omninext" not in dataset:
                with open(os.path.join(save_dir, f"{i+j}.json"), "w") as fw:
                    json.dump(items[i + j], fw, indent=4)
                all_items.append(items[i + j])
        
    with open(save_file, "w") as fw:
        json.dump(all_items, fw, indent=4)