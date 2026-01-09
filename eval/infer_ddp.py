import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, List
from collections import defaultdict
import os
import pdb
import argparse
import json
from typing import Optional, Union
import re
BASE_SYS = 'Carefully watch this video and pay attention to every detail. '
BASE_SYS2 = 'Carefully listen to this audio and pay attention to every detail. '
SYS = BASE_SYS + 'Based on your observations, select the best option that accurately addresses the question.'
TEST_PROMPT_VL = """
These are the frames of a video. \
Select the best answer to the following multiple-choice question based on the video. \
Respond with only the letter (A, B, C, D, E, F) of the correct option.
"""

TEST_PROMPT_OMNI1 = """
These are the frames of a video and the corresponding audio.
Please answer the following multiple-choice question based on the video and audio content.
Choose the correct option and respond with ONLY the letter (A, B, C, D) of your choice.
"""
TEST_PROMPT_OMNI2 = """
These are the frames of a video and the corresponding audio.
Please answer the following multiple-choice question based on the video and audio content.
Choose the correct option and respond with ONLY the letter (A, B, C, D, E and F) of your choice.
"""
# TEST_PROMPT_OMNI = """
# These are the frames of a video and corresponding audio. \
# Select the best answer to the following multiple-choice question based on the video. \
# Respond with only the letter (A, B, C, D, E, F) of the correct option.
# """
MIN_PIXELS = 128 * 28 * 28
MAX_PIXELS = 768 * 28 * 28
TOTAL_PIXELS = 32 * 768 * 28 * 28
NFRAMES = 32
USE_AUDIO_IN_VIDEO = False

class VideoDataset(Dataset):
    """Custom Dataset for loading video data"""
    def __init__(self, df, root = None,dataset="futureomni",model_type="qwen2_5omni"):
        self.df = df
        self.root = root
        self.dataset = dataset
        self.model_type = model_type
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if isinstance(self.df, pd.DataFrame):
            row = self.df.iloc[idx].to_dict()
        elif isinstance(self.df, list):
            row = self.df[idx]
        else:
            pass
        item = {}
        
        
        if self.dataset == "futureomni":
            
            item['options'] = []
            alphas = ['A.','B.','C.','D.','E.','F.']
            message = []
            message.append(dict(type='text', value=TEST_PROMPT_OMNI2))
            
            source = row['source']
            qid = row['qid']
            
            
            question_str = row['question'] +  '\n'.join(row['options'])
            
            
            prompt = 'Question: {}\nAnswer: '.format(question_str)


            if self.mode == 'caption':
                caption = item['caption']
                prompt = f'Caption:{caption}' + prompt
            elif self.mode == 'subtitle':
                subtitle = item['subtitle']
                prompt = f'Subtitle:{subtitle}' + prompt
                
                
            
            message.append(dict(type='text', value=prompt))
            if self.model_type == 'qwen2_5omni':
                
                message.append(dict(type='video', value=row['video']))
                message.append(dict(type='audio', value=row['video']))
            else:
                nid = row['_index']
                video_path = f"{self.root}/{nid}.mp4"
                message.append(dict(type='video', value=video_path))
                if self.model_type == "qwen2_5omni" or self.model_type == "qwen3omni":
                    
                    message.append(dict(type='audio', value=video_path))
                
                    
            message.append(row['seconds'])
        else:
            raise NotImplementedError(f"Dataset {self.dataset} Not Implemented")

        return message, idx

def setup_distributed():
    """Initialize distributed training environment"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        print("Not using distributed training")
        return None, None, None
    
    # Initialize the process group
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

def cleanup():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def _read_data(file, histories=None):
    if file.endswith("tsv"):
        df = pd.read_csv(file, sep="\t")
        return df
    elif file.endswith("json"):
        ks = None
        if histories is not None:
            ks = []
            for history in histories:
                with open(history, "r") as fr:
                    runs = json.load(fr)
                ks.extend([x['source'][:-5] + '-' + str(x['qid']) if x['source'].endswith("json") else x['source'] + '-' + str(x['qid']) for x in runs])
        with open(file, "r") as fr:
            df = json.load(fr)
        if ks is not None:
            ndf = []
            for x in df:
                k = x['source'][:-5] + '-' + str(x['qid']) if x['source'].endswith("json") else x['source'] + '-' + str(x['qid'])
                if k not in ks:
                    ndf.append(x)
            print(f"filtering {len(df) - len(ndf)} samples")
            return ndf
        else:
            return df
                    
def _load_model(model_path, local_rank=None):
    """Load model with DDP support"""
    if "Qwen3-Omni" in model_path:
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
        
        # Load model on specific GPU
        if local_rank is not None:
            device = torch.device(f"cuda:{local_rank}")
            model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                model_path, 
                dtype="auto",
                device_map=None,  # Don't use device_map with DDP
                attn_implementation="flash_attention_2"
            ).to(device)
        else:
            model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                model_path, 
                dtype="auto",
                device_map="auto",
                attn_implementation="flash_attention_2"
            )
        
        processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    elif "Qwen3-VL" in model_path:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        if local_rank is not None:
            device = torch.device(f"cuda:{local_rank}")
            model = AutoModelForImageTextToText.from_pretrained(
                model_path, 
                torch_dtype='auto',
                device_map=None,  # Don't use device_map with DDP
                attn_implementation='flash_attention_2'
            ).to(device)
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                model_path, 
                torch_dtype='auto',
                device_map="auto",
                attn_implementation='flash_attention_2'
            )
        processor = AutoProcessor.from_pretrained(model_path)
    elif 'Qwen2.5-VL' in model_path:
        from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
        if local_rank is not None:
            device = torch.device(f"cuda:{local_rank}")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, 
                torch_dtype='auto',
                device_map=None,  # Don't use device_map with DDP
                attn_implementation='flash_attention_2'
            ).to(device)
        else:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, 
                torch_dtype='auto',
                device_map="auto",
                attn_implementation='flash_attention_2'
            )
        processor = AutoProcessor.from_pretrained(model_path)
    elif "Qwen2-VL" in model_path:
        from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
        if local_rank is not None:
            device = torch.device(f"cuda:{local_rank}")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path, 
                torch_dtype='auto',
                device_map=None,  # Don't use device_map with DDP
                attn_implementation='flash_attention_2'
            ).to(device)
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path, 
                torch_dtype='auto',
                device_map="auto",
                attn_implementation='flash_attention_2'
            )
        processor = AutoProcessor.from_pretrained(model_path)
    elif "Qwen2.5-Omni" in model_path:
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        
        # Load model on specific GPU
        if local_rank is not None:
            device = torch.device(f"cuda:{local_rank}")
            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                model_path, 
                torch_dtype='auto',
                device_map=None,  # Don't use device_map with DDP
                attn_implementation='flash_attention_2'
            ).to(device)
        else:
            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                model_path, 
                torch_dtype='auto',
                device_map="auto",
                attn_implementation='flash_attention_2'
            )
        
        processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    else:
        raise ValueError("Model not supported yet")
    
    model.eval()
    
    # Wrap model with DDP if using distributed training
    if local_rank is not None:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    return model, processor

def get_video_path(inputs: list[dict[str, str]]):
    for s in inputs:
        if s['type'] == 'video':
            return s['value']
    return None
def _prepare_content(inputs: list[dict[str, str]], dataset:str, model_type:str = "qwen2_5omni") -> list[dict[str, str]]:
    """Prepare content for processing"""
    content = []
    if dataset == "futureomni":
        seconds = inputs[-1]
        assert seconds is not None, isinstance(seconds, float)   
        inputs.pop()    
        for s in inputs:
            if s['type'] == 'video':
                if model_type == "qwen2_5omni":
                    item = {
                        'type': 'video',
                        'video': s['value'],
                        'min_pixels': MIN_PIXELS,
                        'max_pixels': MAX_PIXELS,
                        'total_pixels': TOTAL_PIXELS,
                        'max_frames': NFRAMES,
                        'video_end': seconds
                    }
                else:
                    item = {
                        'type': 'video',
                        'video': s['value'],
                        'min_pixels': MIN_PIXELS,
                        'max_pixels': MAX_PIXELS,
                        'total_pixels': TOTAL_PIXELS,
                        'max_frames': NFRAMES,
                    }
                    
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            elif s['type'] == 'audio':
                item = {'type': 'audio', 'audio': s['value'], 'audio_end': seconds}
            elif s['type'] in ['ans', 'frame']:
                continue
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
    else:
        raise NotImplementedError()
        
    return content

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    messages, indices = zip(*batch)
    return list(messages), list(indices)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, 
                       default="Qwen2.5-Omni-7B")
    parser.add_argument('--data_file', type=str, 
                       default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--sid', type=int, default=0)
    parser.add_argument('--skip_rows', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument("--histories", nargs='+', type=str, help="history paths")
    parser.add_argument('--dataset', type=str, default="worldsense")
    parser.add_argument('--model_type', type=str, default="qwen2_5omni")

    args = parser.parse_args()
    if "Qwen2.5-Omni" in args.model_path or 'Qwen3-Omni' in args.model_path:
        # model_type = "qwen2_5"
        from qwen_omni_utils import process_mm_info
    elif 'Qwen3-VL' in args.model_path or 'Qwen2.5-VL' in args.model_path or 'Qwen2-VL' in args.model_path:
        # model_type = "qwen3-vl"
        from qwen_vl_utils import process_vision_info
    
    # Setup distributed training
    df = _read_data(args.data_file,  histories=args.histories)
    print(f"running sid:{args.sid} len:{len(df)}")
    rank, world_size, local_rank = setup_distributed()
    
    
    # Create dataset and dataloader
    dataset = VideoDataset(df,dataset=args.dataset, model_type=args.model_type)
    
    if rank is not None:
        # Use DistributedSampler for DDP
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=1,
            pin_memory=True
        )
    else:
        # Single GPU mode
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=1,
            pin_memory=True
        )
    
    # Load model
    print(f"Loading Model:{args.model_path}")
    model, processor = _load_model(args.model_path, local_rank)
    
    # Get the actual model (unwrap DDP if necessary)
    actual_model = model.module if hasattr(model, 'module') else model
    
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    

    # Process data
    for batch_idx, (messages, indices) in enumerate(dataloader):
        for msg_idx, (message, original_idx) in enumerate(zip(messages, indices)):
            # Calculate global question ID
            if rank is not None:
                qid = args.sid + original_idx + (rank * len(dataset) // world_size)
            else:
                qid = args.sid + original_idx
            
            # Prepare message
            new_message = []
            new_message.append({'role': 'user', 'content': _prepare_content(message, dataset=args.dataset, model_type=args.model_type)})
            
            # Process with model
            text = processor.apply_chat_template(
                [new_message], 
                tokenize=False, 
                add_generation_prompt=True
            )
            if 'Qwen3-VL' in args.model_path:
                images, videos = process_vision_info(new_message, image_patch_size=16)
                inputs = processor(text=text, images=images, videos=videos, do_resize=False, return_tensors="pt")
            elif 'Qwen2.5-VL' in args.model_path or 'Qwen2-VL' in args.model_path:
                # try:
                images, videos = process_vision_info(new_message)
                
                inputs = processor(
                    text=text,
                    images=images,
                    videos=videos,
                    padding=True,
                    return_tensors="pt",
                )
            elif "Qwen2.5-Omni" in args.model_path:
                audios, images, videos = process_mm_info(new_message, use_audio_in_video=True)
                assert videos is not None
                assert audios is not None
                
                
                inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=False, use_audio_in_video=True)

            
            if local_rank is not None:
                device = torch.device(f"cuda:{local_rank}")
                inputs = {k: v.to(device) if torch.is_tensor(v) else v 
                         for k, v in inputs.items()}
            else:
                inputs = inputs.to('cuda')
            
            
            # Generate output
            with torch.no_grad():
                if hasattr(model, 'module'):
                    if 'Qwen2.5-Omni' in args.model_path or 'Qwen3-Omni' in args.model_path:
                        generated_ids = model.module.generate(
                            **inputs, 
                            use_audio_in_video=True if args.mode == 'all' else False, 
                            return_audio=False, 
                            do_sample=False
                        )
                    else:
                        generated_ids = model.module.generate(
                            **inputs,
                            do_sample=False

                        )
                        
                else:
                    if 'Qwen2.5-Omni' in args.model_path or 'Qwen3-Omni' in args.model_path:
                        generated_ids = model.generate(
                            **inputs, 
                            use_audio_in_video=True if args.mode == 'all' else False, 
                            return_audio=False,
                            do_sample=False

                        )
                    else:
                        generated_ids = model.generate(
                            **inputs,
                            do_sample=False

                        )
            
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(inputs['input_ids'], generated_ids)
            ]
            
            out = processor.tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            response = out[0]
            
            # Save attention scores with rank-specific directory
            if rank is not None:
                save_dir = f"{args.output_dir}/rank_{rank}"
            else:
                save_dir = f"{args.output_dir}"
            
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{original_idx}.json")
            x = {'pred' : response}
            x.update(df[original_idx].items())
            with open(save_path, "w") as fw:
                json.dump(x, fw, indent=4)
            # Log progress on rank 0
            if rank == 0 or rank is None:
                print(f"Processed question {qid}, Response: {response[:50]}...")
    
    # Cleanup
    
    # Wait for all processes to complete
    if rank is not None:
        dist.barrier()
    
    # Cleanup distributed training
    cleanup()
    
    if rank == 0 or rank is None:
        print("Processing complete!")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()
