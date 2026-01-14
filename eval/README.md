# FutureOmni Evaluation Inference Scripts

This directory contains inference scripts for evaluating multimodal models (video + audio) on the FutureOmni dataset. The scripts support both open-source and closed-source models:

## Open-Source Models

1. **`infer_ddp.py`**: Distributed Data Parallel (DDP) inference using PyTorch
2. **`infer_vllm.py`**: High-performance inference using vLLM

Both scripts support multiple Qwen model variants and can process video-audio multimodal inputs for question-answering tasks.

## Closed-Source Models

3. **`close_source/inference.py`**: Unified inference script for closed-source API-based models (Claude and Gemini)

This script supports commercial API-based models that don't require local model deployment. See the [Closed-Source Models](#closed-source-models) section below for details.

## Overview

### `infer_ddp.py` - DDP-based Inference

- Uses PyTorch's DistributedDataParallel for multi-GPU inference
- Supports distributed training across multiple nodes
- Processes videos sequentially in batches
- Good for: Research, debugging, and scenarios requiring fine-grained control

### `infer_vllm.py` - vLLM-based Inference

- Uses vLLM for optimized inference performance
- Automatic tensor parallelism and batching
- Faster throughput for large-scale evaluation
- Good for: Production inference, large-scale evaluation, and maximum throughput

## Requirements

### Common Dependencies

```bash
pip install torch torchvision transformers
pip install pandas numpy tqdm
pip install soundfile librosa
pip install opencv-python
```

### For `infer_ddp.py`

- PyTorch with NCCL backend (for multi-GPU)
- Flash Attention 2 (recommended)

### For `infer_vllm.py`

- vLLM (version with Qwen Omni support)
- Requires custom vLLM installation with Qwen Omni model support

### Utility Dependencies

The scripts require the following utility modules (should be in Python path):
- `qwen_omni_utils`: Contains `process_mm_info` function for processing multimodal inputs
- `qwen_vl_utils`: Contains `process_vision_info` function for vision-only models
- `constructor`: Contains time conversion utilities (e.g., `trans_seconds2`)
- `utils`: Contains `load_dataset` function

## Usage

### `infer_ddp.py` - Distributed Inference

#### Basic Usage (Single GPU)

```bash
python infer_ddp.py \
    --model_path "Qwen2.5-Omni-7B" \
    --data_file "test.json" \
    --output_dir "./results" \
    --dataset "futureomni" \
    --model_type "qwen2_5omni" \
    --batch_size 1
```

#### Multi-GPU (Distributed)

```bash
torchrun --nproc_per_node=4 infer_ddp.py \
    --model_path "Qwen2.5-Omni-7B" \
    --data_file "test.json" \
    --output_dir "./results" \
    --dataset "futureomni" \
    --model_type "qwen2_5omni" \
    --batch_size 1 \
    --sid 0
```

#### Multi-Node Distributed

```bash
# Node 0
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 \
    --master_addr="<node0_ip>" --master_port=29500 \
    infer_ddp.py --model_path "Qwen2.5-Omni-7B" ...

# Node 1
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=4 \
    --master_addr="<node0_ip>" --master_port=29500 \
    infer_ddp.py --model_path "Qwen2.5-Omni-7B" ...
```

#### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | str | "Qwen2.5-Omni-7B" | Path to model (HuggingFace ID or local path) |
| `--data_file` | str | None | **Required**: Path to JSON/TSV data file |
| `--batch_size` | int | 1 | Batch size per GPU |
| `--sid` | int | 0 | Starting ID for question numbering |
| `--skip_rows` | int | 0 | Number of rows to skip (legacy) |
| `--output_dir` | str | None | **Required**: Directory to save results |
| `--histories` | list[str] | None | List of history JSON files to filter out (skip already processed) |
| `--dataset` | str | "worldsense" | Dataset name (e.g., "futureomni") |
| `--model_type` | str | "qwen2_5omni" | Model type identifier |

#### Supported Model Types

- `qwen2_5omni`: Qwen2.5-Omni models (multimodal: video + audio)
- `qwen3omni`: Qwen3-Omni models (multimodal: video + audio)
- Qwen2.5-VL, Qwen2-VL, Qwen3-VL: Vision-language models (video only)

#### Output Format

Results are saved in the output directory:
```
output_dir/
├── rank_0/          # Results from GPU rank 0
│   ├── 0.json
│   ├── 1.json
│   └── ...
├── rank_1/          # Results from GPU rank 1
│   └── ...
└── ...
```

Each JSON file contains:
```json
{
    "pred": "A",          # Model prediction (letter)
    "qid": 123,          # Question ID
    "question": "...",    # Original question
    "options": [...],     # Answer options
    "video": "...",       # Video path
    "source": "...",      # Data source
    "seconds": 30.0       # Video duration/segment
}
```

---

### `infer_vllm.py` - vLLM Inference

#### Basic Usage

```bash
python infer_vllm.py \
    --model_path "Qwen2.5-Omni-7B" \
    --data_file "test.json" \
    --dataset "futureomni" \
    --model_type "qwen2_5omni" \
    --root "/path/to/videos" \
    --feature_dir "/path/to/features" \
    --batch_size 4 \
    --max_frames 32 \
    --gpu_device "0,1,2,3"
```

#### Command Line Arguments

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--model_path` | str | "Qwen2.5-Omni-7B" | No | Path to model |
| `--data_file` | str | None | Yes | Path to dataset JSON file |
| `--dataset` | str | None | Yes | Dataset name (e.g., "futureomni") |
| `--model_type` | str | None | Yes | Model type (e.g., "qwen2_5omni", "qwen3omni") |
| `--batch_size` | int | 1 | No | Batch size for inference |
| `--gpu_device` | str | None | No | GPU devices (e.g., "0,1,2,3") |
| `--max_frames` | int | 32 | No | Maximum frames per video |
| `--root` | str | - | **Yes** | Root directory containing video files |
| `--feature_dir` | str | None | No | Directory with pre-extracted features (video/, audio/, feature/) |

#### Supported Model Types

- `qwen2_5omni`: Qwen2.5-Omni (multimodal)
- `qwen3omni`: Qwen3-Omni (multimodal)
- `qwen3_vl`: Qwen3-VL (vision only)
- `qwen2_5_vl`: Qwen2.5-VL (vision only)

#### Feature Directory Structure

If `--feature_dir` is provided, the script expects:
```
feature_dir/
├── video/
│   ├── {qid}.pt      # Preprocessed video tensors (FutureOmni)
│   └── {video_name}.pt   # (other datasets)
├── audio/
│   ├── {qid}.pt      # Preprocessed audio tensors
│   └── ...
└── feature/
    └── ...           # (optional) Combined features
```

#### Output Format

Results are saved incrementally:
```
results/
├── {model_type}/
│   └── {dataset}_{max_frames}/
│       ├── 0.json
│       ├── 1.json
│       └── ...
└── {dataset}_{model_type}_{max_frames}.json  # Final aggregated results
```

Each JSON contains the same format as `infer_ddp.py` output.

---

## Closed-Source Models

### `close_source/inference.py` - API-Based Inference

The `close_source/inference.py` script provides a unified interface for evaluating closed-source multimodal models through their APIs. This is useful for comparing against commercial models without requiring local GPU resources or model weights.

#### Supported Providers

- **Claude (Anthropic)**: Supports Claude Haiku 4 models
- **Gemini (Google)**: Supports Gemini 2.5 Flash/Pro, and Gemini 3 Flash/Pro models
- **GPT-4o (OpenAI)**: Supports GPT-4o

#### Key Features

1. **Unified Interface**: Single script for multiple API providers
2. **Flexible Input**: Supports frame-based (Claude) or video-based (Gemini) inputs
3. **Audio Support**: Claude models support audio file uploads
4. **Concurrent Processing**: Multi-threaded processing for Gemini (configurable)
5. **Resume Capability**: Automatically skips already processed samples
6. **Error Handling**: Robust error handling with automatic retries

#### Requirements

```bash
# For Claude
pip install anthropic

# For Gemini
pip install requests

# Common dependencies
pip install tqdm
```

#### Basic Usage

**Claude:**
```bash
python close_source/inference.py --provider claude input.json \
    --frame_dir ./frames \
    --audio_dir ./audio \
    --api_key $ANTHROPIC_API_KEY \
    --model "claude-haiku-4.5" \
    --output_dir ./claude_results
```

**Gemini:**
```bash
python close_source/inference.py --provider gemini input.json \
    --video_dir ./videos \
    --api_key $GEMINI_API_KEY \
    --model "gemini-2.5-pro" \
    --output_dir ./gemini_results \
    --max_workers 8
```

#### Command Line Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--provider` | str | Yes | Model provider: `"claude"`, `"gemini"`, or `"gpt4o"` |
| `input_file` | str | Yes | Input JSON file path |
| `--frame_dir` | str | No | Base directory for frames (Claude only, default: `./frames`) |
| `--audio_dir` | str | No | Base directory for audio (Claude only, default: `./audio`) |
| `--video_dir` | str | No | Base directory for videos (Gemini/GPT-4o, default: `./videos`) |
| `--api_key` | str | No | API key (or set `ANTHROPIC_API_KEY`/`GEMINI_API_KEY`/`OPENAI_API_KEY` env var) |
| `--base_url` | str | No | Custom base URL for API endpoint |
| `--model` | str | No | Model name (provider-specific, see defaults below) |
| `--test_prompt` | str | No | Custom test prompt template |
| `--output_dir` | str | No | Output directory (default: `./results`) |
| `--output_file` | str | No | Optional aggregated results file |
| `--max_items` | int | No | Maximum items to process (for testing) |
| `--max_workers` | int | No | Concurrent workers for Gemini (default: 1) |

#### Default Models

- **Claude**: `claude-haiku-4.5` (supports video and audio)
- **Gemini**: `gemini-2.5-flash` (supports video and audio)
- **GPT-4o**: `gpt-4o` (supports video)

#### Input Format

The script expects the same JSON format as other inference scripts (see [Input Data Format](#input-data-format) below). For Claude, it uses frame images and optionally audio files. For Gemini, it uses video files directly.

#### Output Format

Results are saved in the output directory with the same format as other inference scripts:
```
output_dir/
├── 0.json
├── 1.json
└── ...
```

Each JSON file contains:
```json
{
    "status": "success",
    "answer": "A",
    "qid": 123,
    "nid": 456,
    "num_frames": 32,
    "has_audio": true,
    "_index": 0
}
```

#### Notes

- Claude models require frame extraction and optionally audio extraction beforehand
- Gemini models work directly with video files (automatically encodes as base64)
- GPT-4o supports direct video inputs via API
- API rate limits may apply; adjust `--max_workers` accordingly
- Large video files (>20MB) may cause issues with Gemini's inline method
- GPT-4o has specific file size and format requirements; check OpenAI documentation for details

---

## Input Data Format

### FutureOmni Dataset Format

The JSON file should contain a list of dictionaries:

```json
[
    {
        "qid": 0,
        "source": "train.json",
        "question": "What happens in the video?",
        "options": [
            "A. Person walks into room",
            "B. Person exits room",
            "C. Person sits down",
            "D. Person stands up"
        ],
        "video": "/path/to/video.mp4",
        "seconds": 30.0,
        "_index": 0
    },
    ...
]
```

**Note**: For `infer_ddp.py`, if using feature directory, videos should be organized with IDs matching the dataset. For `infer_vllm.py`, the script uses `item['id']` for FutureOmni or extracts video name from path for other datasets.

## Configuration Constants

### Video Processing Parameters

```python
MIN_PIXELS = 128 * 28 * 28   # Minimum video resolution
MAX_PIXELS = 768 * 28 * 28   # Maximum video resolution
TOTAL_PIXELS = 32 * 768 * 28 * 28  # Total pixels for frame sequence
NFRAMES = 32                 # Number of frames to extract
```

### Prompts

The scripts use predefined prompts for different scenarios:
- `TEST_PROMPT_OMNI1`: For 4-option questions (A, B, C, D)
- `TEST_PROMPT_OMNI2`: For 6-option questions (A, B, C, D, E, F)
- `PROMPT_WITH_SIX_OPTION`: vLLM script variant

## Model Loading

### Supported Models

#### Multimodal (Video + Audio)
- **Qwen2.5-Omni**: `Qwen2.5-Omni-7B`
- **Qwen3-Omni**: `Qwen3-Omni-30B-A3B`, etc.

#### Vision-Language (Video Only)
- **Qwen2.5-VL**: `Qwen/Qwen2.5-VL-7B-Instruct`
- **Qwen2-VL**: `Qwen/Qwen2-VL-7B-Instruct`
- **Qwen3-VL**: `Qwen/Qwen3-VL-72B-Instruct`

All models use Flash Attention 2 for efficiency.

## Key Features

### `infer_ddp.py`

1. **Distributed Processing**: Automatic data sharding across GPUs
2. **Progress Tracking**: Per-rank progress logging
3. **History Filtering**: Skip already processed samples via `--histories`
4. **Flexible Data Sources**: Supports JSON and TSV formats
5. **Error Resilience**: Continues processing even if individual samples fail

### `infer_vllm.py`

1. **High Throughput**: Optimized batching and tensor parallelism
2. **Feature Caching**: Uses pre-extracted features if available
3. **Incremental Saving**: Saves results after each batch
4. **Skip Existing**: Automatically skips already processed samples
5. **Memory Efficient**: Configurable GPU memory utilization

## Performance Tips

### For `infer_ddp.py`

1. **Batch Size**: Start with `batch_size=1` for large videos, increase if memory allows
2. **Workers**: Set `num_workers=1` (already default) to avoid multiprocessing issues
3. **Memory**: Monitor GPU memory; reduce `batch_size` if OOM occurs
4. **Multi-Node**: Use high-bandwidth interconnect (InfiniBand) for best performance

### For `infer_vllm.py`

1. **Batch Size**: Larger batches (4-8) typically improve throughput
2. **Tensor Parallelism**: Automatically uses all available GPUs
3. **GPU Memory**: Set `gpu_memory_utilization=0.95` (default) or lower if needed
4. **Feature Pre-extraction**: Pre-extract features for faster inference


## Differences Between Scripts

| Feature | `infer_ddp.py` | `infer_vllm.py` |
|---------|---------------|-----------------|
| Backend | PyTorch DDP | vLLM |
| Multi-GPU | Manual setup | Automatic |
| Batch Processing | Sequential | Optimized batching |
| Throughput | Moderate | High |
| Memory Efficiency | Good | Excellent |
| Flexibility | High | Moderate |
| Best For | Research, debugging | Production, scale |

## Example Workflows

### Workflow 1: DDP

```bash
# Extract features first (optional but recommended)
python feature/extract.py --data_file train.json --save_dir ./features

# Run inference
torchrun --nproc_per_node=4 infer_ddp.py \
    --model_path "Qwen2.5-Omni-7B" \
    --data_file "test.json" \
    --output_dir "./results_ddp" \
    --dataset "futureomni" \
    --model_type "qwen2_5omni"
```

### Workflow 2: vLLM

```bash
# Extract features
python feature/extract.py --data_file test.json --save_dir ./features

# Run inference with vLLM
python infer_vllm.py \
    --model_path "Qwen2.5-Omni-7B" \
    --data_file "test.json" \
    --dataset "futureomni" \
    --model_type "qwen2_5omni" \
    --root "/data/videos" \
    --feature_dir "./features" \
    --batch_size 8 \
    --gpu_device "0,1,2,3,4,5,6,7"
```

### Workflow 3: Resume Processing

```bash
# DDP: Use histories to skip processed samples
python infer_ddp.py \
    --data_file "test.json" \
    --histories ./results/rank_0/*.json ./results/rank_1/*.json \
    ...

# vLLM: Automatically skips existing files
python infer_vllm.py \
    --data_file "test.json" \
    --output_dir "./results" \
    ...
```

## Notes

- Both scripts process videos frame-by-frame and extract audio tracks
- For FutureOmni, videos are segmented based on `seconds` field
- Predictions are saved as single letters (A, B, C, D, E, F)
- Results can be aggregated and evaluated using separate evaluation scripts
- The `VideoDataset` class in `infer_ddp.py` has a reference to `self.mode` that should be initialized if using caption/subtitle modes



## Other Omnimodal Model Repositories

For evaluation on FutureOmni with other state-of-the-art omnimodal models, you may find the following repositories useful:

### [Ola](https://github.com/Ola-Omni/Ola)

**Ola** is an Omni-modal language model that achieves competitive performance across image, video, and audio understanding. It supports simultaneous processing of text, image, video, and audio inputs with competitive performance on understanding tasks for all modalities.

- **Repository**: https://github.com/Ola-Omni/Ola
- **HuggingFace**: [Ola-7b](https://huggingface.co/THUdyh/Ola-7b), [Ola-Image](https://huggingface.co/THUdyh/Ola-Image), [Ola-Video](https://huggingface.co/THUdyh/Ola-Video)
- **Key Features**:
  - Progressive modality alignment training strategy
  - Supports text, image, video, and audio modalities
  - Real-time streaming decoding for texts and speeches
  - 7B parameter model size
- **Inference**: Use `inference/infer.py` with multiple input types
- **Paper**: [arXiv:2502.04328](https://arxiv.org/abs/2502.04328)

### [video-SALMONN 2](https://github.com/bytedance/video-SALMONN-2)

**video-SALMONN 2** is a powerful audio-visual large language model that generates high-quality audio-visual video captions, developed by Tsinghua University and ByteDance.

- **Repository**: https://github.com/bytedance/video-SALMONN-2
- **Key Features**:
  - Captioning-enhanced audio-visual understanding
  - Supports video and audio inputs
  - Generates detailed video captions
- **Paper**: [arXiv:2506.15220](https://arxiv.org/abs/2506.15220)

### [MiniCPM-o-2.6](https://huggingface.co/openbmb/MiniCPM-o-2_6)

**MiniCPM-o-2.6** is a streaming omnimodal model from OpenBMB that supports multiple modalities.

- **HuggingFace**: https://huggingface.co/openbmb/MiniCPM-o-2_6
- **Key Features**:
  - Compact model size (8B parameters)
  - Supports text, image, video, and audio
  - Efficient inference

### [VideoLLaMA3](https://github.com/DAMO-NLP-SG/VideoLLaMA3)

**VideoLLaMA3** is a frontier multimodal foundation model for image and video understanding, developed by Alibaba DAMO Academy. It represents the latest iteration of the VideoLLaMA series with improved performance.

- **Repository**: https://github.com/DAMO-NLP-SG/VideoLLaMA3
- **HuggingFace**: [VideoLLaMA3-7B](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA3-7B)
- **Key Features**:
  - Multimodal foundation model for image and video understanding
  - Supports both image and video inputs
  - Improved spatial-temporal modeling
  - Multi-stage training strategy
- **Inference**: Use `inference/` scripts for evaluation
- **Paper**: [arXiv:2501.13106](https://arxiv.org/abs/2501.13106)

### [VideoLLaMA2](https://github.com/DAMO-NLP-SG/VideoLLaMA2)

**VideoLLaMA2** is a video understanding model that extends LLaMA for video-language tasks, developed by Alibaba DAMO Academy.

- **Repository**: https://github.com/DAMO-NLP-SG/VideoLLaMA2
- **Key Features**:
  - Video-language understanding
  - Multi-stage training strategy
  - Supports video-text tasks
- **Paper**: [arXiv:2406.07476](https://arxiv.org/abs/2406.07476)

### [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)

**LLaVA-NeXT** (also known as LLaVA-1.6) is the next-generation Large Language and Vision Assistant that improves upon the original LLaVA model with enhanced vision-language capabilities and better multimodal understanding.

- **Repository**: https://github.com/LLaVA-VL/LLaVA-NeXT
- **HuggingFace**: [llava-v1.6-vicuna-7b](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b), [llava-v1.6-vicuna-13b](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-13b)
- **Key Features**:
  - Enhanced vision-language understanding
  - Support for high-resolution images
  - Improved instruction following
  - Multiple model sizes (7B, 13B variants)
- **Inference**: Use `llava/eval/` scripts for evaluation
- **Paper**: [arXiv:2310.03744](https://arxiv.org/abs/2310.03744) (LLaVA-1.6)
### [AVicuna](https://arxiv.org/abs/2403.16276)

**AVicuna** is an audio-visual large language model framework that enables fine-grained temporal understanding in untrimmed videos. It empowers LLMs to align time, audio-visual events, and language by training on large-scale pseudo-untrimmed videos with precise temporal annotations.

- **Repository**: https://github.com/yunlong10/AVicuna
- **Key Features**:
  - Pseudo-untrimmed video generation from trimmed clips
  - Audio-Visual Token Interleaver for temporal synchronization
  - Progressive multi-stage modality and time-event alignment training
  - Supports joint audio, video, and text reasoning
- **Paper**: [[arXiv:2502.04328](https://arxiv.org/abs/2502.04328)]

### Integration Notes

To evaluate these models on FutureOmni:

1. **Check Model Compatibility**: Ensure the model supports video + audio inputs (FutureOmni requires both modalities)
2. **Adapt Input Format**: Convert FutureOmni data format to the model's expected input format
3. **Feature Extraction**: Some models may require pre-extracted features similar to the Qwen models
4. **API Integration**: For API-based models, consider using the `close_source/inference.py` pattern

For models that don't natively support the exact FutureOmni format, you may need to:
- Pre-process videos/frames to match model requirements
- Adapt prompts to match model instruction format
- Post-process outputs to extract answers in the expected format (single letter: A-F)

## Citation

If you use these scripts, please cite the FutureOmni dataset and respective model papers.

