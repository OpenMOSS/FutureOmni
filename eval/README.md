# FutureOmni Evaluation Inference Scripts

This directory contains two inference scripts for evaluating multimodal models (video + audio) on the FutureOmni dataset:

1. **`infer_ddp.py`**: Distributed Data Parallel (DDP) inference using PyTorch
2. **`infer_vllm.py`**: High-performance inference using vLLM

Both scripts support multiple Qwen model variants and can process video-audio multimodal inputs for question-answering tasks.

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
- **Qwen2.5-Omni**: `Qwen2.5-Omni-7B`, `Qwen2.5-Omni-32B`
- **Qwen3-Omni**: `Qwen3-Omni-MoE-A14.5B`, etc.

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

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'qwen_omni_utils'`

**Solution**: Ensure utility modules are in Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/SILVR/FutureOmni"
```

#### 2. DDP Initialization Failed

**Problem**: `NCCL error` or `init_process_group failed`

**Solution**:
- Ensure NCCL is properly installed
- Check network connectivity between nodes
- Verify `CUDA_VISIBLE_DEVICES` is not set when using DDP

#### 3. Out of Memory (OOM)

**Solution**:
- Reduce `batch_size`
- Reduce `max_frames` (fewer frames per video)
- Use gradient checkpointing (if training)
- Process in smaller chunks using `--sid` and multiple runs

#### 4. Video Processing Errors

**Problem**: `Cannot open video` or `Invalid video format`

**Solution**:
- Verify video file paths are correct
- Check video codec compatibility (H.264, H.265 recommended)
- Ensure OpenCV can read the videos

#### 5. Feature Loading Errors

**Problem**: `FileNotFoundError` for `.pt` files

**Solution**:
- Pre-extract features using feature extraction scripts
- Or remove `--feature_dir` to process videos on-the-fly

#### 6. vLLM Compatibility

**Problem**: `vLLM doesn't support Qwen Omni`

**Solution**:
- Ensure you have a custom vLLM build with Qwen Omni support
- Check `vllm/model_executor/models/` for `qwen2_5_omni_thinker.py`
- May need to build vLLM from source with custom model support

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

### Workflow 1: Research Evaluation (DDP)

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

### Workflow 2: Large-Scale Evaluation (vLLM)

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

## License

[Add license information]

## Citation

If you use these scripts, please cite the FutureOmni dataset and respective model papers.

