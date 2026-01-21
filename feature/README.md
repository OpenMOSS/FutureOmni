# Video and Audio Feature Extraction Tool

A multiprocessing-based tool for extracting multimodal features (video and audio) from video files using Qwen2_5OmniProcessor or Qwen3OmniMoeProcessor. This tool is designed for batch processing large video datasets with parallel processing capabilities.

## Overview

This tool processes video files to extract:
- **Video features**: Extracts frames, resizes them while maintaining aspect ratio, and processes them through video processors
- **Audio features**: Extracts and processes audio tracks from videos
- **Multimodal features**: Combines video and audio features for downstream tasks

The tool supports multiple datasets and model types, with special handling for dynamic video segments (FutureOmni dataset).

## Features

- ✅ **Multiprocessing support**: Parallel processing for faster feature extraction
- ✅ **Multiple model types**: Supports `qwen2_5omni` and `qwen3omni` processors
- ✅ **Smart video resizing**: Maintains aspect ratio while ensuring dimensions are divisible by a factor (default: 28)
- ✅ **Flexible frame extraction**: Configurable number of frames per video
- ✅ **Dynamic video segments**: Special handling for time-segmented videos (FutureOmni)
- ✅ **Progress tracking**: Real-time progress monitoring with tqdm
- ✅ **Error handling**: Skips existing files and tracks failed videos
- ✅ **Multiple dataset formats**: Supports general datasets and FutureOmni format

## Requirements

### Dependencies

```
Pillow (PIL)
numpy
librosa
pandas
torch
torchvision
transformers
opencv-python (cv2)
tqdm
```

### Installation

```bash
pip install pillow numpy librosa pandas torch torchvision transformers opencv-python tqdm
```

## Usage

### Basic Usage

#### For General Datasets (e.g., worldsense, omnivideobench)

```bash
python feature_extractor.py \
    --processor_path "Qwen2.5-Omni-7B" \
    --dataset "worldsense" \
    --video_root "/path/to/videos" \
    --save_dir "/path/to/save/features" \
    --duration_dict "/path/to/duration_dict.json" \
    --max_frames 32 \
    --model_type "qwen2_5omni"
```

#### For FutureOmni Dataset

```bash
python feature_extractor.py \
    --processor_path "Qwen2.5-Omni-7B" \
    --dataset "futureomni" \
    --data_path "/path/to/futureomni_data.json" \
    --video_root "/path/to/videos" \
    --save_dir "/path/to/save/features" \
    --train \
    --max_frames 32 \
    --model_type "qwen2_5omni"
```

### Command Line Arguments

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--processor_path` | str | "Qwen2.5-Omni-7B" | No | Path to the processor model (HuggingFace model ID or local path) |
| `--dataset` | str | "omnivideobench" | No | Dataset name (e.g., "worldsense", "futureomni", "omnivideobench") |
| `--data_path` | str | None | Conditional* | Path to dataset JSON file (required for FutureOmni) |
| `--video_root` | str | - | **Yes** | Root directory containing video files |
| `--save_dir` | str | - | **Yes** | Directory to save extracted features |
| `--duration_dict` | str | None | Conditional* | Path to JSON file mapping video names to durations (required for general datasets) |
| `--max_frames` | int | 32 | No | Maximum number of frames to extract per video |
| `--model_type` | str | "qwen2_5omni" | No | Model type: "qwen2_5omni" or "qwen3omni" |
| `--train` | flag | False | No | Use training data format (for FutureOmni) |
| `--start_id` | int | 0 | No | Starting index for processing (for batch processing) |
| `--qid` | int | 0 | No | Specific video ID to process (for debugging) |
| `--debug` | flag | False | No | Debug mode: process single video only |

*Conditional: Required based on dataset type

### Debug Mode

To process a single video for testing:

```bash
python feature_extractor.py \
    --processor_path "Qwen2.5-Omni-7B" \
    --dataset "worldsense" \
    --video_root "/path/to/videos" \
    --save_dir "/path/to/save/features" \
    --duration_dict "/path/to/duration_dict.json" \
    --qid 0 \
    --debug
```

## Input Data Formats

### General Dataset Format

**Duration Dictionary** (`duration_dict.json`):
```json
{
    "video1.mp4": 120.5,
    "video2.mp4": 45.3,
    "video3.mp4": 180.0
}
```

### FutureOmni Dataset Format

**Training Data** (`data.json` with `--train`):
```json
[
    {
        "videos": ["path/to/video.mp4", "00:01:30-00:02:00", "qid_123", 30.0]
    }
]
```

**Test Data** (`data.json` without `--train`):
```json
[
    {
        "id": "qid_123",
        "original_video": "path/to/video.mp4",
        "split_point": 90.0,
        "duration": 30.0
    }
]
```

## Output Structure

Features are saved in the following directory structure:

```
save_dir/
├── {dataset}_{max_frames}/  (or {source}_{max_frames} for dynamic mode)
│   ├── feature/
│   │   ├── {video_name}.pt      # Combined multimodal features
│   │   └── ...
│   ├── video/
│   │   ├── {video_name}.pt      # Processed video tensors
│   │   └── ...
│   └── audio/
│       ├── {video_name}.pt      # Processed audio tensors
│       └── ...
└── failed_videos.txt            # List of failed videos (if any)
```

### Feature File Contents

Each feature file (`.pt`) contains a dictionary with:
- `pixel_values`: Video frame tensors (processed and resized)
- `video_second_per_grid`: Temporal resolution information
- `input_features`: Audio features (if audio is available)
- `feature_attention_mask`: Audio attention mask (if audio is available)
- Additional model-specific inputs

## Configuration Constants

The script uses several configurable constants:

```python
VIDEO_MIN_PIXELS = 128 * 28 * 28   # Minimum video resolution
VIDEO_TRAIN_PIXELS = 128 * 28 * 28 # Training resolution
VIDEO_MAX_PIXELS = 768 * 28 * 28   # Maximum video resolution
IMAGE_FACTOR = 28                   # Dimension divisibility factor
MAX_RATIO = 200                     # Maximum aspect ratio allowed
```

## Key Functions

### `smart_resize(height, width, factor, min_pixels, max_pixels)`
Resizes images while:
1. Ensuring dimensions are divisible by `factor`
2. Maintaining aspect ratio
3. Keeping pixel count within `[min_pixels, max_pixels]` range

### `read_frames(video_path, frame_num, end_sec)`
Extracts frames from video:
- Evenly spaced frames if `end_sec` is None
- Frames from 0 to `end_sec` if specified

### `feature_extract_single(video_info, ...)`
Processes a single video:
- Extracts and processes video frames
- Extracts and processes audio (if available)
- Combines features using the processor
- Saves results to disk

### `batch_feature_extract(video_infos, ...)`
Parallel batch processing:
- Uses multiprocessing Pool
- Progress tracking with queue-based monitoring
- Error handling and result statistics

## Processing Details

### Video Processing
1. Frame extraction using OpenCV
2. Frame normalization (shape: `[frames, channels, height, width]`)
3. Smart resizing with aspect ratio preservation
4. Conversion to tensor format
5. Processing through video processor

### Audio Processing
1. Audio extraction using librosa
2. Resampling to target sampling rate (default: 16kHz)
3. Duration-based truncation (for dynamic segments)
4. Feature extraction using audio feature extractor
5. Padding and attention mask generation

### Dynamic Mode (FutureOmni)
- Extracts specific time segments from videos
- Uses `end_sec` parameter to limit frame extraction range
- Processes audio for the specific segment duration
- Saves features with question ID (qid) as filename

## Performance Tips

1. **Multiprocessing**: The default uses `min(cpu_count() // 2, 8)` workers. Adjust based on available memory.

2. **Memory Management**: For large videos, reduce `max_frames` or process in batches using `--start_id`.

3. **Caching**: Processed video and audio tensors are cached in `video/` and `audio/` subdirectories to avoid recomputation.

4. **Skip Existing**: The tool automatically skips videos that have already been processed.

## Error Handling

- Failed videos are logged to `failed_videos.txt`
- Processing continues even if individual videos fail
- Progress is saved incrementally
- Detailed error messages are printed for debugging

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `max_workers` or `max_frames`
2. **Video Cannot Open**: Check video file paths and formats
3. **Aspect Ratio Error**: Videos with extreme aspect ratios (>200:1) will fail
4. **Audio Processing Error**: Some datasets may not have audio tracks

### Debugging

Use `--debug` and `--qid` flags to test single video processing:
```bash
python feature_extractor.py ... --debug --qid 0
```

## Dependencies on Utilities

This script requires utility functions from `utils.py`:
- `trans_seconds2`: Alternative time format conversion
- `trans2seconds`: Seconds conversion (from `utils.constructor`)

Make sure these utilities are available in your Python path.
