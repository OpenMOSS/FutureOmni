#!/usr/bin/env python3
"""
Unified inference script for closed-source models (Claude and Gemini).
Supports video-audio inference with frames and/or video files.
"""

import json
import os
import argparse
import base64
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict, Counter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob

# Claude imports
try:
    from anthropic import Anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False


# ==================== Common Utility Functions ====================

def encode_image(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_audio(audio_path: str) -> str:
    """Encode audio to base64 string."""
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")


def encode_video(video_path: str) -> str:
    """Encode video to base64 string."""
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")


def get_image_media_type(image_path: str) -> str:
    """Get media type from image file extension."""
    ext = Path(image_path).suffix.lower().lstrip('.')
    if ext in ["jpg", "jpeg"]:
        return "image/jpeg"
    elif ext == "png":
        return "image/png"
    elif ext == "gif":
        return "image/gif"
    elif ext == "webp":
        return "image/webp"
    else:
        return "image/jpeg"  # default


def get_video_mime_type(video_path: str) -> str:
    """Get MIME type from video file extension."""
    ext = Path(video_path).suffix.lower().lstrip('.')
    if ext == "mp4":
        return "video/mp4"
    elif ext == "avi":
        return "video/x-msvideo"
    elif ext == "mov":
        return "video/quicktime"
    elif ext == "webm":
        return "video/webm"
    elif ext == "mkv":
        return "video/x-matroska"
    else:
        return "video/mp4"  # default


def construct_prompt(item: Dict, test_prompt: str) -> str:
    """Construct the full prompt from the test prompt and question/options."""
    question = item.get("question", "")
    options = item.get("options", [])
    
    prompt = test_prompt
    prompt += f"\n\nQuestion: {question}\n\n"
    prompt += "Options:\n"
    for option in options:
        prompt += f"{option}\n"
    
    return prompt


# ==================== Claude Functions ====================

def load_frames(frame_dir: str, nid: str) -> List[Dict]:
    """Load all frames from the frame directory for a given nid."""
    nid_dir = os.path.join(frame_dir, str(nid))
    if not os.path.exists(nid_dir):
        return []
    
    # Find all frame images
    frame_pattern = os.path.join(nid_dir, "frame_*.jpg")
    frame_files = sorted(glob.glob(frame_pattern))
    
    frames = []
    for frame_path in frame_files:
        if not os.path.exists(frame_path):
            print(f"Warning: Frame file {frame_path} does not exist, skipping")
            continue
        
        try:
            with open(frame_path, "rb") as img_file:
                image_data = base64.b64encode(img_file.read()).decode("utf-8")
                media_type = get_image_media_type(frame_path)
                frames.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                })
        except Exception as e:
            print(f"Warning: Failed to load frame {frame_path}: {e}")
            continue
    
    return frames


def load_audio_path(audio_dir: str, nid: str) -> Optional[str]:
    """Return audio file path."""
    audio_path = os.path.join(audio_dir, str(nid), f"{nid}_audio.wav")
    if not os.path.exists(audio_path):
        return None
    return audio_path


def call_claude(
    client: Anthropic,
    frames: List[Dict],
    audio_path: Optional[str],
    prompt: str,
    model: str = "claude-sonnet-4-20250514"
) -> Dict:
    """Call Claude API with frames, audio, and prompt."""
    content = []
    
    # Add frames
    content.extend(frames)
    
    # Check if model supports audio
    supports_audio = "sonnet-4" in model.lower() or "opus-4" in model.lower()
    
    # Add audio if available
    use_audio = False
    if audio_path and supports_audio:
        if os.path.exists(audio_path):
            try:
                with open(audio_path, "rb") as audio_file:
                    uploaded_audio = client.files.create(
                        file=audio_file,
                        purpose="analysis"
                    )
                    content.append({
                        "type": "file",
                        "file_id": uploaded_audio.id
                    })
                    use_audio = True
            except Exception as e:
                print(f"Warning: Failed to upload audio: {e}. Continuing without audio.")
        else:
            print(f"Warning: Audio file {audio_path} does not exist, skipping")
    
    # Add text prompt
    content.append({
        "type": "text",
        "text": prompt
    })
    
    try:
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": content
            }]
        )
        
        answer_text = message.content[0].text.strip()
        
        return {
            "status": "success",
            "answer": answer_text,
            "full_response": answer_text,
            "model": model,
            "audio_used": use_audio
        }
    
    except Exception as e:
        error_str = str(e)
        # Retry without audio if audio caused the error
        if ("audio" in error_str.lower() or "does not match" in error_str.lower() or "file" in error_str.lower()) and audio_path is not None and use_audio:
            try:
                content_no_audio = frames.copy()
                content_no_audio.append({
                    "type": "text",
                    "text": prompt
                })
                message = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    messages=[{
                        "role": "user",
                        "content": content_no_audio
                    }]
                )
                answer_text = message.content[0].text.strip()
                return {
                    "status": "success",
                    "answer": answer_text,
                    "full_response": answer_text,
                    "model": model,
                    "audio_skipped": True,
                    "audio_used": False
                }
            except Exception as e2:
                return {
                    "status": "error",
                    "error": f"Original: {error_str}, Retry: {str(e2)}",
                    "answer": None
                }
        else:
            return {
                "status": "error",
                "error": error_str,
                "answer": None
            }


def process_claude_item(
    item: Dict,
    client: Anthropic,
    frame_dir: str,
    audio_dir: str,
    test_prompt: str,
    model: str = "claude-sonnet-4-20250514"
) -> Dict:
    """Process a single JSON item with Claude."""
    nid = item.get("nid", item.get("qid", "unknown"))
    qid = item.get("qid", item.get("nid", "unknown"))
    
    # Load frames
    frames = load_frames(frame_dir, nid)
    if not frames:
        return {
            "status": "error",
            "qid": qid,
            "nid": nid,
            "error": f"No frames found for nid {nid}",
            "answer": None
        }
    
    # Load audio path
    audio_path = load_audio_path(audio_dir, nid)
    
    # Construct prompt
    prompt = construct_prompt(item, test_prompt)
    
    # Call Claude
    result = call_claude(client, frames, audio_path, prompt, model)
    
    # Add metadata
    result["qid"] = qid
    result["nid"] = nid
    result["num_frames"] = len(frames)
    result["has_audio"] = audio_path is not None
    
    return result


# ==================== Gemini Functions ====================

def load_video_inline(video_path: str) -> Optional[Dict]:
    """Load video file and return inline_data format dict."""
    if not os.path.exists(video_path):
        return None
    
    try:
        video_base64 = encode_video(video_path)
        mime_type = get_video_mime_type(video_path)
        
        file_size = os.path.getsize(video_path)
        file_size_mb = file_size / (1024 * 1024)
        if file_size_mb > 20:
            print(f"Warning: Video file is {file_size_mb:.2f} MB. Large files may cause issues.")
        
        return {
            "inline_data": {
                "mime_type": mime_type,
                "data": video_base64
            }
        }
    except Exception as e:
        print(f"Warning: Failed to load video {video_path}: {e}")
        return None


def call_gemini_inline(
    api_key: str,
    base_url: str,
    video: Optional[Dict],
    prompt: str,
    model: str = "gemini-2.5-flash"
) -> Dict:
    """Call Gemini API with inline video data using base64 inline_data."""
    parts = [{"text": prompt}]
    
    if video:
        parts.append(video)
    else:
        return {
            "status": "error",
            "error": "No video data provided",
            "answer": None
        }
    
    payload = {
        "contents": [{
            "parts": parts
        }]
    }
    
    api_url = f"{base_url}/v1beta/models/{model}:generateContent"
    if api_key:
        api_url += f"?key={api_key}"
    
    try:
        response = requests.post(
            api_url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=300
        )
        
        response.raise_for_status()
        result = response.json()
        
        try:
            answer_text = result["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            return {
                "status": "error",
                "error": f"Failed to parse response: {e}, Response: {result}",
                "answer": None
            }
        
        return {
            "status": "success",
            "answer": answer_text.strip(),
            "full_response": answer_text.strip(),
            "model": model,
            "video_used": True
        }
    
    except requests.exceptions.RequestException as e:
        return {
            "status": "error",
            "error": str(e),
            "answer": None
        }


def process_gemini_item(
    item: Dict,
    api_key: str,
    base_url: str,
    video_base_dir: str,
    test_prompt: str,
    model: str = "gemini-2.5-flash"
) -> Dict:
    """Process a single JSON item with Gemini."""
    nid = item.get("nid", item.get("qid", "unknown"))
    qid = item.get("qid", item.get("nid", "unknown"))
    
    # Load video
    video_path = os.path.join(video_base_dir, f"{nid}.mp4")
    video = load_video_inline(video_path)
    
    if not video:
        return {
            "status": "error",
            "qid": qid,
            "nid": nid,
            "error": f"No video found at {video_path}",
            "answer": None
        }
    
    # Construct prompt
    prompt = construct_prompt(item, test_prompt)
    
    # Call Gemini
    result = call_gemini_inline(api_key, base_url, video, prompt, model)
    
    # Add metadata
    result["qid"] = qid
    result["nid"] = nid
    result["has_video"] = video is not None
    
    return result


# ==================== Main Function ====================

def main():
    parser = argparse.ArgumentParser(description="Unified inference script for Claude and Gemini models")
    
    # Model selection
    parser.add_argument("--provider", type=str, required=True, choices=["claude", "gemini"],
                       help="Model provider: 'claude' or 'gemini'")
    
    # Data arguments
    parser.add_argument("input_file", type=str, help="Input JSON file")
    parser.add_argument("--frame_dir", type=str, default="./frames",
                       help="Base directory for frames (Claude only)")
    parser.add_argument("--audio_dir", type=str, default="./audio",
                       help="Base directory for audio (Claude only)")
    parser.add_argument("--video_dir", type=str, default="./videos",
                       help="Base directory for videos (Gemini only)")
    
    # API configuration
    parser.add_argument("--api_key", type=str, default=None,
                       help="API key (or set ANTHROPIC_API_KEY/GEMINI_API_KEY env var)")
    parser.add_argument("--base_url", type=str, default=None,
                       help="Custom base URL for API endpoint")
    parser.add_argument("--model", type=str, default=None,
                       help="Model name (provider-specific)")
    
    # Prompt
    parser.add_argument("--test_prompt", type=str, default=None,
                       help="Test prompt (or use default)")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="Output directory to save individual result files")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Optional: Also save aggregated results to this file")
    parser.add_argument("--max_items", type=int, default=None,
                       help="Maximum number of items to process (for testing)")
    
    # Processing options
    parser.add_argument("--max_workers", type=int, default=1,
                       help="Maximum number of concurrent workers (Gemini only, default: 1)")
    
    args = parser.parse_args()
    
    # Default test prompts
    default_test_prompt_claude = """These are the frames of a video and the corresponding audio.

Please answer the following multiple-choice question based on the video and audio content.

Choose the correct option and respond with ONLY the letter (A, B, C, D, E and F) of your choice."""
    
    default_test_prompt_gemini = """This is a video with audio.

Please answer the following multiple-choice question based on the video and audio content.

Choose the correct option and respond with ONLY the letter (A, B, C, D, E and F) of your choice."""
    
    # Set defaults based on provider
    if args.provider == "claude":
        if not CLAUDE_AVAILABLE:
            print("Error: Anthropic package not available. Install with: pip install anthropic")
            return
        
        if args.model is None:
            args.model = "claude-sonnet-4-20250514"
        if args.test_prompt is None:
            args.test_prompt = default_test_prompt_claude
        
        # Get API key
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: Anthropic API key not provided. Set ANTHROPIC_API_KEY env var or use --api_key")
            return
        
        # Initialize client
        if args.base_url:
            client = Anthropic(base_url=args.base_url, api_key=api_key)
        else:
            client = Anthropic(api_key=api_key)
    
    elif args.provider == "gemini":
        if args.model is None:
            args.model = "gemini-2.5-flash"
        if args.test_prompt is None:
            args.test_prompt = default_test_prompt_gemini
        
        # Get API key
        api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("Error: Gemini API key not provided. Set GEMINI_API_KEY env var or use --api_key")
            return
        
        # Get base URL
        base_url = args.base_url or "https://generativelanguage.googleapis.com"
        base_url = base_url.rstrip('/')
        print(f"Using API key with base URL: {base_url}")
    
    # Read input JSON file
    print(f"Reading input file: {args.input_file}")
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input_file}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}")
        return
    
    # Handle different JSON formats
    if isinstance(data, dict):
        items = [data]
    elif isinstance(data, list):
        items = data
    else:
        print("Error: JSON must be a list or dict")
        return
    
    if args.max_items:
        items = items[:args.max_items]
    
    print(f"Found {len(items)} items to process")
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model}")
    if args.provider == "claude":
        print(f"Frame directory: {args.frame_dir}")
        print(f"Audio directory: {args.audio_dir}")
    else:
        print(f"Video directory: {args.video_dir}")
    print("")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check for existing results (for resume)
    processed_indices = set()
    if os.path.exists(args.output_dir):
        existing_files = glob.glob(os.path.join(args.output_dir, "*.json"))
        for file_path in existing_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing_result = json.load(f)
                    if existing_result.get('status') == 'error':
                        continue
                    idx = existing_result.get("_index")
                    if idx is not None:
                        processed_indices.add(idx)
            except Exception:
                continue
    
    if processed_indices:
        original_count = len(items)
        items = [item for item in items if item.get("_index") not in processed_indices]
        print(f"Resuming: Found {len(processed_indices)} existing results, {len(items)} items remaining (skipped {original_count - len(items)})")
    
    # Process items
    results = []
    
    if args.provider == "claude":
        # Sequential processing for Claude
        for item in tqdm(items, desc="Processing", unit="item"):
            result = process_claude_item(
                item,
                client,
                args.frame_dir,
                args.audio_dir,
                args.test_prompt,
                args.model
            )
            
            result["_index"] = item.get("_index")
            results.append(result)
            
            # Print status
            nid = result.get("nid", "unknown")
            idx = result.get("_index", "unknown")
            status = result.get("status", "unknown")
            answer = result.get("answer", "N/A")
            if status == "success":
                tqdm.write(f"[OK] Index: {idx}, NID: {nid}, Answer: {answer}")
            else:
                error = result.get("error", "Unknown error")
                tqdm.write(f"[ERROR] Index: {idx}, NID: {nid}, Error: {error}")
            
            # Save individual result
            idx = result.get("_index")
            if idx is not None:
                try:
                    individual_file = os.path.join(args.output_dir, f"{idx}.json")
                    with open(individual_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    tqdm.write(f"Warning: Failed to save result for index {idx}: {e}")
    
    elif args.provider == "gemini":
        # Concurrent processing for Gemini
        def process_item_with_save(item):
            result = process_gemini_item(
                item,
                api_key,
                base_url,
                args.video_dir,
                args.test_prompt,
                args.model
            )
            result["_index"] = item.get("_index")
            
            # Print status
            nid = result.get("nid", "unknown")
            idx = result.get("_index", "unknown")
            status = result.get("status", "unknown")
            answer = result.get("answer", "N/A")
            if status == "success":
                tqdm.write(f"[OK] Index: {idx}, NID: {nid}, Answer: {answer}")
            else:
                error = result.get("error", "Unknown error")
                tqdm.write(f"[ERROR] Index: {idx}, NID: {nid}, Error: {error}")
            
            # Save individual result
            idx = result.get("_index")
            if idx is not None:
                try:
                    individual_file = os.path.join(args.output_dir, f"{idx}.json")
                    with open(individual_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    tqdm.write(f"Warning: Failed to save result for index {idx}: {e}")
            
            return result
        
        print(f"Processing with {args.max_workers} concurrent workers...")
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_item = {executor.submit(process_item_with_save, item): item for item in items}
            
            with tqdm(total=len(items), desc="Processing", unit="item") as pbar:
                for future in as_completed(future_to_item):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        item = future_to_item[future]
                        idx = item.get("_index", "unknown")
                        nid = item.get("nid", item.get("qid", "unknown"))
                        error_result = {
                            "status": "error",
                            "error": f"Exception during processing: {str(e)}",
                            "answer": None,
                            "qid": item.get("qid", item.get("nid", "unknown")),
                            "nid": nid,
                            "_index": idx
                        }
                        results.append(error_result)
                        tqdm.write(f"[EXCEPTION] Index: {idx}, NID: {nid}, Error: {str(e)}")
                    finally:
                        pbar.update(1)
    
    # Print summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    successful = sum(1 for r in results if r.get("status") == "success")
    failed = len(results) - successful
    print(f"Total items: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    # Optionally save aggregated results
    if args.output_file:
        try:
            output_data = {
                "summary": {
                    "total": len(results),
                    "successful": successful,
                    "failed": failed
                },
                "results": results
            }
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\nAggregated results saved to: {args.output_file}")
        except Exception as e:
            print(f"Warning: Failed to save aggregated results: {e}")
    
    print(f"\nIndividual results saved to: {args.output_dir}")
    
    # Show answer distribution
    if successful > 0:
        print("\nAnswer distribution:")
        answers = {}
        for r in results:
            if r.get("status") == "success":
                answer = r.get("answer", "").strip()
                letter = answer[0] if answer and answer[0].isalpha() else "Unknown"
                answers[letter] = answers.get(letter, 0) + 1
        for letter, count in sorted(answers.items()):
            print(f"  {letter}: {count}")


if __name__ == "__main__":
    main()

