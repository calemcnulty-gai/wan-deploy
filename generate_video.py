import torch
import argparse
import os
import glob
from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.models import ModelManager
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import CPUOffload

def parse_args():
    parser = argparse.ArgumentParser(description="Generate photorealistic video with Wan2.1-T2V-14B")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation")
    parser.add_argument("--ckpt_dir", type=str, default="./models/Wan2.1-T2V-14B", help="Path to model checkpoint")
    parser.add_argument("--width", type=int, default=1280, help="Video width")
    parser.add_argument("--height", type=int, default=720, help="Video height")
    parser.add_argument("--num_frames", type=int, default=48, help="Number of frames")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Diffusion steps")
    parser.add_argument("--sample_guide_scale", type=float, default=5.0, help="CFG scale")
    parser.add_argument("--sample_shift", type=float, default=10.0, help="Sample shift for quality")
    parser.add_argument("--use_prompt_extend", action="store_true", help="Enable prompt extension")
    parser.add_argument("--prompt_extend_method", type=str, default="local_qwen", help="Prompt extension method")
    parser.add_argument("--offload_model", action="store_true", help="Offload model to CPU")
    parser.add_argument("--t5_cpu", action="store_true", help="Run T5 on CPU")
    return parser.parse_args()

def print_memory_usage(local_rank):
    if local_rank == 0:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            print(f"Debug: GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB, Total: {total:.2f} GB")
        else:
            print("Debug: GPU not available for memory stats.")

def main():
    args = parse_args()
    torch.cuda.empty_cache()

    # Initialize distributed training
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)

    # Resolve absolute path for ckpt_dir to avoid path issues
    ckpt_dir = os.path.abspath(args.ckpt_dir)
    if not os.path.exists(ckpt_dir):
        if "Wan2.1-T2V-14B" in ckpt_dir:
            potential_home_path = os.path.expanduser("~/Wan2.1-T2V-14B")
            if os.path.exists(potential_home_path):
                ckpt_dir = potential_home_path
                if local_rank == 0:
                    print(f"Debug: Adjusted checkpoint directory to {ckpt_dir} as original path {args.ckpt_dir} not found.")

    # Debug: Print checkpoint directory info
    if local_rank == 0:
        print(f"Debug: Using checkpoint directory (absolute path): {ckpt_dir}")
        if not os.path.exists(ckpt_dir):
            print(f"Error: Checkpoint directory {ckpt_dir} does not exist!")
            torch.distributed.destroy_process_group()
            exit(1)

    # Print initial memory usage
    print_memory_usage(local_rank)

    # Initialize ModelManager directly on GPU to avoid CPU tensors
    model_manager = ModelManager(device="cuda", torch_dtype=torch.float16)

    # Load models from checkpoint directory
    if local_rank == 0:
        print("Debug: Loading models from checkpoint directory...")
    diffusion_files = sorted(glob.glob(os.path.join(ckpt_dir, "diffusion_pytorch_model-*.safetensors")))
    t5_file = os.path.join(ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth")
    vae_file = os.path.join(ckpt_dir, "Wan2.1_VAE.pth")

    if not diffusion_files:
        if local_rank == 0:
            print(f"Error: No diffusion model files found in {ckpt_dir} with pattern 'diffusion_pytorch_model-*.safetensors'")
            print(f"Debug: Directory contents: {os.listdir(ckpt_dir) if os.path.exists(ckpt_dir) else 'Directory not found'}")
        torch.distributed.destroy_process_group()
        exit(1)
    if not os.path.exists(t5_file):
        if local_rank == 0:
            print(f"Warning: T5 model file not found at {t5_file}, proceeding without it if optional.")
        t5_file = None
    if not os.path.exists(vae_file):
        if local_rank == 0:
            print(f"Warning: VAE model file not found at {vae_file}, proceeding without it if optional.")
        vae_file = None

    model_paths = []
    if diffusion_files:
        model_paths.append(diffusion_files)
    if t5_file:
        model_paths.append(t5_file)
    if vae_file:
        model_paths.append(vae_file)

    if local_rank == 0:
        print(f"Debug: Loading models with paths: {model_paths}")

    try:
        if diffusion_files:
            if local_rank == 0:
                print("Debug: Loading diffusion models...")
            model_manager.load_models([diffusion_files], torch_dtype=torch.float16)
            print_memory_usage(local_rank)
        if t5_file:
            if local_rank == 0:
                print("Debug: Loading T5 model...")
            model_manager.load_models([t5_file], torch_dtype=torch.float16)
            print_memory_usage(local_rank)
        if vae_file:
            if local_rank == 0:
                print("Debug: Loading VAE model...")
            model_manager.load_models([vae_file], torch_dtype=torch.float16)
            print_memory_usage(local_rank)
    except Exception as e:
        if local_rank == 0:
            print(f"Error loading models: {e}")
        torch.distributed.destroy_process_group()
        exit(1)

    # Initialize pipeline using from_model_manager
    if local_rank == 0:
        print("Debug: Initializing pipeline...")
    pipeline = WanVideoPipeline.from_model_manager(
        model_manager=model_manager,
        torch_dtype=torch.float16,
        device="cuda",
        use_usp=False
    )
    print_memory_usage(local_rank)

    # Skip VRAM management and offloading since we want everything on GPU
    if local_rank == 0:
        print("Debug: Skipping CPU offloading, keeping everything on GPU.")
    print_memory_usage(local_rank)

    # Wrap models with FSDP without CPU offloading, specify device_id for GPU initialization
    if hasattr(pipeline, 'dit'):
        if local_rank == 0:
            print("Debug: Wrapping dit with FSDP without CPU offloading...")
        pipeline.dit = FSDP(pipeline.dit, device_id=torch.cuda.current_device())
    if hasattr(pipeline, 'text_encoder'):
        if local_rank == 0:
            print("Debug: Wrapping text_encoder with FSDP without CPU offloading...")
        pipeline.text_encoder = FSDP(pipeline.text_encoder, device_id=torch.cuda.current_device())
    print_memory_usage(local_rank)

    # Ensure all components are on GPU (should be redundant but added for safety)
    if local_rank == 0:
        print("Debug: Ensuring pipeline components are on GPU...")
    if hasattr(pipeline, 'dit'):
        pipeline.dit = pipeline.dit.to("cuda")
        print_memory_usage(local_rank)
    if hasattr(pipeline, 'text_encoder'):
        pipeline.text_encoder = pipeline.text_encoder.to("cuda")
        print_memory_usage(local_rank)
    if hasattr(pipeline, 'vae'):
        pipeline.vae = pipeline.vae.to("cuda")
        print_memory_usage(local_rank)

    # Generate video (only on rank 0 to avoid duplicate outputs)
    if local_rank == 0:
        try:
            if local_rank == 0:
                print("Debug: Starting video generation...")
            video = pipeline(
                prompt=args.prompt,
                width=args.width,
                height=args.height,
                num_frames=args.num_frames,
                fps=args.fps,  # Ensure fps is passed correctly
                num_inference_steps=args.num_inference_steps,
                sample_guide_scale=args.sample_guide_scale,
                sample_shift=args.sample_shift,
                use_prompt_extend=args.use_prompt_extend,
                prompt_extend_method=args.prompt_extend_method,
            )
            try:
                from diffsynth import save_video
                if local_rank == 0:
                    print("Debug: Saving video...")
                save_video(video, "output_video.mp4", fps=args.fps)
            except ImportError:
                print("Warning: save_video not found, attempting to save manually or skipping.")
                if isinstance(video, list):
                    print("Video is a list of frames, manual saving not implemented yet.")
                else:
                    print("Video format unknown, unable to save.")
        except Exception as e:
            print(f"Error during video generation: {e}")

    # Clean up
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()