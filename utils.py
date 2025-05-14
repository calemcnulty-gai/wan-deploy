import torch
import os
import logging
from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.models import ModelManager
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist

logger = logging.getLogger(__name__)

def setup_distributed(rank, world_size):
    """Initialize distributed process group and set device."""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    logger.info(f"Rank {rank}: Distributed process group initialized.")

def load_model(rank, ckpt_dir, world_size):
    """Load the Wan2.1-T2V-14B model with FSDP on GPUs, keeping T5 on CPU to save GPU memory."""
    logger.info(f"Rank {rank}: Loading model from {ckpt_dir}...")
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    logger.info(f"Rank {rank}: Set PYTORCH_CUDA_ALLOC_CONF to expandable_segments:True")
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(rank) / 1024**3
        reserved = torch.cuda.memory_reserved(rank) / 1024**3
        total = torch.cuda.get_device_properties(rank).total_memory / 1024**3
        logger.info(f"Rank {rank}: GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB, Total: {total:.2f} GB")
    
    logger.info(f"Rank {rank}: Loading model on CPU first to avoid GPU OOM...")
    model_manager = ModelManager(device="cpu")
    diffusion_files = [
        os.path.join(ckpt_dir, f"diffusion_pytorch_model-0000{i}-of-00006.safetensors")
        for i in range(1, 7)
    ]
    t5_file = os.path.join(ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth")
    vae_file = os.path.join(ckpt_dir, "Wan2.1_VAE.pth")
    model_manager.load_models(
        [diffusion_files, t5_file, vae_file],
        torch_dtype=torch.float16,
    )
    pipeline = WanVideoPipeline.from_model_manager(
        model_manager, torch_dtype=torch.float16, device="cpu"
    )
    dist.barrier()
    logger.info(f"Rank {rank}: Model loaded on CPU, synchronizing before moving to GPU and FSDP wrap.")
    
    # Move only diffusion model to GPU, keep T5 on CPU
    pipeline.dit = pipeline.dit.to("cuda")
    pipeline.dit = FSDP(pipeline.dit, device_id=torch.cuda.current_device())
    # Keep T5 on CPU, wrap with FSDP if needed (though it might not be necessary)
    pipeline.t5 = FSDP(pipeline.t5, device_id=torch.cuda.current_device(), use_orig_params=True)
    # Move other components to GPU if they exist
    if hasattr(pipeline, 'vae'):
        pipeline.vae = pipeline.vae.to("cuda")
    
    dist.barrier()
    logger.info(f"Rank {rank}: Diffusion model moved to GPU with FSDP, T5 kept on CPU.")
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(rank) / 1024**3
        reserved = torch.cuda.memory_reserved(rank) / 1024**3
        total = torch.cuda.get_device_properties(rank).total_memory / 1024**3
        logger.info(f"Rank {rank}: Post-Load GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB, Total: {total:.2f} GB")
    
    return pipeline