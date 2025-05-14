import torch
import argparse
from fastapi import FastAPI, Response, HTTPException, status
from fastapi.responses import FileResponse
import uvicorn
import os
import asyncio
import logging
import sys
import torch.multiprocessing as mp
from typing import List
from models import BatchPromptRequest, JobStatus, ServerStatus
from worker import JobQueueWorker
from utils import setup_distributed, load_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("server.log"),
    logging.StreamHandler(sys.stdout)
])
logger = logging.getLogger(__name__)

app = FastAPI()
worker_instance: JobQueueWorker = None
server_state = "starting"

@app.on_event("startup")
async def startup_event():
    """Start the worker queue processing on startup (only rank 0)."""
    global worker_instance, server_state
    if worker_instance.rank == 0:
        asyncio.create_task(worker_instance.process_queue())
        logger.info("Rank 0: Worker queue processing started.")
    server_state = "ready"
    logger.info(f"Rank {worker_instance.rank}: Server startup complete, state: {server_state}")

@app.post("/prompt")
async def submit_prompts(request: BatchPromptRequest) -> List[str]:
    """Submit a batch of prompts and return job IDs."""
    global worker_instance, server_state
    if worker_instance.rank != 0:
        return []  # Only rank 0 handles requests

    if server_state == "starting":
        raise HTTPException(status_code=503, detail="Server is still starting, please wait.")

    job_ids = []
    for prompt_req in request.prompts:
        job_id = await worker_instance.add_job(prompt_req)
        job_ids.append(job_id)
    server_state = worker_instance.get_server_status()
    return job_ids

@app.get("/status")
async def get_status() -> ServerStatus:
    """Get server status and list of all jobs."""
    global worker_instance, server_state
    if worker_instance.rank != 0:
        return ServerStatus(server="ready", jobs=[])
    server_state = worker_instance.get_server_status()
    return ServerStatus(server=server_state, jobs=worker_instance.get_all_jobs())

@app.get("/status/{job_id}")
async def get_job_status(job_id: str) -> JobStatus:
    """Get status of a specific job."""
    global worker_instance
    if worker_instance.rank != 0:
        return JobStatus(id=job_id, prompt="N/A", status="ignored")
    return worker_instance.get_job_status(job_id)

@app.get("/video/{job_id}")
async def get_video(job_id: str) -> Response:
    """Retrieve the generated video for a job if available."""
    global worker_instance
    if worker_instance.rank != 0:
        raise HTTPException(status_code=404, detail="Videos only available on rank 0.")

    job_status = worker_instance.get_job_status(job_id)
    if job_status.status != "done" or not job_status.output_path or not os.path.exists(job_status.output_path):
        raise HTTPException(status_code=404, detail="Video not found or job not completed.")

    return FileResponse(job_status.output_path, media_type="video/mp4", filename=f"video_{job_id}.mp4")

def run_server(rank, world_size, ckpt_dir, port):
    """Run the FastAPI server for a given rank."""
    global worker_instance
    setup_distributed(rank, world_size)
    pipeline = load_model(rank, ckpt_dir, world_size)
    worker_instance = JobQueueWorker(pipeline, rank)
    if rank == 0:
        logger.info(f"Rank 0: Starting FastAPI server on port {port}...")
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    else:
        logger.info(f"Rank {rank}: Worker process initialized, waiting for rank 0 to handle requests.")
        # Keep non-zero ranks alive to maintain distributed model
        while True:
            torch.cuda.synchronize()
            time.sleep(10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a FastAPI server for Wan2.1-T2V-14B video generation")
    parser.add_argument("--ckpt_dir", type=str, default="~/Wan2.1-T2V-14B", help="Path to model checkpoint")
    parser.add_argument("--port", type=int, default=8000, help="Port for the FastAPI server")
    parser.add_argument("--world_size", type=int, default=4, help="Number of GPUs/processes")
    args = parser.parse_args()

    # Expand home directory if used in ckpt_dir
    args.ckpt_dir = os.path.expanduser(args.ckpt_dir)
    logger.info(f"Using checkpoint directory: {args.ckpt_dir}")

    mp.spawn(
        run_server,
        args=(args.world_size, args.ckpt_dir, args.port),
        nprocs=args.world_size,
        join=True
    )