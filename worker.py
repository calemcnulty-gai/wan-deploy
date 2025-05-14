import asyncio
import logging
from typing import Dict, List
import uuid
from models import JobStatus, PromptRequest
from diffsynth.pipelines.wan_video import WanVideoPipeline
import os

logger = logging.getLogger(__name__)

class JobQueueWorker:
    def __init__(self, pipeline: WanVideoPipeline, rank: int, output_dir: str = "generated_videos"):
        self.pipeline = pipeline
        self.rank = rank
        self.queue = asyncio.Queue()
        self.jobs: Dict[str, JobStatus] = {}
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.is_processing = False

    async def add_job(self, request: PromptRequest) -> str:
        """Add a job to the queue and return its ID."""
        job_id = str(uuid.uuid4())
        job_status = JobStatus(
            id=job_id,
            prompt=request.prompt,
            status="queued"
        )
        self.jobs[job_id] = job_status
        await self.queue.put((job_id, request))
        logger.info(f"Rank {self.rank}: Job {job_id} queued for prompt: {request.prompt}")
        return job_id

    async def process_queue(self):
        """Process jobs from the queue indefinitely (only on rank 0)."""
        if self.rank != 0:
            logger.info(f"Rank {self.rank}: Not processing queue, only rank 0 processes jobs.")
            return

        while True:
            if self.queue.empty():
                if self.is_processing:
                    self.is_processing = False
                    logger.info("Rank 0: Queue is empty, server status back to ready.")
                await asyncio.sleep(1)
                continue

            self.is_processing = True
            job_id, request = await self.queue.get()
            self.jobs[job_id].status = "running"
            logger.info(f"Rank 0: Processing job {job_id} with prompt: {request.prompt}")

            try:
                video = self.pipeline(
                    prompt=request.prompt,
                    width=request.width,
                    height=request.height,
                    num_frames=request.num_frames,
                    fps=request.fps,
                    num_inference_steps=request.num_inference_steps,
                    sample_guide_scale=request.sample_guide_scale,
                    sample_shift=request.sample_shift,
                    use_prompt_extend=request.use_prompt_extend,
                    prompt_extend_method=request.prompt_extend_method,
                )
                output_path = os.path.join(self.output_dir, f"video_{job_id}.mp4")
                video.save(output_path)
                self.jobs[job_id].status = "done"
                self.jobs[job_id].output_path = output_path
                logger.info(f"Rank 0: Job {job_id} completed, video saved to {output_path}")
            except Exception as e:
                self.jobs[job_id].status = "failed"
                self.jobs[job_id].error_message = str(e)
                logger.error(f"Rank 0: Job {job_id} failed with error: {str(e)}")

            self.queue.task_done()

    def get_job_status(self, job_id: str) -> JobStatus:
        """Get the status of a specific job."""
        return self.jobs.get(job_id, JobStatus(id=job_id, prompt="Unknown", status="not_found"))

    def get_all_jobs(self) -> List[JobStatus]:
        """Get status of all jobs."""
        return list(self.jobs.values())

    def get_server_status(self) -> str:
        """Get server status based on current activity."""
        if self.rank != 0:
            return "ready"  # Non-zero ranks are always ready as they don't process
        if self.is_processing:
            return "working"
        return "ready"