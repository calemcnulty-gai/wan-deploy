from pydantic import BaseModel
from typing import List, Optional

class PromptRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 832
    height: int = 480
    num_frames: int = 24
    fps: int = 24
    num_inference_steps: int = 50
    sample_guide_scale: float = 5.0
    sample_shift: float = 10.0
    use_prompt_extend: bool = True
    prompt_extend_method: str = "local_qwen"

class BatchPromptRequest(BaseModel):
    prompts: List[PromptRequest]

class JobStatus(BaseModel):
    id: str
    prompt: str
    status: str  # queued, running, done, failed
    output_path: Optional[str] = None
    error_message: Optional[str] = None

class ServerStatus(BaseModel):
    server: str  # starting, ready, working
    jobs: List[JobStatus]