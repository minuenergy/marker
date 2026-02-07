import os
import subprocess
import torch

from marker.logger import get_logger
from marker.settings import settings

logger = get_logger()


def cuda_empty_cache():
    """Clear CUDA cache to free up GPU memory between pipeline stages."""
    if torch.cuda.is_available() and "cuda" in settings.TORCH_DEVICE_MODEL:
        torch.cuda.empty_cache()


def get_gpu_free_memory_gb() -> float:
    """Get free GPU memory in GB. Returns 0 if not using CUDA."""
    if not torch.cuda.is_available() or "cuda" not in settings.TORCH_DEVICE_MODEL:
        return 0

    try:
        free_mem, _ = torch.cuda.mem_get_info()
        return free_mem / (1024 ** 3)
    except Exception:
        return 0


def get_gpu_total_memory_gb() -> float:
    """Get total GPU memory in GB. Returns 0 if not using CUDA."""
    if not torch.cuda.is_available() or "cuda" not in settings.TORCH_DEVICE_MODEL:
        return 0

    try:
        _, total_mem = torch.cuda.mem_get_info()
        return total_mem / (1024 ** 3)
    except Exception:
        return 0


def scale_batch_size(default_batch_size: int, low_vram_batch_size: int, vram_threshold_gb: float = 14.0) -> int:
    """Scale batch size based on available GPU memory.

    Args:
        default_batch_size: Batch size for GPUs with >= vram_threshold_gb total VRAM.
        low_vram_batch_size: Batch size for GPUs with < vram_threshold_gb total VRAM.
        vram_threshold_gb: VRAM threshold in GB to decide between default and low_vram batch sizes.
            GPUs below this threshold (e.g. 1080 Ti 11GB, 3060 12GB) get reduced batch sizes.

    Returns:
        Appropriate batch size for the current GPU.
    """
    if "cuda" not in settings.TORCH_DEVICE_MODEL:
        return low_vram_batch_size

    total_vram = get_gpu_total_memory_gb()
    if total_vram < vram_threshold_gb:
        return low_vram_batch_size

    return default_batch_size


class GPUManager:
    default_gpu_vram: int = 8

    def __init__(self, device_idx: int):
        self.device_idx = device_idx
        self.original_compute_mode = None
        self.mps_server_process = None

    def __enter__(self):
        if self.using_cuda():
            self.start_mps_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.using_cuda():
            self.cleanup()

    @staticmethod
    def using_cuda():
        return "cuda" in settings.TORCH_DEVICE_MODEL

    def check_cuda_available(self) -> bool:
        if not torch.cuda.is_available():
            return False
        try:
            subprocess.run(["nvidia-smi", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def get_gpu_vram(self):
        if not self.using_cuda():
            return self.default_gpu_vram

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(self.device_idx),
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            vram_mb = int(result.stdout.strip())
            vram_gb = int(vram_mb / 1024)
            return vram_gb

        except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
            return self.default_gpu_vram

    def start_mps_server(self) -> bool:
        if not self.check_cuda_available():
            return False

        try:
            # Set MPS environment with chunk-specific directories
            env = os.environ.copy()
            pipe_dir = f"/tmp/nvidia-mps-{self.device_idx}"
            log_dir = f"/tmp/nvidia-log-{self.device_idx}"
            env["CUDA_MPS_PIPE_DIRECTORY"] = pipe_dir
            env["CUDA_MPS_LOG_DIRECTORY"] = log_dir

            # Create directories
            os.makedirs(pipe_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)

            # Start MPS control daemon
            self.mps_server_process = subprocess.Popen(
                ["nvidia-cuda-mps-control", "-d"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            logger.info(f"Started NVIDIA MPS server for chunk {self.device_idx}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(
                f"Failed to start MPS server for chunk {self.device_idx}: {e}"
            )
            return False

    def stop_mps_server(self) -> None:
        try:
            # Stop MPS server
            env = os.environ.copy()
            env["CUDA_MPS_PIPE_DIRECTORY"] = f"/tmp/nvidia-mps-{self.device_idx}"
            env["CUDA_MPS_LOG_DIRECTORY"] = f"/tmp/nvidia-log-{self.device_idx}"

            subprocess.run(
                ["nvidia-cuda-mps-control"],
                input="quit\n",
                text=True,
                env=env,
                timeout=10,
            )

            if self.mps_server_process:
                self.mps_server_process.terminate()
                try:
                    self.mps_server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.mps_server_process.kill()
                self.mps_server_process = None

            logger.info(f"Stopped NVIDIA MPS server for chunk {self.device_idx}")
        except Exception as e:
            logger.warning(
                f"Failed to stop MPS server for chunk {self.device_idx}: {e}"
            )

    def cleanup(self) -> None:
        self.stop_mps_server()
