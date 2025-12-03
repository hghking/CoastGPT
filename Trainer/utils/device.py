import os
from typing import List, Optional

import torch


ACCELERATOR_MAP = {
    "gpu": "cuda",
    "cuda": "cuda",
    "npu": "npu",
    "cpu": "cpu",
    "mps": "mps",
}


def _get_visible_device(env_keys: List[str]) -> Optional[int]:
    for key in env_keys:
        if key in os.environ:
            raw = os.environ[key]
            ids = [item.strip() for item in raw.split(",") if item.strip()]
            if len(ids) == 1 and ids[0].isdigit():
                return int(ids[0])
    return None


def get_device(
    accelerator: str,
    *,
    is_distribute: bool = False,
    local_rank: Optional[int] = None,
    index: Optional[int] = None,
) -> torch.device:
    device_type = ACCELERATOR_MAP.get(accelerator, accelerator)

    if is_distribute and local_rank is not None:
        return torch.device(device_type, local_rank)

    env_keys: List[str] = []
    if device_type == "cuda":
        env_keys = ["CUDA_VISIBLE_DEVICES", "CUDA_VISABLE_DEVICES"]
    elif device_type == "npu":
        env_keys = ["NPU_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES"]

    device_index = index
    if device_index is None:
        device_index = _get_visible_device(env_keys)

    return torch.device(device_type, device_index) if device_index is not None else torch.device(device_type)


def get_autocast_device_type(accelerator: str) -> str:
    device_type = ACCELERATOR_MAP.get(accelerator, accelerator)
    if device_type in ("cuda", "npu"):
        return device_type
    return "cpu"


def is_npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()
