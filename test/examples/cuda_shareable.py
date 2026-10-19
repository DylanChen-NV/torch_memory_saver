import os

import torch
from cuda.bindings import driver

from torch_memory_saver import torch_memory_saver


def _handle_type(tensor):
    status, handle = driver.cuMemRetainAllocationHandle(tensor.data_ptr())
    assert status == driver.CUresult.CUDA_SUCCESS, status
    try:
        status, prop = driver.cuMemGetAllocationPropertiesFromHandle(handle)
        assert status == driver.CUresult.CUDA_SUCCESS, status
        return int(prop.requestedHandleTypes)
    finally:
        (status,) = driver.cuMemRelease(handle)
        assert status == driver.CUresult.CUDA_SUCCESS, status


def run(hook_mode):
    torch_memory_saver.hook_mode = hook_mode
    enabled = os.environ["TMS_CUMEM_ENABLE_SHAREABLE_HANDLE"] == "1"
    with torch_memory_saver.region(tag="shareable", enable_cpu_backup=True):
        tensor = torch.full((1 << 20,), 165, dtype=torch.uint8, device="cuda")
    address = tensor.data_ptr()
    expected_types = (
        {
            int(driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
            int(driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC),
        }
        if enabled else {0}
    )
    assert _handle_type(tensor) in expected_types
    assert torch.all(tensor == 165).item()
    torch_memory_saver.pause(tag="shareable")
    torch_memory_saver.resume(tag="shareable")
    assert tensor.data_ptr() == address
    assert _handle_type(tensor) in expected_types
    assert torch.all(tensor == 165).item()
