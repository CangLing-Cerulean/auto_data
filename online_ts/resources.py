"""Hard worker limits, separate from the model allocation preflight."""
from __future__ import annotations

import os
import subprocess

from .config import GateError


def _available_memory_mb() -> int | None:
    if os.name != "nt":
        try:
            return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024 ** 2)
        except (AttributeError, OSError, ValueError):
            return None
    import ctypes
    from ctypes import wintypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64)]

    value = MemoryStatus()
    value.dwLength = ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        return None
    return int(value.ullAvailPhys / 1024 ** 2)


def local_execution_status(expected_gpu: str = "RTX 5060") -> dict:
    """Return bounded resource facts only; no paths, usernames or environment values."""
    gpu = {"detected": False, "expected_match": False}
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=True, creationflags=flags,
        )
        first = result.stdout.splitlines()[0]
        name, total, free, utilization = [part.strip() for part in first.split(",", 3)]
        gpu = {"detected": True, "name": name[:120], "memory_total_mb": int(total),
               "memory_free_mb": int(free), "utilization_percent": int(utilization),
               "expected_match": not expected_gpu or expected_gpu.casefold() in name.casefold()}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    return {"target": "local", "available": True, "cpu_count": os.cpu_count() or 1,
            "memory_available_mb": _available_memory_mb(), "gpu": gpu,
            "training_backend": "numpy-cpu", "gpu_accelerated": False}


def admit_foundation_workload(config, candidate, estimate: dict) -> dict:
    """Hard admission from verified package size plus current machine facts."""
    status = local_execution_status(config.execution.expected_local_gpu)
    available_ram = status.get("memory_available_mb")
    if available_ram is not None and available_ram < estimate["minimum_ram_mb"]:
        raise GateError("FOUNDATION_RAM_INSUFFICIENT")
    if config.budget.max_memory_mb < estimate["minimum_ram_mb"]:
        raise GateError("FOUNDATION_MEMORY_BUDGET")
    if config.foundation.device == "cuda":
        gpu = status["gpu"]
        if not gpu.get("detected") or not gpu.get("expected_match"):
            raise GateError("GPU_UNAVAILABLE_OR_UNEXPECTED")
        if gpu.get("memory_free_mb", 0) < estimate["minimum_vram_mb"]:
            raise GateError("FOUNDATION_VRAM_INSUFFICIENT")
        status.update(training_backend="chronos2-torch-cuda", gpu_accelerated=True)
    else:
        status.update(training_backend="chronos2-torch-cpu", gpu_accelerated=False)
    status["admission_estimate"] = estimate
    status["model_id"] = candidate.model_id
    status["adaptation_mode"] = candidate.adaptation_mode
    return status


def memory_limit(megabytes: int):
    limit = megabytes * 1024 ** 2
    if os.name != "nt":
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        return None
    import ctypes
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

    class IOCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", IOCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateJobObjectW(None, None)
    if not handle:
        raise GateError("WORKER_MEMORY_LIMIT_UNAVAILABLE")
    info = ExtendedLimits()
    info.BasicLimitInformation.LimitFlags = 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
    info.ProcessMemoryLimit = limit
    if not kernel.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)) or not kernel.AssignProcessToJobObject(handle, kernel.GetCurrentProcess()):
        kernel.CloseHandle(handle)
        raise GateError("WORKER_MEMORY_LIMIT_UNAVAILABLE")
    # Keep the handle alive for the worker's entire lifetime; OS closes it on exit.
    return handle
