"""Windows 下 Paddle GPU 动态库路径处理。"""

from __future__ import annotations

import os
import sys
from pathlib import Path


# os.add_dll_directory 返回的对象必须在进程存活期间保持打开。
_DLL_DIRECTORY_HANDLES: list[object] = []


def find_cuda_dll_directories(environment_prefix: Path) -> list[Path]:
    """找出 Paddle GPU wheel 安装的 CUDA 和 cuDNN 动态库目录。"""
    nvidia_packages = environment_prefix / "Lib" / "site-packages" / "nvidia"
    candidates = sorted(nvidia_packages.glob("cu*/bin/x86_64"))
    candidates.append(nvidia_packages / "cudnn" / "bin")
    return [path for path in candidates if path.is_dir()]


def configure_windows_cuda_dll_paths() -> list[Path]:
    """让 Windows 进程能够找到 Paddle wheel 自带的 CUDA 动态库。"""
    if os.name != "nt":
        return []

    dll_directories = find_cuda_dll_directories(Path(sys.prefix))
    if not dll_directories:
        return []

    os.environ["PATH"] = os.pathsep.join(map(str, dll_directories)) + os.pathsep + os.environ["PATH"]
    for directory in dll_directories:
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(directory)))
    return dll_directories

