import tempfile
import unittest
from pathlib import Path

from src.retrieval.windows_cuda import find_cuda_dll_directories


class WindowsCudaPathTests(unittest.TestCase):
    def test_finds_existing_cuda_and_cudnn_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            prefix = Path(temporary_directory)
            cuda_bin = prefix / "Lib" / "site-packages" / "nvidia" / "cu13" / "bin" / "x86_64"
            cudnn_bin = prefix / "Lib" / "site-packages" / "nvidia" / "cudnn" / "bin"
            cuda_bin.mkdir(parents=True)
            cudnn_bin.mkdir(parents=True)

            result = find_cuda_dll_directories(prefix)

            self.assertEqual(result, [cuda_bin, cudnn_bin])

    def test_ignores_missing_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = find_cuda_dll_directories(Path(temporary_directory))

            self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()

