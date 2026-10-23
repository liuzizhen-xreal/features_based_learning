"""查看当前 Python 解释器路径和常用库版本。"""

import importlib
import sys


def main():
    print(f"Python 解释器路径: {sys.executable}")
    print(f"Python 版本: {sys.version.split()[0]}")

    for name, module_name in (
        ("NumPy", "numpy"),
        ("OpenCV", "cv2"),
        ("PyTorch", "torch"),
    ):
        try:
            module = importlib.import_module(module_name)
            print(f"{name} 版本: {module.__version__}")
        except ImportError as error:
            print(f"{name} 无法导入: {error}")


if __name__ == "__main__":
    main()
