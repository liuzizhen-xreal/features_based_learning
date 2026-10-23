"""读取图片，进行灰度转换、取反和中心裁剪。"""

from pathlib import Path

import cv2
import numpy as np


def main():
    # 根据脚本所在目录定位图片，不受运行命令时的工作目录影响。
    base_dir = Path(__file__).resolve().parent
    image_path = base_dir.parent / "dataset" / "images.jpeg"
    output_dir = base_dir / "outputs"

    # OpenCV 默认读取为 BGR 三通道彩色图像。
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图片，请检查文件路径和内容: {image_path}")

    height, width, channels = image.shape
    print(f"宽: {width}, 高: {height}, 通道数: {channels}")
    print(f"数据类型: {image.dtype}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 转成灰度图，形状由 (高, 宽, 3) 变为 (高, 宽)。
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. 用双重循环逐个像素取反。
    inverted_loop = np.empty_like(gray)
    for row in range(height):
        for col in range(width):
            inverted_loop[row, col] = 255 - gray[row, col]

    print(f"[50,100]处的BGR像素值：{image[50,100]},[50,100]处的灰度像素值：{gray[50,100]}");

    # 3. NumPy 数组运算会对所有像素同时应用取反公式。
    inverted_numpy = 255 - gray

    # 比较每个像素，只有所有像素完全一致才返回 True。
    identical = np.array_equal(inverted_loop, inverted_numpy)
    print(f"循环与 NumPy 取反结果完全一致: {identical}")
    if not identical:
        raise AssertionError("两种取反方法的结果不一致")

    # 4. 裁剪原彩色图中心区域，宽、高各取原图的一半（至少 1 像素）。
    crop_width = max(1, width // 2)
    crop_height = max(1, height // 2)
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    center_crop = image[top:top + crop_height, left:left + crop_width]

    # PNG 为无损格式，可以保留精确的像素值。
    results = {
        "gray.png": gray,
        "inverted_loop.png": inverted_loop,
        "inverted_numpy.png": inverted_numpy,
        "center_crop.png": center_crop,
    }
    for filename, result in results.items():
        output_path = output_dir / filename
        if not cv2.imwrite(str(output_path), result):
            raise OSError(f"图片保存失败: {output_path}")
        print(f"已保存: {output_path}")


if __name__ == "__main__":
    main()
