"""比较同一张图中两个不同关键点的 ORB 描述子汉明距离。"""

from pathlib import Path

import cv2
import numpy as np


def main():
    base_dir = Path(__file__).resolve().parent
    image_path = base_dir.parent / "dataset" / "images.jpg"
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    # WTA_K=2 是默认值，此处显式指定，使用普通汉明距离。
    orb = cv2.ORB_create(nfeatures=1000, WTA_K=2)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) < 2:
        raise ValueError("至少需要两个可用关键点才能比较描述子")

    # 每行对应一个关键点，取第 0 行与第 1 行的描述子。
    des1 = descriptors[0]
    des2 = descriptors[1]
    print(f"WTA_K: {orb.getWTA_K()}")
    print(f"关键点 0 的位置: {keypoints[0].pt}")
    print(f"关键点 1 的位置: {keypoints[1].pt}")
    print(f"des1 形状: {des1.shape}, 数据类型: {des1.dtype}")
    print(f"des2 形状: {des2.shape}, 数据类型: {des2.dtype}")

    # ORB 的每个描述子有 32 个字节，即 256 个二进制位。
    bits1 = np.unpackbits(des1)
    bits2 = np.unpackbits(des2)
    print(f"des1 展开后的形状: {bits1.shape}, 数据类型: {bits1.dtype}")
    print("des1 的二进制位（每组 8 位对应一个字节）：")
    for byte_bits in bits1.reshape(-1, 8):
        print("".join(str(int(bit)) for bit in byte_bits), end=" ")
    print()

    # 比较对应位，统计不同位的数量，而不是不同字节的数量。
    manual_distance = int(np.count_nonzero(bits1 != bits2))
    opencv_distance = cv2.norm(des1, des2, cv2.NORM_HAMMING)
    print(f"自己计算的汉明距离: {manual_distance}")
    print(f"OpenCV 计算的汉明距离: {opencv_distance}")
    identical = manual_distance == opencv_distance
    print(f"两种结果完全一致: {identical}")
    if not identical:
        raise AssertionError("手算结果与 OpenCV 汉明距离不一致")


if __name__ == "__main__":
    main()
