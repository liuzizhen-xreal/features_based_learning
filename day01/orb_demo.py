"""Day 1：读图 → 检测与描述 → 匹配 → 保存可视化。

无参数时生成一对合成图，便于先验证环境；真实图片用 --image1/--image2。
今天只做描述子匹配，不包含 RANSAC，连线不保证是正确对应。
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def synthetic_pair():
    # 固定种子，使重复运行产生相同图像。
    rng = np.random.default_rng(42)
    image = np.full((480, 640), 235, dtype=np.uint8)
    for _ in range(150):
        x, y = rng.integers([35, 35], [605, 445])
        radius = int(rng.integers(3, 14))
        color = int(rng.integers(20, 180))
        cv2.circle(image, (int(x), int(y)), radius, color, -1)
    cv2.putText(image, 'DAY 1: ORB', (100, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 3)
    transform = cv2.getRotationMatrix2D((320, 240), 12, 0.9)
    other = cv2.warpAffine(image, transform, (640, 480), borderValue=235)
    return image, other


def read_gray(path):
    # imdecode 支持含中文的文件路径。
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f'无法读取图片：{path}')
    return image


def save_image(path, image):
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise RuntimeError(f'编码失败：{path}')
    encoded.tofile(str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image1', type=Path)
    parser.add_argument('--image2', type=Path)
    parser.add_argument('--nfeatures', type=int, default=1000)
    parser.add_argument('--draw', type=int, default=50)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'outputs')
    args = parser.parse_args()
    if bool(args.image1) != bool(args.image2):
        parser.error('--image1 和 --image2 需要同时提供')
    if args.nfeatures <= 0 or args.draw <= 0:
        parser.error('--nfeatures 和 --draw 必须为正数')
    image1, image2 = ((read_gray(args.image1), read_gray(args.image2))
                      if args.image1 else synthetic_pair())
    args.output.mkdir(parents=True, exist_ok=True)

    # 1. 灰度图像是 H×W 的 uint8 数组，像素范围为 0～255。
    print('image1:', image1.shape, image1.dtype)
    print('image2:', image2.shape, image2.dtype)

    # 2. 每个关键点有位置等属性；每个描述子描述对应点周围的外观。
    # 默认 WTA_K=2，ORB 每个描述子占 32 字节，即 256 位。
    orb = cv2.ORB_create(nfeatures=args.nfeatures)
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)
    print('关键点数量:', len(kp1), len(kp2))
    print('des1:', None if des1 is None else (des1.shape, str(des1.dtype)))
    print('des2:', None if des2 is None else (des2.shape, str(des2.dtype)))

    # 3. 双向最近邻：两点必须互为最近邻，才能保留。
    # 汉明距离统计二进制位的不同数量，不是像素坐标之间的距离。
    matches = []
    if des1 is not None and des2 is not None:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(matcher.match(des1, des2), key=lambda m: m.distance)
    if matches:
        m = matches[0]
        print('首条匹配: queryIdx/trainIdx/distance =', m.queryIdx, m.trainIdx, m.distance)
        print('对应坐标:', kp1[m.queryIdx].pt, kp2[m.trainIdx].pt)
    else:
        print('没有可用匹配，请检查图像纹理与重叠区域。')

    # 4. 只画距离较小的前 N 条，避免图像过于杂乱；不是准确率评测。
    cv2.setRNGSeed(42)
    drawing = cv2.drawMatches(image1, kp1, image2, kp2, matches[:args.draw], None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    save_image(args.output / 'image1.png', image1)
    save_image(args.output / 'image2.png', image2)
    save_image(args.output / 'keypoints1.png', cv2.drawKeypoints(image1, kp1, None))
    save_image(args.output / 'matches.png', drawing)
    summary = {'input': 'real' if args.image1 else 'synthetic',
               'nfeatures_requested': args.nfeatures,
               'keypoints1': len(kp1), 'keypoints2': len(kp2),
               'matches': len(matches), 'drawn_matches': min(len(matches), args.draw),
               'note': '未做几何验证，匹配数量不代表准确率'}
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('结果目录:', args.output.resolve())


if __name__ == '__main__':
    main()
