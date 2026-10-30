"""分别提取两张图片的 SuperPoint 特征，绘制全部关键点，不执行匹配。"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from lightglue import SuperPoint
from lightglue.utils import load_image, rbd


@torch.inference_mode()
def main():
    base_dir = Path(__file__).resolve().parent
    dataset_dir = base_dir.parent / "dataset"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image0", type=Path, default=dataset_dir / "20261023-104938.jpg")
    parser.add_argument("--image1", type=Path, default=dataset_dir / "20261023-105003.jpg")
    parser.add_argument("--max-keypoints", type=int, default=1000)
    parser.add_argument("--resize", type=int, default=1024,
                        help="提取时将长边缩放到此尺寸；0 表示使用原始分辨率")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=base_dir / "outputs" / "extract_pair")
    args = parser.parse_args()
    if args.max_keypoints <= 0 or args.resize < 0:
        parser.error("max-keypoints 必须大于 0，resize 不能小于 0")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}", flush=True)
    # 默认与匹配脚本使用相同的提取设置，方便比较。
    extractor = SuperPoint(max_num_keypoints=args.max_keypoints).eval().to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for index, image_path in enumerate((args.image0, args.image1)):
        original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if original is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        image = load_image(image_path).to(device)
        features = rbd(extractor.extract(image, resize=args.resize or None))
        arrays = {
            name: features[name].cpu().numpy()
            for name in ("keypoints", "keypoint_scores", "descriptors", "image_size")
        }

        # extract 已将坐标映射回原图；画出全部点，不按得分截取前 100 个。
        preview = original.copy()
        for x, y in arrays["keypoints"]:
            cv2.circle(preview, (int(round(x)), int(round(y))), 5,
                       (0, 255, 0), 1, cv2.LINE_AA)

        prefix = f"image{index}_{image_path.stem}"
        preview_path = args.output_dir / f"{prefix}_keypoints.png"
        if not cv2.imwrite(str(preview_path), preview):
            raise OSError(f"图片保存失败: {preview_path}")
        np.savez_compressed(args.output_dir / f"{prefix}_features.npz", **arrays)
        print(f"图片 {index}: {image_path.name}，原图 {original.shape[1]}×{original.shape[0]}，"
              f"提取并绘制 {len(arrays['keypoints'])} 个关键点")
        print(f"可视化: {preview_path}", flush=True)


if __name__ == "__main__":
    main()
