"""用 SuperPoint 提取特征，用 LightGlue 匹配，并保存结果与可视化图片。"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd


def save_image(path, image):
    """压缩 PNG；仍超过 20 MB 时缩小可视化，确保单张图片不超限。"""
    max_bytes = 20_000_000
    preview = image
    while True:
        success, encoded = cv2.imencode(
            ".png", preview, [cv2.IMWRITE_PNG_COMPRESSION, 6],
        )
        if not success:
            raise OSError(f"图片编码失败: {path}")
        if encoded.nbytes <= max_bytes:
            path.write_bytes(encoded.tobytes())
            height, width = preview.shape[:2]
            print(f"图片已保存: {path}，{width}×{height}，{encoded.nbytes / 1_000_000:.2f} MB")
            return
        height, width = preview.shape[:2]
        if width == 1 and height == 1:
            raise OSError(f"图片无法压缩到 20 MB 以内: {path}")
        scale = min(0.9, (max_bytes / encoded.nbytes) ** 0.5 * 0.95)
        preview = cv2.resize(
            preview, (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )


@torch.inference_mode()
def main():
    project_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image0", type=Path,
                        default=project_dir / "dataset" / "20261023-104938.jpg")
    parser.add_argument("--image1", type=Path,
                        default=project_dir / "dataset" / "20261023-105003.jpg")
    parser.add_argument("--max-keypoints", type=int, default=1000)
    parser.add_argument("--resize", type=int, default=1024,
                        help="提取时的长边尺寸，默认 1024；0 表示保留原始分辨率")
    parser.add_argument("--show-top", type=int, default=300,
                        help="可视化得分最高的关键点和匹配数量，0 表示显示全部")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "outputs" / "matching")
    args = parser.parse_args()
    if args.max_keypoints <= 0 or args.show_top < 0 or args.resize < 0:
        parser.error("max-keypoints 必须大于 0，show-top 和 resize 不能小于 0")

    # 保留原图用于画图；LightGlue 返回的关键点坐标对应原图尺寸。
    originals = []
    for path in (args.image0, args.image1):
        original = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if original is None:
            raise FileNotFoundError(f"无法读取图片: {path}")
        originals.append(original)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}", flush=True)
    # 首次运行时，模型会自动下载预训练权重。
    extractor = SuperPoint(max_num_keypoints=args.max_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    image0 = load_image(args.image0).to(device)
    image1 = load_image(args.image1).to(device)
    print(f"输入 image0: {tuple(image0.shape)}，image1: {tuple(image1.shape)}")
    print(f"提取分辨率设置: {'原始分辨率' if args.resize == 0 else f'长边 {args.resize}'}")
    features0 = extractor.extract(image0, resize=args.resize or None)
    features1 = extractor.extract(image1, resize=args.resize or None)
    for index, features in enumerate((features0, features1)):
        for name in ("keypoints", "keypoint_scores", "descriptors"):
            print(f"image{index}.{name}: {tuple(features[name].shape)}")
    matches01 = matcher({"image0": features0, "image1": features1})
    # 模型输入输出有 batch 维度，单张图像的结果用 rbd 移除该维度。
    features0, features1, matches01 = [
        rbd(result) for result in (features0, features1, matches01)
    ]
    matches = matches01["matches"].cpu().numpy()  # [K, 2]，两列分别是两图关键点下标
    scores = matches01["scores"].cpu().numpy()  # 每对匹配的置信分数
    print(f"匹配下标形状: {matches.shape}，匹配分数形状: {scores.shape}")
    print(f"前 5 对匹配下标:\n{matches[:5]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv_keypoints = []
    saved = {"matches": matches, "match_scores": scores}
    for index, (features, original) in enumerate(zip((features0, features1), originals)):
        arrays = {
            name: features[name].cpu().numpy()
            for name in ("keypoints", "keypoint_scores", "descriptors", "image_size")
        }
        for name, value in arrays.items():
            saved[f"image{index}_{name}"] = value
        points = [cv2.KeyPoint(float(x), float(y), 4) for x, y in arrays["keypoints"]]
        cv_keypoints.append(points)
        order = np.argsort(-arrays["keypoint_scores"], kind="stable")
        if args.show_top:
            order = order[:args.show_top]
        preview = cv2.drawKeypoints(
            original, [points[i] for i in order], None, color=(0, 255, 0),
        )
        save_image(args.output_dir / f"image{index}_keypoints.png", preview)
        print(f"图片 {index}: 提取 {len(points)} 个关键点，绘制 {len(order)} 个")

    # 只限制画出的匹配数量，npz 始终保存全部匹配和全部特征。
    order = np.argsort(-scores, kind="stable")
    if args.show_top:
        order = order[:args.show_top]
    drawing_matches = [
        cv2.DMatch(int(matches[i, 0]), int(matches[i, 1]), float(1 - scores[i]))
        for i in order
    ]
    preview = cv2.drawMatches(
        originals[0], cv_keypoints[0], originals[1], cv_keypoints[1],
        drawing_matches, None, matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    save_image(args.output_dir / "matches.png", preview)
    saved["image0_path"] = str(args.image0.resolve())
    saved["image1_path"] = str(args.image1.resolve())
    np.savez_compressed(args.output_dir / "features_matches.npz", **saved)
    print(f"LightGlue 匹配数: {len(matches)}，绘制匹配数: {len(drawing_matches)}")
    print(f"结果已保存到: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
