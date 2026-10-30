from lightglue import SuperPoint
from lightglue.utils import load_image
from pathlib import Path

import cv2
import numpy as np

base_dir = Path(__file__).resolve().parent
image_path = base_dir.parent / "dataset" / "images.jpg"
output_dir = base_dir / "outputs"
output_path = output_dir / f"{image_path.stem}_superpoint.npz"

extractor = SuperPoint(max_num_keypoints=1000).eval().to("cuda")
image = load_image(image_path).to("cuda")

features = extractor.extract(image, resize=None)

for name in ["keypoints", "keypoint_scores", "descriptors"]:
    value = features[name]
    print(name, value.shape, value.dtype, value.device)

# 移除单张图片的 batch 维度，并转到 CPU，方便后续直接用 NumPy 读取。
saved_features = {
    name: features[name][0].detach().cpu().numpy()
    for name in ["keypoints", "keypoint_scores", "descriptors", "image_size"]
}
output_dir.mkdir(parents=True, exist_ok=True)
np.savez_compressed(output_path, **saved_features)
print(f"特征已保存: {output_path}")

# 仅可视化得分最高的 100 个关键点，npz 中仍保留全部特征。
top_indices = np.argsort(-saved_features["keypoint_scores"], kind="stable")[:300]
visualization = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
if visualization is None:
    raise FileNotFoundError(f"无法读取图片: {image_path}")
keypoints = [
    cv2.KeyPoint(float(x), float(y), 4)
    for x, y in saved_features["keypoints"][top_indices]
]
visualization = cv2.drawKeypoints(
    visualization,
    keypoints,
    None,
    color=(0, 255, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,
)
visualization_path = output_dir / f"{image_path.stem}_superpoint_keypoints.png"
if not cv2.imwrite(str(visualization_path), visualization):
    raise OSError(f"图片保存失败: {visualization_path}")
print(f"已绘制得分最高的 {len(keypoints)} 个关键点，可视化已保存: {visualization_path}")
