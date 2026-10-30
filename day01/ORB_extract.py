"""用 ORB 提取 images.jpg 的关键点，按返回顺序绘制前 300 个。"""

from pathlib import Path

import cv2


def main():
    base_dir = Path(__file__).resolve().parent
    image_path = base_dir.parent / "dataset" / "images.jpg"
    output_dir = base_dir / "outputs"
    output_path = output_dir / "images_orb_top300.png"

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 最多提取 1000 个关键点，直接按 ORB 返回顺序取前 300 个。
    orb = cv2.ORB_create(nfeatures=1000)
    keypoints = orb.detect(gray, None)
    top_keypoints = keypoints[:300]

    visualization = cv2.drawKeypoints(
        image, top_keypoints, None, color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), visualization):
        raise OSError(f"图片保存失败: {output_path}")

    print(f"ORB 提取关键点数量: {len(keypoints)}")
    print(f"绘制关键点数量: {len(top_keypoints)}")
    print(f"可视化已保存: {output_path}")


if __name__ == "__main__":
    main()
