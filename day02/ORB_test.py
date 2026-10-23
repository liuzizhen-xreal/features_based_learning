from collections import Counter
from pathlib import Path

import cv2
import numpy

def main():
    # 根据脚本所在目录定位图片，不受运行命令时的工作目录影响。
    base_dir = Path(__file__).resolve().parent
    # 图片放在项目根目录的 dataset 中，先通过 parent 返回上一级。
    image_path = base_dir.parent / "dataset" / "images.jpg"
    output_dir = base_dir / "outputs"


    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=1000)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
  
    print(f"关键点总数: {len(keypoints)}")

    # 这里的前 10 个是检测结果的原始顺序，不是响应排序后的顺序。
    print("前 10 个关键点的属性：")
    for index, point in enumerate(keypoints[:10], start=1):
        print(
            f"{index:2d}: pt=({point.pt[0]:.2f}, {point.pt[1]:.2f}), "
            f"response={point.response:.8f}, octave={point.octave}, "
            f"size={point.size:.2f}, angle={point.angle:.2f}"
        )

    # octave 是关键点所在的图像金字塔层级，0 表示原始尺度。
    octave_counts = Counter(point.octave for point in keypoints)
    print("各 octave 的关键点数量：")
    for octave in range(orb.getNLevels()):
        print(f"octave={octave}: {octave_counts[octave]}")

    # response 越大，响应越强；不足 20 个时取全部。
    top20 = sorted(keypoints, key=lambda point: point.response, reverse=True)[:20]
    print(f"单独绘制响应最高的 {len(top20)} 个点")

    # 圆心表示位置，圆的直径表示尺度，圆内短线表示方向。
    all_image = cv2.drawKeypoints(
        image, keypoints, None, color=(0, 0, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    top20_image = cv2.drawKeypoints(
        image, top20, None, color=(0, 0, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, result in (
        ("orb_test_all.png", all_image),
        ("orb_test_top20.png", top20_image),
    ):
        output_path = output_dir / filename
        if not cv2.imwrite(str(output_path), result):
            raise OSError(f"图片保存失败: {output_path}")
        print(f"已保存: {output_path}")


if __name__ == "__main__":
    main()
