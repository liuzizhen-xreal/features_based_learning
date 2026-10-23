#ORB特征点提取与匹配实验
from pathlib import Path

import numpy
import cv2

def main():
    # 根据脚本所在目录定位图片，不受运行命令时的工作目录影响。
    base_dir = Path(__file__).resolve().parent
    image1_path = base_dir.parent / "dataset" / "20261023-105003.jpg"
    image2_path = base_dir.parent / "dataset" / "20261023-104938.jpg"
    output_dir = base_dir / "outputs" 

    output_dir.mkdir(parents=True, exist_ok=True)

    image1 = cv2.imread(str(image1_path), cv2.IMREAD_COLOR)
    image2 = cv2.imread(str(image2_path), cv2.IMREAD_COLOR)
    if image1 is None:
        raise FileNotFoundError(f"无法读取图片: {image1_path}")
    if image2 is None:
        raise FileNotFoundError(f"无法读取图片: {image2_path}")

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # ORB 返回关键点列表和二进制描述子数组。
    orb = cv2.ORB_create(nfeatures=1000)
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    for name, keypoints, descriptors in (
        ("图片1", keypoints1, descriptors1),
        ("图片2", keypoints2, descriptors2),
    ):
        print(f"{name}关键点数量: {len(keypoints)}")
        if descriptors is None:
            print(f"{name}描述子: None（未检测到可用特征）")
        else:
            print(f"{name}描述子形状: {descriptors.shape}")
            print(f"{name}描述子数据类型: {descriptors.dtype}")

    # 用红色小圆标记关键点位置，不显示尺度和方向。
    keypoints_image1 = cv2.drawKeypoints(
        image1, keypoints1, None, color=(0, 0, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,
    )
    keypoints_image2 = cv2.drawKeypoints(
        image2, keypoints2, None, color=(0, 0, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,
    )

    # 汉明距离比较二进制描述子；crossCheck=True 只保留互为最近邻的匹配。
    matches = []
    if descriptors1 is not None and descriptors2 is not None:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(
            matcher.match(descriptors1, descriptors2),
            key=lambda match: match.distance,
        )

    # 距离越小越相似；匹配不足 50 条时绘制全部匹配。
    best_matches = matches[:50]
    print(f"双向最近邻匹配数量: {len(matches)}")
    print(f"绘制匹配数量: {len(best_matches)}")
    matches_image = cv2.drawMatches(
        image1, keypoints1, image2, keypoints2, best_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    results = {
        "orb_keypoints1.png": keypoints_image1,
        "orb_keypoints2.png": keypoints_image2,
        "orb_matches_top50.png": matches_image,
    }
    for filename, result in results.items():
        output_path = output_dir / filename
        if not cv2.imwrite(str(output_path), result):
            raise OSError(f"图片保存失败: {output_path}")
        print(f"已保存: {output_path}")


if __name__ == "__main__":
    main()
