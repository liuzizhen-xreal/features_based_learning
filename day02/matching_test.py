"""固定图像对，比较 ORB 双向最近邻匹配与最近邻距离比检验。"""

import csv
from pathlib import Path

import cv2


def main():
    base_dir = Path(__file__).resolve().parent
    dataset_dir = base_dir.parent / "dataset"
    image1_path = dataset_dir / "20261023-104938.jpg"
    image2_path = dataset_dir / "20261023-105003.jpg"
    output_dir = base_dir / "outputs" / "matching_test"

    image1 = cv2.imread(str(image1_path), cv2.IMREAD_COLOR)
    image2 = cv2.imread(str(image2_path), cv2.IMREAD_COLOR)
    if image1 is None:
        raise FileNotFoundError(f"无法读取图片: {image1_path}")
    if image2 is None:
        raise FileNotFoundError(f"无法读取图片: {image2_path}")

    # 所有方法复用同一组关键点和描述子，以便公平比较。
    orb = cv2.ORB_create(nfeatures=1000, WTA_K=2)
    keypoints1, descriptors1 = orb.detectAndCompute(
        cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), None,
    )
    keypoints2, descriptors2 = orb.detectAndCompute(
        cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY), None,
    )
    print(f"图片1: {image1_path.name}，关键点: {len(keypoints1)}")
    print(f"图片2: {image2_path.name}，关键点: {len(keypoints2)}")

    cross_matches = []
    neighbors = []
    if descriptors1 is not None and descriptors2 is not None:
        # 互为对方的最近邻，才保留匹配。
        cross_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        cross_matches = cross_matcher.match(descriptors1, descriptors2)

        # 每个图1描述子在图2中寻找两个最近邻，供各阈值复用。
        ratio_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        neighbors = ratio_matcher.knnMatch(descriptors1, descriptors2, k=2)
    else:
        print("至少一张图没有描述子，各方法将输出 0 条匹配。")

    results = [("cross_check", "", cross_matches)]
    skipped = sum(len(pair) < 2 for pair in neighbors)
    print(f"不足两个邻居、跳过比值检验的描述子数量: {skipped}")
    for ratio in (0.6, 0.75, 0.9):
        good_matches = []
        for pair in neighbors:
            # 不能直接写 for first, second in neighbors，否则不足两个会报错。
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < ratio * second.distance:
                good_matches.append(first)
        results.append((f"ratio_{ratio:g}", ratio, good_matches))

    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for method, ratio, matches in results:
        matches = sorted(matches, key=lambda match: match.distance)
        # 绘制全部通过筛选的匹配，图上数量与记录数量一致。
        visualization = cv2.drawMatches(
            image1, keypoints1, image2, keypoints2, matches, None,
            matchColor=(0, 255, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        filename = f"{method}.png"
        output_path = output_dir / filename
        if not cv2.imwrite(str(output_path), visualization):
            raise OSError(f"图片保存失败: {output_path}")
        print(f"{method}: {len(matches)} 条匹配，已保存: {output_path}")
        records.append({
            "image1": image1_path.name,
            "image2": image2_path.name,
            "method": method,
            "ratio": ratio,
            "match_count": len(matches),
            "skipped_insufficient_neighbors": skipped if ratio != "" else "",
            "output_image": filename,
        })

    csv_path = output_dir / "match_counts.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"匹配数量记录已保存: {csv_path}")


if __name__ == "__main__":
    main()
