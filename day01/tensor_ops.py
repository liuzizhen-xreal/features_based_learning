#读取照片、转成tensor、再恢复原图
from pathlib import Path

import cv2;
import numpy as np;
import torch;
import argparse;

def main():
    # 根据脚本所在目录定位图片，不受运行命令时的工作目录影响。
    base_dir = Path(__file__).resolve().parent
    image_path = base_dir.parent / "dataset" / "images.jpeg"
    output_dir = base_dir / "outputs"

    # OpenCV 默认读取为 BGR 三通道彩色图像。
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图片，请检查文件路径和内容: {image_path}")

    #将BRG转为RBG
    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #将numpy转为tensor
    image_tensor = torch.from_numpy(image_RGB).clone()

    print(f"转成tensor后数组大小：{image_tensor.shape}")     
    print(f"转成tensor后数据类型：{image_tensor.dtype}")
    print(f"Tensor 所在设备：{image_tensor.device}")

    #转为float并归一化
    image_tensor = image_tensor.float()/255.0 #float32,范围[0，1]
    print(f"转成float之后数据类型：{image_tensor.dtype}")

    #(H,W,C)-->(C,H,W)
    image_tensor = image_tensor.permute(2,0,1)
    print(f"转成(C,H,W)后数组大小：{image_tensor.shape}")   

    image_tensor = image_tensor.unsqueeze(0)
    print(f"转成(1,C,H,W)后数组大小：{image_tensor.shape}") 

    # 按相反顺序还原：(1,C,H,W) -> (C,H,W) -> (H,W,C)。
    restored_tensor = image_tensor.squeeze(0)
    restored_tensor = restored_tensor.permute(1, 2, 0)

    # 恢复像素值：先乘 255，再四舍五入，最后转为 uint8。
    restored_tensor = (restored_tensor * 255.0).round().to(torch.uint8)
    restored_rgb = restored_tensor.numpy()

    # OpenCV 保存彩色图片时使用 BGR 顺序。
    restored_bgr = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tensor_restored.png"
    if not cv2.imwrite(str(output_path), restored_bgr):
        raise OSError(f"图片保存失败: {output_path}")
    print(f"已保存: {output_path}")
    print(f"还原图片与原图像素完全一致: {np.array_equal(image, restored_bgr)}")
    print(f"Tensor 所在设备：{image_tensor.device}")

if __name__ == "__main__":
    main()
