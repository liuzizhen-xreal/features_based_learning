"""先预测结果，再运行。完成 NumPy 后再加 --torch 对照 Tensor。"""
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--torch', action='store_true')
args = parser.parse_args()

# 用小数组表示一张 H=2、W=3、C=3 的彩色图像。
image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
print('HWC 图像:\n', image)
print('shape / dtype:', image.shape, image.dtype)
print('第 0 行、第 1 列的三个通道:', image[0, 1, :])
normalized = image.astype(np.float32) / 255.0
chw = normalized.transpose(2, 0, 1)
batch = chw[None, ...]
print('归一化后 dtype:', normalized.dtype)
print('CHW / NCHW:', chw.shape, batch.shape)
a = np.array([[1., 2.], [3., 4.]])
print('逐元素乘法:\n', a * a)
print('矩阵乘法:\n', a @ a)

if args.torch:
    import torch
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor = tensor.to(device)
    print('Tensor shape / dtype / device:', tensor.shape, tensor.dtype, tensor.device)
    print('注意：permute 改变维度顺序，reshape 不能替代这个操作。')
