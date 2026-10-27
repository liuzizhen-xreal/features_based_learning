"""检查 CUDA、卷积反向传播和 torchvision 的 GPU 算子。"""

import torch
import torchvision
from torchvision.ops import nms


def main():
    print(f"PyTorch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA 不可用。请在本机终端运行；受限沙箱可能无法访问 GPU 设备。"
        )

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    print(f"GPU: {properties.name}")
    print(f"VRAM: {properties.total_memory / 1024**3:.2f} GiB")

    # 同时验证 CUDA 运算、cuDNN 卷积和自动求导。
    torch.manual_seed(42)
    model = torch.nn.Conv2d(1, 8, 3, padding=1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    weight_before = model.weight.detach().clone()
    image = torch.randn(2, 1, 64, 64, device=device)
    output = model(image)
    loss = output.square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert output.shape == (2, 8, 64, 64)
    assert torch.isfinite(loss).item()
    assert model.weight.grad is not None
    assert torch.isfinite(model.weight.grad).all().item()
    optimizer.step()
    assert not torch.equal(weight_before, model.weight)
    print(f"CUDA forward + backward + update: PASS, loss={loss.item():.6f}")

    # 验证 torch/torchvision 的 CUDA 二进制配套。
    boxes = torch.tensor(
        [[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]],
        dtype=torch.float32,
        device=device,
    )
    scores = torch.tensor([0.9, 0.8, 0.7], device=device)
    kept = nms(boxes, scores, iou_threshold=0.5)
    assert kept.cpu().tolist() == [0, 2]
    print("torchvision CUDA NMS: PASS")


if __name__ == "__main__":
    main()
