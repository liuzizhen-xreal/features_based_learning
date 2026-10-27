"""第一步：查看 Linear -> ReLU -> Linear 的数据与参数形状，不训练。"""

import torch
from torch import nn

import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

def generate_data(seed=42):
    # 固定随机种子，使数据、噪声和划分结果可以复现。
    generator = torch.Generator().manual_seed(seed)

    # 200 个 x 均匀采样自 [-1, 1)，每一行是一个样本。
    x = torch.rand(200, 1, generator=generator) * 2 - 1
    # 小幅高斯噪声：均值为 0，标准差为 0.1。
    epsilon = torch.randn(200, 1, generator=generator) * 0.1
    y = x * x + epsilon

    # 随机打乱索引，按 80% / 20% 划分；x、y 使用同一组索引。
    indices = torch.randperm(200, generator=generator)
    train_indices = indices[:160]
    val_indices = indices[160:]
    x_train, y_train = x[train_indices], y[train_indices]
    x_val, y_val = x[val_indices], y[val_indices]
    return x_train, y_train, x_val, y_val

def main():
    x_train, y_train, x_val, y_val = generate_data(seed=42)
    print(f"训练集: x={x_train.shape}, y={y_train.shape}")
    print(f"验证集: x={x_val.shape}, y={y_val.shape}")
    print(f"数据类型: {x_train.dtype}")
    torch.manual_seed(42)

    # 每个样本输入 1 个 x；隐藏层有 16 个神经元；输出 1 个预测值。
    model = nn.Sequential(
        nn.Linear(1, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    epochs = 500
    history = []

    def record_losses(epoch):
            # 训练和验证损失都使用当前参数计算，不参与梯度计算。
            model.eval()
            with torch.no_grad():
                train_loss = criterion(model(x_train), y_train).item()
                val_loss = criterion(model(x_val), y_val).item()
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            print(f"epoch={epoch:3d}, train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
    
    record_losses(0)
    for epoch in range(1, epochs + 1):
        model.train()
        predictions = model(x_train)             # 1. 前向预测
        loss = criterion(predictions, y_train)   # 2. 计算损失
        optimizer.zero_grad()                   # 3. 清空梯度
        loss.backward()                         # 4. 反向传播
    
        optimizer.step()                        # 5. 更新参数
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            record_losses(epoch)
    
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "MLP_loss_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)
    print(f"损失记录已保存: {history_path}")

# 绘制已记录轮次的训练、验证 MSE；对数纵轴便于观察后期变化。
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    recorded_epochs = [row["epoch"] for row in history]
    ax.plot(recorded_epochs, [row["train_loss"] for row in history],
            "o-", label="Training")
    ax.plot(recorded_epochs, [row["val_loss"] for row in history],
            "s--", label="Validation")
    ax.set(xlabel="Epoch", ylabel="MSE (log scale)",
           yscale="log", title="MLP loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    loss_path = output_dir / "MLP_loss_curve.png"
    fig.savefig(loss_path, dpi=160)
    plt.close(fig)

    # 输入按从小到大排列，绘制拟合直线和无噪声的目标直线。
    line_x = torch.linspace(-1, 1, 200).reshape(-1, 1)
    with torch.no_grad():
        fitted_y = model(line_x)
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    ax.scatter(x_train.numpy().ravel(), y_train.numpy().ravel(),
               s=18, alpha=0.5, label="Training data")
    ax.scatter(x_val.numpy().ravel(), y_val.numpy().ravel(),
               s=25, marker="x", label="Validation data")
    ax.plot(line_x.numpy().ravel(), fitted_y.numpy().ravel(),
            color="crimson", label="Learned model")
    ax.set(xlabel="x", ylabel="y", title="inspect_MLP")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fit_path = output_dir / "mlp_fit.png"
    fig.savefig(fit_path, dpi=160)
    plt.close(fig)
    print(f"损失曲线已保存: {loss_path}")
    print(f"拟合图已保存: {fit_path}")

    model.zero_grad(set_to_none=True)

    z = model[0](x_train)
    h = model[1](z)
    prediction = model[2](h)

    loss = (prediction - y_train).square().mean()
    loss.backward()

    print("激活前：", z.detach())
    print("激活后：", h.detach())

    for name, parameter in model.named_parameters():
        print(name, parameter.grad.shape, parameter.grad.norm().item())


if __name__ == "__main__":
    main()
