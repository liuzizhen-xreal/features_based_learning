"""生成带噪声的线性数据，使用 SGD 学习 y = wx + b。"""
import csv
from pathlib import Path

import torch
import matplotlib

import matplotlib.pyplot as plt


def generate_data(seed=42):
    # 固定随机种子，使数据、噪声和划分结果可以复现。
    generator = torch.Generator().manual_seed(seed)

    # 200 个 x 均匀采样自 [-1, 1)，每一行是一个样本。
    x = torch.rand(200, 1, generator=generator) * 2 - 1
    # 小幅高斯噪声：均值为 0，标准差为 0.1。
    epsilon = torch.randn(200, 1, generator=generator) * 0.1
    y = 3 * x + 2 + epsilon

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
    print("训练集前 5 对 (x, y)：")
    for x, y in zip(x_train[:5], y_train[:5]):
        print(f"x={x.item():.4f}, y={y.item():.4f}")

    # 固定模型初始化的随机种子，使训练过程可以复现。
    torch.manual_seed(42)
    model = torch.nn.Linear(1, 1)
    print(f"训练前: w={model.weight.item():.6f}, b={model.bias.item():.6f}")

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    epochs = 200
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

        # backward 后才有本轮梯度，在第一次 step 更新参数前打印。
        if epoch == 1:
            print(f"第一次更新前 weight.grad: {model.weight.grad}")
            print(f"第一次更新前 bias.grad: {model.bias.grad}")

        optimizer.step()                        # 5. 更新参数
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            record_losses(epoch)

    print(f"训练后: w={model.weight.item():.6f}, b={model.bias.item():.6f}")

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "linear_loss_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)
    print(f"损失记录已保存: {history_path}")

    # 只保存模型参数；加载时需要创建相同结构的模型。
    model_path = output_dir / "linear_model.pth"
    torch.save(model.state_dict(), model_path)
    loaded_model = torch.nn.Linear(1, 1)
    loaded_model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    loaded_model.eval()

    fixed_x = torch.tensor([[-1.0], [-0.5], [0.0], [0.5], [1.0]])
    with torch.no_grad():
        original_predictions = model(fixed_x)
        loaded_predictions = loaded_model(fixed_x)
    identical = torch.equal(original_predictions, loaded_predictions)
    print(f"模型参数已保存: {model_path}")
    print(f"固定输入: {fixed_x.flatten().tolist()}")
    print(f"原模型预测: {original_predictions.flatten().tolist()}")
    print(f"加载后预测: {loaded_predictions.flatten().tolist()}")
    print(f"两者预测完全一致: {identical}")
    if not identical:
        raise AssertionError("加载模型后的预测与原模型不一致")

    # 绘制已记录轮次的训练、验证 MSE；对数纵轴便于观察后期变化。
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    recorded_epochs = [row["epoch"] for row in history]
    ax.plot(recorded_epochs, [row["train_loss"] for row in history],
            "o-", label="Training")
    ax.plot(recorded_epochs, [row["val_loss"] for row in history],
            "s--", label="Validation")
    ax.set(xlabel="Epoch", ylabel="MSE (log scale)",
           yscale="log", title="Linear regression loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    loss_path = output_dir / "linear_loss_curve.png"
    fig.savefig(loss_path, dpi=160)
    plt.close(fig)

    # 输入按从小到大排列，绘制拟合直线和无噪声的目标直线。
    line_x = torch.linspace(-1, 1, 200).reshape(-1, 1)
    with torch.no_grad():
        fitted_y = loaded_model(line_x)
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    ax.scatter(x_train.numpy().ravel(), y_train.numpy().ravel(),
               s=18, alpha=0.5, label="Training data")
    ax.scatter(x_val.numpy().ravel(), y_val.numpy().ravel(),
               s=25, marker="x", label="Validation data")
    ax.plot(line_x.numpy().ravel(), fitted_y.numpy().ravel(),
            color="crimson", label="Learned model")
    ax.plot(line_x.numpy().ravel(), (3 * line_x + 2).numpy().ravel(),
            "--", color="black", label="Target: y = 3x + 2")
    ax.set(xlabel="x", ylabel="y", title="Linear regression fit")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fit_path = output_dir / "linear_fit.png"
    fig.savefig(fit_path, dpi=160)
    plt.close(fig)
    print(f"损失曲线已保存: {loss_path}")
    print(f"拟合图已保存: {fit_path}")


if __name__ == "__main__":
    main()
