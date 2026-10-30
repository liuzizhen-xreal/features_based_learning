from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from inspect_conv import SimpleCNN

root = Path(__file__).resolve().parents[1] / "dataset" / "mnist"

dataset = datasets.MNIST(
    root=root,
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)

train_set, val_set = random_split(
    dataset,
    [55000, 5000],
    generator=torch.Generator().manual_seed(42),
)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

images, labels = next(iter(train_loader))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

images = images.to(device)
labels = labels.to(device)

logits = model(images)          # [64, 10]
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, labels)

predicted = logits.argmax(dim=1)  # 每张图中，分数最高的类别
accuracy = (predicted == labels).float().mean()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
epochs = 5
history = []

output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

for epoch in range(epochs):
    # ---------- 训练 ----------
    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_count = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        train_loss_sum += loss.item() * batch_size
        train_correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()
        train_count += batch_size

    # ---------- 验证 ----------
    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_count = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            val_loss_sum += loss.item() * batch_size
            val_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            val_count += batch_size

    train_loss = train_loss_sum / train_count
    train_acc = train_correct / train_count
    val_loss = val_loss_sum / val_count
    val_acc = val_correct / val_count

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
    })

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"train_loss={train_loss:.4f}, "
        f"train_acc={train_acc:.2%} | "
        f"val_loss={val_loss:.4f}, "
        f"val_acc={val_acc:.2%}"
    )

torch.save({
    "epoch": epoch + 1,
    "model_state_dict": model.state_dict(),
    "val_loss": val_loss,
    "val_acc": val_acc,
}, output_dir / "best_cnn.pt")

checkpoint = torch.load(
output_dir / "best_cnn.pt",
map_location=device,
weights_only=True,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

import csv

with (output_dir / "history.csv").open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(file, fieldnames=history[0].keys())
    writer.writeheader()
    writer.writerows(history)

errors = []

with torch.no_grad():
    for images, labels in val_loader:
        logits = model(images.to(device))
        predictions = logits.argmax(dim=1).cpu()
        labels = labels.cpu()

        wrong_indices = (
            predictions != labels
        ).nonzero(as_tuple=True)[0]

        for i in wrong_indices.tolist():
            errors.append((
                images[i].cpu().squeeze(0),
                labels[i].item(),
                predictions[i].item(),
            ))

            if len(errors) >= 16:
                break

        if len(errors) >= 16:
            break

import matplotlib.pyplot as plt

if errors:
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))

    for ax in axes.flat:
        ax.axis("off")

    for ax, (image, true_label, predicted_label) in zip(
        axes.flat, errors
    ):
        ax.imshow(image.numpy(), cmap="gray")
        ax.set_title(
            f"True: {true_label} / Pred: {predicted_label}",
            color="red",
        )

    fig.tight_layout()
    save_path = output_dir / "wrong_predictions.png"
    fig.savefig(save_path, dpi=160)
    plt.close(fig)

    print("错误样本图已保存：", save_path)
else:
    print("验证集中没有发现错误预测。")

