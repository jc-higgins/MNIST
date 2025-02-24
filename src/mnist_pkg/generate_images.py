import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import torch
from typing import List
from torch.utils.data import DataLoader
from torch.nn import Module

from mnist_pkg.utils import load_model


def ensure_assets_dir() -> None:
    assets_dir = "assets"
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)


def plot_confusion_matrix(
    y_true: List[int], y_pred: List[int], save_path: str = "assets/confusion_matrix.png"
) -> None:
    ensure_assets_dir()

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Convert to percentages
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create figure and axes
    plt.figure(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=range(10),
        yticklabels=range(10),
    )

    # Add labels
    plt.title("MNIST Confusion Matrix (%)", pad=20)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Ensure the plot is tight and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_model_visualizations(
    model: Module, test_loader: DataLoader, device: str = "mps"
) -> None:
    # Collect predictions
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    plot_confusion_matrix(y_true, y_pred)


if __name__ == "__main__":
    from mnist_pkg.cnn_model import Net
    from mnist_pkg.data_loader import MnistDataloader

    model = Net().to("mps")
    opt = torch.optim.Adam(model.parameters())
    load_model(model, opt, "best_model.pth")
    test_loader = MnistDataloader().get_test_dataloader()

    generate_model_visualizations(model, test_loader, "mps")
