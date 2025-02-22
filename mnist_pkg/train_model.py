import torch
from mnist_pkg.cnn_model import Net
from mnist_pkg.data_loader import get_dataloader
from mnist_pkg.utils import set_device

def train() -> None:
    device = set_device()

    # Instantiate model
    model = Net().to(device)

    # Create Loss function and Optimiser
    loss = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader = get_dataloader()

    for data, label in train_loader:
        data = data.to(device)
        label = label.to(device)

        optimiser.zero_grad()
        output = model(data)
        loss_value = loss(output, label)
        loss_value.backward()
        optimiser.step()

        print(f"Loss: {loss_value.item():.4f}")


if __name__ == "__main__":
    train()
