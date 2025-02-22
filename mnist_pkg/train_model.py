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
    running_loss = 0.0
    print_every = 100
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader = get_dataloader()

    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(device)
        label = label.to(device)

        optimiser.zero_grad()
        output = model(data)
        loss_value = loss(output, label)
        running_loss += loss_value
        loss_value.backward()
        optimiser.step()

        if batch_idx % print_every == 0 and batch_idx > 0:
            avg_loss = running_loss / print_every
            print(f'Batch {batch_idx}, Average Loss: {avg_loss:.4f}')
            running_loss = 0.0


if __name__ == "__main__":
    train()
