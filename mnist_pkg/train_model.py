import torch
from mnist_pkg.cnn_model import Net
from mnist_pkg.data_loader import MnistDataloader
from mnist_pkg.utils import save_model, set_device

def train() -> None:
    device = set_device()

    # Instantiate model
    model = Net().to(device)

    # Create Loss function and Optimiser
    loss = torch.nn.CrossEntropyLoss()
    running_loss = 0.0
    print_every = 100
    epochs = 5
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    # Validation
    best_validation_loss = float("inf")

    dataloader = MnistDataloader()
    train_loader = dataloader.get_training_dataloader()
    validation_loader = dataloader.get_validation_dataloader()
    test_loader = dataloader.get_test_dataloader()

    for epoch in range(epochs):
        validation_total = validation_correct = validation_running_loss = 0

        # Train
        model.train()
        for batch_idx, (data, label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)

            optimiser.zero_grad()
            output = model(data)
            loss_value = loss(output, label)
            running_loss += loss_value.item()
            loss_value.backward()
            optimiser.step()

            if batch_idx % print_every == 0 and batch_idx > 0:
                avg_loss = running_loss / print_every
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Average Loss: {avg_loss:.4f}')
                running_loss = 0.0
        
        # Validate
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(validation_loader):
                data = data.to(device)
                labels = labels.to(device)

                outputs = model(data)
                validation_loss = loss(outputs, labels)
                validation_running_loss += validation_loss.item()

                _, predicted = torch.max(outputs.data, 1)
                validation_total += labels.size(0)
                validation_correct += (predicted == labels).sum().item()

        ave_validation_loss = validation_running_loss / len(validation_loader)
        validation_accuracy = 100 * validation_correct / validation_total
        print(f'Validation Loss: {ave_validation_loss:.4f}, Accuracy: {validation_accuracy:.2f}%')

        # Save if better
        if ave_validation_loss < best_validation_loss:
            best_validation_loss = ave_validation_loss
            save_model(
                model=model,
                epoch=epoch,
                optimizer=optimiser,
                loss=best_validation_loss,
                path="data/MNIST/models/best_model.pth"
            )

if __name__ == "__main__":
    train()
