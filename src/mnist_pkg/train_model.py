import torch
from mnist_pkg.cnn_model import Net
from mnist_pkg.constants import MODELS_PATH
from mnist_pkg.data_loader import MnistDataloader
from mnist_pkg.utils import load_model, save_model, set_device

def model_train() -> None:
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
                optimiser=optimiser,
                loss=best_validation_loss,
                model_filename="best_model.pth"
            )


def model_test() -> None:
    device = set_device()

    per_class_correct = torch.zeros(10).to(device)
    per_class_total = torch.zeros(10).to(device)
    # Rows are correct, Columns are predictions
    confusion_matrix = torch.zeros(10, 10).to(device)

    model = Net().to(device)
    optimiser = torch.optim.Adam(model.parameters())
    loss = torch.nn.CrossEntropyLoss()

    if not (MODELS_PATH / "best_model.pth").exists():
        raise FileNotFoundError("Please create a best_model by running train().")

    load_model(model, optimiser, "best_model.pth")

    dataloader = MnistDataloader()
    test_loader = dataloader.get_test_dataloader()

    running_test_loss = test_correct = test_total = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_loader):
            data = data.to(device)
            labels = labels.to(device)

            output = model(data)
            test_loss = loss(output, labels)
            running_test_loss += test_loss.item()

            _, predicted = torch.max(output, 1)

            for label, pred in zip(labels, predicted):
                per_class_total[label] += 1
                per_class_correct[label] += (label == pred)
                confusion_matrix[label][pred] += 1

            test_total += len(data)
            test_correct += (predicted == labels).sum().item()
    
    ave_test_loss = running_test_loss / len(test_loader)
    test_accuracy = 100 * test_correct/test_total 
    print(f"Test Performance: {test_correct}/{test_total} correct, {test_accuracy:.2f}")
    print(f"Average Test Loss: {ave_test_loss:.4f}")
    
    # Per-class results
    for i in range(10):
        class_acc = 100 * per_class_correct[i] / per_class_total[i]
        print(f"Accuracy of digit {i}: {class_acc:.2f}%")

    print("\nConfusion Matrix:")
    print("    Pred-> 0    1    2    3    4    5    6    7    8    9")
    print("True")
    for i in range(10):
        row = confusion_matrix[i].int()
        print(f"{i:4d}     {row[0]:4d} {row[1]:4d} {row[2]:4d} {row[3]:4d} {row[4]:4d} {row[5]:4d} {row[6]:4d} {row[7]:4d} {row[8]:4d} {row[9]:4d}")

if __name__ == "__main__":
    # model_train()
    model_test()