import torch
from mnist_pkg.train_model import train
from mnist_pkg.cnn_model import Net

def test_validation_loss_calculation():
    model = Net()
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Create dummy data
    batch_size = 32
    x = torch.randn(batch_size, 1, 28, 28)
    y = torch.randint(0, 10, (batch_size,))
    
    # Forward pass
    output = model(x)
    loss = loss_fn(output, y)
    
    assert not torch.isnan(loss)
    assert loss.item() > 0
