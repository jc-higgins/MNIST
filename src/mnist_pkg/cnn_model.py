import torch
from torch import Tensor

class Net(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Layers
        self.fir_conv_layer = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.sec_conv_layer = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.relu_layer     = torch.nn.ReLU()
        self.pooling_layer  = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1            = torch.nn.Linear(in_features=64*12*12, out_features=128)
        self.fc2            = torch.nn.Linear(in_features=128, out_features=10)


    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
        Returns:
            Tensor of shape (batch_size, 10)
        """

        # Forward pass
        x = self.fir_conv_layer(x)
        x = self.relu_layer(x)
        x = self.sec_conv_layer(x)
        x = self.relu_layer(x)
        x = self.pooling_layer(x)

        x = x.view(-1, 64*12*12)
        x = self.fc1(x)
        x = self.relu_layer(x)
        x = self.fc2(x)
        return x
        
