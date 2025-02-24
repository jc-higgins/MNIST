import torch
from mnist_pkg.cnn_model import Net


def test_model_output_shape() -> None:
    model = Net()
    batch_size = 32
    x = torch.randn(batch_size, 1, 28, 28)
    output = model(x)

    assert output.shape == (batch_size, 10)


def test_model_forward_pass() -> None:
    model = Net()
    batch_size = 32
    x = torch.randn(batch_size, 1, 28, 28)
    output = model(x)

    assert not torch.isnan(output).any()
