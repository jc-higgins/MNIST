from mnist_pkg.data_loader import MnistDataloader


def test_dataloader_initialization() -> None:
    loader = MnistDataloader()
    train_loader = loader.get_training_dataloader()
    validation_loader = loader.get_validation_dataloader()
    test_loader = loader.get_test_dataloader()

    assert train_loader is not None, "Training DataLoader is empty on initialization"
    assert (
        validation_loader is not None
    ), "Validation DataLoader is empty on initialization"
    assert test_loader is not None, "Test DataLoader is empty on initialization"


def test_data_shapes() -> None:
    loader = MnistDataloader()
    train_loader = loader.get_training_dataloader()

    # Get first batch
    images, labels = next(iter(train_loader))
    assert images.shape[0] == 32  # batch size
    assert images.shape[1] == 1  # channels
    assert images.shape[2] == 28  # height
    assert images.shape[3] == 28  # width
