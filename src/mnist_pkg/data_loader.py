import torch
import torchvision
import os 
from pathlib import Path
from torch.utils.data import DataLoader

class MnistDataloader:

    def __init__(self) -> None:
        # Set torch hub directory to avoid permission issues
        torch.hub.set_dir(str(Path.home() / ".cache" / "torch"))
        
        # Override MNIST URLs to use direct S3 links
        torchvision.datasets.MNIST.mirrors = ["https://ossci-datasets.s3.amazonaws.com/mnist/"]
        torchvision.datasets.MNIST.resources = [
            ('train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
            ('train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
            ('t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
            ('t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
        ]
        
        # Download MNIST with explicit transforms
        dataset = Path(os.getcwd()) / "data"
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        try:
            # Create training dataset and loader
            train_dataset = torchvision.datasets.MNIST(
                root=str(dataset),  # Convert Path to string
                train=True, 
                download=True,
                transform=transform
            )
            self.training_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

            # Create validation and test loaders
            full_test_dataset = torchvision.datasets.MNIST(
                root=str(dataset),  # Convert Path to string
                train=False, 
                download=True,
                transform=transform
            )

            validation_set_size = len(full_test_dataset) // 2
            test_set_size = len(full_test_dataset) - validation_set_size
            validation_dataset, test_dataset = torch.utils.data.random_split(full_test_dataset, [validation_set_size, test_set_size])

            self.validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=32, shuffle=False)
            self.test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
            
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            # Try to clean up potentially corrupted files
            import shutil
            if dataset.exists():
                shutil.rmtree(dataset)
            print("Cleaned up data directory. Please run the script again.")


    def get_training_dataloader(self) -> DataLoader:
        return self.training_dataloader

    def get_validation_dataloader(self) -> DataLoader:
        return self.validation_dataloader

    def get_test_dataloader(self) -> DataLoader:
        return self.test_dataloader
