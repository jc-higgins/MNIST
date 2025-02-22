import torch
import torchvision
import os 
from pathlib import Path
from torch.utils.data import DataLoader

def get_dataloader() -> DataLoader:
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
        train_dataset = torchvision.datasets.MNIST(
            root=str(dataset),  # Convert Path to string
            train=True, 
            download=True,
            transform=transform
        )

        # Create a dataloader
        return DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

        # Use the dataloader
        # for batch_idx, (data, labels) in enumerate(train_loader):
        #     print(f"Batch shape: {data.shape}")
        #     print(f"Labels shape: {labels.shape}")
        #     print(f"First few labels: {labels[:5]}")
        #     break
            
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        # Try to clean up potentially corrupted files
        import shutil
        if dataset.exists():
            shutil.rmtree(dataset)
        print("Cleaned up data directory. Please run the script again.")
