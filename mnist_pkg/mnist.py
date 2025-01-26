import torchvision
from constants import CHECKOUT_HOME


def main() -> None:
    # Download MNIST
    torchvision.datasets.MNIST(root=CHECKOUT_HOME / "dataset", train=True, download=False, )


if __name__ == "__main__":
    main()