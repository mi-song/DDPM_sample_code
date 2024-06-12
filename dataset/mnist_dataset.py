import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchvision

class MnistDataset(Dataset):
    r"""
    Nothing special here. Just a simple dataset class for MNIST images.
    Created a dataset class rather using torchvision to allow
    replacement with any other image dataset
    """

    def __init__(self, split, transform=None):
        r"""
        Init method for initializing the dataset properties
        :param split: 'train' or 'test' to download the respective dataset
        :param transform: transformations to be applied to the images
        """
        self.split = split
        self.transform = transform

        if self.split == 'train':
            self.dataset = torchvision.datasets.MNIST(
                root='./data', train=True, download=True, transform=transforms.ToTensor()
            )
        else:
            self.dataset = torchvision.datasets.MNIST(
                root='./data', train=False, download=True, transform=transforms.ToTensor()
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        im, label = self.dataset[index]

        if self.transform:
            im = self.transform(im)

        # Convert input to -1 to 1 range.
        im = (2 * im) - 1
        return im, label
