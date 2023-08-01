import torchvision
from torchvision import datasets
from torchvision import transforms

transform = transforms.Compose([
                transforms.ToTensor(),
            ])

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Cifar10SearchDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label

def get_CIFAR10(train_transform=transform, test_transform=transform):
    train_data = Cifar10SearchDataset(root='./data', train=True, download=True, transform=train_transform)
    test_data = Cifar10SearchDataset(root='./data', train=False, download=True, transform=test_transform)
    return train_data, test_data

def get_CIFAR10_classes():
    return classes