import cv2
import torchvision
import torch
import torchvision.transforms as transforms
# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class Cifar10SearchDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label

def get_train_transforms(pixel_means, pixel_stds):
  train_transforms = A.Compose(
      [
          A.PadIfNeeded(min_height=36, min_width=36, p=1),
          A.RandomCrop(32, 32, True, 1),
          A.Normalize(
              mean = pixel_means,
              std = pixel_stds,
              p =1.0
          ),
          A.HorizontalFlip(p=0.5),
          A.CoarseDropout(max_holes = 1, max_height=8, max_width=8,
                          min_holes = 1, min_height=8, min_width=8,
                          fill_value=pixel_means, mask_fill_value = None),
          ToTensorV2()
      ],
      p=1.0
  )
  return train_transforms

def get_test_transforms(pixel_means, pixel_stds):
  test_transforms = A.Compose(
      [
          A.Normalize(
              mean = pixel_means,
              std = pixel_stds,
              p =1.0
          ),
          ToTensorV2()
      ],
      p=1.0
  )
  return test_transforms

class AlbumentationsTransform(pl.LightningDataModule):
    def __init__(self, train_transforms, test_transforms, train_loader, test_loader, val_loader):
        super().__init__()
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader

    def train_transform(self, image):
        return self.train_transforms(image=image)["image"]

    def test_transform(self, image):
        return self.test_transforms(image=image)["image"]

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

