import cv2
import torchvision
import torch
import torchvision.transforms as transforms
# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

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
          A.HorizontalFlip(p=0.5),
          A.ShiftScaleRotate(),
          A.CoarseDropout(max_holes = 1, max_height=16, max_width=16,
                          min_holes = 1, min_height=16, min_width=16,
                          fill_value=pixel_means, mask_fill_value = None),
          A.Normalize(
              mean = pixel_means,
              std = pixel_stds,
              p =1.0
          ),
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
