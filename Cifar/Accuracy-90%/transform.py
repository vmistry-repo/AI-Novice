import cv2
import torchvision
import torch
import torchvision.transforms as transforms
# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

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
