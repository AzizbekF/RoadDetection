import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class RoadsDataset(Dataset):
    """
        Args:
        df (DataFrame): DataFrame containing images / labels paths
        augmentation (albumentations.Compose): data transformation pipeline
                    (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
                    (e.g. normalization, shape manipulation, etc.)
    """

    def __init__(
            self,
            df,
            augmentation=None,
            preprocessing=None,
    ):
        self.image_paths = df['sat_image_path'].tolist()
        self.mask_paths = df['mask_path'].tolist()
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # Read images and masks
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask_gray = cv2.imread(self.mask_paths[i], cv2.IMREAD_GRAYSCALE)
        mask = (mask_gray > 128).astype(np.uint8)  # 1=road, 0=background

        # Apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # Apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.image_paths)


def get_training_augmentation(height=512, width=512):
    """
    Training augmentation pipeline.

    Args:
        height (int): Output image height
        width (int): Output image width

    Returns:
        albumentations.Compose: Augmentation pipeline
    """
    train_transform = [
        A.RandomCrop(height=height, width=width, always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1.0),
                A.RandomGamma(p=1.0),
                A.HueSaturationValue(p=1.0),
            ],
            p=0.5,
        ),
        A.OneOf(
            [
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1.0),
                A.GridDistortion(p=1.0),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1.0),
            ],
            p=0.3,
        ),
    ]

    return A.Compose(train_transform)


def get_validation_augmentation(height=512, width=512):
    """
    Validation augmentation - just resize image and mask to a specified size.

    Args:
        height (int): Output image height
        width (int): Output image width

    Returns:
        albumentations.Compose: Augmentation pipeline for validation
    """
    return A.Compose([
        A.CenterCrop(height, width)
    ])


def get_preprocessing():
    """
    Prepare image for input to neural network:
    - Convert to PyTorch tensor
    - Normalize with ImageNet mean and std

    Returns:
        albumentations.Compose: Preprocessing pipeline
    """
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def calculate_class_weights(dataset, num_classes=2):
    """
    Calculate class weights for imbalanced datasets.

    Args:
        dataset: Dataset containing the masks
        num_classes (int): Number of classes

    Returns:
        torch.Tensor: Class weights
    """
    counts = torch.zeros(num_classes)

    for i in range(len(dataset)):
        _, mask = dataset[i]

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        counts[0] += (mask == 0).sum()
        counts[1] += (mask == 1).sum()

    # Calculate weights as inverse of frequency
    weights = 1.0 / counts
    weights = weights / weights.sum()  # Normalize

    return weights