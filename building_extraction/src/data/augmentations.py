import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_building_augmentations(mode: str = "train", image_size: int = 512):
    """
    Get augmentation pipeline for building extraction.

    Args:
        mode: "train" or "val"
        image_size: target image size
    """
    if mode == "train":
        return A.Compose([
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05,
                p=0.5
            ),
            A.GaussNoise(p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_light_augmentations(image_size: int = 512):
    """Light augmentation for faster inference."""
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])