import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class AugmentedDataset(Dataset):
    """
    A custom Dataset class that applies data augmentation techniques
    to expand the original dataset.
    """
    
    def __init__(self, images, labels, augment=True):
        """
        Initialize the dataset with images and their corresponding labels.
        
        Arguments:
            images (np.ndarray): Image data of shape (N, H, W, C) or (N, H, W)
            labels (np.ndarray): Labels of shape (N,)
            augment (bool): Whether to apply augmentation
        """
        self.images = images
        self.labels = labels
        self.augment = augment
        
        # Basic transformations (always applied)
        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # Augmentation transformations
        self.augmentation_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        
        # Convert to PIL Image for transformations
        if len(img.shape) == 2:  # Grayscale
            img = Image.fromarray(img, mode='L')
        else:  # RGB
            img = Image.fromarray(img)
            
        # Apply transformations
        if self.augment:
            img = self.augmentation_transforms(img)
        else:
            img = self.basic_transform(img)
            
        return img, label


def create_augmented_loader(images, labels, batch_size=64, augment=True, shuffle=True, 
                           pin_memory=False, num_workers=0):
    """
    Creates a DataLoader with optional data augmentation.
    
    Arguments:
        images (np.ndarray): Image data
        labels (np.ndarray): Labels
        batch_size (int): Batch size for training
        augment (bool): Whether to apply augmentations
        shuffle (bool): Whether to shuffle the dataset
        pin_memory (bool): Whether to pin memory (useful for GPU training)
        num_workers (int): Number of workers for data loading
        
    Returns:
        DataLoader: PyTorch DataLoader with augmentation
    """
    dataset = AugmentedDataset(images, labels, augment=augment)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers
    )
    return dataloader


def create_mixed_dataset(images, labels, augment_factor=1):
    """
    Creates an enlarged dataset by adding augmented versions of the original data.
    
    Arguments:
        images (np.ndarray): Original images
        labels (np.ndarray): Original labels
        augment_factor (int): How many augmented versions to create per original image
        
    Returns:
        tuple: (augmented_images, augmented_labels)
    """
    # For demonstration, this applies simple augmentations directly
    # Create empty arrays to hold both original and augmented data
    aug_images = []
    aug_labels = []
    
    # Add original data
    aug_images.append(images)
    aug_labels.append(labels)
    
    # Apply basic augmentations for each factor
    for i in range(augment_factor):
        # Here we apply simple numpy-based augmentations
        # Horizontal flip
        if i % 3 == 0:
            flipped = np.flip(images, axis=2)
            aug_images.append(flipped)
            aug_labels.append(labels)
        # Vertical flip
        elif i % 3 == 1:
            flipped = np.flip(images, axis=1)
            aug_images.append(flipped)
            aug_labels.append(labels)
        # Rotation (90 degrees)
        else:
            rotated = np.rot90(images, k=1, axes=(1, 2))
            aug_images.append(rotated)
            aug_labels.append(labels)
    
    # Concatenate all augmented data
    final_images = np.concatenate(aug_images, axis=0)
    final_labels = np.concatenate(aug_labels, axis=0)
    
    return final_images, final_labels
