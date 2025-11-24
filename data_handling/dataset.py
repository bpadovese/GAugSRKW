import torch
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets import DatasetFolder

class ImageDataset(DatasetFolder):
    def __init__(self, paths, label, transform=None):
        """
        Generic dataset for combining folders dynamically for any class.

        Args:
            paths (list): List of folder paths to include in the dataset.
                          The label is inferred from the last subfolder (e.g., '.../0', '.../1').
            label: the numeric label to use for all these folders
            transform (callable, optional): Transformation to apply to the input data.
        """
        self.transform = transform
        # Collect all files and their inferred labels
        self.samples = self._make_dataset(paths, label)

        if not self.samples:
            raise ValueError(f"No valid samples found in provided paths: {paths}")
        
    def _make_dataset(self, paths, label):
        samples = []
        for folder_path in paths:

            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"Dataset path does not exist: {folder_path}")

            if not os.path.isdir(folder_path):
                raise NotADirectoryError(f"Expected a directory but got a file: {folder_path}")

            # Collect all valid files
            if os.path.isdir(folder_path):
                # We already know the label from `label` param
                for root, _, filenames in os.walk(folder_path):
                    for filename in filenames:
                        path = os.path.join(root, filename)
                        samples.append((path, label))
        return samples

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path)  # Adjust if not working with images
        if self.transform:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)
    
class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

class ConditionalResize:
    def __init__(self, target_size):
        self.target_size = target_size
        self.resize_transform = transforms.Resize(target_size)

    def __call__(self, image):
        if image.size != self.target_size:
            return self.resize_transform(image)
        return image