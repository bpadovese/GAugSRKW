# =============================================================================
# Imports
# =============================================================================

import time
import torch
import sys
import torch.nn as nn
import re
import random
import numpy as np
import os
from torchvision import transforms
from dev_utils.nn import resnet18_for_single_channel
from data_handling.dataset import Subset, ImageDataset, ConditionalResize
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
from pathlib import Path
from diffusers.optimization import get_cosine_schedule_with_warmup 
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger
import torcheval.metrics as metrics

def train(fabric, model, optimizer, loss_func, train_loader, lr_scheduler=None, val_loader=None, num_epochs=20, num_classes=2):
    print("Training starting...")
    for epoch in range(num_epochs):
        start_time = time.time()  
        model.train()

        # Training step
        train_loss, train_accuracy, train_precision, train_recall = train_epoch(
            fabric, model, optimizer, loss_func, train_loader, epoch, lr_scheduler=lr_scheduler, num_classes=num_classes
        )

        # Validation step
        avg_val_loss, val_accuracy, val_precision, val_recall = validate(
            fabric, model, val_loader, loss_func, num_classes
        )

        end_time = time.time()  
        epoch_duration = end_time - start_time

        print(f'Epoch {epoch + 1}, Training Loss: {train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
        print(f'Training Accuracy: {train_accuracy:.2f}%, Training Precision: {train_precision:.4f}, Training Recall: {train_recall:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.2f}%, Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}')
        print(f'Epoch Duration: {epoch_duration:.2f} seconds')
        print()

        fabric.log_dict({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'validation_loss': avg_val_loss,
            'validation_accuracy': val_accuracy,
            'validation_precision': val_precision,
            'validation_recall': val_recall,
        })

def train_epoch(fabric, model, optimizer, loss_func, train_loader, epoch, lr_scheduler=None, num_classes=2):
    running_loss = 0.0

    # Initializing metrics
    precision_metric = metrics.MulticlassPrecision(num_classes=num_classes, average=None).to(fabric.device)
    recall_metric = metrics.MulticlassRecall(num_classes=num_classes, average=None).to(fabric.device)
    accuracy_metric = metrics.MulticlassAccuracy(num_classes=num_classes, average='macro').to(fabric.device)

    for _, (inputs,labels) in enumerate(train_loader, 0):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        fabric.backward(loss)
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()
        
        # Update running loss
        running_loss += loss.item()

        precision_metric.update(outputs, labels)
        recall_metric.update(outputs, labels)
        accuracy_metric.update(outputs, labels)

    # Compute average loss for the epoch
    avg_loss = running_loss / len(train_loader)

    # Compute metrics for the entire epoch
    precision = precision_metric.compute()
    recall = recall_metric.compute()
    accuracy = accuracy_metric.compute().item()

    precision = precision[1].item()  # Precision for class 1 (KW)
    recall = recall[1].item()  # Recall for class 1 (KW)
    
    return avg_loss, accuracy, precision, recall 

@torch.no_grad()
def validate(fabric, model, dataloader, loss_func, num_classes):
    model.eval()
    test_loss = 0

    precision_metric = metrics.MulticlassPrecision(num_classes=num_classes, average=None).to(fabric.device)
    recall_metric = metrics.MulticlassRecall(num_classes=num_classes, average=None).to(fabric.device)
    accuracy_metric = metrics.MulticlassAccuracy(num_classes=num_classes, average='macro').to(fabric.device)

    for inputs, labels in dataloader:
        outputs = model(inputs)
        test_loss += loss_func(outputs, labels).item()

        precision_metric.update(outputs, labels)
        recall_metric.update(outputs, labels)
        accuracy_metric.update(outputs, labels)

    avg_val_loss = test_loss / len(dataloader)
    precision = precision_metric.compute()
    recall = recall_metric.compute()
    accuracy = accuracy_metric.compute().item()
    
    precision = precision[1].item()
    recall = recall[1].item()
    return avg_val_loss, accuracy, precision, recall

def get_next_version(model_name, output_folder):
    """
    Finds the highest version of the model in the output folder and returns the next version number.
    """
    version_pattern = re.compile(rf"{model_name}_v(\d+)\.pt")

    # List all model files in the output folder
    version_numbers = []
    for file in output_folder.iterdir():
        match = version_pattern.match(file.name)
        if match:
            version_numbers.append(int(match.group(1)))

    return max(version_numbers, default=-1) + 1

def create_balanced_indices(labels):
    """
    Create balanced indices for each label.

    Args:
        labels: List of labels.

    Returns:
        List of balanced indices.
    """
    # Gather all indices by class
    class_indices = {}
    for idx, label in enumerate(labels):
        class_indices.setdefault(label, []).append(idx)

    # Determine the minimum class size
    min_count = min(len(indices) for indices in class_indices.values())

    # Randomly sample min_count indices from each class
    balanced_indices = []
    for cls, indices in class_indices.items():
        # randperm will create a tensor length indices with random integers from 0 to len(indices).
        # From this tensor we will take the first min_count indices which effectively corresponds to
        # sampling min_count indicies (similar behaviour to numpy.random.choice)
        sampled_relative_indices = torch.randperm(len(indices))[:min_count].tolist()
        sampled_indices = [indices[i] for i in sampled_relative_indices]  # Map to global indices
        balanced_indices.extend(sampled_indices)

    return balanced_indices

def create_per_class_datasets(paths, transform=None, num_samples_per_dataset=None):
    """ 
    Returns a list of datasets, each corersponding to a different class, with an optional balanced sampling.
    
    Args:
        paths (list): List of folder paths containing class data.
        transform (callable, optional): Transformations applied to the data.
        num_samples_per_class (int, optional): Number of samples to take per class.
        
    Returns:
        list: List of datasets, each corresponding to a specific class.
    """
    datasets = []
    
    for i, folder_path in enumerate(paths):
        if not os.path.exists(folder_path):
                raise FileNotFoundError(f"Dataset path does not exist: {folder_path}")

        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"Expected a directory but got a file: {folder_path}")
        
        label = int(os.path.basename(folder_path))  # Label is inferred from folder name
        dataset = ImageDataset(paths=[folder_path], label=label, transform=transform)

        # Sample a fixed number of instances if specified
        if num_samples_per_dataset is not None and num_samples_per_dataset[i] is not None:
            num_samples = min(num_samples_per_dataset[i], len(dataset))
            sampled_indices = random.sample(range(len(dataset)), num_samples)
            dataset = Subset(dataset, sampled_indices)

        datasets.append(dataset)

    return datasets 

def main(dataset, train_set='/train', val_set='/test', num_samples_per_dataset=None, output_folder=None, 
         train_batch_size=32, input_shape=(128,128), eval_batch_size=32, num_epochs=20, model_name='my_model', 
         versioning=False, seed=None):

    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    if output_folder is None:
        output_folder = Path('.').resolve()
    else:
        output_folder = Path(output_folder).resolve()

    output_folder.mkdir(parents=True, exist_ok=True)

    logger = CSVLogger(output_folder, name='logs')

    fabric = Fabric(loggers=logger)
    num_classes = 2

    # Model file naming logic based on versioning parameter
    if versioning:
        # If versioning is True, version the model by checking existing files
        version = get_next_version(model_name, output_folder)

        model_path = output_folder / f"{model_name}_v{version}.pt"
    else:
        # If versioning is False, always use the same model name
        model_path = output_folder / f"{model_name}.pt"

    train_transform = train_transform = transforms.Compose([
        ConditionalResize(input_shape),
        transforms.ToTensor(),
    ])

    train_set = [os.path.normpath(table).lstrip(os.sep) for table in train_set]
    train_paths = [os.path.join(dataset, table) for table in train_set]

    train_datasets = create_per_class_datasets(train_paths, transform=train_transform, num_samples_per_dataset=num_samples_per_dataset)
    train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    
    val_set = [os.path.normpath(table).lstrip(os.sep) for table in val_set]
    val_paths = [os.path.join(dataset, table) for table in val_set]

    test_datasets = create_per_class_datasets(val_paths, transform=train_transform)
    test_dataset = ConcatDataset(test_datasets) if len(test_datasets) > 1 else test_datasets[0]

    labels = [train_dataset[idx][1] for idx in range(len(train_dataset))]  # Extract labels

    # Create balanced indices after combining datasets
    print("Creating balanced subset for training dataset...")
    balanced_indices = create_balanced_indices(labels) 

    print("Setting up Fabric...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        sampler=SubsetRandomSampler(balanced_indices),
        shuffle=False, 
        num_workers=4,
    )
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=4)

    print("Setting up DataLoaders...")
    train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)   

    model = resnet18_for_single_channel()

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_loader) * num_epochs),
    )

    model, optimizer, lr_scheduler = fabric.setup(model, optimizer, lr_scheduler)
    
    train(fabric, model, optimizer, loss_func, train_loader, val_loader=test_loader, lr_scheduler=lr_scheduler, num_epochs=num_epochs, num_classes=num_classes)

    logger.finalize("success")
    
    state = { # fabric will automatically unwrap the state_dict() when necessary
        "model": model, 
        "optimizer": optimizer, 
        "lr_scheduler": lr_scheduler, 
        "epoch": num_epochs
    }
    fabric.save(model_path, state)
    
    print('Finished Training')

if __name__ == "__main__":
    import argparse
    
    def parse_num_samples(value):
        """Convert 'None' to None and other values to integers."""
        return None if value.lower() == 'none' else int(value)

    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('dataset', help='Path to the root image folder')
    parser.add_argument('--train_set', type=str, nargs='+', default=None, help=(
        'path(s) to training image folders.'))
    parser.add_argument('--val_set', type=str, nargs='+', default=None, help=(
        'path(s) to validation image folders.'))
    parser.add_argument('--num_samples_per_dataset', type=parse_num_samples, nargs='+', default=None, help="List specifying the number of samples to extract from each dataset. If None, extract all samples.")
    parser.add_argument('--output_folder', default=None, type=str, help='Folder to save the trained model.')
    parser.add_argument('--train_batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Batch size for evaluation.')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--input_shape', type=int, nargs='+', default=[128, 128], help='Input shape as width and height (e.g., --input_shape 128 128).')
    parser.add_argument('--model_name', type=str, default='my_model', help='name of the model')
    parser.add_argument('--versioning', action='store_true', help='If set, the model name will be versioned (e.g., v_0, v_1, etc.) based on the models already saved in the output path.')
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    args = parser.parse_args()

    if len(args.input_shape) == 1:
        input_shape = (args.input_shape[0], args.input_shape[0])
    elif len(args.input_shape) == 2:
        input_shape = tuple(args.input_shape) 
    else:
        parser.error("--input_shape must be one or two integers.")

    main(args.dataset, train_set=args.train_set, val_set=args.val_set, 
         num_samples_per_dataset= args.num_samples_per_dataset, output_folder=args.output_folder, 
         train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size,
         num_epochs=args.num_epochs, input_shape=input_shape, 
         model_name=args.model_name, versioning=args.versioning, seed=args.seed)
