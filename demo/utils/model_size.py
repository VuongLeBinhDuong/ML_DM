import torch
import os
import numpy as np
from typing import Union, Dict, Any

def get_model_size(model: Union[torch.nn.Module, Dict[str, Any]], format: str = 'MB') -> float:
    """
    Calculate the size of a PyTorch model or state dict in specified format
    
    Args:
        model: PyTorch model or state dict
        format: Size format ('B', 'KB', 'MB', 'GB')
    
    Returns:
        float: Size of the model in specified format
    """
    # Convert model to state dict if it's a nn.Module
    if isinstance(model, torch.nn.Module):
        state_dict = model.state_dict()
    else:
        state_dict = model
    
    # Calculate total size in bytes
    total_size = 0
    for param in state_dict.values():
        total_size += param.nelement() * param.element_size()
    
    # Convert to specified format
    if format == 'B':
        return total_size
    elif format == 'KB':
        return total_size / 1024
    elif format == 'MB':
        return total_size / (1024 * 1024)
    elif format == 'GB':
        return total_size / (1024 * 1024 * 1024)
    else:
        raise ValueError("Format must be one of: 'B', 'KB', 'MB', 'GB'")

def print_model_info(model: Union[torch.nn.Module, Dict[str, Any]], model_name: str = "Model"):
    """
    Print detailed information about model size and parameters
    
    Args:
        model: PyTorch model or state dict
        model_name: Name of the model for display
    """
    # Get model size in different formats
    size_b = get_model_size(model, 'B')
    size_kb = get_model_size(model, 'KB')
    size_mb = get_model_size(model, 'MB')
    size_gb = get_model_size(model, 'GB')
    
    # Get number of parameters
    if isinstance(model, torch.nn.Module):
        total_params = sum(p.numel() for p in model.parameters())
    else:
        total_params = sum(p.numel() for p in model.values())
    
    # Print information
    print(f"\n{model_name} Information:")
    print("-" * 50)
    print(f"Total Parameters: {total_params:,}")
    print(f"Model Size:")
    print(f"  - Bytes: {size_b:,.2f} B")
    print(f"  - Kilobytes: {size_kb:,.2f} KB")
    print(f"  - Megabytes: {size_mb:,.2f} MB")
    print(f"  - Gigabytes: {size_gb:.4f} GB")
    print("-" * 50)

def get_file_size(file_path: str, format: str = 'MB') -> float:
    """
    Get the size of a file in specified format
    
    Args:
        file_path: Path to the file
        format: Size format ('B', 'KB', 'MB', 'GB')
    
    Returns:
        float: Size of the file in specified format
    """
    size_bytes = os.path.getsize(file_path)
    
    if format == 'B':
        return size_bytes
    elif format == 'KB':
        return size_bytes / 1024
    elif format == 'MB':
        return size_bytes / (1024 * 1024)
    elif format == 'GB':
        return size_bytes / (1024 * 1024 * 1024)
    else:
        raise ValueError("Format must be one of: 'B', 'KB', 'MB', 'GB'")

def print_file_info(file_path: str, file_name: str = None):
    """
    Print detailed information about file size
    
    Args:
        file_path: Path to the file
        file_name: Name of the file for display (defaults to filename)
    """
    if file_name is None:
        file_name = os.path.basename(file_path)
    
    # Get file size in different formats
    size_b = get_file_size(file_path, 'B')
    size_kb = get_file_size(file_path, 'KB')
    size_mb = get_file_size(file_path, 'MB')
    size_gb = get_file_size(file_path, 'GB')
    
    # Print information
    print(f"\n{file_name} File Information:")
    print("-" * 50)
    print(f"File Size:")
    print(f"  - Bytes: {size_b:,.2f} B")
    print(f"  - Kilobytes: {size_kb:,.2f} KB")
    print(f"  - Megabytes: {size_mb:,.2f} MB")
    print(f"  - Gigabytes: {size_gb:.4f} GB")
    print("-" * 50) 