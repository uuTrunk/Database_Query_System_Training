from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from training.process_data import process_output_list
from utils.paths import ASK_GRAPH_LOG_CSV


class DatasetSuccessRate(Dataset):
    """PyTorch dataset where x is question text and y is success_rate."""

    def __init__(self, data_dict: Dict[str, list]):
        """Initialize dataset from processed question statistics.

        Args:
            data_dict: Mapping from question text to aggregated metric values.

        Returns:
            None: Stores keys and values for indexed access.
        """
        self.keys = list(data_dict.keys())
        self.values = [value for value in data_dict.values()]

    def __len__(self) -> int:
        """Return total number of question samples.

        Args:
            None.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.keys)

    def __getitem__(self, idx: int):
        """Fetch one sample by index.

        Args:
            idx: Zero-based sample index.

        Returns:
            tuple: ``(question_text, success_rate)``.
        """
        x = self.keys[idx]
        y = self.values[idx][1]
        return x, y


def _subset_texts(subset) -> set:
    """Extract question texts contained in a ``random_split`` subset.

    Args:
        subset: A torch ``Subset`` object produced by ``random_split``.

    Returns:
        set: Set of question strings in the subset.
    """
    return {subset.dataset.keys[idx] for idx in subset.indices}


def _split_sizes(total_size: int) -> Tuple[int, int, int]:
    """Compute train/validation/test split sizes with small-data safeguards.

    Args:
        total_size: Total number of available samples.

    Returns:
        tuple[int, int, int]: ``(train_size, val_size, test_size)``.

    Raises:
        ValueError: If ``total_size`` is zero or negative.
    """
    if total_size <= 0:
        raise ValueError("Dataset is empty. Please generate logs before training.")

    if total_size < 5:
        return total_size, 0, 0

    val_size = max(1, int(0.1 * total_size))
    test_size = max(1, int(0.1 * total_size))
    train_size = total_size - val_size - test_size
    if train_size <= 0:
        train_size = max(total_size - 2, 1)
        remaining = total_size - train_size
        val_size = 1 if remaining > 0 else 0
        test_size = max(remaining - val_size, 0)
    return train_size, val_size, test_size


def build_dataloaders(
    output_file: Path = ASK_GRAPH_LOG_CSV,
    batch_size: int = 8,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """Build train/validation/test dataloaders from graph request logs.

    Args:
        output_file: Path to the aggregated ask-graph CSV log file.
        batch_size: Batch size for all generated dataloaders.
        seed: Random seed used for deterministic dataset splitting.

    Returns:
        tuple: ``(train_loader, val_loader, test_loader, split_stats)`` where
        ``split_stats`` contains sizes and overlap diagnostics.

    Raises:
        FileNotFoundError: If the log file does not exist.
        ValueError: If no valid training records are available.
    """
    output_path = Path(output_file)
    if not output_path.exists():
        raise FileNotFoundError(f"Training log file not found: {output_path}")

    data_dict = process_output_list(str(output_path))
    if not data_dict:
        raise ValueError(f"No valid records found in training log: {output_path}")

    dataset_success_rate = DatasetSuccessRate(data_dict)
    total_size = len(dataset_success_rate)
    train_size, val_size, test_size = _split_sizes(total_size)

    split_generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset_success_rate,
        [train_size, val_size, test_size],
        generator=split_generator,
    )

    train_texts = _subset_texts(train_dataset)
    val_texts = _subset_texts(val_dataset)
    test_texts = _subset_texts(test_dataset)

    split_stats = {
        "dataset_size": len(dataset_success_rate),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "train_val_overlap": len(train_texts & val_texts),
        "train_test_overlap": len(train_texts & test_texts),
        "val_test_overlap": len(val_texts & test_texts),
    }

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader, split_stats


# Backward-compatible module-level names.
train_dataloader_success_rate = None
val_dataloader_success_rate = None
test_dataloader_success_rate = None
split_stats = {}


def initialize_default_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """Populate legacy module-level dataloaders for backward compatibility.

    Args:
        None.

    Returns:
        tuple: ``(train_loader, val_loader, test_loader, split_stats)`` from
        ``build_dataloaders()``.
    """
    global train_dataloader_success_rate
    global val_dataloader_success_rate
    global test_dataloader_success_rate
    global split_stats

    (
        train_dataloader_success_rate,
        val_dataloader_success_rate,
        test_dataloader_success_rate,
        split_stats,
    ) = build_dataloaders()
    return (
        train_dataloader_success_rate,
        val_dataloader_success_rate,
        test_dataloader_success_rate,
        split_stats,
    )


if __name__ == "__main__":
    _, _, _, stats = initialize_default_dataloaders()
    print(stats)
