import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import datetime
import random

import torch
from torch import nn
from torch.optim import Adam
from transformers import BertModel, BertTokenizer

from training.dataset import build_dataloaders
from utils.paths import BEST_MODEL_PATH, SAVES_DIR, TRAIN_LOG_DIR, ensure_runtime_directories
from utils.write_csv import write_csv_from_list

_TOKENIZER = None
_BERT_BACKBONE = None


def get_tokenizer() -> BertTokenizer:
    """Lazy-load and cache the BERT tokenizer.

    Args:
        None.

    Returns:
        BertTokenizer: Tokenizer instance for bert-base-multilingual-uncased.
    """
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
    return _TOKENIZER


def get_bert_backbone() -> BertModel:
    """Lazy-load and cache the BERT backbone model.

    Args:
        None.

    Returns:
        BertModel: Pretrained bert-base-multilingual-uncased backbone.
    """
    global _BERT_BACKBONE
    if _BERT_BACKBONE is None:
        _BERT_BACKBONE = BertModel.from_pretrained("bert-base-multilingual-uncased")
    return _BERT_BACKBONE


class BertRegressionModel(nn.Module):
    """BERT pooled_output -> MLP regressor for success-rate prediction."""

    def __init__(self):
        """Initialize the regression head on top of the BERT backbone.

        Args:
            None.

        Returns:
            None: Builds the model layers in-place.
        """
        super().__init__()
        self.bert = get_bert_backbone()
        self.regressor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask):
        """Run forward inference and produce one regression score per sample.

        Args:
            input_ids: Token id tensor with shape ``[batch, seq_len]``.
            attention_mask: Attention mask tensor matching ``input_ids`` shape.

        Returns:
            torch.Tensor: Regression output tensor with shape ``[batch, 1]``.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.regressor(outputs.pooler_output)


def _build_inputs(text_batch, tokenizer: BertTokenizer, device):
    """Tokenize a batch of texts and move tensors to target device.

    Args:
        text_batch: Iterable of input question strings.
        tokenizer: Tokenizer used to encode the texts.
        device: Torch device where tensors should be placed.

    Returns:
        tuple: ``(input_ids, attention_mask)`` tensors on ``device``.
    """
    inputs = tokenizer(list(text_batch), return_tensors="pt", padding=True, truncation=True, max_length=512)
    return inputs["input_ids"].to(device), inputs["attention_mask"].to(device)


def _compute_train_mean_label(train_dataloader) -> float:
    """Compute mean label value from the training dataloader.

    Args:
        train_dataloader: Dataloader yielding ``(text, label)`` batches.

    Returns:
        float: Mean of all training labels.
    """
    mean_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for _, labels in train_dataloader:
            labels = labels.float()
            mean_sum += labels.sum().item()
            sample_count += labels.numel()
    return mean_sum / max(sample_count, 1)


def training(epochs: int = 100, threshold: float = 0.1, lr: float = 1e-5) -> None:
    """Train the BERT regression model and persist checkpoints/logs.

    Args:
        epochs: Number of training epochs.
        threshold: Absolute-error threshold used for accuracy-like reporting.
        lr: Learning rate for Adam optimizer.

    Returns:
        None: Saves logs and model checkpoints to runtime directories.

    Raises:
        ValueError: If the training dataloader is empty.
        RuntimeError: If output and label batch sizes mismatch.
    """
    random.seed(42)
    torch.manual_seed(42)

    ensure_runtime_directories()
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    log_file_name = TRAIN_LOG_DIR / f"log_{timestamp}.csv"

    (
        train_dataloader_success_rate,
        val_dataloader_success_rate,
        test_dataloader_success_rate,
        split_stats,
    ) = build_dataloaders()

    if len(train_dataloader_success_rate) == 0:
        raise ValueError("Training dataloader is empty. Please check input data.")

    model = BertRegressionModel()
    tokenizer = get_tokenizer()
    device = torch.device("cpu")
    model.to(device)

    loss_function = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    best_val_error = float("inf")
    best_model_path = BEST_MODEL_PATH

    print(
        f"Dataset split stats -> total: {split_stats['dataset_size']}, "
        f"train/val/test: {split_stats['train_size']}/{split_stats['val_size']}/{split_stats['test_size']}, "
        f"overlap(train,val/test,val-test): "
        f"{split_stats['train_val_overlap']}/{split_stats['train_test_overlap']}/{split_stats['val_test_overlap']}"
    )
    if split_stats["val_size"] < 5 or split_stats["test_size"] < 5:
        print("[Warning] Validation/Test set is very small; metrics can look unrealistically perfect.")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_predictions_train = 0
        total_samples_train = 0
        sum_abs_error_train = 0.0

        for x, y in train_dataloader_success_rate:
            y = y.float().to(device)
            total_samples_train += len(y)

            input_ids, attention_mask = _build_inputs(x, tokenizer, device)
            outputs = model(input_ids, attention_mask)
            if outputs.shape[0] != y.shape[0]:
                raise RuntimeError(f"Batch mismatch: outputs={outputs.shape}, labels={y.shape}")

            loss = loss_function(outputs.float(), y.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            abs_error = torch.abs(outputs.flatten() - y)
            correct_predictions_train += (abs_error < threshold).sum().item()
            sum_abs_error_train += abs_error.sum().item()

        train_accuracy = correct_predictions_train / max(total_samples_train, 1)
        train_avg_abs_error = sum_abs_error_train / max(total_samples_train, 1)
        train_loss_avg = train_loss / max(len(train_dataloader_success_rate), 1)

        print(
            f"Epoch: {epoch}, Train Loss: {train_loss_avg}, "
            f"Train Accuracy: {train_accuracy}, Train Average Error: {train_avg_abs_error}"
        )

        model.eval()
        correct_predictions_val = 0
        sum_abs_error_val = 0.0
        total_samples_val = 0

        with torch.no_grad():
            for x, y in val_dataloader_success_rate:
                y = y.float().to(device)
                total_samples_val += len(y)
                input_ids, attention_mask = _build_inputs(x, tokenizer, device)
                outputs = model(input_ids, attention_mask)
                predicted = outputs.flatten()

                abs_error = torch.abs(predicted - y)
                correct_predictions_val += (abs_error < threshold).sum().item()
                sum_abs_error_val += abs_error.sum().item()

        if total_samples_val > 0:
            val_accuracy = correct_predictions_val / total_samples_val
            val_avg_abs_error = sum_abs_error_val / total_samples_val
            train_mean_y = _compute_train_mean_label(train_dataloader_success_rate)

            val_baseline_abs_error = 0.0
            with torch.no_grad():
                for _, y_val in val_dataloader_success_rate:
                    y_val = y_val.float()
                    val_baseline_abs_error += torch.abs(y_val - train_mean_y).sum().item()
            val_baseline_abs_error = val_baseline_abs_error / max(total_samples_val, 1)

            print(
                f" Validation Accuracy: {val_accuracy}, Validation Average Error: {val_avg_abs_error}, "
                f"Val Baseline Avg Error: {val_baseline_abs_error}"
            )

            if val_avg_abs_error < best_val_error:
                best_val_error = val_avg_abs_error
                torch.save(model.state_dict(), best_model_path)
                print(f" ---> New best model saved! (Val Error: {best_val_error:.4f})")
        else:
            val_accuracy = 0.0
            val_avg_abs_error = 0.0
            print(" Validation Skipped (no samples)")

        write_csv_from_list(
            str(log_file_name),
            [epoch, train_loss_avg, train_accuracy, train_avg_abs_error, val_accuracy, val_avg_abs_error],
        )

        if epoch % 10 == 9:
            checkpoint_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            checkpoint_path = SAVES_DIR / f"model_{epoch}_{checkpoint_timestamp}.pth"
            torch.save(model.state_dict(), checkpoint_path)

    if best_model_path.exists():
        print(f"\nLoading best model from {best_model_path} for testing...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    model.eval()
    correct_predictions = 0
    sum_abs_error_test = 0.0
    total_samples_test = 0

    with torch.no_grad():
        for x, y in test_dataloader_success_rate:
            y = y.float().to(device)
            total_samples_test += len(y)

            input_ids, attention_mask = _build_inputs(x, tokenizer, device)
            outputs = model(input_ids, attention_mask)
            predicted = outputs.flatten()

            abs_error = torch.abs(predicted - y)
            correct_predictions += (abs_error < threshold).sum().item()
            sum_abs_error_test += abs_error.sum().item()
            print(f"Predicted: {outputs}, Actual: {y}")

        if total_samples_test > 0:
            test_accuracy = correct_predictions / total_samples_test
            test_avg_abs_error = sum_abs_error_test / total_samples_test
            print(f"Test Accuracy: {test_accuracy}, Test Average Absolute Error: {test_avg_abs_error}")
            write_csv_from_list(str(log_file_name), [test_accuracy, test_avg_abs_error])
        else:
            print("Test Skipped (no samples)")

    final_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    final_model_path = SAVES_DIR / f"model_final_{final_timestamp}.pth"
    torch.save(model.state_dict(), final_model_path)


if __name__ == "__main__":
    training()
