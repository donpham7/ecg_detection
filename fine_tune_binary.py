import sys
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import json
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
)
import io

sys.path.append("..")
import models.ecg_cnn as cnn
from models.focal_loss import FocalLoss
from helpers.data_helper import prepare_data, set_all_seeds, load_data

SEEDS = [1, 7, 21, 41, 89]

# Configuration for binary classification
BINARY_SETTINGS = {
    "base_model": cnn.ECGCNN,  # Which model to load
    "layers_count": 5,
    "base_model_path": "model_ECGCNN_final.pt",  # Path to saved weights
    "learning_rate": 0.0001,  # Lower LR for fine-tuning
    "epochs": 10,  # Fewer epochs for fine-tuning
    "batch_size": 128,
    "freeze_backbone": True,  # Freeze all layers except final classification
    "loss_function": "focal_loss",
    "focal_loss_alpha": 1.0,
    "focal_loss_gamma": 3.0,
    "binary_task": "normal_vs_abnormal",  # Options: "normal_vs_abnormal", "custom"
    "augmentation_technique": "base",  # Options: "smote", "base", "none"
    "sampling_strategy": {
        0: 60000,
        1: 25000,
        2: 18000,
        3: 12000,  # Reduce (too many synthetic → overprediction)
        4: 16000,
    },
    "sampling_config": {
        "jitter": True,
        "jitter_rate": 0.8,  # 80% chance per sample
        "jitter_sigma": 0.08,  # Gentler noise
        "scale": True,
        "scale_rate": 0.7,
        "scale_low": 0.8,
        "scale_high": 1.2,
        "shift": True,
        "shift_rate": 0.7,
        "shift_max": 5,
    },
    "quantize_post_finetune": True,
    # For custom binary classification, define mapping:
    # "class_mapping": {0: 0, 1: 1, 2: 1, 3: 1, 4: 1},  # Normal=0, All others=1
}


def convert_to_binary_labels(
    data: pl.DataFrame, task: str = "normal_vs_abnormal"
) -> pl.DataFrame:
    """
    Convert multi-class labels to binary.

    Args:
        data: DataFrame with 'label' column
        task: Type of binary classification
            - "normal_vs_abnormal": Normal (0) vs All abnormal (1,2,3,4)
            - "custom": Use custom mapping from BINARY_SETTINGS

    Returns:
        DataFrame with binary labels (0 or 1)
    """
    if task == "normal_vs_abnormal":
        # Class 0 (Normal) = 0, All others (Abnormal) = 1
        binary_data = data.with_columns(
            pl.when(pl.col("label") == 0).then(0).otherwise(1).alias("label")
        )
        print(f"Binary conversion: Normal (0) vs Abnormal (1)")

    elif task == "custom" and "class_mapping" in BINARY_SETTINGS:
        # Custom mapping
        mapping = BINARY_SETTINGS["class_mapping"]
        print(f"Custom binary mapping: {mapping}")

        # Apply mapping
        binary_data = data.with_columns(
            pl.col("label")
            .map_elements(lambda x: mapping.get(x, 0), return_dtype=pl.Int64)
            .alias("label")
        )
    else:
        raise ValueError(f"Unknown binary task: {task}")

    # Print class distribution
    label_counts = binary_data.group_by("label").agg(pl.count()).sort("label")
    print(f"Binary class distribution:")
    for row in label_counts.iter_rows():
        print(f"  Class {row[0]}: {row[1]} samples")

    return binary_data


def modify_model_for_binary(
    model: nn.Module, freeze_backbone: bool = True
) -> nn.Module:
    """
    Modify a multi-class model for binary classification.

    Args:
        model: Trained multi-class model
        freeze_backbone: If True, freeze all layers except final classifier

    Returns:
        Modified model with binary output
    """
    # Freeze all parameters first
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        print("✅ Backbone frozen - only training final layer")
    else:
        print("⚠️  Fine-tuning entire model")

    # Find and replace the final layer
    # Different models have different final layer names
    in_features = 0
    # Find the last fc layer and replace it
    if hasattr(model, "fc3"):
        # Models with fc3 (3-layer classifier)
        in_features = model.fc3.in_features
        model.fc3 = nn.Linear(in_features, 2)  # Binary: 2 classes
        in_features = model.fc2.in_features
        out_features = model.fc2.out_features
        model.fc2 = nn.Linear(in_features, out_features)  # Adjust fc2 as needed

        print(f"Replaced fc3 (final layer): {in_features} -> 2")
    elif hasattr(model, "fc2"):
        # Models with fc2 (2-layer classifier)
        in_features = model.fc2.in_features
        model.fc2 = nn.Linear(in_features, 2)  # Binary: 2 classes
        print(f"Replaced fc2 (final layer): {in_features} -> 2")
        in_features = model.fc1.in_features
        out_features = model.fc1.out_features
        model.fc1 = nn.Linear(in_features, out_features)  # Adjust fc1 as needed
        print(f"Adjusted fc1 layer: {in_features} -> {out_features}")
    else:
        raise ValueError("Model does not have fc2 or fc3 layer")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100*trainable_params/total_params:.2f}%)"
    )

    return model


def load_pretrained_model(
    model_class: nn.Module,
    model_path: str,
    freeze_backbone: bool = True,
    device: str = "cuda",
) -> nn.Module:
    """
    Load a pretrained model and prepare it for binary fine-tuning.
    """
    # Initialize model with original architecture
    model = model_class(
        num_classes=5, num_conv_layers=BINARY_SETTINGS["layers_count"]
    )  # Original had 5 classes

    # Load pretrained weights
    print(f"Loading pretrained weights from: {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print("✅ Pretrained weights loaded")

    # Modify for binary classification
    model = modify_model_for_binary(model, freeze_backbone)

    model = model.to(device)
    return model


class BinaryECGTrainer:
    """Trainer for binary classification fine-tuning"""

    def __init__(
        self,
        model: nn.Module,
        train_data: pl.DataFrame,
        validation_data: pl.DataFrame,
        test_data: pl.DataFrame,
        criterion: nn.Module,
        learning_rate: float = 0.0001,
        batch_size: int = 128,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)

        # Only optimize unfrozen parameters
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            weight_decay=1e-4,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )

        # Create dataloaders
        self.train_data = DataLoader(
            TensorDataset(
                torch.tensor(
                    train_data.drop("label").to_numpy(), dtype=torch.float32
                ).unsqueeze(1),
                torch.tensor(train_data["label"].to_numpy(), dtype=torch.long),
            ),
            batch_size=batch_size,
            shuffle=True,
        )

        self.validation_data = DataLoader(
            TensorDataset(
                torch.tensor(
                    validation_data.drop("label").to_numpy(), dtype=torch.float32
                ).unsqueeze(1),
                torch.tensor(validation_data["label"].to_numpy(), dtype=torch.long),
            ),
            batch_size=batch_size,
            shuffle=False,
        )

        self.test_data = DataLoader(
            TensorDataset(
                torch.tensor(
                    test_data.drop("label").to_numpy(), dtype=torch.float32
                ).unsqueeze(1),
                torch.tensor(test_data["label"].to_numpy(), dtype=torch.long),
            ),
            batch_size=batch_size,
            shuffle=False,
        )

    def train(self, num_epochs: int = 10):
        """Train the binary classifier"""
        best_val_loss = float("inf")
        patience_counter = 0
        patience = 5

        for epoch in range(num_epochs):
            # Training
            self.model.train()
            total_loss = 0

            for inputs, labels in self.train_data:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                # Handle MoE models that return aux_loss
                if self.model.__class__.__name__ in [
                    "ECGCNN_MoE",
                    "ECGCNN_MoE_Large",
                    "ECGCNN_MoE_Small",
                ]:
                    outputs, _ = self.model(inputs)
                else:
                    outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_data)

            # Validation
            val_loss, val_accuracy, val_f1 = self.validate()

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            print(
                f"Epoch [{epoch+1}/{num_epochs}], "
                f"Train Loss: {avg_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Accuracy: {val_accuracy:.4f}, "
                f"Val F1: {val_f1:.4f}"
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.validation_data:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                if self.model.__class__.__name__ in [
                    "ECGCNN_MoE",
                    "ECGCNN_MoE_Large",
                    "ECGCNN_MoE_Small",
                ]:
                    outputs, _ = self.model(inputs)
                else:
                    outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.validation_data)
        accuracy = correct / total
        f1 = f1_score(all_labels, all_predictions, average="binary")

        return avg_loss, accuracy, f1

    def evaluate(self):
        """Evaluate on test set"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in self.test_data:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                if self.model.__class__.__name__ in [
                    "ECGCNN_MoE",
                    "ECGCNN_MoE_Large",
                    "ECGCNN_MoE_Small",
                ]:
                    outputs, _ = self.model(inputs)
                else:
                    outputs = self.model(inputs)

                probs = torch.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())

                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / len(self.test_data)
        accuracy = correct / total
        cm = confusion_matrix(all_labels, all_predictions)

        # Binary metrics
        f1 = f1_score(all_labels, all_predictions, average="binary")
        precision = precision_score(
            all_labels, all_predictions, average="binary", zero_division=0
        )
        recall = recall_score(
            all_labels, all_predictions, average="binary", zero_division=0
        )

        # Specificity (True Negative Rate)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # ROC AUC
        all_probs_array = np.array(all_probs)
        roc_auc = roc_auc_score(
            all_labels, all_probs_array[:, 1]
        )  # Prob of positive class

        metrics = {
            "avg_loss": avg_loss,
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "roc_auc": roc_auc,
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }

        return metrics


def quantize_model(model: nn.Module, device: str = "cpu") -> nn.Module:
    """Quantize the model using dynamic quantization"""
    print("Quantizing model...")
    model.to("cpu")  # Quantization works on CPU
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_before = buffer.getbuffer().nbytes
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    print("✅ Model quantized")

    # print size reduction
    buffer = io.BytesIO()
    torch.save(quantized_model.state_dict(), buffer)
    size_after = buffer.getbuffer().nbytes
    reduction = 100 * (size_before - size_after) / size_before
    print(
        f"Model size before: {size_before / 1e6:.2f} MB, "
        f"after: {size_after / 1e6:.2f} MB "
        f"({reduction:.2f}% reduction)"
    )
    return quantized_model.to(device)


def main():
    """Main fine-tuning pipeline"""

    metrics_list = []
    for seed in SEEDS:
        print(f"\n{'='*80}")
        print(f"Running binary fine-tuning with seed: {seed}")
        print(f"{'='*80}\n")

        set_all_seeds(seed)

        # Load and convert data to binary
        print("Loading data...")
        train_data, validation_data, test_data = prepare_data(
            config=BINARY_SETTINGS, balance=True, seed=seed
        )

        train_data = convert_to_binary_labels(
            train_data, BINARY_SETTINGS["binary_task"]
        )
        validation_data = convert_to_binary_labels(
            validation_data, BINARY_SETTINGS["binary_task"]
        )
        test_data = convert_to_binary_labels(test_data, BINARY_SETTINGS["binary_task"])

        # Load pretrained model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nUsing device: {device}")

        model = load_pretrained_model(
            model_class=BINARY_SETTINGS["base_model"],
            model_path=BINARY_SETTINGS["base_model_path"],
            freeze_backbone=BINARY_SETTINGS["freeze_backbone"],
            device=device,
        )

        # Setup criterion
        if BINARY_SETTINGS["loss_function"] == "focal_loss":
            criterion = FocalLoss(
                alpha=BINARY_SETTINGS["focal_loss_alpha"],
                gamma=BINARY_SETTINGS["focal_loss_gamma"],
            )
        else:
            criterion = nn.CrossEntropyLoss()

        # Create trainer
        trainer = BinaryECGTrainer(
            model=model,
            train_data=train_data,
            validation_data=validation_data,
            test_data=test_data,
            criterion=criterion,
            learning_rate=BINARY_SETTINGS["learning_rate"],
            batch_size=BINARY_SETTINGS["batch_size"],
            device=device,
        )

        # Train
        print("\nStarting fine-tuning...")
        start_time = time.time()
        trainer.train(num_epochs=BINARY_SETTINGS["epochs"])
        training_time = time.time() - start_time

        # Evaluate
        print("\nEvaluating on test set...")
        start_time = time.time()
        metrics = trainer.evaluate()
        evaluation_time = time.time() - start_time

        # Print results
        print("\n" + "=" * 80)
        print("BINARY CLASSIFICATION RESULTS")
        print("=" * 80)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall (Sensitivity): {metrics['recall']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN={metrics['true_negatives']}, FP={metrics['false_positives']}")
        print(f"  FN={metrics['false_negatives']}, TP={metrics['true_positives']}")
        print(f"\nTraining time: {training_time:.2f}s")
        print(f"Evaluation time: {evaluation_time:.2f}s")
        print("=" * 80)

        # Save metrics
        serializable_metrics = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in metrics.items()
        }

        metrics_entry = {
            "seed": seed,
            "metrics": serializable_metrics,
            "training_time": training_time,
            "evaluation_time": evaluation_time,
        }

        metrics_list.append(metrics_entry)

    # Save to file
    output_file = f"binary_metrics_{BINARY_SETTINGS['base_model'].__name__}.json"
    with open(output_file, "w") as f:
        json.dump(metrics_list, f, indent=2)

    print(f"\n✅ Metrics saved to: {output_file}")

    # Save fine-tuned model
    model_file = f"binary_model_{BINARY_SETTINGS['base_model'].__name__}.pt"
    torch.save(model.state_dict(), model_file)
    print(f"✅ Model saved to: {model_file}\n")


if __name__ == "__main__":
    main()
