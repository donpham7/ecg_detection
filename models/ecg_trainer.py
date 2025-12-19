import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.ecg_cnn import ECGCNN, ECGCNN_MoE
from torch.utils.data import DataLoader, TensorDataset
import polars as pl
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    multilabel_confusion_matrix,
)


def specificity_score_multiclass(y_true, y_pred):
    # Get a separate confusion matrix for each class
    mcm = multilabel_confusion_matrix(y_true, y_pred)

    # Extract TN and FP for each class from the confusion matrices
    # For a binary confusion matrix:
    # [[TN, FP],
    #  [FN, TP]]
    tn = mcm[:, 0, 0]
    fp = mcm[:, 0, 1]

    # Calculate specificity for each class: TN / (TN + FP)
    # Use np.divide and np.nan_to_num to handle potential zero divisions gracefully
    specificity_per_class = np.divide(
        tn, (tn + fp), out=np.zeros_like(tn, dtype=float), where=(tn + fp) != 0
    )

    # Return the average specificity across all classes (macro-average)
    return np.mean(specificity_per_class)


class ECGTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_data: pl.DataFrame,
        test_data: pl.DataFrame,
        input_length=187,
        num_classes=5,
        learning_rate=0.001,
        criterion=nn.CrossEntropyLoss,
        batch_size=32,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
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
        self.test_data = DataLoader(
            TensorDataset(
                torch.tensor(
                    test_data.drop("label").to_numpy(), dtype=torch.float32
                ).unsqueeze(1),
                torch.tensor(test_data["label"].to_numpy(), dtype=torch.long),
            ),
            batch_size=batch_size,
            shuffle=True,
        )

    def train(self, num_epochs=10):
        self.model.train()
        if type(self.model) == ECGCNN_MoE:
            self.model.training = True  # Enable training-specific layers like dropout
        for epoch in range(num_epochs):
            total_loss = 0
            for inputs, labels in self.train_data:
                loss = self.train_step(inputs, labels)
                total_loss += loss
            avg_loss = total_loss / len(self.train_data)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    def train_step(self, inputs, labels):
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        balance_loss = 0
        self.model.train()
        self.optimizer.zero_grad()
        if type(self.model) == ECGCNN_MoE:
            outputs, balance_loss = self.model(inputs)
        else:
            outputs = self.model(inputs)
        loss = self.criterion(outputs, labels) + balance_loss
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self):
        self.model.eval()
        if type(self.model) == ECGCNN_MoE:
            self.model.training = False  # Disable training-specific layers like dropout
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
                balance_loss = 0
                if type(self.model) == ECGCNN_MoE:
                    outputs, balance_loss = self.model(inputs)
                else:
                    outputs = self.model(inputs)

                probs = torch.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())

                loss = self.criterion(outputs, labels) + balance_loss
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.test_data)
        accuracy = correct / total

        # Build confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)

        # Calculate various metrics
        f1_macro = f1_score(all_labels, all_predictions, average="macro")
        precision_macro = precision_score(
            all_labels, all_predictions, average="macro", zero_division=0
        )
        recall_macro = recall_score(
            all_labels, all_predictions, average="macro", zero_division=0
        )
        specificity_macro = specificity_score_multiclass(all_labels, all_predictions)
        # Per-class metrics (useful for imbalanced ECG datasets)
        f1_per_class = f1_score(
            all_labels, all_predictions, average=None, zero_division=0
        )
        precision_per_class = precision_score(
            all_labels, all_predictions, average=None, zero_division=0
        )
        recall_per_class = recall_score(
            all_labels, all_predictions, average=None, zero_division=0
        )

        # ROC AUC (one-vs-rest for multiclass)
        try:
            all_probs_array = np.array(all_probs)
            roc_auc_ovr = roc_auc_score(
                all_labels, all_probs_array, multi_class="ovr", average="macro"
            )
            roc_auc_ovo = roc_auc_score(
                all_labels, all_probs_array, multi_class="ovo", average="macro"
            )
        except:
            roc_auc_ovr = None
            roc_auc_ovo = None

        metrics = {
            "avg_loss": avg_loss,
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "f1_macro": f1_macro,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "specificity_macro": specificity_macro,
            "f1_per_class": f1_per_class.tolist(),
            "precision_per_class": precision_per_class.tolist(),
            "recall_per_class": recall_per_class.tolist(),
            "roc_auc_ovr": roc_auc_ovr,
            "roc_auc_ovo": roc_auc_ovo,
        }

        return metrics
