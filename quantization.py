import json
import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.quantization as quantization
from helpers.data_helper import load_data, prepare_data
import models.ecg_cnn as cnn
import torch.nn as nn
from models.focal_loss import FocalLoss
import models.ecg_trainer as trainer
import os

RUN_SETTINGS = {
    "model": cnn.ECGCNN_MoE,
    "quantization_bits": 8,
    "dtype": torch.qint8,
    "seed": 42,
}


def quantize_saved_model(
    model_path: str, model_class: nn.Module, dtype: torch.dtype
) -> nn.Module:
    # Load the model
    model = model_class(device="cpu")
    model_weights = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(model_weights)
    model.eval()

    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv1d},  # Only quantize Linear layers
        dtype=dtype,  # 8-bit integers
    )

    # Print size of the original
    original_size = os.path.getsize(model_path)
    print(f"Original model size: {original_size} bytes")

    return quantized_model


def evaluate_quantized_model(
    model: nn.Module,
    test_data: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_data:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_data)
    accuracy = correct / total
    f1_macro = f1_score(all_labels, all_predictions, average="macro")
    return avg_loss, accuracy, f1_macro


def main():
    quantized_model = quantize_saved_model(
        model_path=f"model_{RUN_SETTINGS['model'].__name__}_final.pt",
        model_class=RUN_SETTINGS["model"],
        dtype=RUN_SETTINGS["dtype"],
    )
    device = "cpu"
    quantized_model.to(device)
    # Load test data and criterion as per your existing setup
    train_data, validation_data, test_data = prepare_data(
        config={}, balance=False, seed=RUN_SETTINGS["seed"]
    )
    ecg_trainer = trainer.ECGTrainer(
        model=quantized_model,
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
        criterion=FocalLoss(alpha=1.0, gamma=3.0),
        device=device,
    )
    metrics = ecg_trainer.evaluate()

    # Save to pt file
    saved_file = f"quantized_model_{RUN_SETTINGS['model'].__name__}_{RUN_SETTINGS['quantization_bits']}.pt"
    torch.save(
        quantized_model.state_dict(),
        saved_file,
    )

    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            serializable_metrics[key] = value.item()
        else:
            serializable_metrics[key] = value

    # Get size of the quantized model
    size_in_bytes = os.path.getsize(saved_file)
    serializable_metrics["model_size_bytes"] = size_in_bytes

    metrics_entry = {
        "seed": RUN_SETTINGS["seed"],
        "metrics": serializable_metrics,
    }

    # save metrics to json file
    with open(
        f"quantization_metrics_{RUN_SETTINGS['model'].__name__}_{RUN_SETTINGS['quantization_bits']}.json",
        "w",
    ) as f:
        json.dump(metrics_entry, f)


if __name__ == "__main__":
    main()
