from collections import Counter
import sys
import numpy as np
import polars as pl

sys.path.append("..")
import models.ecg_cnn as cnn
import models.ecg_trainer as trainer
import torch
import time
import json
from imblearn.over_sampling import SMOTE
import torch.nn as nn
import torch.nn.functional as F
from models.focal_loss import FocalLoss
import helpers.visualizations as viz
from helpers.data_augmentation import SimpleAugmentor
from helpers.data_helper import prepare_data, set_all_seeds
import pickle

SEEDS = [1, 7, 21, 41, 89]
RUN_SETTINGS = [
    {
        "loss_function": "focal_loss",  # Options: "cross_entropy", "focal_loss"
        "focal_loss_alpha": 1,
        "focal_loss_gamma": 3.0,
        "learning_rate": 0.001,
        "epochs": 25,
        "model": cnn.ECGCNN,
        "batch_size": 128,
        "layers_count": 5,
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
    },
    # Optimized settings for CNN 1M
    {
        "loss_function": "focal_loss",  # Options: "cross_entropy", "focal_loss"
        "focal_loss_alpha": 1,
        "focal_loss_gamma": 3.0,
        "learning_rate": 0.0005,
        "epochs": 25,
        "model": cnn.ECGCNN_1M,
        "batch_size": 128,
        "layers_count": 5,
        "augmentation_technique": "base",  # Options: "smote", "base", "adasyn", "none"
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
    },
    {
        # Optimal settings for CNN 3M
        "loss_function": "focal_loss",  # Options: "cross_entropy", "focal_loss"
        "focal_loss_alpha": 1,
        "focal_loss_gamma": 3.0,
        "learning_rate": 0.001,
        "epochs": 25,
        "model": cnn.ECGCNN_3M,
        "batch_size": 128,
        "layers_count": 5,
        "augmentation_technique": "base",  # Options: "smote", "base", "adasyn", "none"
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
    },
    {
        # Optimal settings for CNN 4M
        "loss_function": "focal_loss",  # Options: "cross_entropy", "focal_loss"
        "focal_loss_alpha": 1,
        "focal_loss_gamma": 3.0,
        "learning_rate": 0.001,
        "epochs": 25,
        "model": cnn.ECGCNN_4M,
        "batch_size": 128,
        "layers_count": 5,
        "augmentation_technique": "base",  # Options: "smote", "base", "adasyn", "none"
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
    },
    # Optimal settings for CNN MoE
    {
        "loss_function": "focal_loss",  # Options: "cross_entropy", "focal_loss"
        "focal_loss_alpha": 1,
        "focal_loss_gamma": 3.0,
        "learning_rate": 0.0005,
        "epochs": 25,
        "model": cnn.ECGCNN_MoE,
        "batch_size": 128,
        "layers_count": 5,
        "augmentation_technique": "base",  # Options: "smote", "base", "adasyn", "none"
        "sampling_strategy": {
            0: 60000,
            1: 27000,
            2: 18000,
            3: 14000,  # Reduce (too many synthetic → overprediction)
            4: 16000,
        },
        "sampling_config": {
            "jitter": True,
            "jitter_rate": 0.8,  # 80% chance per sample
            "jitter_sigma": 0.08,
            "scale": True,
            "scale_rate": 0.7,
            "scale_low": 0.8,
            "scale_high": 1.2,
            "shift": True,
            "shift_rate": 0.7,
            "shift_max": 5,
        },
    },
    # Optimal settings for CNN MoE Large
    {
        "loss_function": "focal_loss",  # Options: "cross_entropy", "focal_loss"
        "focal_loss_alpha": 1,
        "focal_loss_gamma": 3.0,
        "learning_rate": 0.0005,
        "epochs": 25,
        "model": cnn.ECGCNN_MoE_Large,
        "batch_size": 128,
        "layers_count": 5,
        "augmentation_technique": "base",  # Options: "smote", "base", "adasyn", "none"
        "sampling_strategy": {
            0: 60000,
            1: 27000,
            2: 18000,
            3: 14000,  # Reduce (too many synthetic → overprediction)
            4: 16000,
        },
        "sampling_config": {
            "jitter": True,
            "jitter_rate": 0.8,  # 80% chance per sample
            "jitter_sigma": 0.08,
            "scale": True,
            "scale_rate": 0.7,
            "scale_low": 0.8,
            "scale_high": 1.2,
            "shift": True,
            "shift_rate": 0.7,
            "shift_max": 5,
        },
    },
    {
        "loss_function": "focal_loss",  # Options: "cross_entropy", "focal_loss"
        "focal_loss_alpha": 1,
        "focal_loss_gamma": 3.0,
        "learning_rate": 0.0005,
        "epochs": 25,
        "model": cnn.ECGCNN_MoE_Small,
        "batch_size": 128,
        "layers_count": 5,
        "augmentation_technique": "base",  # Options: "smote", "base", "adasyn", "none"
        "sampling_strategy": {
            0: 60000,
            1: 27000,
            2: 18000,
            3: 14000,  # Reduce (too many synthetic → overprediction)
            4: 16000,
        },
        "sampling_config": {
            "jitter": True,
            "jitter_rate": 0.8,  # 80% chance per sample
            "jitter_sigma": 0.08,
            "scale": True,
            "scale_rate": 0.7,
            "scale_low": 0.8,
            "scale_high": 1.2,
            "shift": True,
            "shift_rate": 0.7,
            "shift_max": 5,
        },
    },
]


def main():

    for settings in RUN_SETTINGS:
        if settings["model"] != cnn.ECGCNN:
            continue  # Only run MoE for now
        metrics_list = []
        for seed in SEEDS:
            print(f"\n\n=== Running with seed: {seed} ===")
            settings["seed"] = seed
            # Prepare data and set seeds
            train_data, validation_data, test_data = prepare_data(
                config=settings,
                balance=settings["augmentation_technique"] != "none",
            )
            set_all_seeds(settings["seed"])

            # Initialize model, criterion, and trainer
            model = settings["model"](num_conv_layers=settings["layers_count"])
            print("Model Size:", sum(p.numel() for p in model.parameters()))

            criterion = None
            if settings["loss_function"] == "focal_loss":
                criterion = FocalLoss(
                    alpha=settings["focal_loss_alpha"],
                    gamma=settings["focal_loss_gamma"],
                )
            else:
                criterion = nn.CrossEntropyLoss()

            ecg_trainer = trainer.ECGTrainer(
                model,
                train_data=train_data,
                validation_data=validation_data,
                test_data=test_data,
                criterion=criterion,
                learning_rate=settings["learning_rate"],
                batch_size=settings["batch_size"],
            )

            # Train and evaluate the model
            start_time = time.time()
            ecg_trainer.train(num_epochs=settings["epochs"])
            end_time = time.time()
            training_time = end_time - start_time

            start_time = time.time()
            metrics = ecg_trainer.evaluate()
            end_time = time.time()
            evaluation_time = end_time - start_time

            # Print results
            print("Final Evaluation Metrics:")
            for key, value in metrics.items():
                if key != "confusion_matrix":
                    print(f"{key}: {value}")
            print(f"Training time (s): {training_time}")
            print(f"Evaluation time (s): {evaluation_time}")

            # Convert non-serializable objects to serializable types
            serializable_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    serializable_metrics[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    serializable_metrics[key] = value.item()
                else:
                    serializable_metrics[key] = value

            metrics_entry = {
                "seed": seed,
                "metrics": serializable_metrics,
                "training_time": training_time,
                "evaluation_time": evaluation_time,
                "config": {
                    k: v if not isinstance(v, type) else v.__name__
                    for k, v in settings.items()
                },
            }
            metrics_list.append(metrics_entry)

        try:
            # Save all metrics to a file
            with open(f"metrics_{settings['model'].__name__}.json", "w") as f:
                json.dump(metrics_list, f, indent=2)
        except Exception as e:
            print(f"Error saving metrics to file: {e}")
            # Save object as pkl if json fails
            with open("metrics.pkl", "wb") as f:
                pickle.dump(metrics_list, f)
        # Save model weights to .pt file
        try:
            torch.save(
                model.state_dict(),
                f"model_{settings['model'].__name__}_final.pt",
            )
        except Exception as e:
            print(f"Error saving model weights: {e}")


if __name__ == "__main__":
    main()
