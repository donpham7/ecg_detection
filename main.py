from collections import Counter
import random
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

SEEDS = [42]
RUN_SETTINGS = {
    "loss_function": "focal_loss",  # Options: "cross_entropy", "focal_loss"
    "focal_loss_alpha": 1,
    "focal_loss_gamma": 3.0,
    "learning_rate": 0.0001,
    "epochs": 100,
    "model": cnn.ECGCNN_1M,
    "batch_size": 128,
    "layers_count": 5,
    "augmentation_technique": "base",  # Options: "smote", "base", "adasyn", "none"
    "sampling_strategy": {
        0: 72472,
        1: 23000,
        2: 21000,
        3: 9000,  # Reduce (too many synthetic → overprediction)
        4: 19000,
    },
    "sampling_config": {
        "jitter": True,
        "jitter_rate": 0.8,  # 80% chance per sample
        "jitter_sigma": [0.02, 0.08, 0.15],
        "scale": True,
        "scale_rate": 0.7,
        "scale_low": [0.8, 0.9],
        "scale_high": [1.1, 1.2],
        "shift": True,
        "shift_rate": 0.7,
        "shift_max": [5, 15],
    },
}


def random_search(n_trials=100, seed=None):
    """
    Generate n_trials random configurations from RUN_SETTINGS

    Args:
        n_trials: Number of random configurations to generate
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)

    # Extract list parameters
    gamma_values = RUN_SETTINGS["focal_loss_gamma"]
    ss1_values = RUN_SETTINGS["sampling_strategy"][1]
    ss2_values = RUN_SETTINGS["sampling_strategy"][2]
    ss3_values = RUN_SETTINGS["sampling_strategy"][3]
    ss4_values = RUN_SETTINGS["sampling_strategy"][4]
    sigma_values = RUN_SETTINGS["sampling_config"]["jitter_sigma"]

    for i in range(n_trials):
        config = {
            "loss_function": RUN_SETTINGS["loss_function"],
            "focal_loss_alpha": RUN_SETTINGS["focal_loss_alpha"],
            "focal_loss_gamma": random.choice(gamma_values),
            "learning_rate": RUN_SETTINGS["learning_rate"],
            "epochs": RUN_SETTINGS["epochs"],
            "model": RUN_SETTINGS["model"],
            "batch_size": RUN_SETTINGS["batch_size"],
            "layers_count": RUN_SETTINGS["layers_count"],
            "augmentation_technique": RUN_SETTINGS["augmentation_technique"],
            "sampling_strategy": {
                0: RUN_SETTINGS["sampling_strategy"][0],
                1: random.choice(ss1_values),
                2: random.choice(ss2_values),
                3: random.choice(ss3_values),
                4: random.choice(ss4_values),
            },
            "sampling_config": {
                "jitter": True,
                "jitter_rate": 1.0,
                "jitter_sigma": random.choice(sigma_values),
                "scale": True,
                "scale_rate": 1.0,
                "scale_low": RUN_SETTINGS["sampling_config"]["scale_low"],
                "scale_high": RUN_SETTINGS["sampling_config"]["scale_high"],
                "shift": True,
                "shift_rate": 1.0,
                "shift_max": RUN_SETTINGS["sampling_config"]["shift_max"],
            },
        }

        yield config


def get_all_combinations():
    """
    Generate all combinations of parameters from RUN_SETTINGS
    """
    combinations = []
    jitter_sigmas = RUN_SETTINGS["sampling_config"]["jitter_sigma"]
    scale_lows = RUN_SETTINGS["sampling_config"]["scale_low"]
    scale_highs = RUN_SETTINGS["sampling_config"]["scale_high"]
    shift_maxs = RUN_SETTINGS["sampling_config"]["shift_max"]

    for sigma in jitter_sigmas:
        for scale_low in scale_lows:
            for scale_high in scale_highs:
                for shift_max in shift_maxs:
                    config = {
                        "loss_function": RUN_SETTINGS["loss_function"],
                        "focal_loss_alpha": RUN_SETTINGS["focal_loss_alpha"],
                        "focal_loss_gamma": RUN_SETTINGS["focal_loss_gamma"],
                        "learning_rate": RUN_SETTINGS["learning_rate"],
                        "epochs": RUN_SETTINGS["epochs"],
                        "model": RUN_SETTINGS["model"],
                        "batch_size": RUN_SETTINGS["batch_size"],
                        "layers_count": RUN_SETTINGS["layers_count"],
                        "augmentation_technique": RUN_SETTINGS[
                            "augmentation_technique"
                        ],
                        "sampling_strategy": RUN_SETTINGS["sampling_strategy"],
                        "sampling_config": {
                            "jitter": True,
                            "jitter_rate": 1.0,
                            "jitter_sigma": sigma,
                            "scale": True,
                            "scale_rate": 1.0,
                            "scale_low": scale_low,
                            "scale_high": scale_high,
                            "shift": True,
                            "shift_rate": 1.0,
                            "shift_max": shift_max,
                        },
                    }
                    combinations.append(config)
    return combinations


def main():
    metrics_list = []
    for seed in SEEDS:
        for i, config in enumerate(get_all_combinations()):
            print(f"\n=== Trial {i+1} with seed {seed} ===")
            print("Configuration:", config)
            # Prepare data and set seeds
            train_data, test_data = prepare_data(
                config=config,
                balance=config["augmentation_technique"] != "none",
                seed=seed,
            )

            print(
                "Class distribution in training data:\n",
                train_data["label"].value_counts().sort("label"),
            )

            set_all_seeds(seed)

            # Initialize model, criterion, and trainer
            model = config["model"](num_conv_layers=config["layers_count"])

            criterion = None
            if config["loss_function"] == "focal_loss":
                criterion = FocalLoss(
                    alpha=config["focal_loss_alpha"],
                    gamma=config["focal_loss_gamma"],
                )
            else:
                criterion = nn.CrossEntropyLoss()

            ecg_trainer = trainer.ECGTrainer(
                model,
                train_data=train_data,
                test_data=test_data,
                criterion=criterion,
                learning_rate=config["learning_rate"],
                batch_size=config["batch_size"],
            )

            # Train and evaluate the model
            start_time = time.time()
            ecg_trainer.train(num_epochs=config["epochs"])
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

            metrics_list.append(metrics)

    # Save all metrics to a file
    with open("metrics.json", "w") as f:
        json.dump(metrics_list, f)


if __name__ == "__main__":
    main()
