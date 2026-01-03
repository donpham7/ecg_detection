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

RUN_SETTINGS = (
    {
        "loss_function": "focal_loss",  # Options: "cross_entropy", "focal_loss"
        "focal_loss_alpha": 1,
        "focal_loss_gamma": 2.5,
        "learning_rate": 0.0001,
        "epochs": 25,
        "model": cnn.ECGCNN_MoE,
        "class_count": 2,
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
            "jitter_sigma": 0.1,
            "scale": True,
            "scale_rate": 0.7,
            "scale_low": 0.7,
            "scale_high": 1.3,
            "shift": True,
            "shift_rate": 0.7,
            "shift_max": 10,
        },
    },
)


def load_ptbdb_data(data_path: str):
    normal_data = pl.read_csv(f"data/ptbdb_normal.csv", has_header=False)
    abnormal_data = pl.read_csv(f"data/ptbdb_abnormal.csv", has_header=False)
    return normal_data, abnormal_data


def prepare_ptbdb_data(normal_data: pl.DataFrame, abnormal_data: pl.DataFrame):
    # Label normal data as 0 and abnormal data as 1
    normal_data = normal_data.with_columns(pl.lit(0).alias("label"))
    abnormal_data = abnormal_data.with_columns(pl.lit(1).alias("label"))
    return normal_data, abnormal_data


def combine_and_shuffle_data(normal_data: pl.DataFrame, abnormal_data: pl.DataFrame):
    combined_data = pl.concat([normal_data, abnormal_data])
    return combined_data.sample(
        fraction=1.0, with_replacement=False
    )  # Shuffle the data


def evaluate_model_from_pt(
    model_path: str, test_data: pl.DataFrame, batch_size=32, device=None
):
    model = cnn.ECGCNN_MoE(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    ecg_trainer = trainer.ECGTrainer(
        model=model,
        train_data=test_data,  # Dummy, not used
        validation_data=test_data,  # Dummy, not used
        test_data=test_data,
        device=device,
        batch_size=batch_size,
        criterion=FocalLoss(alpha=1, gamma=3.0).to(device),
    )
    metrics = ecg_trainer.evaluate()
    return metrics


def main():
    set_all_seeds(42)
    normal_data, abnormal_data = load_ptbdb_data("data/ptbdb")
    normal_data, abnormal_data = prepare_ptbdb_data(normal_data, abnormal_data)
    combined_data = combine_and_shuffle_data(normal_data, abnormal_data)

    model_path = "model_binary_ECGCNN_MoE_final.pt"
    metrics = evaluate_model_from_pt(model_path, combined_data)
    print("Evaluation Metrics:", metrics)


if __name__ == "__main__":
    main()
