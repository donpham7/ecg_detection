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
from sklearn.utils.class_weight import compute_class_weight


SEEDS = [1, 7, 21, 41, 89]
RUN_SETTINGS = [
    {
        "loss_function": "focal_loss",  # Options: "cross_entropy", "focal_loss"
        "focal_loss_alpha": 1,
        "focal_loss_gamma": 3.0,
        "learning_rate": 0.0001,
        "epochs": 100,
        "model": cnn.ECGCNN,
        "class_count": 5,
        "batch_size": 128,
        "layers_count": 5,
        "augmentation_technique": "none",  # Options: "smote", "base", "none"
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
            "cutout": True,
            "cutout_rate": 0.5,
            "cutout_size": 10,
        },
    },
    {
        "loss_function": "focal_loss",  # Options: "cross_entropy", "focal_loss"
        "focal_loss_alpha": 1,
        "focal_loss_gamma": 3.0,
        "learning_rate": 0.0001,
        "epochs": 100,
        "model": cnn.ECGCNN,
        "class_count": 5,
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
            "cutout": True,
            "cutout_rate": 0.5,
            "cutout_size": 10,
        },
    },
    {
        "loss_function": "focal_loss",  # Options: "cross_entropy", "focal_loss"
        "focal_loss_alpha": 1,
        "focal_loss_gamma": 3.0,
        "learning_rate": 0.0001,
        "epochs": 100,
        "model": cnn.ECGCNN,
        "class_count": 5,
        "batch_size": 128,
        "layers_count": 5,
        "augmentation_technique": "smote",  # Options: "smote", "base", "none"
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
            "cutout": True,
            "cutout_rate": 0.5,
            "cutout_size": 10,
        },
    },
    # Optimized settings for CNN 1M
    {
        "loss_function": "focal_loss",  # Options: "cross_entropy", "focal_loss"
        "focal_loss_alpha": 1,
        "focal_loss_gamma": 3.0,
        "learning_rate": 0.0001,
        "epochs": 100,
        "model": cnn.ECGCNN_1M,
        "class_count": 5,
        "batch_size": 128,
        "layers_count": 5,
        "augmentation_technique": "rgan",  # Options: "smote", "base", "adasyn", "none"
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
        "loss_function": "focal_loss",  # Options: "cross_entropy", "focal_loss"
        "focal_loss_alpha": 1,
        "focal_loss_gamma": 3.0,
        "learning_rate": 0.0001,
        "epochs": 100,
        "model": cnn.ECGCNN_1M,
        "class_count": 5,
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
        "loss_function": "focal_loss",  # Options: "cross_entropy", "focal_loss"
        "focal_loss_alpha": 1,
        "focal_loss_gamma": 3.0,
        "learning_rate": 0.0001,
        "epochs": 100,
        "model": cnn.ECGCNN_1M,
        "class_count": 5,
        "batch_size": 128,
        "layers_count": 5,
        "augmentation_technique": "smote",  # Options: "smote", "base", "adasyn", "none"
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
        "learning_rate": 0.0001,
        "epochs": 100,
        "model": cnn.ECGCNN_3M,
        "class_count": 5,
        "batch_size": 128,
        "layers_count": 5,
        "augmentation_technique": "rgan",  # Options: "smote", "base", "adasyn", "none"
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
        "learning_rate": 0.0001,
        "epochs": 100,
        "model": cnn.ECGCNN_3M,
        "class_count": 5,
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
        "learning_rate": 0.0001,
        "epochs": 100,
        "model": cnn.ECGCNN_3M,
        "class_count": 5,
        "batch_size": 128,
        "layers_count": 5,
        "augmentation_technique": "smote",  # Options: "smote", "base", "adasyn", "none"
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
        "epochs": 100,
        "model": cnn.ECGCNN_5M,
        "class_count": 5,
        "batch_size": 64,
        "layers_count": 5,
        "augmentation_technique": "rgan",  # Options: "smote", "base", "adasyn", "none"
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
        "epochs": 100,
        "model": cnn.ECGCNN_5M,
        "class_count": 5,
        "batch_size": 64,
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
            "jitter_rate": 0.5,  # Down from 0.8
            "jitter_sigma": 0.03,  # Down from 0.08
            "scale": True,
            "scale_rate": 0.5,
            "scale_low": 0.9,
            "scale_high": 1.1,
            "shift": True,
            "shift_rate": 0.5,
            "shift_max": 5,
        },
    },
    {
        # Optimal settings for CNN 4M
        "loss_function": "focal_loss",  # Options: "cross_entropy", "focal_loss"
        "focal_loss_alpha": 1,
        "focal_loss_gamma": 3.0,
        "learning_rate": 0.0001,
        "epochs": 100,
        "model": cnn.ECGCNN_5M,
        "class_count": 5,
        "batch_size": 128,
        "layers_count": 5,
        "augmentation_technique": "smote",  # Options: "smote", "base", "adasyn", "none"
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
        "focal_loss_gamma": 2.5,
        "learning_rate": 0.0005,
        "epochs": 100,
        "model": cnn.ECGCNN_MoE,
        "class_count": 5,
        "batch_size": 128,
        "layers_count": 5,
        "augmentation_technique": "rgan",  # Options: "smote", "base", "adasyn", "none"
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
    {
        "loss_function": "focal_loss",  # Options: "cross_entropy", "focal_loss"
        "focal_loss_alpha": 1,
        "focal_loss_gamma": 2.5,
        "learning_rate": 0.001,
        "epochs": 100,
        "model": cnn.ECGCNN_MoE,
        "class_count": 5,
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
    {
        "loss_function": "focal_loss",  # Options: "cross_entropy", "focal_loss"
        "focal_loss_alpha": 1,
        "focal_loss_gamma": 2.5,
        "learning_rate": 0.001,
        "epochs": 100,
        "model": cnn.ECGCNN_MoE,
        "class_count": 5,
        "batch_size": 128,
        "layers_count": 5,
        "augmentation_technique": "smote",  # Options: "smote", "base", "adasyn", "none"
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
    # Optimal settings for CNN MoE Small
    {
        "loss_function": "focal_loss",  # Options: "cross_entropy", "focal_loss"
        "focal_loss_alpha": 1,
        "focal_loss_gamma": 3.0,
        "learning_rate": 0.0001,
        "epochs": 100,
        "model": cnn.ECGCNN_MoE_Small,
        "class_count": 5,
        "batch_size": 128,
        "layers_count": 5,
        "augmentation_technique": "rgan",  # Options: "smote", "base", "adasyn", "none"
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
        "learning_rate": 0.0001,
        "epochs": 100,
        "model": cnn.ECGCNN_MoE_Small,
        "class_count": 5,
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
        "learning_rate": 0.0001,
        "epochs": 100,
        "model": cnn.ECGCNN_MoE_Small,
        "class_count": 5,
        "batch_size": 128,
        "layers_count": 5,
        "augmentation_technique": "smote",  # Options: "smote", "base", "adasyn", "none"
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
    # Optimal settings for CNN MoE Small
    {
        "loss_function": "focal_loss",  # Options: "cross_entropy", "focal_loss"
        "focal_loss_alpha": 1,
        "focal_loss_gamma": 3.0,
        "learning_rate": 0.0001,
        "epochs": 100,
        "model": cnn.ECGCNN_MoE_LSTM,
        "class_count": 5,
        "batch_size": 128,
        "layers_count": 5,
        "augmentation_technique": "rgan",  # Options: "smote", "base", "adasyn", "none"
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
        "learning_rate": 0.0001,
        "epochs": 100,
        "model": cnn.ECGCNN_MoE_LSTM,
        "class_count": 5,
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
        "learning_rate": 0.0001,
        "epochs": 100,
        "model": cnn.ECGCNN_MoE_LSTM,
        "class_count": 5,
        "batch_size": 128,
        "layers_count": 5,
        "augmentation_technique": "smote",  # Options: "smote", "base", "adasyn", "none"
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
        "learning_rate": 0.0001,
        "epochs": 100,
        "model": cnn.ECGCNN_MoE_Small_LSTM,
        "class_count": 5,
        "batch_size": 128,
        "layers_count": 5,
        "augmentation_technique": "rgan",  # Options: "smote", "base", "adasyn", "none"
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
        "learning_rate": 0.0001,
        "epochs": 100,
        "model": cnn.ECGCNN_MoE_Small_LSTM,
        "class_count": 5,
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
        "learning_rate": 0.0001,
        "epochs": 100,
        "model": cnn.ECGCNN_MoE_Small_LSTM,
        "class_count": 5,
        "batch_size": 128,
        "layers_count": 5,
        "augmentation_technique": "smote",  # Options: "smote", "base", "adasyn", "none"
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
        "learning_rate": 0.0001,
        "epochs": 100,
        "model": cnn.ECGCNN_LSTM,
        "class_count": 5,
        "batch_size": 128,
        "layers_count": 5,
        "augmentation_technique": "rgan",  # Options: "smote", "base", "adasyn", "none"
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
            "cutout": True,
            "cutout_rate": 0.5,
            "cutout_size": 10,
        },
    },
    {
        "loss_function": "focal_loss",  # Options: "cross_entropy", "focal_loss"
        "focal_loss_alpha": 1,
        "focal_loss_gamma": 3.0,
        "learning_rate": 0.0001,
        "epochs": 100,
        "model": cnn.ECGCNN_LSTM,
        "class_count": 5,
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
            "cutout": True,
            "cutout_rate": 0.5,
            "cutout_size": 10,
        },
    },
    {
        "loss_function": "focal_loss",  # Options: "cross_entropy", "focal_loss"
        "focal_loss_alpha": 1,
        "focal_loss_gamma": 3.0,
        "learning_rate": 0.0001,
        "epochs": 100,
        "model": cnn.ECGCNN_LSTM,
        "class_count": 5,
        "batch_size": 128,
        "layers_count": 5,
        "augmentation_technique": "smote",  # Options: "smote", "base", "adasyn", "none"
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
            "cutout": True,
            "cutout_rate": 0.5,
            "cutout_size": 10,
        },
    },
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    for settings in RUN_SETTINGS:
        torch.set_default_dtype(settings.get("torch_dtype", torch.float32))
        if settings["model"] != cnn.ECGCNN_MoE:
            continue  # Only run MoE for now
        metrics_list = []
        for seed in SEEDS:
            print(
                f"\n\n=== Running with seed: {seed} - Model: {settings['model'].__name__} - Augmentation: {settings['augmentation_technique']} ==="
            )
            settings["seed"] = seed
            # Prepare data and set seeds
            train_data, validation_data, test_data = prepare_data(
                config=settings,
                balance=settings["augmentation_technique"] != "none",
                isBinary=False,
            )
            set_all_seeds(settings["seed"])

            # Initialize model, criterion, and trainer
            model = settings["model"](
                num_conv_layers=settings["layers_count"],
                num_classes=settings["class_count"],
                device=device,
            )
            print("Model Size:", sum(p.numel() for p in model.parameters()))

            criterion = None
            if settings["loss_function"] == "focal_loss":
                criterion = FocalLoss(
                    alpha=settings["focal_loss_alpha"],
                    gamma=settings["focal_loss_gamma"],
                )
            else:
                class_labels = train_data["label"].unique().to_numpy()
                class_weights = compute_class_weight(
                    class_weight="balanced",
                    classes=class_labels,
                    y=train_data["label"].to_numpy(),
                )
                print("Class Weights:", class_weights)
                class_weight_tensor = torch.tensor(
                    class_weights, dtype=torch.float32
                ).to(device)
                criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)

            ecg_trainer = trainer.ECGTrainer(
                model,
                train_data=train_data,
                validation_data=validation_data,
                test_data=test_data,
                criterion=criterion,
                learning_rate=settings["learning_rate"],
                batch_size=settings["batch_size"],
                device=device,
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

            # Single inference time measurement
            num_warmup = 3
            num_samples = 10
            samples = test_data.drop("label").to_numpy()[0 : num_warmup + num_samples]

            model.eval()
            with torch.no_grad():
                # Warmup (GPU needs to "warm up" for accurate timing)
                for i in range(num_warmup):
                    sample_tensor = (
                        torch.tensor(samples[i], dtype=torch.float32)
                        .view(1, 1, -1)
                        .to(device)
                    )
                    _ = model(sample_tensor)

                # Actual timing
                if device.type == "cuda":
                    torch.cuda.synchronize()  # Ensure GPU finishes warmup

                total_inference_time = 0
                for i in range(num_warmup, num_warmup + num_samples):
                    sample_tensor = (
                        torch.tensor(samples[i], dtype=torch.float32)
                        .view(1, 1, -1)
                        .to(device)
                    )

                    if device.type == "cuda":
                        torch.cuda.synchronize()  # Wait for GPU before timing

                    start_time = time.time()
                    _ = model(sample_tensor)

                    if device.type == "cuda":
                        torch.cuda.synchronize()  # Wait for GPU to finish

                    end_time = time.time()
                    total_inference_time += end_time - start_time

            single_inference_time = total_inference_time / num_samples

            # Print results
            print("Final Evaluation Metrics:")
            for key, value in metrics.items():
                if key != "confusion_matrix":
                    print(f"{key}: {value}")
            print(f"Training time (s): {training_time}")
            print(f"Evaluation time (s): {evaluation_time}")
            print(f"Single inference time (s): {single_inference_time}")

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
                "single_inference_time": single_inference_time,
                "config": {
                    k: v if not isinstance(v, type) else v.__name__
                    for k, v in settings.items()
                },
            }
            metrics_list.append(metrics_entry)
        folder = "metrics_multiclass"
        model_name = (
            f"{settings['model'].__name__}_{settings['augmentation_technique']}"
        )
        try:
            # Save all metrics to a file
            with open(f"{folder}/metrics_multiclass_{model_name}.json", "w") as f:
                json.dump(metrics_list, f, indent=2)
        except Exception as e:
            print(f"Error saving metrics to file: {e}")
            # Save object as pkl if json fails
            with open(f"{folder}/metrics_multiclass.pkl", "wb") as f:
                pickle.dump(metrics_list, f)
        # Save model weights to .pt file
        try:
            torch.save(
                model.state_dict(),
                f"model_multiclass_{model_name}_final.pt",
            )
        except Exception as e:
            print(f"Error saving model weights: {e}")


if __name__ == "__main__":
    main()
