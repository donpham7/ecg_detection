from random import random
from typing import Counter
import polars as pl
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN
import torch
from helpers.data_augmentation import SimpleAugmentor


def load_data(config: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    train_data = pl.read_csv("data/mitbih_train.csv", has_header=False)
    train_data = train_data.rename({"column_188": "label"})

    test_data = pl.read_csv("data/mitbih_test.csv", has_header=False)
    test_data = test_data.rename({"column_188": "label"})
    return train_data, test_data


def balance_data(data: pl.DataFrame, config: dict, seed: int) -> pl.DataFrame:
    # Convert to numpy arrays for SMOTE
    X = data.drop("label").to_numpy().astype(np.float32)
    y = data["label"].to_numpy().astype(np.int64)

    print(f"Original class distribution: {Counter(y)}")
    if config["augmentation_technique"] == "smote":
        print("Applying augmentation technique: SMOTE")
        smote = SMOTE(
            sampling_strategy=config["sampling_strategy"],
            k_neighbors=3,
            random_state=seed,
        )
        X_resampled, y_resampled = smote.fit_resample(X, y)
    elif config["augmentation_technique"] == "base":
        print("Applying augmentation technique: BASE")
        augmentor = SimpleAugmentor(
            sampling_config=config["sampling_config"],
            sampling_strategy=config["sampling_strategy"],
            seed=seed,
        )
        X_resampled, y_resampled = augmentor.augment_dataset(X, y)
    elif config["augmentation_technique"] == "adasyn":
        print("Applying augmentation technique: ADASYN")
        adasyn = ADASYN(
            sampling_strategy="auto",
            n_neighbors=3,
            random_state=seed,
        )
        X_resampled, y_resampled = adasyn.fit_resample(X, y)
    else:
        X_resampled, y_resampled = X, y

    print(f"Resampled class distribution: {Counter(y_resampled)}")
    feature_cols = [f"column_{i}" for i in range(X_resampled.shape[1])]

    resampled_data = pl.DataFrame(X_resampled, schema=feature_cols)
    resampled_data = resampled_data.with_columns(pl.Series("label", y_resampled))

    return resampled_data


def prepare_data(
    config: dict, balance: bool = True, seed: int = 42
) -> tuple[pl.DataFrame, pl.DataFrame]:
    # Set all seeds for reproducibility
    set_all_seeds(seed)

    train_data, test_data = load_data(config)
    if balance:
        train_data = balance_data(train_data, config, seed=seed)
    return train_data, test_data


def set_all_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
