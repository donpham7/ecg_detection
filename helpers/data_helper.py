import os
from random import random
from typing import Counter
import polars as pl
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN
import torch
from helpers.data_augmentation import (
    GAN,
    GANDiscriminator_LSTM,
    GANGenerator_LSTM,
    SimpleAugmentor,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def load_data() -> tuple[pl.DataFrame, pl.DataFrame]:
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
    elif config["augmentation_technique"] == "rgan":
        print("Applying augmentation technique: RGAN")
        X_resampled, y_resampled = X, y
        for label in config["sampling_strategy"].keys():
            # Build Generator and Discriminator
            generator = GANGenerator_LSTM(
                noise_dim=187, signal_length=X.shape[1], hidden_dim=128
            )
            discriminator = GANDiscriminator_LSTM(
                signal_length=X.shape[1], hidden_dim=128
            )

            # Dataloader for class according to label
            label_dataset = TensorDataset(
                torch.tensor(X[y == label]), torch.tensor(y[y == label])
            )
            label_dataloader = DataLoader(
                label_dataset, batch_size=X[y == label].shape[0] // 100, shuffle=True
            )

            rgan = GAN(
                noise_dim=187,
                signal_length=X.shape[1],
                random_seed=seed,
                generator=generator,
                discriminator=discriminator,
            )

            model_prefix = f"label_{label}"
            gen_model_suffix, disc_model_suffix = "_generator.pt", "_discriminator.pt"
            # Check if .pt files exist for generator and discriminator
            if not os.path.exists(
                f"{model_prefix}{gen_model_suffix}"
            ) and not os.path.exists(f"{model_prefix}{disc_model_suffix}"):
                # Train RGAN
                print(f"Training RGAN for class {label}")
                rgan.train(
                    label_dataloader,
                    num_epochs=2500,
                    save_path=model_prefix,
                )
            else:
                print(f"Loading pre-trained models for class {label}")
                rgan.load_pretrained_models(model_prefix)

            # Generate synthetic samples
            num_samples = config["sampling_strategy"][label] - y[y == label].shape[0]
            print(f"Generating {num_samples} samples for class {label}")
            X_samples = rgan.generate_samples(num_samples)
            X_resampled = np.concatenate([X_resampled, X_samples])
            y_resampled = np.concatenate([y_resampled, np.full(num_samples, label)])
    else:
        X_resampled, y_resampled = X, y

    print(f"Resampled class distribution: {Counter(y_resampled)}")
    feature_cols = [f"column_{i}" for i in range(X_resampled.shape[1])]

    resampled_data = pl.DataFrame(X_resampled, schema=feature_cols)
    resampled_data = resampled_data.with_columns(pl.Series("label", y_resampled))

    return resampled_data


def prepare_data(
    config: dict, balance: bool = True, seed: int = 42, isBinary: bool = False
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    # Set all seeds for reproducibility
    set_all_seeds(seed)

    train_data, test_data = load_data()
    train_data, validation_data = train_test_split(
        train_data.to_pandas(),
        test_size=0.2,
        random_state=seed,
        stratify=train_data["label"].to_numpy(),
    )
    train_data = pl.DataFrame(train_data)
    validation_data = pl.DataFrame(validation_data)
    if balance:
        train_data = balance_data(train_data, config, seed=seed)
    if isBinary:
        train_data = train_data.with_columns(
            pl.when(pl.col("label") == 0).then(0).otherwise(1).alias("label")
        )
        validation_data = validation_data.with_columns(
            pl.when(pl.col("label") == 0).then(0).otherwise(1).alias("label")
        )
        test_data = test_data.with_columns(
            pl.when(pl.col("label") == 0).then(0).otherwise(1).alias("label")
        )
    return train_data, validation_data, test_data


def set_all_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
