import numpy as np


class SimpleAugmentor:
    def __init__(self, sampling_config: dict, sampling_strategy: dict, seed: int = 42):
        self.sampling_config = sampling_config
        self.sampling_strategy = sampling_strategy
        self.seed = seed
        np.random.seed(self.seed)

    def add_jitter(self, x: np.ndarray, sigma: float = 0.03) -> np.ndarray:
        x_jittered = x + np.random.normal(0, sigma, x.shape)
        return x_jittered

    def augment_scale(
        self, x: np.ndarray, low: float = 0.8, high: float = 1.2
    ) -> np.ndarray:
        scale = np.random.uniform(low, high)
        x_scaled = x * scale
        return x_scaled

    def augment_shift(self, x: np.ndarray, shift_max: int = 5) -> np.ndarray:
        shift = np.random.randint(-shift_max, shift_max)
        x_shifted = np.roll(x, shift)
        return x_shifted

    def augment_datapoint(self, x: np.ndarray) -> np.ndarray:
        isJitter = self.sampling_config.get("jitter", False)
        isScale = self.sampling_config.get("scale", False)
        isShift = self.sampling_config.get("shift", False)

        jitterRate = self.sampling_config.get("jitter_rate", 0.5)
        jitterSigma = self.sampling_config.get("jitter_sigma", 0.03)

        scaleRate = self.sampling_config.get("scale_rate", 0.5)
        scaleLow = self.sampling_config.get("scale_low", 0.8)
        scaleHigh = self.sampling_config.get("scale_high", 1.2)

        shiftRate = self.sampling_config.get("shift_rate", 0.5)
        shiftMax = self.sampling_config.get("shift_max", 5)

        if isJitter and np.random.rand() < jitterRate:
            x = self.add_jitter(x, sigma=jitterSigma)
        if isScale and np.random.rand() < scaleRate:
            x = self.augment_scale(x, low=scaleLow, high=scaleHigh)
        if isShift and np.random.rand() < shiftRate:
            x = self.augment_shift(x, shift_max=shiftMax)

        return x

    def augment_dataset(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        X_augmented = []
        y_augmented = []

        for class_label, target_count in self.sampling_strategy.items():
            X_class = X[y == class_label]
            current_count = X_class.shape[0]
            samples_needed = target_count - current_count

            if samples_needed <= 0:
                # Downsample if needed
                X_class = X_class[
                    np.random.choice(current_count, target_count, replace=False)
                ]
                X_augmented.append(X_class)
                y_augmented.extend([class_label] * target_count)
                continue

            for _ in range(samples_needed):
                idx = np.random.randint(0, current_count)
                x_original = X_class[idx]
                x_augmented = self.augment_datapoint(x_original)
                X_augmented.append(x_augmented)
                y_augmented.append(class_label)

        return np.concatenate([X, np.array(X_augmented)]), np.concatenate(
            [y, np.array(y_augmented)]
        )
