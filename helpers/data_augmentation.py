from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as parameterizations
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns


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

    def augment_cutout(self, x: np.ndarray, size: int = 10) -> np.ndarray:
        h = len(x)
        y = np.random.randint(0, h)
        # Cutout 1D array
        x_cutout = x.copy()
        x_cutout[y : y + size] = 0

        return x_cutout

    def augment_datapoint(self, x: np.ndarray) -> np.ndarray:
        isJitter = self.sampling_config.get("jitter", False)
        isScale = self.sampling_config.get("scale", False)
        isShift = self.sampling_config.get("shift", False)
        isCutout = self.sampling_config.get("cutout", False)

        jitterRate = self.sampling_config.get("jitter_rate", 0.5)
        jitterSigma = self.sampling_config.get("jitter_sigma", 0.03)

        scaleRate = self.sampling_config.get("scale_rate", 0.5)
        scaleLow = self.sampling_config.get("scale_low", 0.8)
        scaleHigh = self.sampling_config.get("scale_high", 1.2)

        shiftRate = self.sampling_config.get("shift_rate", 0.5)
        shiftMax = self.sampling_config.get("shift_max", 5)

        cutoutRate = self.sampling_config.get("cutout_rate", 0.5)
        cutoutCount = self.sampling_config.get("cutout_size", 10)

        if isJitter and np.random.rand() < jitterRate:
            x = self.add_jitter(x, sigma=jitterSigma)
        if isScale and np.random.rand() < scaleRate:
            x = self.augment_scale(x, low=scaleLow, high=scaleHigh)
        if isShift and np.random.rand() < shiftRate:
            x = self.augment_shift(x, shift_max=shiftMax)
        if isCutout and np.random.rand() < cutoutRate:
            x = self.augment_cutout(x, size=cutoutCount)

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


class GANGenerator_FF(nn.Module):
    def __init__(self, noise_dim: int, signal_length: int):
        super(GANGenerator_FF, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noise_dim = noise_dim
        self.signal_length = signal_length

        self.generator = nn.Sequential(
            parameterizations.spectral_norm(nn.Linear(self.noise_dim, 32)),
            nn.LeakyReLU(),
            parameterizations.spectral_norm(nn.Linear(32, 128)),
            nn.LeakyReLU(),
            parameterizations.spectral_norm(nn.Linear(128, self.signal_length)),
            nn.Tanh(),
        ).to(self.device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.to(self.device)
        out = self.generator(z)
        output = (out + 1) / 2
        return output


class GANDiscriminator_FF(nn.Module):
    def __init__(self, signal_length: int):
        super(GANDiscriminator_FF, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.signal_length = signal_length
        self.discriminator = nn.Sequential(
            nn.Linear(self.signal_length, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
        ).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        return self.discriminator(x)


class GANGenerator_LSTM(nn.Module):
    def __init__(self, noise_dim: int, signal_length: int, hidden_dim: int):
        super(GANGenerator_LSTM, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.noise_dim = noise_dim
        self.signal_length = signal_length
        self.hidden_dim = hidden_dim

        self.rnn = nn.LSTM(
            input_size=signal_length,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        ).to(self.device)
        self.layer1 = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.LeakyReLU(),
        ).to(self.device)
        self.layer2 = nn.Sequential(
            nn.Linear(256, 256), nn.LeakyReLU(), nn.Dropout(0.2)
        ).to(self.device)
        self.layer3 = nn.Sequential(nn.Linear(256, signal_length), nn.Sigmoid()).to(
            self.device
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.to(self.device)
        z = z.unsqueeze(1)
        rnn_out, _ = self.rnn(z)
        out = self.layer1(rnn_out[:, -1, :])
        out = self.layer2(out)
        out = self.layer3(out)
        return out


class GANDiscriminator_LSTM(nn.Module):
    def __init__(self, signal_length: int, hidden_dim: int):
        super(GANDiscriminator_LSTM, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rnn = nn.LSTM(
            input_size=signal_length,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        ).to(self.device)
        self.layer1 = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.LeakyReLU(),
        ).to(self.device)
        self.layer2 = nn.Sequential(
            nn.Linear(256, 256), nn.LeakyReLU(), nn.Dropout(0.2)
        ).to(self.device)
        self.layer3 = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid()).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        rnn_out, _ = self.rnn(x)
        out = self.layer1(rnn_out[:, -1, :])
        out = self.layer2(out)
        out = self.layer3(out)
        return out


class GAN:
    def __init__(
        self,
        noise_dim: int,
        signal_length: int,
        random_seed: int,
        generator: nn.Module,
        discriminator: nn.Module,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noise_dim = noise_dim
        self.signal_length = signal_length
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)
        # Generator
        self.generator = generator
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=0.0005,
            betas=(0.5, 0.999),  # Add momentum for stability
        )

        # Discriminator
        self.discriminator = discriminator
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=0.0001,
            betas=(0.5, 0.999),  # Add momentum for stability
        )

        # Loss function
        self.criterion = nn.BCELoss()

    def train(
        self, dataloader: DataLoader, num_epochs: int, save_path=None, visualizer=False
    ):
        training_history = {
            "g_loss": [],
            "d_loss": [],
            "d_real_acc": [],
            "d_fake_acc": [],
        }
        for epoch in range(num_epochs):
            g_losses = []
            d_losses = []
            d_real_accs = []
            d_fake_accs = []

            self.generator.train()
            self.discriminator.train()
            for i, real_data in enumerate(dataloader):
                data, labels = real_data
                loss_disc, loss_gen, disc_real_acc, disc_fake_acc = self.train_step(
                    data,
                    discriminator_steps=1,
                    generator_steps=1,
                )
                g_losses.append(loss_gen)
                d_losses.append(loss_disc)
                d_real_accs.append(disc_real_acc)
                d_fake_accs.append(disc_fake_acc)

            # Log epoch results
            training_history["g_loss"].append(np.mean(g_losses))
            training_history["d_loss"].append(np.mean(d_losses))
            training_history["d_real_acc"].append(np.mean(d_real_accs))
            training_history["d_fake_acc"].append(np.mean(d_fake_accs))

            print(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"G Loss: {training_history['g_loss'][-1]:.4f}, "
                f"D Loss: {training_history['d_loss'][-1]:.4f}, "
                f"D Real Acc: {training_history['d_real_acc'][-1]:.4f}, "
                f"D Fake Acc: {training_history['d_fake_acc'][-1]:.4f}"
            )

            if visualizer and epoch % 100 == 0:
                self.visualize_samples()

        if save_path:
            # Save Generator and Discriminator models
            torch.save(self.generator.state_dict(), f"{save_path}_generator.pt")
            torch.save(self.discriminator.state_dict(), f"{save_path}_discriminator.pt")

        return training_history

    def train_step(
        self, real_data: torch.Tensor, discriminator_steps: int, generator_steps: int
    ):
        batch_size = real_data.size(0)

        # Train discriminator
        for _ in range(discriminator_steps):
            z_disc = torch.randn(batch_size, self.noise_dim).to(self.device)
            fake_data_discriminator = self.generator(z_disc).detach()
            self.discriminator.zero_grad()

            # Add noise for discriminator
            noise = torch.randn_like(real_data).to(self.device) * 0.1
            real_data.to(self.device)
            fake_data_discriminator.to(self.device)

            # Real data + noise
            real_data_noisy = real_data.to(self.device) + noise.to(self.device)
            real_pred_disc = self.discriminator(real_data_noisy)

            # Fake data + noise
            fake_data_noisy = fake_data_discriminator + noise.to(self.device)
            fake_pred_disc = self.discriminator(fake_data_noisy)

            # Compute losses
            real_labels = (
                torch.FloatTensor(batch_size, 1).uniform_(0.95, 1.0).to(self.device)
            )
            fake_labels = (
                torch.FloatTensor(batch_size, 1).uniform_(0.0, 0.05).to(self.device)
            )
            real_loss_disc = self.criterion(real_pred_disc, real_labels)
            fake_loss_disc = self.criterion(fake_pred_disc, fake_labels)
            loss_disc = (real_loss_disc + fake_loss_disc) / 2

            loss_disc.backward()
            self.discriminator_optimizer.step()

        # Train generator
        for _ in range(generator_steps):
            self.generator.zero_grad()

            # Generate fake data
            z_gen = torch.randn(batch_size, self.noise_dim).to(self.device)
            fake_data_gen = self.generator(z_gen)
            fake_pred_gen = self.discriminator(fake_data_gen)
            loss_gen = self.criterion(fake_pred_gen, torch.ones_like(fake_pred_gen))
            loss_gen.backward()
            self.generator_optimizer.step()

        disc_real_acc = (real_pred_disc > 0.5).float().mean().item()
        disc_fake_acc = (fake_pred_disc < 0.5).float().mean().item()
        return loss_disc.item(), loss_gen.item(), disc_real_acc, disc_fake_acc

    def generate_samples(self, num_samples: int) -> np.ndarray:
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.generator.noise_dim).to(self.device)
            samples = self.generator(z)
        return samples.cpu().numpy()

    def visualize_samples(self, num_samples: int = 5):
        samples = self.generate_samples(num_samples)

        # Visualize 1D samples on one graph with different colors
        plt.figure(figsize=(15, 5))
        for i in range(num_samples):
            x_ms = np.array(range(187)) * 8
            sns.lineplot(x=x_ms, y=samples[i].flatten())
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude")
        plt.title("Generated Samples")
        plt.show()

    def load_pretrained_models(self, model_prefix: str):
        self.generator.load_state_dict(torch.load(f"{model_prefix}_generator.pt"))
        self.discriminator.load_state_dict(
            torch.load(f"{model_prefix}_discriminator.pt")
        )
