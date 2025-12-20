import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ECGCNN(nn.Module):
    def __init__(self, input_length=187, num_classes=5, num_conv_layers=3):
        super(ECGCNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_norm_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        out_channels = 2
        resolution = input_length
        for _ in range(num_conv_layers):
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                )
            )
            self.batch_norm_layers.append(nn.BatchNorm1d(num_features=out_channels))
            in_channels = out_channels
            out_channels *= 2
            resolution = resolution + 2 * 2 - 5 + 1  # padding and kernel size effect
            resolution = ((resolution - 2) // 2) + 1  # max pooling effect
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(resolution * in_channels, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(p=0.03)
        self.conv_dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.to(self.device)
        for conv, bn in zip(self.conv_layers, self.batch_norm_layers):
            x = F.relu(conv(x))
            x = bn(x)
            x = self.pool(x)
            # x = self.conv_dropout(x)
        # size is (batch_size, 8, 23)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ECGCNN_1M(nn.Module):
    def __init__(self, input_length=187, num_classes=5, num_conv_layers=3):
        super(ECGCNN_1M, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear((input_length // 8) * 512, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ECGCNN_3M(nn.Module):
    def __init__(self, input_length=187, num_classes=5, num_conv_layers=3):
        super(ECGCNN_3M, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear((input_length // 16) * 1024, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ECGCNN_4M(nn.Module):
    def __init__(self, input_length=187, num_classes=5, num_conv_layers=3):
        super(ECGCNN_4M, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear((input_length // 16) * 1024, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ECGCNN_MoE(nn.Module):
    def __init__(
        self, input_length=187, num_classes=5, num_experts=8, top_k=3, num_conv_layers=3
    ):
        super(ECGCNN_MoE, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_experts = num_experts
        self.top_k = top_k

        # Initial layer to process input before experts
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)

        # Experts
        self.experts = nn.ModuleList([ECGCNN_MoE_Expert() for _ in range(num_experts)])

        # Routing network
        self.router_pooling = nn.AdaptiveAvgPool1d(1)
        self.training = True
        self.router_noise_std = 0.05
        self.router = nn.Linear(16, num_experts)

        self.conv2 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        final_length = (input_length // 8) // 2  # After expert blocks and conv2
        self.fc1 = nn.Linear(final_length * 1024, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.05)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.to(self.device)

        # Initial convolution
        x = F.relu(self.conv1(x))  # (batch, 16, length)

        # Routing
        pooled = self.router_pooling(x).squeeze(-1)
        if self.training and self.router_noise_std > 0:
            noise = torch.randn_like(pooled) * self.router_noise_std
            pooled = pooled + noise
        routing_logits = self.router(pooled)
        routing_probs = F.softmax(routing_logits, dim=-1)

        # Get top-k
        top_k_gates, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        top_k_gates = top_k_gates / top_k_gates.sum(dim=-1, keepdim=True)

        # MOST EFFICIENT: Only compute needed experts
        # Group samples by expert to maximize batch processing
        expert_outputs = torch.zeros(
            batch_size, 512, x.size(2) // 8, device=self.device
        )

        # For each expert, find which samples use it
        for expert_idx in range(self.num_experts):
            # Find which samples selected this expert
            expert_mask = top_k_indices == expert_idx  # (batch, top_k)

            if expert_mask.any():
                # Get batch indices and k positions where this expert is used
                batch_idx, k_idx = torch.where(expert_mask)

                if len(batch_idx) > 0:
                    # Get inputs for these samples
                    expert_input = x[batch_idx]  # (n_samples, 16, length)

                    # Process in one batch
                    expert_out = self.experts[expert_idx](
                        expert_input
                    )  # (n_samples, 512, length//8)

                    # Get corresponding gate values
                    gates = top_k_gates[batch_idx, k_idx]  # (n_samples,)

                    # Accumulate weighted outputs
                    expert_outputs[batch_idx] += expert_out * gates.view(-1, 1, 1)

        # Final layers
        x = self.conv2(expert_outputs)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x, self._load_balancing_loss(routing_probs)

    def _load_balancing_loss(self, routing_probs, epsilon=1e-10):
        """
        Encourage uniform expert usage across the batch.
        Uses coefficient of variation (CV) loss.
        """
        # Average probability assigned to each expert
        mean_probs = routing_probs.mean(dim=0)  # (num_experts,)

        # Coefficient of variation: std / mean
        # Lower CV = more balanced usage
        cv = mean_probs.std() / (mean_probs.mean() + epsilon)

        return cv**2  # Squared to make it stronger


class ECGCNN_MoE_Expert(nn.Module):
    def __init__(self):
        super(ECGCNN_MoE_Expert, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.block1 = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x
