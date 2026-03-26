from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from holosoma.utils.safe_torch_import import torch


@dataclass
class TrainConfig:
    data_dir: str = "/home/rllab/haechan/holosoma/logs/WholeBodyTracking/20260326_114156-g1_29dof_wbt_manager-ir/telemetry/"
    condition_text: str = "Push the suitcase, and set it back down."
    output_path: str = "encoder_only.pt"
    latent_dim: int = 32
    hidden_dims: tuple[int, int] = (256, 128)
    text_hash_dim: int = 256
    condition_dim: int = 64
    batch_size: int = 256
    epochs: int = 100
    learning_rate: float = 1e-3
    kl_weight: float = 1e-4
    weight_decay: float = 1e-6
    seed: int = 42
    device: str = "auto"
    log_interval: int = 10


class TextConditionEncoder(nn.Module):
    def __init__(self, input_dim: int, condition_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, condition_dim),
            nn.LayerNorm(condition_dim),
            nn.Tanh(),
        )

    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        return self.net(text_features)


class UWindowEncoder(nn.Module):
    def __init__(self, input_dim: int, condition_dim: int, hidden_dims: Sequence[int], latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dims[1], latent_dim)
        self.logvar = nn.Linear(hidden_dims[1], latent_dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(torch.cat([x, condition], dim=-1))
        return self.mu(h), self.logvar(h)


class UWindowDecoder(nn.Module):
    def __init__(self, latent_dim: int, condition_dim: int, hidden_dims: Sequence[int], output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], output_dim),
        )

    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, condition], dim=-1))


class UWindowCVAE(nn.Module):
    def __init__(self, input_dim: int, text_hash_dim: int, condition_dim: int, hidden_dims: Sequence[int], latent_dim: int):
        super().__init__()
        self.text_encoder = TextConditionEncoder(text_hash_dim, condition_dim)
        self.encoder = UWindowEncoder(input_dim, condition_dim, hidden_dims, latent_dim)
        self.decoder = UWindowDecoder(latent_dim, condition_dim, hidden_dims, input_dim)

    def encode(self, x: torch.Tensor, text_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        condition = self.text_encoder(text_features)
        return self.encoder(x, condition)

    def decode(self, z: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        condition = self.text_encoder(text_features)
        return self.decoder(z, condition)

    def forward(self, x: torch.Tensor, text_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, text_features)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        recon = self.decode(z, text_features)
        return recon, mu, logvar


class SavedUWindowEncoder(nn.Module):
    def __init__(self, input_dim: int, text_hash_dim: int, condition_dim: int, hidden_dims: Sequence[int], latent_dim: int):
        super().__init__()
        self.text_encoder = TextConditionEncoder(text_hash_dim, condition_dim)
        self.encoder = UWindowEncoder(input_dim, condition_dim, hidden_dims, latent_dim)

    def forward(self, x: torch.Tensor, text_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        condition = self.text_encoder(text_features)
        return self.encoder(x, condition)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def hash_text_to_features(text: str, dim: int) -> np.ndarray:
    features = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[A-Za-z0-9']+", text.lower())
    if not tokens:
        features[0] = 1.0
        return features

    for token in tokens:
        index = hash(token) % dim
        features[index] += 1.0

    norm = np.linalg.norm(features)
    if norm > 0.0:
        features /= norm
    return features


def extract_all_u_windows(data_dir: Path) -> np.ndarray:
    json_paths = sorted(data_dir.glob("episode_env*_idx*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No episode JSON files found under: {data_dir}")

    windows: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None

    for json_path in json_paths:
        with json_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        entries = payload.get("entries", [])
        for entry_index, entry in enumerate(entries):
            u_window = entry.get("u_window")
            if u_window is None:
                continue

            u_window_array = np.asarray(u_window, dtype=np.float32)
            if u_window_array.ndim != 2:
                raise ValueError(
                    f"Expected u_window to have rank 2, got shape {u_window_array.shape} "
                    f"in {json_path} entry {entry_index}."
                )

            if expected_shape is None:
                expected_shape = (int(u_window_array.shape[0]), int(u_window_array.shape[1]))
            elif u_window_array.shape != expected_shape:
                raise ValueError(
                    f"Inconsistent u_window shape. Expected {expected_shape}, got {u_window_array.shape} "
                    f"in {json_path} entry {entry_index}."
                )

            windows.append(u_window_array)

    if not windows:
        raise ValueError(f"No u_window entries found under: {data_dir}")

    stacked = np.stack(windows, axis=0)
    logger.info(f"Loaded {stacked.shape[0]} u_window samples from {len(json_paths)} episode files with shape {stacked.shape[1:]}")
    return stacked


def create_dataloader(x: torch.Tensor, batch_size: int) -> DataLoader:
    dataset = TensorDataset(x)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


def train_encoder(config: TrainConfig) -> Path:
    set_seed(config.seed)
    device = resolve_device(config.device)
    data_dir = Path(config.data_dir)
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    u_windows = extract_all_u_windows(data_dir)
    num_samples, time_steps, control_dim = u_windows.shape
    flattened = u_windows.reshape(num_samples, -1)

    x = torch.tensor(flattened, dtype=torch.float32)
    feature_mean = x.mean(dim=0)
    feature_std = x.std(dim=0).clamp_min(1e-6)
    x_norm = (x - feature_mean) / feature_std

    text_features_np = hash_text_to_features(config.condition_text, config.text_hash_dim)
    text_features = torch.tensor(text_features_np, dtype=torch.float32).unsqueeze(0)
    text_features = text_features.repeat(num_samples, 1)

    dataloader = create_dataloader(x_norm, config.batch_size)

    model = UWindowCVAE(
        input_dim=x_norm.shape[1],
        text_hash_dim=config.text_hash_dim,
        condition_dim=config.condition_dim,
        hidden_dims=config.hidden_dims,
        latent_dim=config.latent_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    reconstruction_loss_fn = nn.MSELoss(reduction="mean")

    logger.info(
        f"Training CVAE on {num_samples} windows, input_shape=({time_steps}, {control_dim}), "
        f"flattened_dim={x_norm.shape[1]}, latent_dim={config.latent_dim}, device={device}"
    )

    x_norm = x_norm.to(device)
    text_features = text_features.to(device)

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_recon = 0.0
        epoch_kl = 0.0
        epoch_total = 0.0
        seen = 0

        for (batch_x,) in dataloader:
            batch_x = batch_x.to(device)
            batch_size = batch_x.shape[0]
            batch_text = text_features[:batch_size]

            recon, mu, logvar = model(batch_x, batch_text)
            recon_loss = reconstruction_loss_fn(recon, batch_x)
            kl_loss = kl_divergence(mu, logvar)
            total_loss = recon_loss + config.kl_weight * kl_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            epoch_recon += recon_loss.item() * batch_size
            epoch_kl += kl_loss.item() * batch_size
            epoch_total += total_loss.item() * batch_size
            seen += batch_size

        if epoch % config.log_interval == 0 or epoch == 1 or epoch == config.epochs:
            logger.info(
                f"epoch={epoch:04d} recon={epoch_recon / seen:.6f} "
                f"kl={epoch_kl / seen:.6f} total={epoch_total / seen:.6f}"
            )

    encoder_only = SavedUWindowEncoder(
        input_dim=x_norm.shape[1],
        text_hash_dim=config.text_hash_dim,
        condition_dim=config.condition_dim,
        hidden_dims=config.hidden_dims,
        latent_dim=config.latent_dim,
    )
    encoder_only.text_encoder.load_state_dict(model.text_encoder.state_dict())
    encoder_only.encoder.load_state_dict(model.encoder.state_dict())
    encoder_only.eval()

    save_payload = {
        "model_type": "u_window_cvae_encoder",
        "config": asdict(config),
        "input_shape": [time_steps, control_dim],
        "flattened_dim": int(x_norm.shape[1]),
        "num_samples": int(num_samples),
        "feature_mean": feature_mean.cpu(),
        "feature_std": feature_std.cpu(),
        "encoder_state_dict": encoder_only.state_dict(),
    }
    torch.save(save_payload, output_path)
    logger.info(f"Saved encoder-only checkpoint to: {output_path}")
    return output_path


def load_encoder(checkpoint_path: str, device: str = "cpu") -> tuple[SavedUWindowEncoder, dict]:
    payload = torch.load(checkpoint_path, map_location=device)
    config_dict = payload["config"]
    encoder = SavedUWindowEncoder(
        input_dim=payload["flattened_dim"],
        text_hash_dim=config_dict["text_hash_dim"],
        condition_dim=config_dict["condition_dim"],
        hidden_dims=tuple(config_dict["hidden_dims"]),
        latent_dim=config_dict["latent_dim"],
    )
    encoder.load_state_dict(payload["encoder_state_dict"])
    encoder.to(device)
    encoder.eval()
    return encoder, payload


@torch.no_grad()
def encode_u_window_to_latent(
    checkpoint_path: str,
    u_window: np.ndarray | list,
    condition_text: str,
    device: str = "cpu",
) -> torch.Tensor:
    encoder, payload = load_encoder(checkpoint_path, device=device)
    u_window_array = np.asarray(u_window, dtype=np.float32)
    expected_shape = tuple(payload["input_shape"])
    if tuple(u_window_array.shape) != expected_shape:
        raise ValueError(f"Expected u_window shape {expected_shape}, got {u_window_array.shape}")

    x = torch.tensor(u_window_array.reshape(1, -1), dtype=torch.float32, device=device)
    feature_mean = payload["feature_mean"].to(device=device, dtype=torch.float32)
    feature_std = payload["feature_std"].to(device=device, dtype=torch.float32)
    x = (x - feature_mean.unsqueeze(0)) / feature_std.unsqueeze(0)

    text_features = hash_text_to_features(condition_text, payload["config"]["text_hash_dim"])
    text_features = torch.tensor(text_features, dtype=torch.float32, device=device).unsqueeze(0)

    mu, _ = encoder(x, text_features)
    return mu.squeeze(0).cpu()


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a CVAE encoder over all u_window entries in IR telemetry JSON files.")
    parser.add_argument("--data-dir", type=str, default=TrainConfig.data_dir)
    parser.add_argument("--condition-text", type=str, default=TrainConfig.condition_text)
    parser.add_argument("--output-path", type=str, default=TrainConfig.output_path)
    parser.add_argument("--latent-dim", type=int, default=TrainConfig.latent_dim)
    parser.add_argument("--hidden-dim-1", type=int, default=TrainConfig.hidden_dims[0])
    parser.add_argument("--hidden-dim-2", type=int, default=TrainConfig.hidden_dims[1])
    parser.add_argument("--text-hash-dim", type=int, default=TrainConfig.text_hash_dim)
    parser.add_argument("--condition-dim", type=int, default=TrainConfig.condition_dim)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--kl-weight", type=float, default=TrainConfig.kl_weight)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", type=str, default=TrainConfig.device)
    parser.add_argument("--log-interval", type=int, default=TrainConfig.log_interval)
    args = parser.parse_args()
    return TrainConfig(
        data_dir=args.data_dir,
        condition_text=args.condition_text,
        output_path=args.output_path,
        latent_dim=args.latent_dim,
        hidden_dims=(args.hidden_dim_1, args.hidden_dim_2),
        text_hash_dim=args.text_hash_dim,
        condition_dim=args.condition_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        kl_weight=args.kl_weight,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        log_interval=args.log_interval,
    )


def main() -> None:
    config = parse_args()
    train_encoder(config)


if __name__ == "__main__":
    main()
