"""
TCN Autoencoder for TEP Process Monitoring.
Architecture: Symmetric non-causal dilated TCN encoder-decoder.
Input: (batch, n_vars, window)  Output: (batch, n_vars, window)
"""
import torch
import torch.nn as nn


class DilatedBlock(nn.Module):
    """Non-causal dilated conv block with residual connection."""

    def __init__(self, n_ch: int, kernel: int = 3, dilation: int = 1):
        super().__init__()
        pad = (kernel - 1) * dilation // 2   # symmetric padding (non-causal)
        self.net = nn.Sequential(
            nn.Conv1d(n_ch, n_ch, kernel, dilation=dilation, padding=pad),
            nn.BatchNorm1d(n_ch),
            nn.GELU(),
            nn.Conv1d(n_ch, n_ch, kernel, dilation=dilation, padding=pad),
            nn.BatchNorm1d(n_ch),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class TCNAE(nn.Module):
    """TCN Autoencoder for anomaly scoring."""

    def __init__(
        self,
        n_vars: int = 52,
        window: int = 20,
        n_filters: int = 64,
        latent_dim: int = 24,
        dilations: tuple = (1, 2, 4, 8),
        kernel: int = 3,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.window = window
        self.latent_dim = latent_dim

        # Encoder: n_vars → n_filters via dilated blocks → pool → latent
        enc_blocks = [nn.Conv1d(n_vars, n_filters, 1)]
        for d in dilations:
            enc_blocks.append(DilatedBlock(n_filters, kernel, d))
        enc_blocks += [
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(n_filters, latent_dim),
            nn.GELU(),
        ]
        self.encoder = nn.Sequential(*enc_blocks)

        # Decoder: latent → expand → deconv → n_vars
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, n_filters * window),
            nn.Unflatten(1, (n_filters, window)),
            nn.Conv1d(n_filters, n_filters // 2, kernel, padding=kernel // 2),
            nn.GELU(),
            nn.Conv1d(n_filters // 2, n_vars, kernel, padding=kernel // 2),
        )

    def forward(self, x: torch.Tensor):
        """x: (B, V, W)  returns  recon: (B, V, W), z: (B, latent_dim)"""
        z = self.encoder(x)
        recon = self.decoder(z)
        # Trim/pad recon to match input window exactly
        recon = recon[..., : self.window]
        return recon, z

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE reconstruction error. Shape: (B,)"""
        recon, _ = self(x)
        return ((x - recon) ** 2).mean(dim=(1, 2))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
