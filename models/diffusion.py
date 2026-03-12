import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine noise schedule as proposed in:
    'Improved Denoising Diffusion Probabilistic Models' (Nichol & Dhariwal, 2021).
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """Linear noise schedule from the original DDPM paper."""
    return torch.linspace(beta_start, beta_end, timesteps)


class GaussianDiffusion(nn.Module):
    """
    Gaussian diffusion process for MoCoDiff.

    Implements the forward (noising) and reverse (denoising) diffusion processes
    with support for DDPM and DDIM sampling.

    Args:
        model: The denoising network (UNet).
        timesteps: Total number of diffusion timesteps.
        schedule: Noise schedule type ('cosine' or 'linear').
        loss_type: Type of training loss ('l2' or 'l1').
    """

    def __init__(self, model, timesteps=1000, schedule="cosine", loss_type="l2"):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.loss_type = loss_type

        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown noise schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # Compute posterior variance q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def _extract(self, a, t, x_shape):
        """Extract values from a 1-D tensor and reshape to broadcast with x."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion: sample x_t from q(x_t | x_0).

        Args:
            x_start: (B, T, C, H, W) clean video frames.
            t: (B,) timesteps.
            noise: Optional pre-sampled noise.

        Returns:
            Noisy video frames at timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(self, x_t, t, noise):
        """Predict x_0 from the predicted noise at timestep t."""
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """Compute the posterior mean and variance of q(x_{t-1} | x_t, x_0)."""
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x_t, t, clip_denoised=True):
        """Compute the mean and variance of the denoising distribution p(x_{t-1} | x_t)."""
        predicted_noise = self.model(x_t, t)
        x_recon = self.predict_start_from_noise(x_t, t, predicted_noise)

        if clip_denoised:
            x_recon = x_recon.clamp(-1, 1)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_recon, x_t, t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x_t, t, clip_denoised=True):
        """Single reverse diffusion step: sample x_{t-1} from p(x_{t-1} | x_t)."""
        model_mean, _, model_log_variance = self.p_mean_variance(
            x_t, t, clip_denoised=clip_denoised
        )
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().reshape(-1, *((1,) * (len(x_t.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, device, clip_denoised=True, show_progress=True):
        """
        Full DDPM reverse diffusion sampling loop.

        Args:
            shape: (B, T, C, H, W) desired output shape.
            device: Torch device.
            clip_denoised: Whether to clip predicted x_0 to [-1, 1].
            show_progress: Whether to display a progress bar.

        Returns:
            Generated video frames.
        """
        x = torch.randn(shape, device=device)
        timesteps = reversed(range(self.timesteps))

        if show_progress:
            timesteps = tqdm(timesteps, desc="Sampling", total=self.timesteps)

        for t_val in timesteps:
            t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
            x = self.p_sample(x, t, clip_denoised=clip_denoised)

        return x

    @torch.no_grad()
    def ddim_sample(self, shape, device, ddim_steps=50, eta=0.0, clip_denoised=True, show_progress=True):
        """
        DDIM sampling for faster inference.

        Args:
            shape: (B, T, C, H, W) desired output shape.
            device: Torch device.
            ddim_steps: Number of DDIM sampling steps.
            eta: DDIM eta parameter (0 = deterministic, 1 = DDPM).
            clip_denoised: Whether to clip predicted x_0 to [-1, 1].
            show_progress: Whether to display a progress bar.

        Returns:
            Generated video frames.
        """
        x = torch.randn(shape, device=device)
        step_indices = np.linspace(0, self.timesteps - 1, ddim_steps, dtype=int)
        step_indices = list(reversed(step_indices))

        iterator = tqdm(step_indices, desc="DDIM Sampling") if show_progress else step_indices

        for i, t_val in enumerate(iterator):
            t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
            predicted_noise = self.model(x, t)
            x_recon = self.predict_start_from_noise(x, t, predicted_noise)

            if clip_denoised:
                x_recon = x_recon.clamp(-1, 1)

            if i < len(step_indices) - 1:
                t_prev = step_indices[i + 1]
                alpha_t = self.alphas_cumprod[t_val]
                alpha_t_prev = self.alphas_cumprod[t_prev]
                sigma_t = eta * torch.sqrt(
                    (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
                )
                c1 = torch.sqrt(alpha_t_prev)
                c2 = torch.sqrt(1 - alpha_t_prev - sigma_t**2)
                noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
                x = c1 * x_recon + c2 * predicted_noise + sigma_t * noise
            else:
                x = x_recon

        return x

    def compute_loss(self, x_start, t=None, noise=None):
        """
        Compute the training loss.

        Args:
            x_start: (B, T, C, H, W) clean video frames in [-1, 1].
            t: Optional (B,) timesteps. If None, sampled uniformly.
            noise: Optional pre-sampled noise.

        Returns:
            Scalar loss value.
        """
        if t is None:
            t = torch.randint(0, self.timesteps, (x_start.shape[0],), device=x_start.device)
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start, t, noise=noise)
        predicted_noise = self.model(x_noisy, t)

        if self.loss_type == "l2":
            loss = nn.functional.mse_loss(predicted_noise, noise)
        elif self.loss_type == "l1":
            loss = nn.functional.l1_loss(predicted_noise, noise)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss

    def forward(self, x_start, t=None, noise=None):
        return self.compute_loss(x_start, t=t, noise=noise)
