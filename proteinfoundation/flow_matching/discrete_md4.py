import math
from typing import Optional

import torch
from torch import Tensor, nn


def _reverse_broadcast(t: Tensor, target_ndim: int) -> Tensor:
    while t.dim() < target_ndim:
        t = t.unsqueeze(-1)
    return t


def _symmetrize_matrix(x: Tensor) -> Tensor:
    if x.dim() < 2:
        return x
    upper = torch.triu(x, diagonal=0)
    diag = torch.diagonal(upper, dim1=-2, dim2=-1)
    return upper + upper.transpose(-1, -2) - torch.diag_embed(diag)


def _distance_fraction(n: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    idx = torch.arange(n, device=device, dtype=dtype)
    dist = (idx[None, :] - idx[:, None]).abs()
    max_dist = max(n - 1, 1)
    return dist / max_dist


def _distance_w(
    n: int, w_min: float, w_max: float, device: torch.device, dtype: torch.dtype
) -> Tensor:
    if w_min <= 0 or w_max <= 0:
        raise ValueError(f"w_min and w_max must be > 0, got {w_min}, {w_max}")
    if w_min == w_max:
        return torch.full((n, n), w_min, device=device, dtype=dtype)
    frac = _distance_fraction(n, device=device, dtype=dtype)
    ratio = w_max / w_min
    return w_min * ratio**frac


def _normalize_position_bias_mode(mode: str) -> str:
    if mode in ("diagonal_distance", "polynomial"):
        return "polynomial"
    if mode in ("diagonal_sigmoid", "sigmoid"):
        return "sigmoid"
    return mode


def _param_sigmoid(x: Tensor, k: Tensor, d: Tensor) -> Tensor:
    return 1.0 - 1.0 / (1.0 + torch.exp(-k * (x - d)))


def _param_sigmoid_prime(x: Tensor, k: Tensor, d: Tensor) -> Tensor:
    s = _param_sigmoid(x, k, d)
    return -k * s * (1.0 - s)


def _sigmoid_alpha(
    t_ext: Tensor, d_pos: Tensor, k: Tensor, eps: float
) -> tuple[Tensor, Tensor]:
    alpha_raw = _param_sigmoid(t_ext, k, d_pos)
    alpha0 = _param_sigmoid(torch.zeros_like(t_ext), k, d_pos)
    alpha1 = _param_sigmoid(torch.ones_like(t_ext), k, d_pos)
    denom = (alpha0 - alpha1).clamp_min(1e-12)
    alpha = (alpha_raw - alpha1) / denom
    if eps > 0:
        alpha = alpha.clamp(min=eps, max=1.0 - eps)
    dalpha = _param_sigmoid_prime(t_ext, k, d_pos) / denom
    return alpha, dalpha


class MaskingSchedule(nn.Module):
    """Scalar masking schedule for MD4."""

    def __init__(self, schedule_fn_type: str = "cosine", eps: float = 1e-4):
        super().__init__()
        self.schedule_fn_type = schedule_fn_type
        self.eps = eps

    def _dalpha(self, t: Tensor) -> Tensor:
        if self.schedule_fn_type == "cosine":
            return -math.pi / 2.0 * torch.sin(math.pi / 2.0 * (1.0 - t))
        if self.schedule_fn_type == "linear":
            return -torch.ones_like(t)
        if self.schedule_fn_type.startswith("poly"):
            exponent = float(self.schedule_fn_type.replace("poly", ""))
            return -exponent * t ** (exponent - 1.0)
        raise NotImplementedError(f"Unknown schedule {self.schedule_fn_type}")

    def dalpha(self, t: Tensor) -> Tensor:
        return (1.0 - 2.0 * self.eps) * self._dalpha(t)

    def _alpha(self, t: Tensor) -> Tensor:
        if self.schedule_fn_type == "linear":
            return 1.0 - t
        if self.schedule_fn_type.startswith("poly"):
            exponent = float(self.schedule_fn_type.replace("poly", ""))
            return 1.0 - t ** exponent
        if self.schedule_fn_type == "cosine":
            return 1.0 - torch.cos(math.pi / 2.0 * (1.0 - t))
        raise NotImplementedError(f"Unknown schedule {self.schedule_fn_type}")

    def alpha(self, t: Tensor) -> Tensor:
        return (1.0 - 2.0 * self.eps) * self._alpha(t) + self.eps

    def dgamma_times_alpha(self, t: Tensor) -> Tensor:
        return self.dalpha(t) / (1.0 - self.alpha(t))

    def forward(self, t: Tensor) -> Tensor:
        alpha = self.alpha(t)
        return torch.log(alpha / (1.0 - alpha))


class LearnableVecMaskingSchedule(nn.Module):
    """Learnable vector-valued masking schedule for GenMD4."""

    def __init__(
        self,
        vocab_size: int,
        schedule_fn_type: str = "poly",
        eps: float = 1e-4,
        power_init: float = 1.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.schedule_fn_type = schedule_fn_type
        self.eps = eps
        if self.schedule_fn_type != "poly":
            raise NotImplementedError(
                f"Only poly schedule supported, got {self.schedule_fn_type}"
            )
        w_init = math.log(math.exp(power_init) - 1.0)
        self.w = nn.Parameter(torch.full((vocab_size,), w_init))

    @property
    def power(self) -> Tensor:
        return torch.nn.functional.softplus(self.w)

    def dalpha(self, t: Tensor) -> Tensor:
        t = t[..., None]
        return -(1.0 - self.eps) * self.power * t ** (self.power - 1.0)

    def alpha(self, t: Tensor) -> Tensor:
        t = t[..., None]
        return 1.0 - (1.0 - self.eps) * t ** self.power

    def dgamma_times_alpha(self, t: Tensor) -> Tensor:
        t = t[..., None].clamp_min(1e-8)
        return -self.power / t

    def forward(self, t: Tensor) -> Tensor:
        alpha = self.alpha(t)
        return torch.log(alpha / (1.0 - alpha))


class MD4DiscreteDiffusion(nn.Module):
    """Simplified masked diffusion (MD4) for discrete contact maps."""

    def __init__(
        self,
        vocab_size: int = 2,
        noise_schedule_type: str = "cosine",
        timesteps: int = 1000,
        cont_time: bool = True,
        sampling_grid: str = "cosine",
        eps: float = 1e-4,
        position_bias: Optional[dict] = None,
        symmetrize: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token = vocab_size
        self.timesteps = timesteps
        self.cont_time = cont_time
        self.sampling_grid = sampling_grid
        self.symmetrize = symmetrize
        self.position_bias = position_bias or {}
        self.position_bias_enabled = bool(self.position_bias.get("enabled", False))
        self.position_bias_mode = _normalize_position_bias_mode(
            self.position_bias.get("mode", "polynomial")
        )
        self.position_bias_w_min = float(self.position_bias.get("w_min", 0.2))
        self.position_bias_w_max = float(self.position_bias.get("w_max", 5.0))
        self.position_bias_k = float(self.position_bias.get("k", 30.0))
        self.noise_schedule = MaskingSchedule(
            schedule_fn_type=noise_schedule_type, eps=eps
        )

    def _position_w(self, n: int, device: torch.device, dtype: torch.dtype) -> Optional[Tensor]:
        if not self.position_bias_enabled:
            return None
        if self.position_bias_mode != "polynomial":
            raise NotImplementedError(
                f"Unknown position bias mode {self.position_bias_mode}"
            )
        return _distance_w(
            n,
            self.position_bias_w_min,
            self.position_bias_w_max,
            device=device,
            dtype=dtype,
        )

    def _alpha_and_dgamma(
        self, t: Tensor, x_shape: tuple, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Optional[Tensor]]:
        alpha = self.noise_schedule.alpha(t)
        alpha = _reverse_broadcast(alpha, len(x_shape))
        if not self.position_bias_enabled:
            dgamma = self.noise_schedule.dgamma_times_alpha(t)
            return alpha, dgamma
        n = x_shape[-1]
        if self.position_bias_mode == "sigmoid":
            t_ext = _reverse_broadcast(t, len(x_shape))
            d_pos = _distance_fraction(n, device=device, dtype=dtype)
            k = torch.tensor(self.position_bias_k, device=device, dtype=dtype)
            alpha_pos, dalpha = _sigmoid_alpha(
                t_ext, d_pos, k, self.noise_schedule.eps
            )
            dgamma_pos = dalpha / (1.0 - alpha_pos).clamp_min(1e-8)
            return alpha_pos, dgamma_pos
        w_pos = self._position_w(n, device=device, dtype=dtype)
        if w_pos is None:
            dgamma = self.noise_schedule.dgamma_times_alpha(t)
            return alpha, dgamma
        t_ext = _reverse_broadcast(t, len(x_shape)).clamp(min=1e-6, max=1.0)
        alpha_pos = 1.0 - t_ext ** w_pos
        alpha_pos = alpha_pos.clamp(
            min=self.noise_schedule.eps, max=1.0 - self.noise_schedule.eps
        )
        dgamma_pos = -w_pos / t_ext
        return alpha_pos, dgamma_pos

    def prior_sample(self, shape: tuple, device: torch.device) -> Tensor:
        return torch.full(shape, self.mask_token, device=device, dtype=torch.long)

    def forward_sample(self, x: Tensor, t: Tensor) -> Tensor:
        alpha_t, _ = self._alpha_and_dgamma(t, x.shape, x.device, x.dtype)
        rand = torch.rand_like(x.float())
        if self.symmetrize:
            rand = _symmetrize_matrix(rand)
        unmask = rand < alpha_t
        zt = torch.where(unmask, x, torch.full_like(x, self.mask_token))
        if self.symmetrize:
            zt = _symmetrize_matrix(zt)
        return zt

    def recon_loss(self, pair_mask: Optional[Tensor]) -> Tensor:
        if pair_mask is None:
            if self.position_bias_enabled:
                raise ValueError(
                    "position_bias requires pair_mask for recon_loss."
                )
            alpha_t1 = self.noise_schedule.alpha(torch.tensor(0.0))
            return (1.0 - alpha_t1) * math.log(self.vocab_size)
        batch_size = pair_mask.shape[0]
        t_dtype = pair_mask.dtype if pair_mask.is_floating_point() else torch.float32
        t1 = torch.zeros(batch_size, device=pair_mask.device, dtype=t_dtype)
        alpha_t1, _ = self._alpha_and_dgamma(
            t1, pair_mask.shape, pair_mask.device, t1.dtype
        )
        mask_prob = (1.0 - alpha_t1) * pair_mask.float()
        loss = mask_prob.sum(dim=tuple(range(1, mask_prob.dim())))
        return loss * math.log(self.vocab_size)

    def latent_loss(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.zeros(batch_size, device=device)

    def diffusion_loss(
        self,
        logits: Tensor,
        x: Tensor,
        zt: Tensor,
        t: Tensor,
        pair_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.vocab_size == 2:
            # Binary case: use sigmoid (original MD4)
            log_p1 = torch.nn.functional.logsigmoid(logits)
            log_p0 = torch.nn.functional.logsigmoid(-logits)
            log_p_true = torch.where(x == 1, log_p1, log_p0)
        else:
            # Multi-class case: use softmax
            log_p = torch.nn.functional.log_softmax(logits, dim=-1)
            one_hot_x = torch.nn.functional.one_hot(x, self.vocab_size).float()
            log_p_true = (one_hot_x * log_p).sum(dim=-1)

        mask = (zt == self.mask_token).float()
        if pair_mask is not None:
            mask = mask * pair_mask.float()

        if not self.cont_time:
            t_discrete = (torch.floor(t * self.timesteps) + 1.0) / self.timesteps
            s = t_discrete - (1.0 / self.timesteps)
            if self.position_bias_enabled:
                alpha_t, _ = self._alpha_and_dgamma(
                    t_discrete, x.shape, x.device, x.dtype
                )
                alpha_s, _ = self._alpha_and_dgamma(
                    s, x.shape, x.device, x.dtype
                )
                gt = torch.log(alpha_t / (1.0 - alpha_t))
                gs = torch.log(alpha_s / (1.0 - alpha_s))
                weight = self.timesteps * torch.expm1(gt - gs) * alpha_s
                return (mask * log_p_true * weight).sum(
                    dim=tuple(range(1, x.dim()))
                )
            gt = self.noise_schedule(t_discrete)
            gs = self.noise_schedule(s)
            weight = (
                self.timesteps
                * torch.expm1(gt - gs)
                * self.noise_schedule.alpha(s)
            )
            masked_neg_cross_ent = (mask * log_p_true).sum(
                dim=tuple(range(1, x.dim()))
            )
            return weight * masked_neg_cross_ent

        alpha_t, dgamma = self._alpha_and_dgamma(t, x.shape, x.device, x.dtype)
        if self.position_bias_enabled:
            return (mask * log_p_true * dgamma).sum(dim=tuple(range(1, x.dim())))
        masked_neg_cross_ent = (mask * log_p_true).sum(
            dim=tuple(range(1, x.dim()))
        )
        return dgamma * masked_neg_cross_ent

    def get_sampling_grid(self, i: int, timesteps: int) -> tuple[float, float]:
        t = (timesteps - i) / timesteps
        s = t - 1.0 / timesteps
        if self.sampling_grid == "cosine":
            t = math.cos(math.pi / 2.0 * (1.0 - t))
            s = math.cos(math.pi / 2.0 * (1.0 - s))
        return s, t

    def sample_step(self, zt: Tensor, logits: Tensor, s: float, t: float) -> Tensor:
        t_tensor = torch.full(
            (zt.shape[0],), t, device=zt.device, dtype=logits.dtype
        )
        s_tensor = torch.full(
            (zt.shape[0],), s, device=zt.device, dtype=logits.dtype
        )
        alpha_t, _ = self._alpha_and_dgamma(
            t_tensor, zt.shape, zt.device, zt.dtype
        )
        alpha_s, _ = self._alpha_and_dgamma(
            s_tensor, zt.shape, zt.device, zt.dtype
        )
        unmask_prob = (alpha_s - alpha_t) / (1.0 - alpha_t)
        unmask_prob = torch.clamp(unmask_prob, 0.0, 1.0)

        is_mask = zt == self.mask_token
        u = torch.rand_like(zt.float())
        if self.symmetrize:
            u = _symmetrize_matrix(u)
        unmask = (u < unmask_prob) & is_mask

        if self.vocab_size == 2:
            # Binary: sample from sigmoid
            p_contact = torch.sigmoid(logits)
            u2 = torch.rand_like(p_contact)
            sampled = torch.where(u2 < p_contact, torch.ones_like(zt), torch.zeros_like(zt))
        else:
            # Multi-class: sample from softmax
            probs = torch.softmax(logits, dim=-1)  # [..., vocab_size]
            flat_probs = probs.reshape(-1, self.vocab_size)
            sampled_flat = torch.multinomial(flat_probs.clamp_min(0), 1).squeeze(-1)
            sampled = sampled_flat.reshape(zt.shape)

        zt_updated = torch.where(unmask, sampled, zt)
        if self.symmetrize:
            zt_updated = _symmetrize_matrix(zt_updated)
        return zt_updated

    def decode(self, zt: Tensor, logits: Tensor) -> Tensor:
        is_mask = zt == self.mask_token
        zt_clipped = torch.where(is_mask, torch.zeros_like(zt), zt)
        if self.vocab_size == 2:
            p_contact = torch.sigmoid(logits)
            sampled = (p_contact >= 0.5).long()
        else:
            sampled = logits.argmax(dim=-1)
        return torch.where(is_mask, sampled, zt_clipped)


class UDLMDiscreteDiffusion(nn.Module):
    """Uniform Diffusion Language Model (Schiff et al., ICLR 2025).

    Uniform noise discrete diffusion where tokens transition to uniformly
    random values instead of a special mask token (absorbing diffusion).

    Forward process: q(z_t | x) = alpha(t) * delta(z_t, x) + (1 - alpha(t)) / N
    where N = vocab_size.

    The training loss uses the continuous-time NELBO from Eq. 18/19, and the
    reverse process uses the closed-form posterior from Eq. 15.
    """

    def __init__(
        self,
        vocab_size: int = 2,
        noise_schedule_type: str = "cosine",
        timesteps: int = 1000,
        cont_time: bool = True,
        sampling_grid: str = "cosine",
        eps: float = 1e-4,
        position_bias: Optional[dict] = None,
        symmetrize: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token = None  # No mask token in uniform diffusion
        self.timesteps = timesteps
        self.cont_time = cont_time
        self.sampling_grid = sampling_grid
        self.symmetrize = symmetrize
        self.position_bias = position_bias or {}
        self.position_bias_enabled = bool(self.position_bias.get("enabled", False))
        self.position_bias_mode = _normalize_position_bias_mode(
            self.position_bias.get("mode", "polynomial")
        )
        self.position_bias_w_min = float(self.position_bias.get("w_min", 0.2))
        self.position_bias_w_max = float(self.position_bias.get("w_max", 5.0))
        self.position_bias_k = float(self.position_bias.get("k", 30.0))
        self.noise_schedule = MaskingSchedule(
            schedule_fn_type=noise_schedule_type, eps=eps
        )

    def _position_w(self, n: int, device: torch.device, dtype: torch.dtype) -> Optional[Tensor]:
        if not self.position_bias_enabled:
            return None
        if self.position_bias_mode != "polynomial":
            raise NotImplementedError(
                f"Unknown position bias mode {self.position_bias_mode}"
            )
        return _distance_w(
            n,
            self.position_bias_w_min,
            self.position_bias_w_max,
            device=device,
            dtype=dtype,
        )

    def _alpha_and_dalpha(
        self, t: Tensor, x_shape: tuple, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        """Return (alpha_t, dalpha_t) with proper broadcasting to data shape."""
        alpha_cos = self.noise_schedule.alpha(t)
        dalpha_cos = self.noise_schedule.dalpha(t)
        alpha_cos = _reverse_broadcast(alpha_cos, len(x_shape))
        dalpha_cos = _reverse_broadcast(dalpha_cos, len(x_shape))

        if not self.position_bias_enabled:
            return alpha_cos, dalpha_cos

        if self.position_bias_mode != "sigmoid":
            raise NotImplementedError(
                f"UDLM only supports 'sigmoid' position bias, got '{self.position_bias_mode}'"
            )

        n = x_shape[-1]
        t_ext = _reverse_broadcast(t, len(x_shape))
        d_pos = _distance_fraction(n, device=device, dtype=dtype)
        k = torch.tensor(self.position_bias_k, device=device, dtype=dtype)
        alpha_sig, dalpha_sig = _sigmoid_alpha(
            t_ext, d_pos, k, self.noise_schedule.eps
        )

        # Product schedule: alpha = alpha_cos * alpha_sig
        alpha_t = alpha_cos * alpha_sig
        # Product rule: dalpha = dalpha_cos * alpha_sig + alpha_cos * dalpha_sig
        dalpha_t = dalpha_cos * alpha_sig + alpha_cos * dalpha_sig

        return alpha_t, dalpha_t

    def prior_sample(self, shape: tuple, device: torch.device) -> Tensor:
        """Sample from prior: uniform over vocabulary."""
        return torch.randint(0, self.vocab_size, shape, device=device)

    def forward_sample(self, x: Tensor, t: Tensor) -> Tensor:
        """Corrupt clean tokens using uniform noise.

        Each token is kept with probability alpha(t), replaced with a uniform
        random token from {0, ..., vocab_size-1} with probability 1 - alpha(t).
        """
        alpha_t, _ = self._alpha_and_dalpha(t, x.shape, x.device, x.dtype)
        rand_tokens = torch.randint_like(x, 0, self.vocab_size)
        keep_rand = torch.rand_like(x.float())
        if self.symmetrize:
            keep_rand = _symmetrize_matrix(keep_rand)
            rand_tokens = _symmetrize_matrix(rand_tokens)
        keep_mask = keep_rand < alpha_t
        zt = torch.where(keep_mask, x, rand_tokens)
        return zt

    def recon_loss(self, pair_mask: Optional[Tensor]) -> Tensor:
        """Reconstruction loss.

        In the continuous-time limit (T -> inf), the reconstruction loss goes
        to zero analytically (Section 4.2, Schiff et al. 2025), because
        alpha(1/T) -> 1 implies z_{1/T} = x with probability 1.
        """
        if pair_mask is None:
            return torch.tensor(0.0)
        return torch.zeros(pair_mask.shape[0], device=pair_mask.device)

    def latent_loss(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.zeros(batch_size, device=device)

    def diffusion_loss(
        self,
        logits: Tensor,
        x: Tensor,
        zt: Tensor,
        t: Tensor,
        pair_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute UDLM continuous-time NELBO (Eq. 18/19, Schiff et al. 2025).

        L_inf = integral_0^1 E_q [ (alpha'_t / (N * alpha_t)) *
            [ N / x_bar_i  -  N / x_bar_theta_i
              - sum_{j != i} (x_bar_j / x_bar_i)
                * log( (x_bar_theta_i * x_bar_j) / (x_bar_theta_j * x_bar_i) )
            ]
        ] dt

        where:
            N = vocab_size
            x_bar = N * alpha_t * x + (1 - alpha_t) * 1  (smoothed clean data)
            x_bar_theta = N * alpha_t * x_theta + (1 - alpha_t) * 1  (smoothed pred)
            i = index of z_t (the observed noisy token)
            x_theta = softmax(logits)  (model prediction of clean data distribution)

        The integral is approximated with Monte Carlo using uniformly sampled t.
        The weight alpha'_t / (N * alpha_t) < 0 and the bracket <= 0, so the
        product is non-negative (a valid loss to minimize).
        """
        N = self.vocab_size
        eps = 1e-8  # numerical stability

        # -- noise schedule values --
        alpha_t, dalpha_t = self._alpha_and_dalpha(t, x.shape, x.device, x.dtype)

        # -- model prediction as probability distribution --
        if N == 2:
            p1 = torch.sigmoid(logits)
            x_theta = torch.stack([1.0 - p1, p1], dim=-1)  # [..., 2]
        else:
            x_theta = torch.softmax(logits, dim=-1)  # [..., N]

        # -- one-hot representations --
        x_onehot = torch.nn.functional.one_hot(x.long(), N).float()   # [*, N]
        zt_onehot = torch.nn.functional.one_hot(zt.long(), N).float() # [*, N]

        # -- smoothed quantities: x_bar = N * alpha_t * x + (1 - alpha_t) --
        alpha_ext = alpha_t[..., None]  # add vocab dim
        x_bar = N * alpha_ext * x_onehot + (1.0 - alpha_ext)
        x_bar_theta = N * alpha_ext * x_theta + (1.0 - alpha_ext)

        # -- extract values at the observed noisy token index i --
        x_bar_i = (x_bar * zt_onehot).sum(dim=-1)               # [*]
        x_bar_theta_i = (x_bar_theta * zt_onehot).sum(dim=-1)   # [*]

        # -- Term 1: N / x_bar_i - N / x_bar_theta_i --
        term1 = N / x_bar_i.clamp_min(eps) - N / x_bar_theta_i.clamp_min(eps)

        # -- Term 2: sum_{j != i} (x_bar_j / x_bar_i) * log(...) --
        # log((x_bar_theta_i * x_bar_j) / (x_bar_theta_j * x_bar_i))
        #   = log(x_bar_theta_i / x_bar_i) - log(x_bar_theta_j / x_bar_j)
        log_ratio = (
            torch.log(x_bar_theta.clamp_min(eps))
            - torch.log(x_bar.clamp_min(eps))
        )  # [*, N]
        log_ratio_i = (log_ratio * zt_onehot).sum(dim=-1)  # [*]

        not_i = 1.0 - zt_onehot                              # [*, N]
        coeff = x_bar * not_i / x_bar_i[..., None].clamp_min(eps)  # [*, N]

        term2 = (coeff * (log_ratio_i[..., None] - log_ratio)).sum(dim=-1)  # [*]

        # -- bracket: term1 - term2 --
        bracket = term1 - term2

        # -- weight: alpha'_t / (N * alpha_t) --
        weight = dalpha_t / (N * alpha_t.clamp_min(eps))

        loss_per_pos = weight * bracket

        # -- apply mask --
        if pair_mask is not None:
            loss_per_pos = loss_per_pos * pair_mask.float()

        # Sum over spatial dimensions (per-sample loss)
        return loss_per_pos.sum(dim=tuple(range(1, loss_per_pos.dim())))

    def get_sampling_grid(self, i: int, timesteps: int) -> tuple[float, float]:
        t = (timesteps - i) / timesteps
        s = t - 1.0 / timesteps
        if self.sampling_grid == "cosine":
            t = math.cos(math.pi / 2.0 * (1.0 - t))
            s = math.cos(math.pi / 2.0 * (1.0 - s))
        return s, t

    def sample_step(self, zt: Tensor, logits: Tensor, s: float, t: float) -> Tensor:
        """Denoising step from time t to time s using the UDLM posterior (Eq. 15).

        p_theta(z_s | z_t) = Cat(z_s; p_s) where, for each class j:

        p_s_j = [ N * alpha_t * (z_t)_j * (x_theta)_j
                   + (alpha_{t|s} - alpha_t) * (z_t)_j
                   + (alpha_s - alpha_t) * (x_theta)_j
                   + (alpha_s - alpha_t) * (1 - alpha_s) / (N * alpha_s)
                 ] / Z

        where alpha_{t|s} = alpha_t / alpha_s, and Z normalizes to sum 1.
        """
        N = self.vocab_size
        eps = 1e-8

        t_tensor = torch.full((zt.shape[0],), t, device=zt.device, dtype=logits.dtype)
        s_tensor = torch.full((zt.shape[0],), s, device=zt.device, dtype=logits.dtype)

        # Position-dependent alpha (uses product schedule when PB enabled)
        alpha_t_b, _ = self._alpha_and_dalpha(t_tensor, zt.shape, zt.device, logits.dtype)
        alpha_s_b, _ = self._alpha_and_dalpha(s_tensor, zt.shape, zt.device, logits.dtype)

        alpha_ts = (alpha_t_b / alpha_s_b.clamp_min(eps)).clamp(max=1.0)

        # Model prediction
        if N == 2:
            p1 = torch.sigmoid(logits)
            x_theta = torch.stack([1.0 - p1, p1], dim=-1)
        else:
            x_theta = torch.softmax(logits, dim=-1)

        zt_onehot = torch.nn.functional.one_hot(zt.long(), N).float()

        alpha_t_ext = alpha_t_b[..., None]
        alpha_s_ext = alpha_s_b[..., None]
        alpha_ts_ext = alpha_ts[..., None]
        diff_as_at = alpha_s_ext - alpha_t_ext

        # Numerator terms per Eq. 15
        p1_term = N * alpha_t_ext * zt_onehot * x_theta
        p2_term = (alpha_ts_ext - alpha_t_ext) * zt_onehot
        p3_term = diff_as_at * x_theta
        p4_term = diff_as_at * (1.0 - alpha_s_ext) / (N * alpha_s_ext.clamp_min(eps))

        numerator = p1_term + p2_term + p3_term + p4_term
        numerator = numerator.clamp_min(0.0)

        posterior = numerator / numerator.sum(dim=-1, keepdim=True).clamp_min(eps)

        flat_posterior = posterior.reshape(-1, N)
        sampled_flat = torch.multinomial(flat_posterior.clamp_min(0), 1).squeeze(-1)
        z_new = sampled_flat.reshape(zt.shape)

        if self.symmetrize:
            z_new = _symmetrize_matrix(z_new)

        return z_new

    def decode(self, zt: Tensor, logits: Tensor) -> Tensor:
        """Decode final tokens: argmax of logits."""
        if self.vocab_size == 2:
            return (torch.sigmoid(logits) >= 0.5).long()
        return logits.argmax(dim=-1)

class GenMD4DiscreteDiffusion(nn.Module):
    """Generalized state-dependent masked diffusion (GenMD4)."""

    def __init__(
        self,
        vocab_size: int = 2,
        noise_schedule_type: str = "poly",
        power_init: float = 1.0,
        t1: float = 1e-3,
        eps: float = 1e-4,
        position_bias: Optional[dict] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token = vocab_size
        self.t1 = t1
        self.position_bias = position_bias or {}
        self.position_bias_enabled = bool(self.position_bias.get("enabled", False))
        self.position_bias_mode = _normalize_position_bias_mode(
            self.position_bias.get("mode", "polynomial")
        )
        self.position_bias_w_min = float(self.position_bias.get("w_min", 0.2))
        self.position_bias_w_max = float(self.position_bias.get("w_max", 5.0))
        self.position_bias_k = float(self.position_bias.get("k", 30.0))
        self.noise_schedule = LearnableVecMaskingSchedule(
            vocab_size=vocab_size,
            schedule_fn_type=noise_schedule_type,
            power_init=power_init,
            eps=eps,
        )

    def _position_w(self, n: int, device: torch.device, dtype: torch.dtype) -> Optional[Tensor]:
        if not self.position_bias_enabled:
            return None
        if self.position_bias_mode != "polynomial":
            raise NotImplementedError(
                f"Unknown position bias mode {self.position_bias_mode}"
            )
        return _distance_w(
            n,
            self.position_bias_w_min,
            self.position_bias_w_max,
            device=device,
            dtype=dtype,
        )

    def _alpha_and_dgamma(
        self, t: Tensor, x_shape: tuple, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        t_ext = _reverse_broadcast(t, len(x_shape) + 1)
        alpha = self.noise_schedule.alpha(t_ext)
        if not self.position_bias_enabled:
            return alpha, self.noise_schedule.dgamma_times_alpha(t_ext)
        n = x_shape[-1]
        if self.position_bias_mode == "sigmoid":
            d_pos = _distance_fraction(n, device=device, dtype=dtype).unsqueeze(0).unsqueeze(-1)
            k = torch.tensor(self.position_bias_k, device=device, dtype=dtype)
            w_vec = self.noise_schedule.power
            k_eff = k * w_vec.view(1, 1, 1, -1)
            alpha_pos, dalpha = _sigmoid_alpha(
                t_ext, d_pos, k_eff, self.noise_schedule.eps
            )
            dgamma_pos = dalpha / (1.0 - alpha_pos).clamp_min(1e-8)
            return alpha_pos, dgamma_pos
        w_pos = self._position_w(n, device=device, dtype=dtype)
        if w_pos is None:
            return alpha, self.noise_schedule.dgamma_times_alpha(t_ext)
        w_vec = self.noise_schedule.power
        w_pos = w_pos.unsqueeze(0).unsqueeze(-1)
        w_eff = w_vec.view(1, 1, 1, -1) * w_pos
        t_ext = t_ext.clamp(min=1e-6, max=1.0)
        alpha_pos = 1.0 - (1.0 - self.noise_schedule.eps) * t_ext ** w_eff
        alpha_pos = alpha_pos.clamp(
            min=self.noise_schedule.eps, max=1.0 - self.noise_schedule.eps
        )
        dgamma_pos = -w_eff / t_ext
        return alpha_pos, dgamma_pos

    def forward_sample(self, x: Tensor, t: Tensor) -> Tensor:
        alpha_t, _ = self._alpha_and_dgamma(t, x.shape, x.device, x.dtype)
        one_hot_x = torch.nn.functional.one_hot(x, self.vocab_size).float()
        unmask_prob = (alpha_t * one_hot_x).sum(dim=-1)
        rand = _symmetrize_matrix(torch.rand_like(unmask_prob))
        unmask = rand < unmask_prob
        zt = torch.where(unmask, x, torch.full_like(x, self.mask_token))
        return _symmetrize_matrix(zt)

    def recon_loss(self, x: Tensor, pair_mask: Optional[Tensor]) -> Tensor:
        batch_size = x.shape[0]
        t1 = torch.full((batch_size,), self.t1, device=x.device, dtype=x.dtype)
        alpha_t1, _ = self._alpha_and_dgamma(t1, x.shape, x.device, x.dtype)
        one_hot_x = torch.nn.functional.one_hot(x, self.vocab_size).float()
        alpha_x = (alpha_t1 * one_hot_x).sum(dim=-1)
        one_minus = (1.0 - alpha_t1).clamp_min(1e-12)
        one_minus_x = (1.0 - alpha_x).clamp_min(1e-12)
        denom = one_minus.sum(dim=-1).clamp_min(1e-12)
        loss_recon = one_minus_x * (torch.log(denom) - torch.log(one_minus_x))
        if pair_mask is not None:
            loss_recon = loss_recon * pair_mask.float()
        return loss_recon.sum(dim=tuple(range(1, x.dim())))

    def latent_loss(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.zeros(batch_size, device=device)

    def diffusion_loss(
        self,
        logits: Tensor,
        x: Tensor,
        zt: Tensor,
        t: Tensor,
        pair_mask: Optional[Tensor] = None,
    ) -> Tensor:
        log_p0 = torch.nn.functional.logsigmoid(-logits)
        log_p1 = torch.nn.functional.logsigmoid(logits)
        log_p = torch.stack([log_p0, log_p1], dim=-1)
        one_hot_x = torch.nn.functional.one_hot(x, self.vocab_size).float()
        neg_cross_ent = (one_hot_x * log_p).sum(dim=-1, keepdim=True)
        integrand = (neg_cross_ent + 1.0) * one_hot_x - log_p.exp()

        mask = (zt == self.mask_token).float()
        if pair_mask is not None:
            mask = mask * pair_mask.float()
        _, dgamma = self._alpha_and_dgamma(t, x.shape, x.device, x.dtype)
        weighted = integrand * dgamma
        masked_integrand = (mask[..., None] * weighted).sum(
            dim=tuple(range(1, x.dim()))
        )
        # masked_integrand is [b] (per-sample); return as-is for torch.mean by caller
        return masked_integrand

    def reinforce_loss(
        self,
        t: Tensor,
        x: Tensor,
        zt_1: Tensor,
        zt_2: Tensor,
        loss_diff_1: Tensor,
        loss_diff_2: Tensor,
        pair_mask: Optional[Tensor] = None,
    ) -> Tensor:
        eps = self.noise_schedule.eps
        one_hot_x = torch.nn.functional.one_hot(x, self.vocab_size).float()
        w = self.noise_schedule.power
        w_x = (w * one_hot_x).sum(dim=-1)
        t_ext = _reverse_broadcast(t, x.dim())
        t_ext = t_ext.clamp(min=1e-6, max=1.0)
        if self.position_bias_enabled and self.position_bias_mode == "sigmoid":
            d_pos = _distance_fraction(x.shape[-1], x.device, x.dtype)
            k = torch.as_tensor(self.position_bias_k, device=x.device, dtype=x.dtype)
            k_eff = k * w_x
            alpha_t_x, _ = _sigmoid_alpha(
                t_ext, d_pos, k_eff, self.noise_schedule.eps
            )
            log_q_mask = torch.log((1.0 - alpha_t_x).clamp_min(1e-12))
            log_q_unmask = torch.log(alpha_t_x.clamp_min(1e-12))
        else:
            w_eff = w_x
            if self.position_bias_enabled:
                w_pos = self._position_w(x.shape[-1], x.device, x.dtype)
                if w_pos is not None:
                    w_eff = w_eff * w_pos
            alpha_t_x = 1.0 - (1.0 - eps) * t_ext ** w_eff
            log_q_mask = math.log(1.0 - eps) + w_eff * torch.log(t_ext)
            log_q_unmask = torch.log(alpha_t_x.clamp_min(1e-12))

        log_q1 = torch.where(zt_1 == self.mask_token, log_q_mask, log_q_unmask)
        log_q2 = torch.where(zt_2 == self.mask_token, log_q_mask, log_q_unmask)

        if pair_mask is not None:
            log_q1 = log_q1 * pair_mask.float()
            log_q2 = log_q2 * pair_mask.float()

        sum_q1 = log_q1.sum(dim=tuple(range(1, x.dim())))
        sum_q2 = log_q2.sum(dim=tuple(range(1, x.dim())))
        diff = (loss_diff_1 - loss_diff_2).detach()
        return 0.5 * diff * (sum_q1 - sum_q2)
