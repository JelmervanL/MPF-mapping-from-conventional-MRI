"""Forward MRI physics: MT-informed apparent T1 and the Bloch signal equations.

This module is the single source of truth for the equations. Both the training/inference
forward pass (qmap.models.qmap_model) and the per-session rescaling anchor
(qmap.data.dataset) evaluate the signal through these functions, so the synthesized contrasts
and the anchor they are rescaled around always come from the same physics.

Conventions (paper Eqs. 3-8):
  - PD / T1 / T2 are per-voxel maps with T1 and T2 in seconds.
  - TR / TE / TI are shape-(1,) tensors in milliseconds, FA in degrees.
  - `f` is the pool-size ratio F = m0s / (1 - m0s); callers convert the MPF m0s to F.
  - `signal_model` returns a (1, 1, H, W) tensor.
"""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MTParams:
    """Fixed two-pool constants (paper Section 2.3.2)."""
    gamma: float          # gyromagnetic ratio [rad s^-1 T^-1]
    t2_bound_s: float     # T2 of the semisolid pool [s]
    t1_bound_ms: float    # T1 of the semisolid pool [ms]
    exchange_rate: float  # Rx [s^-1], normalised to the total magnetisation

    @classmethod
    def from_cfg(cls, physics_cfg):
        return cls(gamma=float(physics_cfg.gamma), t2_bound_s=float(physics_cfg.t2_bound_s),
                   t1_bound_ms=float(physics_cfg.t1_bound_ms),
                   exchange_rate=float(physics_cfg.exchange_rate))


def _exchange_rate(f, mt):
    """Semisolid -> free exchange rate k^s in the free-pool normalisation.

    Rx is normalised to the total magnetisation (k^f = Rx m0s, k^s = Rx (1 - m0s)), whereas the
    apparent-T1 formula below is written with M0^f = 1. Since 1 + F = 1 / (1 - m0s), dividing by
    (1 + F) converts Rx to k^s, and `k^s * F` then equals k^f = Rx m0s exactly.
    """
    return mt.exchange_rate / (1.0 + f)


def _bound_pool_rate(mt):
    """R1 of the semisolid pool, 1 / T1^s."""
    return 1.0 / (mt.t1_bound_ms / 1000.0)


def _saturation_rate(b1_rms_microtesla, mt):
    """Sequence-averaged saturation rate <W> = sqrt(pi/2) gamma^2 T2^s B1rms^2 (paper Eq. 2).

    math.sqrt keeps the prefactor in double precision (a torch.pi tensor would be float32 and
    leave the float64 rescaling anchor ~1e-9 off the forward pass).
    """
    b1_tesla = b1_rms_microtesla * 1e-6
    prefactor = (mt.gamma ** 2) * math.sqrt(math.pi / 2.0) * mt.t2_bound_s
    return prefactor * (b1_tesla ** 2)


def apparent_T1(T1_free_s, f, b1_rms_microtesla, mt):
    """Apparent T1 [s] from the free-pool T1 (Teixeira et al. 2019; paper Eq. 3)."""
    r_ex = _exchange_rate(f, mt)
    r_b = _bound_pool_rate(mt)
    R_rfb = _saturation_rate(b1_rms_microtesla, mt)
    R_A = 1.0 / (T1_free_s + 1e-6)
    k_f = r_ex * f
    denominator = r_b + R_rfb + r_ex
    numerator = r_b + R_rfb
    R_app = R_A + k_f * (numerator / (denominator + 1e-6))
    return 1.0 / (R_app + 1e-6)


def signal_model(PD, T1_app, T2, TR, TE, TI, FA, seq_type):
    """Closed-form Bloch signal of one acquisition (paper Eqs. 4-7), relaxing with T1_app."""
    has_ti = TI is not None and not torch.isnan(TI).any()
    if has_ti:
        TI = TI / 1000
        TI = TI.unsqueeze(-1)
    TR, TE = TR / 1000, TE / 1000
    TR = TR.unsqueeze(-1)
    TE = TE.unsqueeze(-1)
    FA = FA.unsqueeze(-1)
    FA = torch.deg2rad(FA)

    if seq_type == 'MPRAGE':
        out = torch.abs(PD * (1 - 2 * torch.exp(-TI / (T1_app + 1e-6)) + torch.exp(-TR / (T1_app + 1e-6))) * torch.sin(FA) * torch.exp(-TE / (T2 + 1e-6)) / (1 + torch.cos(FA) * torch.exp(-TR / (T1_app + 1e-6))))
    elif seq_type == 'GRE' or seq_type == 'TFE':
        out = PD * torch.sin(FA) * (1 - torch.exp(-TR / (T1_app + 1e-6))) * torch.exp(-TE / (T2 + 1e-6)) / (1 - torch.cos(FA) * torch.exp(-TR / (T1_app + 1e-6)))
    elif seq_type == 'TSE':
        out = PD * torch.exp(-TE / (T2 + 1e-6)) * (1 - torch.exp(-TR / (T1_app + 1e-6)))
    elif seq_type == 'IR':
        out = torch.abs(PD * torch.exp(-TE / (T2 + 1e-6)) * (1 - 2 * torch.exp(-TI / (T1_app + 1e-6)) + torch.exp(-TR / (T1_app + 1e-6))))
    else:
        raise ValueError(f"Sequence type '{seq_type}' is not supported")

    if out.dim() == 2:
        out = out.unsqueeze(0)
    return out[:, None, :, :]
