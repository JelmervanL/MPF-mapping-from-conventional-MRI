"""Physics-guided self-supervised q-map model: network -> q-maps -> synthetic contrasts -> loss."""

import os
import random

import torch
import torchio as tio

from qmap.models import losses, physics
from qmap.models.networks import build_generator

SCAN_TYPES = ('T1w', 'T2w', 'FLAIR')


class QMapModel:
    """Wraps the generator, the MT-informed forward signal model and the training objective.

    Output channels (after the sigmoid) are scaled to PD [0, 1], T1f [0, 5] s, T2f [0, 3] s
    and m0s [0, 0.4] (paper Section 2.3.1).
    """

    def __init__(self, cfg, device, is_train):
        self.cfg = cfg
        self.device = device
        self.is_train = is_train
        self.mt = physics.MTParams.from_cfg(cfg.physics)
        self.scale = cfg.physics.output_scale
        self.conditioned = cfg.model.arch == 'adain'
        self._current_epoch = 0

        self.netG = build_generator(cfg.model, device)

        self.criterionL1 = torch.nn.L1Loss()
        self.criterionMSE = torch.nn.MSELoss()
        self.criterionSSIM = losses.SSIM().to(self.device)

        if is_train:
            optim = cfg.optim
            self.optimizer = torch.optim.Adam(list(self.netG.parameters()), lr=optim.lr,
                                              betas=(optim.beta1, optim.beta2),
                                              weight_decay=optim.weight_decay)

        n_params = sum(p.numel() for p in self.netG.parameters())
        print(f"[netG: {cfg.model.arch}] {n_params / 1e6:.3f} M parameters")

    # --- bookkeeping ---------------------------------------------------------------------

    def set_epoch(self, epoch):
        """Current epoch (0 = warm-up/inference); the T1f prior is only active for epoch > 0."""
        self._current_epoch = epoch

    def eval(self):
        self.netG.eval()

    def checkpoint_path(self, directory, epoch):
        return os.path.join(directory, f"{epoch}_net_G.pth")

    def save(self, directory, epoch):
        path = self.checkpoint_path(directory, epoch)
        torch.save(self.netG.cpu().state_dict(), path)
        self.netG.to(self.device)
        print(f"Saved {path}")

    def load(self, directory, epoch):
        path = self.checkpoint_path(directory, epoch)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        state_dict = torch.load(path, map_location=self.device)
        self.netG.load_state_dict(state_dict, strict=True)
        print(f"Loaded {path}")

    # --- inputs ----------------------------------------------------------------------------

    def _scalar(self, batch, key):
        return torch.as_tensor(batch[key]).unsqueeze(-1).to(dtype=torch.float, device=self.device)

    def set_input(self, batch):
        self.mask_brain = batch['brain_mask'][tio.DATA].to(self.device).squeeze(-1).bool()
        self.mask_wm = batch['wm_mask'][tio.DATA].to(self.device).squeeze(-1).bool() if 'wm_mask' in batch else None

        self.real_T1w = batch['T1w'][tio.DATA].to(dtype=torch.float, device=self.device).squeeze(-1) * self.mask_brain
        self.real_T2w = batch['T2w'][tio.DATA].to(dtype=torch.float, device=self.device).squeeze(-1) * self.mask_brain
        self.real_FLAIR = batch['FLAIR'][tio.DATA].to(dtype=torch.float, device=self.device).squeeze(-1) * self.mask_brain

        # Sequence parameters (TR/TE/TI in ms, FA in degrees, B1rms in uT). The T2w acquisitions
        # have no inversion time.
        self.params = {
            'T1w': {k: self._scalar(batch, f'T1w_{k}') for k in ('TR', 'TE', 'TI', 'FA', 'B1rms')},
            'T2w': {k: self._scalar(batch, f'T2w_{k}') for k in ('TR', 'TE', 'FA', 'B1rms')},
            'FLAIR': {k: self._scalar(batch, f'FLAIR_{k}') for k in ('TR', 'TE', 'TI', 'FA', 'B1rms')},
        }
        self.params['T2w']['TI'] = None
        self.anchor = {s: self._scalar(batch, f'{s}_rescaling_factor') for s in SCAN_TYPES}
        self.seq_types = {s: batch[f'{s}_seq_type'] for s in SCAN_TYPES}

        scans = {'T1w': self.real_T1w, 'T2w': self.real_T2w, 'FLAIR': self.real_FLAIR}
        inputs = [scans[s] for s in SCAN_TYPES]
        if self.conditioned:
            conditions = [self._condition_vector(s) for s in SCAN_TYPES]
        # The contrast order is shuffled whenever the model is in training mode, including the
        # validation passes during training (the fusion is permutation invariant; the shuffle
        # draws from Python's RNG, so it is kept for identical training runs).
        if self.is_train:
            if self.conditioned:
                combined = list(zip(inputs, conditions))
                random.shuffle(combined)
                inputs, conditions = zip(*combined)
                inputs, conditions = list(inputs), list(conditions)
            else:
                random.shuffle(inputs)
        self.G_inputs = inputs
        self.G_conditions = conditions if self.conditioned else None

    def _condition_vector(self, scan):
        """[is_T1w, is_T2w, is_FLAIR, ln TR, ln TE, sin FA, B1rms, has_TI, ln TI] (paper Eq. 9)."""
        p = self.params[scan]
        TR, TE, FA, B1rms, TI = p['TR'], p['TE'], p['FA'], p['B1rms'], p['TI']
        ln_TR = torch.log(TR + 1e-6)
        ln_TE = torch.log(TE + 1e-6)
        sin_FA = torch.sin(torch.deg2rad(FA))
        if TI is not None:
            has_ti = (~torch.isnan(TI)).float()
            ln_TI = torch.log(torch.nan_to_num(TI, nan=0.0) + 1e-6) * has_ti
        else:
            has_ti = torch.zeros_like(TR)
            ln_TI = torch.zeros_like(TR)
        return torch.cat([
            torch.full_like(TR, 1.0 if scan == 'T1w' else 0.0),
            torch.full_like(TR, 1.0 if scan == 'T2w' else 0.0),
            torch.full_like(TR, 1.0 if scan == 'FLAIR' else 0.0),
            ln_TR, ln_TE, sin_FA, B1rms,
            has_ti, ln_TI
        ], dim=1)

    # --- forward model ---------------------------------------------------------------------

    def forward(self):
        if self.conditioned:
            self.fake_B = self.netG(self.G_inputs, self.G_conditions)
        else:
            self.fake_B = self.netG(self.G_inputs)

        s = self.scale
        self.Q1 = (self.fake_B[:, 0, :, :] * s.pd)[:, None, :, :] * self.mask_brain    # PD
        self.Q2 = (self.fake_B[:, 1, :, :] * s.t1_s)[:, None, :, :] * self.mask_brain  # T1f [s]
        self.Q3 = (self.fake_B[:, 2, :, :] * s.t2_s)[:, None, :, :] * self.mask_brain  # T2f [s]
        self.Q4 = (self.fake_B[:, 3, :, :] * s.m0s)[:, None, :, :] * self.mask_brain   # m0s
        self.PD = self.Q1.squeeze(1)
        self.T1_free = self.Q2.squeeze(1)
        self.T2 = self.Q3.squeeze(1)
        self.m0s = self.Q4.squeeze(1)

        # Mean absolute gradient per map, logged during training.
        self.grad_metrics = {}
        if torch.is_grad_enabled():
            for name, t in (('Grad_PD', self.PD), ('Grad_T1_free', self.T1_free),
                            ('Grad_T2', self.T2), ('Grad_m0s', self.m0s)):
                t.register_hook(lambda grad, name=name: self.grad_metrics.update({name: grad.abs().mean().item()})
                                if grad is not None else None)

        self.fake_raw = {scan: self._synthesize(scan) for scan in SCAN_TYPES}

    def _synthesize(self, scan):
        """Synthetic contrast for every slice in the batch (paper Eqs. 3-7)."""
        p = self.params[scan]
        outs = []
        for b in range(self.PD.size(0)):
            f = self.m0s[b] / (1.0 - self.m0s[b] + 1e-6)   # MPF -> pool-size ratio
            T1_app = physics.apparent_T1(self.T1_free[b], f, p['B1rms'][b], self.mt)
            out = physics.signal_model(
                self.PD[b], T1_app, self.T2[b], p['TR'][b], p['TE'][b],
                None if p['TI'] is None else p['TI'][b], p['FA'][b],
                seq_type=str(self.seq_types[scan][b]),
            )
            outs.append(out.squeeze(0))
        return torch.stack(outs, dim=0)

    def _rescale(self, fake, real, anchor):
        """Closed-form least-squares scaling of `fake` onto `real` inside the brain mask.

        'bounded' clamps the factor to anchor * (1 +/- bound) (paper Section 2.3.2).
        """
        bound = self.cfg.loss.rescaling.bound
        lower = (anchor - bound * anchor).squeeze(1)
        upper = (anchor + bound * anchor).squeeze(1)
        mask = self.mask_brain.float()
        num = (mask * fake * real).sum(dim=(1, 2, 3))
        den = (mask * fake * fake).sum(dim=(1, 2, 3)).clamp(min=1e-6)
        factor = (num / den).detach()
        method = self.cfg.loss.rescaling.method
        if method == 'bounded':
            factor = torch.clamp(factor, min=lower, max=upper)
        elif method != 'unbounded':
            raise ValueError(f"Unknown rescaling method '{method}'")
        return fake * factor.view(-1, 1, 1, 1), factor

    # --- objective -------------------------------------------------------------------------

    def compute_losses_and_metrics(self, optimize=False):
        loss_cfg = self.cfg.loss
        metrics = {}
        l1_losses, l2_losses, pearson_losses, ssim_losses = [], [], [], []
        mask_bool = self.mask_brain.bool()
        reals = {'T1w': self.real_T1w, 'T2w': self.real_T2w, 'FLAIR': self.real_FLAIR}

        for name in SCAN_TYPES:
            fake_raw, real = self.fake_raw[name], reals[name]
            fake_rescaled, factor = self._rescale(fake_raw, real, self.anchor[name])
            setattr(self, f'fake_{name}', fake_rescaled)

            fake_masked = fake_rescaled[mask_bool]
            real_masked = real[mask_bool]
            l1 = self.criterionL1(fake_masked, real_masked)
            l2 = self.criterionMSE(fake_masked, real_masked)
            pearson = losses.pearson_loss(fake_raw[mask_bool], real_masked)
            ssim_score = self.criterionSSIM(fake_rescaled, real, self.mask_brain)

            metrics[f'L1_{name}'] = l1.item()
            metrics[f'L2_{name}'] = l2.item()
            metrics[f'Pearson_{name}'] = pearson.item()
            metrics[f'SSIM_{name}'] = ssim_score.item()
            metrics[f'Factor_{name}'] = factor.mean().item()
            l1_losses.append(l1)
            l2_losses.append(l2)
            pearson_losses.append(pearson)
            ssim_losses.append(1.0 - ssim_score)

        loss_l1 = torch.stack(l1_losses).mean()
        metrics['L1_Global'] = loss_l1.item()
        metrics['L2_Global'] = torch.stack(l2_losses).mean().item()
        metrics['Pearson_Global'] = torch.stack(pearson_losses).mean().item()
        metrics['SSIM_Global'] = torch.stack(ssim_losses).mean().item()

        # PD constraint (hinge). Always reported at inference; during training only when used.
        pd_w = loss_cfg.pd_constraint.weight
        loss_pd = losses.pd_constraint_loss(self.PD, self.mask_brain, threshold=loss_cfg.pd_constraint.threshold)
        if pd_w > 0 or not self.is_train:
            metrics['Loss_PD_constraint'] = loss_pd.item()

        # Gaussian prior on T1f in white matter (active after warm-up, i.e. epoch > 0).
        t1_w = loss_cfg.t1_prior.weight
        loss_t1 = None
        if self._current_epoch > 0 and self.mask_wm is not None and t1_w > 0:
            loss_t1 = losses.gaussian_prior_loss(self.T1_free, self.mask_wm,
                                                 loss_cfg.t1_prior.mean_ms / 1000.0,
                                                 loss_cfg.t1_prior.sigma_ms / 1000.0)
            metrics['Loss_t1_prior'] = loss_t1.item()

        if not optimize:
            return 0.0, metrics
        total = 0.0
        if loss_cfg.image_l1 > 0:
            total += loss_l1 * loss_cfg.image_l1
        if pd_w > 0:
            total += loss_pd * pd_w
        if loss_t1 is not None:
            total += loss_t1 * t1_w
        return total, metrics

    def optimize_parameters(self):
        self.grad_metrics = {}
        self.forward()
        self.optimizer.zero_grad()
        loss, metrics = self.compute_losses_and_metrics(optimize=True)
        loss.backward()
        self.optimizer.step()
        metrics.update(self.grad_metrics)
        return metrics

    def evaluate(self):
        with torch.no_grad():
            self.forward()
            _, metrics = self.compute_losses_and_metrics(optimize=False)
        return metrics

    def warmup_optimize_parameters(self, targets):
        """Supervised warm-up step towards constant WM values inside the brain mask."""
        self.forward()
        self.optimizer.zero_grad()
        s = self.scale
        mask = self.mask_brain.squeeze(1)
        val_PD = targets.pd
        val_T1 = targets.t1_free_ms / 1000.0
        val_T2 = targets.t2_ms / 1000.0
        val_m0s = targets.m0s
        loss = 0
        loss += self.criterionL1(self.PD[mask] / 1, torch.full_like(self.PD[mask], val_PD) / 1)
        loss += self.criterionL1(self.T1_free[mask] / 5, torch.full_like(self.T1_free[mask], val_T1) / 5)
        loss += self.criterionL1(self.T2[mask] / 3, torch.full_like(self.T2[mask], val_T2) / 3)
        loss += self.criterionL1(self.m0s[mask] / s.m0s, torch.full_like(self.m0s[mask], val_m0s) / s.m0s)
        loss.backward()
        self.optimizer.step()
