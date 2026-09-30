"""Training loop with per-epoch validation on the clinical validation set."""

import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torchio as tio
from omegaconf import OmegaConf
from tqdm import tqdm

from qmap.config import dataset_options
from qmap.data.dataset import SliceDataset, create_dataset
from qmap.data.groups import manufacturer_seq_group
from qmap.evaluation.inference import infer_volume
from qmap.evaluation.tissue import calc_qstar_stats, erode_masks_from_csf
from qmap.evaluation.visualization import save_validation_figure
from qmap.models.qmap_model import QMapModel
from qmap.utils import enable_deterministic_mode, select_device, set_seed

MAX_VAL_VISUALS = 8


class RunLogger:
    """Appends every logged dict to <run_dir>/metrics.jsonl and, optionally, to Weights & Biases."""

    def __init__(self, run_dir, cfg):
        self._fh = open(os.path.join(run_dir, 'metrics.jsonl'), 'a')
        self.wandb = None
        if cfg.logging.wandb:
            import wandb
            wandb.init(project=cfg.logging.wandb_project, name=cfg.name,
                       config=OmegaConf.to_container(cfg, resolve=True))
            self.wandb = wandb

    def image(self, path):
        return self.wandb.Image(path) if self.wandb else path

    def log(self, log_dict, step):
        record = {'step': step}
        for k, v in log_dict.items():
            record[k] = v if isinstance(v, (int, float, str, type(None))) else str(v)
        self._fh.write(json.dumps(record) + '\n')
        self._fh.flush()
        if self.wandb:
            self.wandb.log(log_dict, step=step)

    def finish(self):
        self._fh.close()
        if self.wandb:
            self.wandb.finish()


def _subject_loader(dataset, num_workers):
    return torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=num_workers,
                                       collate_fn=lambda x: x[0])


def train(cfg):
    """Train one model. The order of the calls below fixes the random number stream (data
    order, initial weights), so a run with the same seed reproduces the published training."""
    if cfg.deterministic:
        enable_deterministic_mode()
    device = select_device(cfg.gpu)
    set_seed(cfg.seed)
    run_dir = cfg.run_dir
    os.makedirs(run_dir, exist_ok=True)
    logger = RunLogger(run_dir, cfg)

    # --- data ---
    n_workers = min(cfg.data.num_workers, 24)
    dataset = create_dataset(dataset_options(cfg, 'train'))
    print(f'Training size: {len(dataset)} sessions')
    loader_train = tio.SubjectsLoader(
        SliceDataset(dataset, cfg.data.slices_per_volume),
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=n_workers,
        persistent_workers=(n_workers > 0),
        prefetch_factor=4 if n_workers > 0 else None,
        pin_memory=True,
    )
    dataset_val = None
    if cfg.train.validate:
        dataset_val = create_dataset(dataset_options(cfg, 'val'))
        print(f'Validation size: {len(dataset_val)} sessions')

    # --- model ---
    model = QMapModel(cfg, device, is_train=True)

    # Supervised warm-up towards constant WM values for a stable start.
    n_warmup = cfg.train.warmup.iters
    if n_warmup > 0:
        model.set_epoch(0)
        warmup_iter = 0
        pbar = tqdm(total=n_warmup, desc="Warm-up")
        while warmup_iter < n_warmup:
            for data in loader_train:
                if warmup_iter >= n_warmup:
                    break
                model.set_input(data)
                model.warmup_optimize_parameters(cfg.train.warmup.targets)
                warmup_iter += 1
                pbar.update(1)
        pbar.close()

    # --- training ---
    global_step = 0
    n_epochs = cfg.train.epochs
    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        model.set_epoch(epoch)

        pbar = tqdm(total=len(loader_train), desc=f"Epoch {epoch} Train")
        for data in loader_train:
            model.set_input(data)
            metrics = model.optimize_parameters()
            for k, v in metrics.items():
                if np.isnan(v):
                    print(f"NaN detected in metric {k} at epoch {epoch}")
            log_dict = {f"Train/{k}": v for k, v in metrics.items()}
            log_dict['epoch'] = epoch
            logger.log(log_dict, step=global_step)
            global_step += 1
            pbar.update(1)
        pbar.close()

        if dataset_val is not None:
            run_validation(model, cfg, dataset_val, epoch, global_step, logger)

        if epoch % cfg.train.save_every == 0 or epoch == n_epochs:
            model.save(run_dir, epoch)
            model.save(run_dir, 'latest')
        print(f"Epoch {epoch} done in {time.time() - epoch_start:.2f}s")

    logger.finish()


def run_validation(model, cfg, dataset_val, epoch, global_step, logger):
    """Mean image metrics over all slices and mean tissue q-values per vendor-by-protocol group."""
    print("Running validation...")
    patch_metrics_acc, subject_metrics_acc = {}, {}
    subject_counts = defaultdict(int)
    total_patches = 0
    vis_image_paths = {}
    visualized_groups = set()

    for subject in tqdm(_subject_loader(dataset_val, min(cfg.data.num_workers, 24)), desc="Validation"):
        mfr, group, mgroup = manufacturer_seq_group(subject)
        # One quick-look figure per manufacturer x T1w-sequence group.
        is_visual = mgroup not in visualized_groups and len(visualized_groups) < MAX_VAL_VISUALS
        if is_visual:
            visualized_groups.add(mgroup)

        qmaps, synth, n_patches = infer_volume(model, subject, cfg.data.batch_size,
                                               metrics_acc=patch_metrics_acc, synth=is_visual)
        total_patches += n_patches

        if is_visual:
            vis_image_paths[mgroup] = save_validation_figure(
                os.path.join(cfg.run_dir, 'web'), epoch, subject, synth,
                qmaps['PD'], qmaps['T1'], qmaps['T2'], qmaps['m0s'], group=mgroup)

        m_wm = subject['wm_mask'][tio.DATA].numpy().squeeze().astype(bool)
        m_gm = subject['gm_mask'][tio.DATA].numpy().squeeze().astype(bool)
        m_csf = subject['csf_mask'][tio.DATA].numpy().squeeze().astype(bool)
        m_ventricles = subject['ventricles_mask'][tio.DATA].numpy().squeeze().astype(bool) if 'ventricles_mask' in subject else None
        q_stats = calc_qstar_stats(qmaps['PD'], qmaps['T1'], qmaps['T2'], qmaps['m0s'], m_wm, m_gm, m_csf, m_ventricles)
        for k, v in q_stats.items():
            for key in (f"QStar_Global/{k}", f"QStar_{mgroup}/{k}"):
                subject_metrics_acc[key] = subject_metrics_acc.get(key, 0.0) + v

        subject_counts['Global'] += 1
        subject_counts[mgroup] += 1

    log_dict = {f"Val/{k}": v / total_patches for k, v in patch_metrics_acc.items()}
    for k, v in subject_metrics_acc.items():
        group_name = k.split('/')[0].replace('QStar_', '')
        if subject_counts[group_name] > 0:
            log_dict[k] = v / subject_counts[group_name]
    for grp, path in vis_image_paths.items():
        log_dict[f"Val/Visual_{grp}"] = logger.image(path)
    log_dict['epoch'] = epoch
    logger.log(log_dict, step=global_step)
