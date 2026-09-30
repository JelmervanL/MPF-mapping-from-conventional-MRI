"""Slice-wise inference over a whole session volume."""

import torch
import torchio as tio

from qmap.data.transforms import PATCH_SIZE

QMAP_KEYS = ('PD', 'T1', 'T2', 'm0s')
_QMAP_ATTR = {'PD': 'Q1', 'T1': 'Q2', 'T2': 'Q3', 'm0s': 'Q4'}


def infer_volume(model, subject, batch_size, metrics_acc=None, synth=False):
    """Run the model on every axial slice of `subject` and stitch the outputs.

    Returns (qmaps, synth_images, n_patches): q-maps as numpy volumes (T1/T2 in ms) and, if
    `synth`, the rescaled synthetic contrasts. When `metrics_acc` is a dict, the per-batch
    metrics of `model.evaluate()` are summed into it; otherwise only the forward pass runs.

    Note: iterating the torchio loader draws from the global torch RNG (loader seed and one
    draw per batch inside torchio's Crop). During training this happens between epochs, so it
    is part of what makes a seeded training run reproducible.
    """
    grid_sampler = tio.inference.GridSampler(subject, (PATCH_SIZE, PATCH_SIZE, 1), patch_overlap=(0, 0, 0))
    loader = tio.SubjectsLoader(grid_sampler, batch_size=batch_size, num_workers=0)
    aggs = {k: tio.inference.GridAggregator(grid_sampler) for k in QMAP_KEYS}
    synth_aggs = {s: tio.inference.GridAggregator(grid_sampler) for s in ('T1w', 'T2w', 'FLAIR')} if synth else {}

    n_patches = 0
    for patch in loader:
        model.set_input(patch)
        if metrics_acc is not None:
            metrics = model.evaluate()
            for k, v in metrics.items():
                metrics_acc[k] = metrics_acc.get(k, 0.0) + v
        else:
            with torch.no_grad():
                model.forward()
        n_patches += 1

        locs = patch[tio.LOCATION]
        for k in QMAP_KEYS:
            aggs[k].add_batch(getattr(model, _QMAP_ATTR[k]).detach().cpu().unsqueeze(-1), locs)
        for s, agg in synth_aggs.items():
            agg.add_batch(getattr(model, f'fake_{s}').detach().cpu().unsqueeze(-1), locs)

    qmaps = {k: agg.get_output_tensor().numpy().squeeze() for k, agg in aggs.items()}
    qmaps['T1'] = qmaps['T1'] * 1000  # s -> ms
    qmaps['T2'] = qmaps['T2'] * 1000
    synth_images = {s: agg.get_output_tensor().numpy().squeeze() for s, agg in synth_aggs.items()}
    return qmaps, synth_images, n_patches
