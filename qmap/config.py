"""Translate the Hydra config into the option objects used by the data pipeline."""

from omegaconf import ListConfig

from qmap.data.dataset import DatasetOptions, WMReference
from qmap.models.physics import MTParams


def _as_int_list(value):
    """Session cap(s): an int (applies to every dataset) or one int per dataset."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, ListConfig)):
        return [int(v) for v in value]
    return [int(value)]


def dataset_sources(cfg, names):
    """[(root, csv_suffix)] for the named datasets in configs/paths."""
    sources = []
    for name in names:
        if name not in cfg.paths.datasets:
            raise ValueError(f"Unknown dataset '{name}'. Available: {', '.join(cfg.paths.datasets)}")
        entry = cfg.paths.datasets[name]
        sources.append((str(entry.root), str(entry.csv_suffix)))
    return sources


def dataset_options(cfg, phase, datasets=None, eval_wmh=False, use_cache=None):
    """DatasetOptions for `phase` ('train' | 'val' | 'test').

    `datasets` defaults to cfg.data.datasets. The manufacturer / T1w-sequence filters of
    cfg.data.filter always apply.
    """
    sources = dataset_sources(cfg, list(datasets if datasets is not None else cfg.data.datasets))
    return DatasetOptions(
        phase=phase,
        sources=sources,
        mt=MTParams.from_cfg(cfg.physics),
        wm_ref=WMReference.from_cfg(cfg.physics.wm_reference),
        modenorm_mask=cfg.data.modenorm_mask,
        use_cache=bool(cfg.data.cache.enabled if use_cache is None else use_cache),
        cache_dir=cfg.data.cache.dir,
        eval_wmh=eval_wmh,
        max_sessions=_as_int_list(cfg.data.max_sessions[phase]),
        seed=int(cfg.seed),
        filter_manufacturer=[str(v) for v in cfg.data.filter.manufacturer],
        filter_seq_group=[str(v) for v in cfg.data.filter.t1w_sequence],
    )
