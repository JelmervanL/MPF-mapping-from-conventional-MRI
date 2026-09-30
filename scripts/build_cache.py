"""Pre-build the preprocessing cache for one phase (optional; train/test fill it lazily).

The preprocessing (reorientation, crop/pad to 224 x 224, slice removal [train], mode
normalization) is deterministic, so every session is processed once and stored as
<dataset root>/preprocessed_cache/<phase>_pad224_wmnorm[_wmh]/<ID>.pt.

    python scripts/build_cache.py                  # training phase
    python scripts/build_cache.py +phase=val
    python scripts/build_cache.py +phase=test +eval_wmh=true data.datasets=[UMCU]
"""
import os
import sys
import time

import hydra
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmap.config import dataset_options  # noqa: E402
from qmap.data.dataset import create_dataset  # noqa: E402


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg):
    phase = cfg.get('phase', 'train')
    dataset = create_dataset(dataset_options(cfg, phase, eval_wmh=cfg.get('eval_wmh', False), use_cache=True))
    n_workers = min(cfg.data.num_workers, 24)
    print(f"Building preprocessing cache | phase={phase} | {len(dataset)} sessions | {n_workers} workers")
    # Each worker preprocesses and writes its session inside __getitem__; only the ID is returned.
    loader = DataLoader(dataset, batch_size=1, num_workers=n_workers,
                        collate_fn=lambda batch: [str(s.get('subject_id', '')) for s in batch])
    t0 = time.time()
    for _ in tqdm(loader, total=len(dataset), desc="Caching"):
        pass
    print(f"Done in {time.time() - t0:.1f}s.")


if __name__ == '__main__':
    # A utility: no Hydra output directory or log file.
    sys.argv += ['hydra.output_subdir=null', 'hydra.run.dir=.', 'hydra/job_logging=disabled']
    main()
