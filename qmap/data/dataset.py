"""Session datasets: CSV metadata + NIfTI images -> preprocessed torchio subjects.

Each dataset root is expected to contain
    <root>/<phase><csv_suffix>              one row per session (sequence parameters, see README)
    <root>/images_stripped/<ID>_{T1w,T2w,FLAIR}.nii.gz
    <root>/masks_tissue/<ID>_{brain,wm,gm,csf,ventricles}.nii.gz
    <root>/masks_wmh/<ID>_wmh.nii.gz   (optional, lesion analysis only)
"""

import hashlib
import os
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torchio as tio

from qmap.data.groups import get_seq_group, normalize_manufacturer
from qmap.data.transforms import PATCH_SIZE, build_transforms, modenorm_mask_name
from qmap.models import physics

# Tissue masks attached to every session (zero-filled when absent on disk).
TISSUE_MASK_KEYS = ('csf_mask', 'gm_mask', 'wm_mask', 'ventricles_mask')

# Per-modality fallback sequence type, shared by the subject and the rescaling anchor.
DEFAULT_SEQ_TYPES = {'T1w': 'GRE', 'T2w': 'TSE', 'FLAIR': 'IR'}


@dataclass(frozen=True)
class WMReference:
    """Representative white-matter parameters that define the rescaling anchor (paper Section 2.3.2)."""
    pd: float
    t2_ms: float
    t1_free_ms: float
    m0s: float

    @classmethod
    def from_cfg(cls, ref_cfg):
        return cls(pd=float(ref_cfg.pd), t2_ms=float(ref_cfg.t2_ms),
                   t1_free_ms=float(ref_cfg.t1_free_ms), m0s=float(ref_cfg.m0s))


@dataclass
class DatasetOptions:
    phase: str                                   # 'train' | 'val' | 'test'
    sources: List[Tuple[str, str]]               # [(dataset root, CSV suffix)]
    mt: physics.MTParams
    wm_ref: WMReference
    modenorm_mask: str = 'wm'
    use_cache: bool = False
    cache_dir: Optional[str] = None              # None -> <root>/preprocessed_cache
    eval_wmh: bool = False                       # also load the WMH lesion masks
    max_sessions: Sequence[int] = ()             # per-dataset session cap (0 / empty = all)
    seed: int = 0
    filter_manufacturer: Sequence[str] = field(default_factory=list)
    filter_seq_group: Sequence[str] = field(default_factory=list)


# --- Preprocessing cache -------------------------------------------------------------------
# The transform pipeline is deterministic, so each session's transformed volumes are written to
# <cache root>/<phase>_pad224[_wmnorm][_wmh]/<ID>.pt once and reloaded on every later epoch/run.
# The phase is part of the key because slice removal only runs in 'train'.

def _subject_cache_path(root, opts, sub_id):
    if not opts.use_cache:
        return None
    base = opts.cache_dir if opts.cache_dir else os.path.join(root, 'preprocessed_cache')
    tag = f"{opts.phase}_pad{PATCH_SIZE}"
    if modenorm_mask_name(opts.modenorm_mask) == 'wm_mask':
        tag += '_wmnorm'
    if opts.eval_wmh:
        tag += '_wmh'
    return os.path.join(base, tag, f"{sub_id}.pt")


def _subject_to_blob(subject):
    """Serialize a transformed tio.Subject to a plain dict (label maps as uint8)."""
    blob = {}
    for key, val in subject.items():
        if isinstance(key, str) and key.startswith('_'):
            continue
        if isinstance(val, tio.Image):
            data = val.data
            is_label = isinstance(val, tio.LabelMap)
            if is_label and data.numel() and float(data.max()) <= 255:
                data = data.round().to(torch.uint8)
            else:
                data = data.clone()
            blob[key] = {
                '__image__': True,
                'label': is_label,
                'data': data,
                'affine': torch.as_tensor(np.asarray(val.affine), dtype=torch.float64),
            }
        else:
            blob[key] = val
    return blob


def _blob_to_subject(blob):
    kwargs = {}
    for key, val in blob.items():
        if isinstance(val, dict) and val.get('__image__'):
            affine = np.asarray(val['affine'])
            if val.get('label'):
                kwargs[key] = tio.LabelMap(tensor=val['data'], affine=affine)
            else:
                kwargs[key] = tio.ScalarImage(tensor=val['data'].float(), affine=affine)
        else:
            kwargs[key] = val
    return tio.Subject(**kwargs)


def _load_subject_from_cache(path):
    return _blob_to_subject(torch.load(path, map_location='cpu', weights_only=False))


def _atomic_save_subject(subject, path):
    """Write to a temporary file first so concurrent workers never read a partial file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(_subject_to_blob(subject), tmp)
    os.replace(tmp, path)


class CachedSubjectsDataset(tio.SubjectsDataset):
    """SubjectsDataset that reads/writes the per-session preprocessing cache.

    `scalar_overrides` (rescaling anchor, pathology labels) are re-applied on every load, so a
    cache entry written under different settings cannot supply stale values for them.
    """

    def __init__(self, subjects, transform=None, cache_paths=None, use_cache=False,
                 scalar_overrides=None):
        super().__init__(subjects, transform=transform)
        self._cache_paths = cache_paths if cache_paths is not None else [None] * len(subjects)
        self._use_cache = use_cache
        self._scalar_overrides = scalar_overrides if scalar_overrides is not None else [{}] * len(subjects)

    def __getitem__(self, index):
        cache_file = self._cache_paths[index] if self._use_cache else None

        subject = None
        if cache_file and os.path.exists(cache_file):
            try:
                subject = _load_subject_from_cache(cache_file)
            except Exception as e:  # corrupt/incompatible cache -> recompute
                print(f"[cache] failed to load {cache_file}: {e}; recomputing")
                subject = None

        if subject is None:
            subject = super().__getitem__(index)  # runs the transform pipeline
            if cache_file is not None:
                try:
                    _atomic_save_subject(subject, cache_file)
                except Exception as e:
                    print(f"[cache] failed to save {cache_file}: {e}")

        for k, v in self._scalar_overrides[index].items():
            subject[k] = v

        # Every session carries the same tissue-mask keys (zero-filled when missing on disk),
        # so torchio's collate never sees subjects with different key sets.
        missing = [k for k in TISSUE_MASK_KEYS if k not in subject]
        if missing:
            ref = subject['brain_mask']
            for k in missing:
                subject.add_image(tio.LabelMap(tensor=torch.zeros_like(ref.data), affine=ref.affine), k)
        return subject


# --- Session selection ---------------------------------------------------------------------

def _resolve_nifti(path):
    """Return `path`, or its .nii <-> .nii.gz alternative when only that exists."""
    if os.path.exists(path):
        return path
    if path.endswith('.nii.gz'):
        alt = path[:-len('.nii.gz')] + '.nii'
    elif path.endswith('.nii'):
        alt = path[:-len('.nii')] + '.nii.gz'
    else:
        return path
    return alt if os.path.exists(alt) else path


def _adjust_te_params(df, root):
    """Replace the nominal TSE echo times by the effective ones (paper Supplementary Section C).

    Uses the CSV's `{T2w,FLAIR}_TEeff` (or `_TEfactor`) columns. CSVs without them fall back
    to the Philips factors 0.9 (T2w) and 0.42 (3D FLAIR).
    """
    for mod in ['T2w', 'FLAIR']:
        if f'{mod}_TE' not in df.columns:
            raise ValueError(f"{mod}_TE missing from CSV at {root}")
        eff_col, fac_col = f'{mod}_TEeff', f'{mod}_TEfactor'
        if eff_col in df.columns:
            df[f'{mod}_TE'] = df[eff_col]
        elif fac_col in df.columns:
            df[f'{mod}_TE'] = df[f'{mod}_TE'] * df[fac_col]
        elif mod == 'FLAIR':
            df['FLAIR_TE'] = df['FLAIR_TE'].apply(lambda x: x * 0.89 if x < 200 else x * 0.42)
        else:
            df['T2w_TE'] = df['T2w_TE'] * 0.9
    return df


def _load_subject_df(root, suffix, phase):
    path = os.path.join(root, phase + suffix)
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def _stable_seed(*parts):
    """Deterministic 32-bit seed (hash() of strings is salted per process)."""
    digest = hashlib.sha256("||".join(str(p) for p in parts).encode()).hexdigest()
    return int(digest[:8], 16)


def _session_cap(opts, ds_index):
    """Session cap for dataset `ds_index`: one value for all datasets, or one per dataset."""
    caps = list(opts.max_sessions)
    if not caps:
        return 0
    if len(caps) == 1:
        cap = caps[0]
    elif ds_index < len(caps):
        cap = caps[ds_index]
    else:
        cap = caps[-1]
    return cap if cap and cap > 0 else 0


def _row_manufacturer(row):
    return normalize_manufacturer(row.get('Manufacturer', 'Unknown'))


def _canonical_filter_values(tokens, classifier, axis, valid):
    """Resolve filter tokens through the same classifier that later groups the results."""
    out = []
    for tok in tokens:
        val = classifier(tok)
        if val not in valid:
            raise ValueError(
                f"filter {axis}: '{tok}' is not a recognised {axis} (it resolves to '{val}'). "
                f"Valid values: {', '.join(sorted(valid))}.")
        out.append(val)
    return out


def _filter_groups(df, opts, root):
    """Keep only the requested manufacturers / T1w sequence groups (ANDed)."""
    want_mfr = list(opts.filter_manufacturer or [])
    want_grp = list(opts.filter_seq_group or [])
    if not want_mfr and not want_grp:
        return df

    mask = pd.Series(True, index=df.index)
    desc = []
    if want_mfr:
        vals = _canonical_filter_values(want_mfr, normalize_manufacturer, 'manufacturer',
                                        {'Siemens', 'Philips', 'GE'})
        mask &= df.apply(_row_manufacturer, axis=1).isin(vals)
        desc.append(f"manufacturer={'|'.join(vals)}")
    if want_grp:
        vals = _canonical_filter_values(want_grp, lambda t: get_seq_group({'T1w_seq_type': t}),
                                        'seq_group', {'MPRAGE', 'GRE_TFE', 'SE'})
        mask &= df.apply(get_seq_group, axis=1).isin(vals)
        desc.append(f"seq_group={'|'.join(vals)}")

    kept = df[mask]
    print(f"[filter] {os.path.basename(os.path.normpath(root))} ({opts.phase}): "
          f"{len(kept)}/{len(df)} sessions kept ({', '.join(desc)})")
    return kept


def _subject_ids(df, opts, root, ds_index):
    all_ids = df['ID'].values.tolist()

    # Seeded random subset, reproducible across runs and processes.
    cap = _session_cap(opts, ds_index)
    if cap and cap < len(all_ids):
        csv_names = tuple(suffix for _, suffix in opts.sources)
        rng = np.random.default_rng(_stable_seed(root, csv_names, opts.phase, opts.seed))
        selected = rng.choice(all_ids, size=cap, replace=False).tolist()
        print(f"[max_sessions.{opts.phase}] {root}: using {len(selected)}/{len(all_ids)} sessions.")
        return selected
    return all_ids


# --- Rescaling anchor ----------------------------------------------------------------------
# The synthesized images are rescaled to the inputs by a least-squares factor that is clamped
# to +/- bound around a per-session reference factor, 1 / (synthetic WM signal). That reference
# is evaluated with the model's own physics for representative WM parameters (float64).

def _seq_type(row, mod):
    seq = row.get(f'{mod}_seq_type', None)
    return str(seq) if pd.notna(seq) else DEFAULT_SEQ_TYPES[mod]


def _wm_reference_T1_s(mod, row, opts):
    dt = torch.float64
    T1_free_s = torch.tensor([[opts.wm_ref.t1_free_ms / 1000.0]], dtype=dt)
    m0s = torch.tensor([[opts.wm_ref.m0s]], dtype=dt)
    f = m0s / (1.0 - m0s + 1e-6)
    b1 = torch.tensor([float(row[f'{mod}_B1rms'])], dtype=dt)
    return float(physics.apparent_T1(T1_free_s, f, b1, opts.mt).reshape(-1)[0])


def _synth_wm_signal(mod, row, opts, T1_s):
    """WM reference signal with the forward model (TE of T2w/FLAIR is already effective)."""
    dt = torch.float64
    PD = torch.tensor([[opts.wm_ref.pd]], dtype=dt)
    T1_app = torch.tensor([[float(T1_s)]], dtype=dt)
    T2 = torch.tensor([[opts.wm_ref.t2_ms / 1000.0]], dtype=dt)
    TR = torch.tensor([float(row[f'{mod}_TR'])], dtype=dt)
    TE = torch.tensor([float(row[f'{mod}_TE'])], dtype=dt)
    FA = torch.tensor([float(row[f'{mod}_FA'])], dtype=dt)
    # The forward pass never uses a T2w TI, so neither does the anchor.
    ti_raw = None if mod == 'T2w' else row.get(f'{mod}_TI', None)
    TI = None if ti_raw is None else torch.tensor([float(ti_raw)], dtype=dt)
    out = physics.signal_model(PD, T1_app, T2, TR, TE, TI, FA, seq_type=_seq_type(row, mod))
    return float(out.reshape(-1)[0])


def rescale_anchor(row, mod, opts):
    """Per-session reference rescaling factor = 1 / synthetic WM signal."""
    sig = _synth_wm_signal(mod, row, opts, _wm_reference_T1_s(mod, row, opts))
    if not (np.isfinite(sig) and sig > 0):
        raise ValueError(f"Non-usable WM reference signal ({sig}) for {mod} of session {row.get('ID', '?')}")
    return 1.0 / sig


def _create_subject(row, root, opts, rescale):
    """Build one torchio Subject (images, masks, sequence parameters) from a CSV row."""
    sub_id = row['ID']
    dir_img = os.path.join(root, 'images_stripped')
    dir_msk = os.path.join(root, 'masks_tissue')

    fnames = {
        'T1w': row.get('T1w_file', sub_id + '_T1w.nii.gz'),
        'T2w': row.get('T2w_file', sub_id + '_T2w.nii.gz'),
        'FLAIR': row.get('FLAIR_file', sub_id + '_FLAIR.nii.gz')
    }

    kwargs = {
        'subject_id': sub_id,
        'brain_mask': tio.LabelMap(_resolve_nifti(os.path.join(dir_msk, sub_id + '_brain.nii.gz'))),

        'T1w_TR': row['T1w_TR'], 'T1w_TE': row['T1w_TE'], 'T1w_FA': row['T1w_FA'], 'T1w_TI': row.get('T1w_TI', None),
        'T1w_B1rms': row['T1w_B1rms'],
        'T2w_TR': row['T2w_TR'], 'T2w_TE': row['T2w_TE'], 'T2w_FA': row['T2w_FA'],
        'T2w_B1rms': row['T2w_B1rms'],
        'FLAIR_TR': row['FLAIR_TR'], 'FLAIR_TE': row['FLAIR_TE'], 'FLAIR_TI': row['FLAIR_TI'], 'FLAIR_FA': row['FLAIR_FA'],
        'FLAIR_B1rms': row['FLAIR_B1rms'],

        'T1w_rescaling_factor': rescale['T1w'],
        'T2w_rescaling_factor': rescale['T2w'],
        'FLAIR_rescaling_factor': rescale['FLAIR'],

        'T1w_seq_type': _seq_type(row, 'T1w'),
        'T2w_seq_type': _seq_type(row, 'T2w'),
        'FLAIR_seq_type': _seq_type(row, 'FLAIR'),

        'manufacturer': row.get('Manufacturer', 'Unknown'),
        'scanner_model': row.get('ManufacturersModelName', 'Unknown'),
    }

    for mod, fname in fnames.items():
        path = _resolve_nifti(os.path.join(dir_img, fname))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {mod}: {path}")
        kwargs[mod] = tio.ScalarImage(path)

    for tissue in ['csf', 'gm', 'wm', 'ventricles']:
        p = _resolve_nifti(os.path.join(dir_msk, f"{sub_id}_{tissue}.nii.gz"))
        if os.path.exists(p):
            kwargs[f"{tissue}_mask"] = tio.LabelMap(p)

    # WMH lesion mask. Not zero-filled: its presence marks the lesion-analysis sessions.
    if opts.eval_wmh:
        p = _resolve_nifti(os.path.join(root, 'masks_wmh', f"{sub_id}_wmh.nii.gz"))
        if os.path.exists(p):
            kwargs['wmh_mask'] = tio.LabelMap(p)

    return tio.Subject(**kwargs)


def create_dataset(opts: DatasetOptions) -> CachedSubjectsDataset:
    """Volume-level dataset of preprocessed sessions from all `opts.sources`."""
    mask_name = modenorm_mask_name(opts.modenorm_mask)
    print(f"Creating {opts.phase} dataset | mode normalization in {mask_name} | "
          f"WM reference: PD={opts.wm_ref.pd}, T2={opts.wm_ref.t2_ms} ms, "
          f"T1f={opts.wm_ref.t1_free_ms} ms, m0s={opts.wm_ref.m0s}")

    all_subjects, cache_paths, scalar_overrides = [], [], []
    groups_seen = set()
    for ds_index, (root, suffix) in enumerate(opts.sources):
        df = _load_subject_df(root, suffix, opts.phase)
        df = _adjust_te_params(df, root)

        # Filtering happens before the session cap, so the cap applies to the filtered pool.
        groups_seen.update(f"{_row_manufacturer(r)}_{get_seq_group(r)}" for _, r in df.iterrows())
        df = _filter_groups(df, opts, root)
        subject_ids = _subject_ids(df, opts, root, ds_index)
        print(f"Loading {len(subject_ids)} sessions from {root}...")

        for sub_id in subject_ids:
            row = df[df['ID'] == sub_id].iloc[0]
            rescale = {mod: rescale_anchor(row, mod, opts) for mod in ('T1w', 'T2w', 'FLAIR')}
            try:
                subj = _create_subject(row, root, opts, rescale)
            except FileNotFoundError as e:
                print(f"Skipping {sub_id}: {e}")
                continue
            all_subjects.append(subj)
            cache_paths.append(_subject_cache_path(root, opts, sub_id))
            scalar_overrides.append({
                'T1w_rescaling_factor': rescale['T1w'],
                'T2w_rescaling_factor': rescale['T2w'],
                'FLAIR_rescaling_factor': rescale['FLAIR'],
                # Pathology labels for the lesion analysis (None when the CSV lacks the columns).
                'is_MS': row.get('is_MS', None),
                'pathology': row.get('pathology', None),
            })

    if not all_subjects and (opts.filter_manufacturer or opts.filter_seq_group):
        raise ValueError(
            f"No sessions matched manufacturer={list(opts.filter_manufacturer) or 'any'} "
            f"seq_group={list(opts.filter_seq_group) or 'any'} in phase '{opts.phase}'. "
            f"Groups present: {', '.join(sorted(groups_seen)) or 'none'}.")

    return CachedSubjectsDataset(
        all_subjects,
        transform=build_transforms(opts.phase, mask_name),
        cache_paths=cache_paths,
        use_cache=opts.use_cache,
        scalar_overrides=scalar_overrides,
    )


class SliceDataset(torch.utils.data.Dataset):
    """Single random axial slices from a volume dataset (training).

    len = n_sessions * slices_per_volume; a shuffled DataLoader over this index space samples
    `slices_per_volume` random slices of every session per epoch.
    """

    def __init__(self, vol_ds, slices_per_volume):
        self.vol_ds = vol_ds
        self.ppv = max(1, int(slices_per_volume))

    def __len__(self):
        return len(self.vol_ds) * self.ppv

    def __getitem__(self, idx):
        full = self.vol_ds[idx // self.ppv]

        # All images share the depth after preprocessing.
        ref = next(v for v in full.values() if isinstance(v, tio.Image))
        z = random.randrange(ref.data.shape[-1])

        kwargs = {}
        for key, val in full.items():
            if isinstance(val, tio.Image):
                sl = val.data[..., z:z + 1].clone()
                cls = tio.LabelMap if isinstance(val, tio.LabelMap) else tio.ScalarImage
                kwargs[key] = cls(tensor=sl, affine=val.affine)
            else:
                kwargs[key] = val
        return tio.Subject(**kwargs)
