"""Inference on a test set: q-map NIfTIs, per-session metrics and tissue/lesion statistics."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchio as tio
from scipy.stats import gaussian_kde, kurtosis, skew
from tqdm import tqdm

from qmap.config import dataset_options
from qmap.data.dataset import create_dataset
from qmap.data.groups import manufacturer_seq_group
from qmap.evaluation.inference import infer_volume
from qmap.evaluation.tissue import erode_masks_from_csf, exclude_lesions_from_masks, get_trimmed_voxels
from qmap.evaluation.visualization import SliceVisualizer
from qmap.models.qmap_model import QMapModel
from qmap.utils import save_nifti, select_device

METRICS_TO_TRACK = ['PD', 'T1_free', 'T2', 'm0s']
LABEL_COLS = ['subject_id', 'seq_group', 'manufacturer', 'mgroup']
# Plot ranges of the KDE figures.
CLIP_RANGES = {'PD': (0.3, 1.1), 'T1_free': (0, 3000), 'T2': (0, 200), 'm0s': (0, 0.4)}


def run_test(cfg):
    device = select_device(cfg.gpu)
    ts = cfg.test_set
    dataset = create_dataset(dataset_options(cfg, 'test', datasets=ts.datasets, eval_wmh=ts.eval_wmh,
                                             use_cache=ts.cache))

    out_dir = cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)
    if cfg.save_figures:
        visualizer = SliceVisualizer()
        vis_dir = os.path.join(out_dir, 'output_figures')
        os.makedirs(vis_dir, exist_ok=True)
    kde_dir = os.path.join(out_dir, 'kde_plots')
    os.makedirs(kde_dir, exist_ok=True)

    model = QMapModel(cfg, device, is_train=False)
    model.load(cfg.checkpoint_dir, cfg.epoch)
    model.eval()

    all_metrics_list, all_qstar_list, all_wmh_list = [], [], []
    # tissue -> seq group -> metric -> [(subject_id, voxel subsample)]
    kde_data = {'WM': {}, 'GM': {}}

    print(f"Testing on {len(dataset)} sessions -> {out_dir}")
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4, collate_fn=lambda x: x[0])
    for subject in tqdm(loader, desc="Testing"):
        subj_id = subject['subject_id']
        if isinstance(subj_id, (list, tuple)):
            subj_id = subj_id[0]
        mfr, group, mgroup = manufacturer_seq_group(subject)
        for tissue in ['WM', 'GM']:
            if group not in kde_data[tissue]:
                kde_data[tissue][group] = {m: [] for m in METRICS_TO_TRACK}

        # --- 1. Inference ---
        subj_metrics = {}
        qmaps, synth, patch_count = infer_volume(model, subject, cfg.data.batch_size,
                                                 metrics_acc=subj_metrics, synth=True)
        subj_metrics = {k: v / patch_count for k, v in subj_metrics.items()}
        subj_metrics['subject_id'] = subj_id
        subj_metrics['seq_group'] = group
        subj_metrics['manufacturer'] = mfr
        subj_metrics['mgroup'] = mgroup
        all_metrics_list.append(subj_metrics)

        full_PD, full_T1, full_T2, full_m0s = qmaps['PD'], qmaps['T1'], qmaps['T2'], qmaps['m0s']
        brain = subject['brain_mask'][tio.DATA].numpy().squeeze()
        real = {s: subject[s][tio.DATA].numpy().squeeze() * brain for s in ('T1w', 'T2w', 'FLAIR')}
        m_wmh = (subject['wmh_mask'][tio.DATA].numpy().squeeze().astype(bool)
                 if 'wmh_mask' in subject else None)
        vols = {'PD': full_PD, 'T1_free': full_T1, 'T2': full_T2, 'm0s': full_m0s}

        # --- 2. Statistics inside WMH lesions (raw voxels, no trimming / CSF exclusion) ---
        if ts.eval_wmh and m_wmh is not None:
            all_wmh_list.append(_lesion_stats(subject, subj_id, m_wmh, vols))

        # --- 3. Tissue statistics: CSF-eroded, lesion-free WM/GM, 0.1-99.9 % trimmed ---
        m_wm_raw = subject['wm_mask'][tio.DATA].numpy().squeeze().astype(bool)
        m_gm_raw = subject['gm_mask'][tio.DATA].numpy().squeeze().astype(bool)
        m_csf = subject['csf_mask'][tio.DATA].numpy().squeeze().astype(bool)
        eroded = erode_masks_from_csf({'wm': m_wm_raw, 'gm': m_gm_raw}, m_csf, subject['T1w'].spacing)
        m_wm, m_gm = eroded['wm'], eroded['gm']
        n_wm_pre, n_gm_pre = int(m_wm.sum()), int(m_gm.sum())
        excluded = cfg.exclude_wmh_from_tissue and m_wmh is not None
        if excluded:
            lesion_free = exclude_lesions_from_masks({'wm': m_wm, 'gm': m_gm}, m_wmh)
            m_wm, m_gm = lesion_free['wm'], lesion_free['gm']

        q_stats = {'subject_id': subj_id, 'seq_group': group, 'manufacturer': mfr, 'mgroup': mgroup}
        q_stats['WM_nvox'], q_stats['GM_nvox'] = int(m_wm.sum()), int(m_gm.sum())
        # NaN when no lesion mask was available (distinguishes "not excluded" from "0 removed").
        q_stats['WM_nvox_wmh_removed'] = (n_wm_pre - q_stats['WM_nvox']) if excluded else np.nan
        q_stats['GM_nvox_wmh_removed'] = (n_gm_pre - q_stats['GM_nvox']) if excluded else np.nan
        for vol_name, vol_data in vols.items():
            wm_mean, wm_std, wm_kde_vals = get_trimmed_voxels(vol_data, m_wm, low_pct=0.1, high_pct=99.9)
            q_stats[f'WM_mean_{vol_name}'] = wm_mean
            q_stats[f'WM_std_{vol_name}'] = wm_std
            kde_data['WM'][group][vol_name].append((subj_id, wm_kde_vals))

            gm_mean, gm_std, gm_kde_vals = get_trimmed_voxels(vol_data, m_gm, low_pct=0.1, high_pct=99.9)
            q_stats[f'GM_mean_{vol_name}'] = gm_mean
            q_stats[f'GM_std_{vol_name}'] = gm_std
            kde_data['GM'][group][vol_name].append((subj_id, gm_kde_vals))

            csf_vals = vol_data[m_csf]
            q_stats[f'CSF_mean_{vol_name}'] = np.mean(csf_vals) if len(csf_vals) > 0 else np.nan

        m_ventricles = subject['ventricles_mask'][tio.DATA].numpy().squeeze().astype(bool) if 'ventricles_mask' in subject else None
        if m_ventricles is not None:
            q_stats['Ventricles_mean_PD'] = np.mean(full_PD[m_ventricles])
            q_stats['Ventricles_mean_m0s'] = np.mean(full_m0s[m_ventricles])
        all_qstar_list.append(q_stats)

        # --- 4. NIfTIs ---
        if cfg.save_nifti:
            subject_dir = os.path.join(out_dir, 'niftis', subj_id)
            os.makedirs(subject_dir, exist_ok=True)
            affine = subject['T1w'].affine
            outputs = {'qPD': full_PD, 'qT1_free': full_T1, 'qT2': full_T2, 'qM0s': full_m0s}
            outputs.update({f'fake_{s}': synth[s] for s in ('T1w', 'T2w', 'FLAIR')})
            outputs.update({f'real_{s}': real[s] for s in ('T1w', 'T2w', 'FLAIR')})
            # The exact masks the tissue statistics were computed over.
            outputs.update({'wm_mask_clean': m_wm, 'gm_mask_clean': m_gm})
            for suffix, data in outputs.items():
                save_nifti(data, affine, os.path.join(subject_dir, f"{subj_id}_{suffix}.nii.gz"))
            if 'wmh_mask' in subject:
                save_nifti(subject['wmh_mask'][tio.DATA].numpy().squeeze(), affine,
                           os.path.join(subject_dir, f"{subj_id}_wmh_mask.nii.gz"))

        # --- 5. Quick-look figure ---
        if cfg.save_figures:
            visualizer.save_subject_figure(
                os.path.join(vis_dir, f"{subj_id}_slice_plot.png"),
                real_imgs=[real['T1w'], real['T2w'], real['FLAIR']],
                fake_imgs=[synth['T1w'], synth['T2w'], synth['FLAIR']],
                q_maps=[full_PD, full_T1, full_T2, full_m0s],
            )

    _write_metric_tables(out_dir, all_metrics_list)
    _write_tissue_tables(out_dir, all_qstar_list, cfg.exclude_wmh_from_tissue)
    _plot_kde(kde_dir, kde_data)
    if ts.eval_wmh and all_wmh_list:
        _write_lesion_tables(out_dir, all_wmh_list)

    print(f"Testing complete. Results saved to {out_dir}")


def _lesion_stats(subject, subj_id, m_wmh, vols):
    nvox = int(m_wmh.sum())
    is_ms = str(subject.get('is_MS')).strip().lower() in ('true', '1', 'yes')
    is_demy = 'demyelinat' in str(subject.get('pathology') or '').lower()
    # Empty masks are kept (NaN statistics) so every masked session appears in the CSV.
    rec = {'subject_id': subj_id, 'is_MS': is_ms, 'is_demyelinating': is_demy, 'WMH_nvox': nvox}
    for vol_name, vol_data in vols.items():
        vals = vol_data[m_wmh] if nvox > 0 else np.array([])
        vals = vals[~np.isnan(vals)]
        n = len(vals)
        sd = float(np.std(vals)) if n else np.nan
        rec[f'WMH_mean_{vol_name}'] = float(np.mean(vals)) if n else np.nan
        rec[f'WMH_std_{vol_name}'] = sd
        if n:
            p5, p25, p50, p75, p95 = (float(x) for x in np.percentile(vals, [5, 25, 50, 75, 95]))
        else:
            p5 = p25 = p50 = p75 = p95 = np.nan
        rec[f'WMH_p5_{vol_name}'] = p5
        rec[f'WMH_p25_{vol_name}'] = p25
        rec[f'WMH_median_{vol_name}'] = p50
        rec[f'WMH_p75_{vol_name}'] = p75
        rec[f'WMH_p95_{vol_name}'] = p95
        rec[f'WMH_iqr_{vol_name}'] = (p75 - p25) if n else np.nan
        shape_ok = n >= 3 and sd > 0
        rec[f'WMH_skew_{vol_name}'] = float(skew(vals)) if shape_ok else np.nan
        rec[f'WMH_kurtosis_{vol_name}'] = float(kurtosis(vals)) if shape_ok else np.nan
    return rec


def _front_labels(df):
    front = [c for c in LABEL_COLS if c in df.columns]
    return df[front + [c for c in df.columns if c not in front]]


def _write_metric_tables(out_dir, all_metrics_list):
    df_metrics = _front_labels(pd.DataFrame(all_metrics_list))
    df_metrics.to_csv(os.path.join(out_dir, "test_metrics_subjects.csv"), index=False)
    numeric_cols = [c for c in df_metrics.columns if c not in LABEL_COLS]
    overall = df_metrics[numeric_cols].mean().reset_index()
    overall.columns = ['metric', 'overall_mean']
    overall.to_csv(os.path.join(out_dir, "test_metrics_overall.csv"), index=False)
    df_metrics.groupby('seq_group')[numeric_cols].mean().reset_index().to_csv(
        os.path.join(out_dir, "test_metrics_by_group.csv"), index=False)
    df_metrics.groupby('manufacturer')[numeric_cols].mean().reset_index().to_csv(
        os.path.join(out_dir, "test_metrics_by_manufacturer.csv"), index=False)


def _write_tissue_tables(out_dir, all_qstar_list, exclude_wmh):
    if not all_qstar_list:
        return
    df_qstar = _front_labels(pd.DataFrame(all_qstar_list))
    df_qstar.to_csv(os.path.join(out_dir, "test_qstar_values_subjects.csv"), index=False)

    removed = df_qstar['WM_nvox_wmh_removed']
    n_excl = int(removed.notna().sum())
    if not exclude_wmh:
        print("WM/GM tissue values: WMH lesions NOT excluded.")
    elif n_excl:
        print(f"WM/GM tissue values: WMH lesions removed from the masks of {n_excl}/{len(df_qstar)} "
              f"sessions (median {np.nanmedian(removed):.0f}, max {np.nanmax(removed):.0f} WM voxels).")
    else:
        print("WM/GM tissue values: no WMH masks loaded, so no lesion voxels were removed.")

    numeric_q = [c for c in df_qstar.columns if c not in LABEL_COLS]
    pd.DataFrame({
        'qstar_measure': numeric_q,
        'overall_mean': df_qstar[numeric_q].mean().values,
        'overall_std': df_qstar[numeric_q].std().values
    }).to_csv(os.path.join(out_dir, "qstar_overall.csv"), index=False)
    df_qstar.groupby('seq_group')[numeric_q].mean().reset_index().to_csv(
        os.path.join(out_dir, "qstar_by_group.csv"), index=False)
    df_qstar.groupby('manufacturer')[numeric_q].mean().reset_index().to_csv(
        os.path.join(out_dir, "qstar_by_manufacturer.csv"), index=False)


def _write_lesion_tables(out_dir, all_wmh_list):
    """Per-session lesion statistics and their means/SDs for the MS / demyelinating groups."""
    df_wmh = pd.DataFrame(all_wmh_list)
    df_wmh.to_csv(os.path.join(out_dir, "wmh_qstar_values_subjects.csv"), index=False)

    stat_cols = [c for c in df_wmh.columns if c.startswith('WMH_')]
    is_ms_col = df_wmh['is_MS'].astype(bool)
    is_demy_col = df_wmh['is_demyelinating'].astype(bool)
    group_rows = []
    for grp_name, grp_mask in [('MS', is_ms_col),
                               ('Demyelinating_nonMS', is_demy_col & ~is_ms_col),
                               ('Demyelinating', is_demy_col)]:
        sub = df_wmh[grp_mask.astype(bool) & (df_wmh['WMH_nvox'] > 0)]
        if len(sub) == 0:
            continue
        row = {'group': grp_name, 'n_subjects': len(sub)}
        for c in stat_cols:
            row[f'{c}_group_mean'] = sub[c].mean()
            row[f'{c}_group_std'] = sub[c].std()
        group_rows.append(row)
    if group_rows:
        pd.DataFrame(group_rows).to_csv(os.path.join(out_dir, "wmh_qstar_by_group.csv"), index=False)
    n_empty = int((df_wmh['WMH_nvox'] == 0).sum())
    print(f"WMH lesion stats: {len(df_wmh)} sessions with WMH masks ({len(df_wmh) - n_empty} non-empty), "
          f"{int(df_wmh['is_MS'].sum())} MS, {int(df_wmh['is_demyelinating'].sum())} demyelinating.")


def _plot_kde(kde_dir, kde_data):
    """Mean +/- SD of the per-session KDEs of each q-map in WM and GM, per T1w sequence group."""
    sns.set_theme(style="whitegrid")
    for metric in METRICS_TO_TRACK:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"Mean Distribution ± SD: {metric} (Trimming: 0.1-99.9%)", fontsize=16)
        current_clip = CLIP_RANGES.get(metric, (0, 1))
        x_grid = np.linspace(current_clip[0], current_clip[1], 500)

        for idx, tissue in enumerate(['WM', 'GM']):
            ax = axes[idx]
            plot_generated = False
            for group_name, group_metrics_dict in kde_data[tissue].items():
                subject_kdes = []
                for _, subj_vals in group_metrics_dict.get(metric, []):
                    vals = np.atleast_1d(subj_vals)
                    if len(vals) > 10:
                        try:
                            clean_vals = vals[~np.isnan(vals)]
                            if len(clean_vals) > 10:
                                subject_kdes.append(gaussian_kde(clean_vals)(x_grid))
                        except Exception:
                            pass
                if not subject_kdes:
                    continue

                kdes = np.vstack(subject_kdes)
                mean_kde, std_kde = np.mean(kdes, axis=0), np.std(kdes, axis=0)
                line, = ax.plot(x_grid, mean_kde, label=f"{group_name} (n={len(subject_kdes)})", linewidth=2)
                ax.fill_between(x_grid, np.maximum(0, mean_kde - std_kde), mean_kde + std_kde,
                                alpha=0.3, color=line.get_color())
                plot_generated = True

            if plot_generated:
                ax.set_title(f"{tissue}", fontsize=14)
                ax.set_xlabel(f"{metric} Value", fontsize=12)
                ax.set_ylabel("Density", fontsize=12)
                ax.legend(title="Sequence Group")
                ax.set_xlim(current_clip)
            else:
                ax.set_title(f"{tissue} - No Data", fontsize=14)
                ax.axis('off')

        plt.tight_layout()
        fig.subplots_adjust(top=0.88)
        plt.savefig(os.path.join(kde_dir, f"KDE_MeanSD_{metric}.png"), dpi=300)
        plt.close(fig)
