"""Tissue masks and tissue-level summary statistics of the q-maps."""

import numpy as np
from scipy.ndimage import distance_transform_edt

# WM/GM voxels within this distance of CSF are excluded (partial-volume guard, paper Section 2.1.3).
CSF_EROSION_MM = 2.0


def erode_masks_from_csf(masks, m_csf, spacing, mm=CSF_EROSION_MM):
    """Drop voxels within `mm` of CSF from each boolean mask in the dict `masks`."""
    far_from_csf = distance_transform_edt(~m_csf, sampling=spacing) > mm
    return {k: (v.astype(bool) & far_from_csf) for k, v in masks.items()}


def exclude_lesions_from_masks(masks, m_lesion):
    """Remove lesion voxels so WM/GM describe normal-appearing tissue (`m_lesion` may be None)."""
    if m_lesion is None:
        return {k: v.astype(bool) for k, v in masks.items()}
    keep = ~np.asarray(m_lesion).astype(bool)
    return {k: (v.astype(bool) & keep) for k, v in masks.items()}


def get_trimmed_voxels(volume, mask, low_pct=0.1, high_pct=99.9, max_samples_for_kde=5000):
    """Mean and SD of the voxels in `mask` after percentile trimming, plus a subsample for KDE plots."""
    vals = volume[mask]
    if len(vals) == 0:
        return np.nan, np.nan, np.array([])

    p_low = np.percentile(vals, low_pct)
    p_high = np.percentile(vals, high_pct)
    trimmed_vals = vals[(vals >= p_low) & (vals <= p_high)]
    if len(trimmed_vals) == 0:
        return np.nan, np.nan, np.array([])

    mean_val = np.mean(trimmed_vals)
    std_val = np.std(trimmed_vals)
    if len(trimmed_vals) > max_samples_for_kde:
        kde_vals = np.random.choice(trimmed_vals, max_samples_for_kde, replace=False)
    else:
        kde_vals = trimmed_vals
    return mean_val, std_val, kde_vals


def calc_qstar_stats(PD, T1_free, T2, m0s, m_wm, m_gm, m_csf, m_ventricles=None):
    """Mean q-map values per tissue for one session (validation logging)."""
    stats = {
        'PD_WM': PD[m_wm].mean(), 'PD_GM': PD[m_gm].mean(), 'PD_CSF': PD[m_csf].mean(),
        'T1free_WM': T1_free[m_wm].mean(), 'T1free_GM': T1_free[m_gm].mean(), 'T1free_CSF': T1_free[m_csf].mean(),
        'T2_WM': T2[m_wm].mean(), 'T2_GM': T2[m_gm].mean(), 'T2_CSF': T2[m_csf].mean(),
        'PD_Ventricles': PD[m_ventricles].mean() if m_ventricles is not None else None,
        'T1free_Ventricles': T1_free[m_ventricles].mean() if m_ventricles is not None else None,
        'T2_Ventricles': T2[m_ventricles].mean() if m_ventricles is not None else None,
    }
    if m0s is not None:
        stats.update({
            'm0s_WM': m0s[m_wm].mean(), 'm0s_GM': m0s[m_gm].mean(), 'm0s_CSF': m0s[m_csf].mean(),
            'm0s_Ventricles': m0s[m_ventricles].mean() if m_ventricles is not None else None,
        })
    return stats
