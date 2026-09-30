"""Quick-look figures: inputs, synthetic contrasts and q-maps of one session (centre slice)."""

import os

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import torchio as tio

COLORMAP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'colormaps')


class SliceVisualizer:
    """3x4 panel figure; T1/T2 use the lipari/navia colormaps (Fuderer et al.)."""

    def __init__(self, colormap_dir=COLORMAP_DIR):
        self.colormap_dir = colormap_dir

    @staticmethod
    def color_log_remap(ori_cmap, loLev, upLev):
        """Re-map a colormap along a log-like curve."""
        assert upLev > 0, "upper level must be positive"
        assert upLev > loLev, "upper level must be larger than lower level"

        map_length = ori_cmap.shape[0]
        e_inv = np.exp(-1.0)
        aVal = e_inv * upLev
        mVal = max(aVal, loLev)
        if aVal >= loLev:
            bVal = (1.0 / map_length) + ((aVal - loLev) / (2 * aVal - loLev))
        else:
            bVal = 1.0 / map_length
        bVal += 1e-7

        log_cmap = np.zeros_like(ori_cmap)
        log_cmap[0, :] = ori_cmap[0, :]
        log_portion = 1.0 / (np.log(mVal) - np.log(upLev))

        for g in range(1, map_length):
            x = (g + 1) * (upLev - loLev) / map_length + loLev
            f = 0.0
            if x > mVal:
                f = map_length * ((np.log(mVal) - np.log(x)) * log_portion * (1 - bVal) + bVal)
            else:
                if (loLev < aVal) and (x > loLev):
                    f = map_length * (((x - loLev) / (aVal - loLev)) * (bVal - (1.0 / map_length))) + 1.0
                if x <= loLev:
                    f = 1.0
            idx = min(map_length, 1 + int(np.floor(f))) - 1
            log_cmap[g, :] = ori_cmap[idx, :]
        return log_cmap

    def relaxation_color_map(self, maptype, x, loLev, upLev):
        """Clip the image and return it with the matching relaxation colormap lookup table."""
        mmm = maptype[0].upper() + maptype[1:]
        if mmm in ['T1', 'R1']:
            csv_filename = os.path.join(self.colormap_dir, 'lipari.csv')
        elif mmm in ['T2', 'T2*', 'R2', 'R2*', 'T1rho', 'T1ρ', 'R1rho', 'R1ρ']:
            csv_filename = os.path.join(self.colormap_dir, 'navia.csv')
        else:
            raise ValueError("Expect 'T1', 'T2', 'R1', or 'R2' as maptype")

        if not os.path.exists(csv_filename):
            print(f"Warning: Colormap file {csv_filename} not found. Using default viridis.")
            return x, 'viridis'

        colortable = np.loadtxt(csv_filename, delimiter=' ', skiprows=1)
        if mmm[0] == 'R':
            colortable = np.flipud(colortable)
        colortable[0, :] = 0.0
        map_length = colortable.shape[0]
        eps_val = (upLev - loLev) / map_length

        x_clip = np.where(x < eps_val,
                          np.where(x < (loLev + eps_val), loLev - eps_val, loLev + eps_val),
                          x)
        if loLev < 0:
            x_clip = np.where(x < eps_val, loLev - eps_val, x)
        return x_clip, self.color_log_remap(colortable, loLev, upLev)

    def save_subject_figure(self, save_path, real_imgs, fake_imgs, q_maps):
        """real_imgs/fake_imgs: [T1w, T2w, FLAIR]; q_maps: [PD, T1f (ms), T2f (ms), m0s]."""
        r_t1, r_t2, r_flair = real_imgs
        f_t1, f_t2, f_flair = fake_imgs
        q_pd, q_t1, q_t2, q_m0s = q_maps
        sl = r_t1.shape[2] // 2

        def get_sl(vol):
            img_2d = np.fliplr(np.rot90(vol[:, :, sl]))
            return img_2d[:, 32:-32]

        fig, axs = plt.subplots(3, 4, figsize=(10.5, 10), facecolor='black', constrained_layout=True,
                                gridspec_kw={'wspace': 0.0, 'hspace': 0.0})

        def plot_ax(ax, img, title, cmap='gray', vmin=None, vmax=None, show_cbar=False):
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, color='white', fontsize=10, pad=5)
            ax.axis('off')
            if show_cbar:
                cbar = plt.colorbar(im, ax=ax, location='bottom', shrink=1, aspect=30, pad=0.02)
                cbar.ax.xaxis.set_tick_params(color='white', labelcolor='white', labelsize=8)
                cbar.outline.set_edgecolor('white')

        plot_ax(axs[0, 0], get_sl(r_t1), 'Real T1w')
        plot_ax(axs[0, 1], get_sl(r_t2), 'Real T2w')
        plot_ax(axs[0, 2], get_sl(r_flair), 'Real FLAIR')
        axs[0, 3].axis('off')

        plot_ax(axs[1, 0], get_sl(f_t1), 'Synth T1w')
        plot_ax(axs[1, 1], get_sl(f_t2), 'Synth T2w')
        plot_ax(axs[1, 2], get_sl(f_flair), 'Synth FLAIR')
        axs[1, 3].axis('off')

        plot_ax(axs[2, 0], get_sl(q_pd), 'Quant. PD', cmap='gray', vmin=0, vmax=1, show_cbar=True)

        T1_lo, T1_up = 0, 3000
        t1_clip, lut_t1 = self.relaxation_color_map('T1', get_sl(q_t1), T1_lo, T1_up)
        cmap_t1 = lut_t1 if isinstance(lut_t1, str) else matplotlib.colors.ListedColormap(lut_t1)
        plot_ax(axs[2, 1], t1_clip, 'Quant. T1 free (ms)', cmap=cmap_t1, vmin=T1_lo, vmax=T1_up, show_cbar=True)

        T2_lo, T2_up = 0, 300
        t2_clip, lut_t2 = self.relaxation_color_map('T2', get_sl(q_t2), T2_lo, T2_up)
        cmap_t2 = lut_t2 if isinstance(lut_t2, str) else matplotlib.colors.ListedColormap(lut_t2)
        plot_ax(axs[2, 2], t2_clip, 'Quant. T2 (ms)', cmap=cmap_t2, vmin=T2_lo, vmax=T2_up, show_cbar=True)

        plot_ax(axs[2, 3], get_sl(q_m0s), 'm0s Fraction', cmap='inferno', vmin=0, vmax=0.3, show_cbar=True)

        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close(fig)


def save_validation_figure(save_dir, epoch, subject, synth, vol_PD, vol_T1_free, vol_T2, vol_m0s, group=""):
    """Quick-look figure of one validation session; returns its path."""
    os.makedirs(save_dir, exist_ok=True)
    mask = subject['brain_mask'][tio.DATA].numpy().squeeze().astype(bool)
    real = [subject[s][tio.DATA].numpy().squeeze() * mask for s in ('T1w', 'T2w', 'FLAIR')]
    fake = [synth[s] for s in ('T1w', 'T2w', 'FLAIR')]
    filename = f"val_epoch_{epoch}_{group}.png" if group else f"val_epoch_{epoch}.png"
    save_path = os.path.join(save_dir, filename)
    SliceVisualizer().save_subject_figure(save_path, real, fake, [vol_PD, vol_T1_free, vol_T2, vol_m0s])
    return save_path
