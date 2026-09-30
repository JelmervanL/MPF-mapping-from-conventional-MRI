"""Deterministic preprocessing pipeline applied to every session before training/inference."""

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import torchio as tio

# In-plane matrix size of the network input (paper Section 2.1.3).
PATCH_SIZE = 224

# Each contrast is divided by the mode of its intensity histogram inside a mask. The
# percentile trims belong to the mask: they only keep the fixed 512-bin histogram range from
# being stretched by tails the mask does not remove.
MODENORM_MASK_NAMES = {'wm': 'wm_mask', 'brain': 'brain_mask'}
MODENORM_THRESHOLDS = {
    'wm_mask': {
        'T1w': {'lower': 1, 'upper': 99},
        'T2w': {'lower': 1, 'upper': 99},
        'FLAIR': {'lower': 1, 'upper': 99},
    },
    'brain_mask': {
        'T1w': {'lower': 5, 'upper': 95},
        'T2w': {'lower': 5, 'upper': 95},
        'FLAIR': {'lower': 10, 'upper': 95},
    },
}


def modenorm_mask_name(choice):
    """Subject key of the mask the mode is fitted in ('wm' -> 'wm_mask', 'brain' -> 'brain_mask')."""
    if choice not in MODENORM_MASK_NAMES:
        raise ValueError(f"modenorm_mask must be one of {sorted(MODENORM_MASK_NAMES)}, got '{choice}'")
    return MODENORM_MASK_NAMES[choice]


def build_transforms(phase, mask_name='wm_mask'):
    transforms = [tio.ToCanonical(), CustomCropPad()]
    if phase == 'train':
        transforms.append(RemoveSparseMaskedSlices(mask_name='brain_mask', min_voxels=5000))
    # The Otsu mask is not used by the model. It is kept so that cache entries written by this
    # code have the same content as the existing preprocessing caches.
    transforms.append(OtsuMaskTransform(t1_image_name='T1w', mask_name='otsu_mask'))
    # brain_mask is the fallback for a session without a WM mask; it carries its own trims.
    transforms.append(
        ModeNormalization(
            masking_method=mask_name,
            thresholds_map=MODENORM_THRESHOLDS[mask_name],
            fallback_masking_method='brain_mask',
            fallback_thresholds_map=MODENORM_THRESHOLDS['brain_mask'],
        )
    )
    return tio.Compose(transforms)


class ModeNormalization(tio.transforms.preprocessing.intensity.NormalizationTransform):
    """Divide each image by the histogram mode of its intensities inside a mask.

    `masking_method` names the preferred LabelMap; when it is absent or has fewer than
    `min_mask_voxels` voxels, `fallback_masking_method` is used instead.
    """
    def __init__(self, masking_method, thresholds_map=None, fallback_masking_method=None,
                 fallback_thresholds_map=None, min_mask_voxels=1000, **kwargs):
        super().__init__(masking_method=masking_method, **kwargs)
        self.args_names = ['masking_method', 'thresholds_map', 'fallback_masking_method',
                           'fallback_thresholds_map', 'min_mask_voxels']
        self.thresholds_map = thresholds_map or {}
        self.fallback_masking_method = fallback_masking_method
        self.fallback_thresholds_map = fallback_thresholds_map or self.thresholds_map
        self.min_mask_voxels = min_mask_voxels

    def resolve_mask(self, subject):
        """(mask, thresholds_map) for this subject: the preferred mask when usable, else the fallback."""
        candidates = [(self.masking_method, self.thresholds_map)]
        if self.fallback_masking_method:
            candidates.append((self.fallback_masking_method, self.fallback_thresholds_map))

        sub_id = subject.get('subject_id', '?')
        for name, thresholds in candidates:
            if name not in subject or not isinstance(subject[name], tio.LabelMap):
                print(f"[modenorm] no '{name}' for subject {sub_id}; trying next mask.")
                continue
            mask = subject[name].data.bool()
            n_voxels = int(mask.sum())
            if n_voxels >= self.min_mask_voxels:
                return mask, thresholds
            print(f"[modenorm] '{name}' has {n_voxels} < {self.min_mask_voxels} voxels for "
                  f"subject {sub_id}; trying next mask.")

        raise RuntimeError(
            f"No usable normalization mask for subject {sub_id} "
            f"(tried {[c[0] for c in candidates]}).")

    def apply_transform(self, subject):
        mask, thresholds_map = self.resolve_mask(subject)
        for image_name in self.get_images_dict(subject):
            self.apply_normalization(subject, image_name, mask, thresholds_map)
        return subject

    def apply_normalization(self, subject, image_name, mask, thresholds_map=None):
        image = subject[image_name]
        data = image.data

        thresholds_map = self.thresholds_map if thresholds_map is None else thresholds_map
        thresholds = thresholds_map.get(image_name, {})
        normalized = self.modenorm(
            data, mask,
            percentile_threshold_lower=thresholds.get('lower'),
            percentile_threshold_upper=thresholds.get('upper')
        )

        if normalized is None:
            if torch.max(data) == 0:
                return subject
            raise RuntimeError(f'Mode is 0 for masked values in image "{image_name}"')

        image.set_data(normalized)

    @staticmethod
    def get_mode(voxels, bins=512, lower_pct=None, upper_pct=None):
        if lower_pct:
            voxels = voxels[voxels >= torch.quantile(voxels, lower_pct / 100.0)]
        if upper_pct:
            voxels = voxels[voxels <= torch.quantile(voxels, upper_pct / 100.0)]

        if voxels.numel() == 0:
            return None

        min_val, max_val = torch.min(voxels), torch.max(voxels)
        hist = torch.histc(voxels, bins=bins, min=min_val.item(), max=max_val.item())
        bin_edges = torch.linspace(min_val, max_val, steps=bins + 1)
        mode_val = 0.5 * (bin_edges[torch.argmax(hist)] + bin_edges[torch.argmax(hist) + 1])
        return mode_val.item()

    @staticmethod
    def modenorm(tensor, mask, percentile_threshold_lower=None, percentile_threshold_upper=None):
        tensor = tensor.clone().float()
        mode_val = ModeNormalization.get_mode(
            tensor[mask],
            lower_pct=percentile_threshold_lower,
            upper_pct=percentile_threshold_upper
        )
        return (tensor / mode_val) if mode_val and mode_val > 0 else None


class OtsuMaskTransform(tio.Transform):
    """Compute an Otsu foreground mask from the T1w image and attach it to the subject."""
    def __init__(self, t1_image_name='T1w', mask_name='otsu_mask'):
        super().__init__()
        self.t1_name = t1_image_name
        self.mask_name = mask_name

    def apply_transform(self, subject):
        t1 = subject[self.t1_name]

        sitk_img = sitk.GetImageFromArray(t1.data.numpy().squeeze(0))
        sitk_img.SetDirection(t1.affine[:3, :3].flatten().tolist())
        sitk_img.SetOrigin(tuple(t1.affine[:3, 3].tolist()))

        sitk_mask = sitk.OtsuThreshold(sitk.InvertIntensity(sitk.Cast(sitk_img, sitk.sitkFloat32)))

        mask_tensor = torch.from_numpy(sitk.GetArrayFromImage(sitk_mask).astype(np.uint8)).unsqueeze(0)
        subject.add_image(
            image=tio.LabelMap(tensor=mask_tensor, affine=t1.affine),
            image_name=self.mask_name
        )
        return subject


class RemoveSparseMaskedSlices(tio.Transform):
    """Drop axial slices with fewer than `min_voxels` brain-mask voxels (training only)."""
    def __init__(self, mask_name='brain_mask', min_voxels=100):
        super().__init__()
        self.mask_name = mask_name
        self.min_voxels = min_voxels

    def apply_transform(self, subject):
        mask = subject[self.mask_name].data
        if mask.ndimension() == 4:
            mask = mask.squeeze(0)

        valid_slices = mask.sum(dim=(0, 1)) >= self.min_voxels
        if not valid_slices.any():
            raise RuntimeError(f"No valid slices for subject {subject.get('subject_id')}")

        for image in subject.get_images(intensity_only=False):
            image.set_data(image.data[..., valid_slices])
        return subject


class CustomCropPad(tio.Transform):
    """Drop the top/bottom 20 % of axial slices and center-crop/zero-pad in-plane to 224 x 224."""
    def __init__(self, top_crop=0.20, bottom_crop=0.20):
        super().__init__()
        self.top_crop = top_crop
        self.bottom_crop = bottom_crop

    def apply_transform(self, subject):
        target_sz = PATCH_SIZE
        for image in subject.get_images(intensity_only=False):
            data = image.data  # (C, H, W, D)
            C, _, _, D = data.shape

            n_top = int(self.top_crop * D) if isinstance(self.top_crop, float) else self.top_crop
            n_bottom = int(self.bottom_crop * D) if isinstance(self.bottom_crop, float) else self.bottom_crop
            if D - n_top - n_bottom <= 0:
                raise RuntimeError(f"Invalid crop amounts: {n_top} + {n_bottom} >= {D}")
            data = data[..., n_top: D - n_bottom]

            h = data.shape[1]
            if h > target_sz:
                start = (h - target_sz) // 2
                data = data[:, start:start + target_sz, :, :]
            elif h < target_sz:
                diff = target_sz - h
                # F.pad order for (C,H,W,D): (D_l, D_r, W_l, W_r, H_l, H_r)
                data = F.pad(data, (0, 0, 0, 0, diff // 2, diff - diff // 2))

            w = data.shape[2]
            if w > target_sz:
                start = (w - target_sz) // 2
                data = data[:, :, start:start + target_sz, :]
            elif w < target_sz:
                diff = target_sz - w
                data = F.pad(data, (0, 0, diff // 2, diff - diff // 2, 0, 0))

            image.set_data(data.contiguous())
        return subject
