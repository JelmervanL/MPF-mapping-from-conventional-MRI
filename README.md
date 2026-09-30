# Retrospective Macromolecular Pool Fraction Mapping from Conventional Clinical MRI

This repository contains the implementation of the paper: **"Retrospective macromolecular pool fraction mapping from conventional clinical MRI: a physics-guided self-supervised deep learning framework"**.

It extends our earlier framework for T1, T2 and PD mapping from conventional MRI ([Quantitative-mapping-from-conventional-MRI](https://github.com/JelmervanL/Quantitative-mapping-from-conventional-MRI)).

<p align="center">
  <img src="figures/reveal.gif" alt="T1w, T2w and FLAIR inputs mapped to MPF maps by the self-supervised physics-guided network" width="600">
</p>

## 📋 Abstract

Macromolecular pool fraction (MPF) is a quantitative imaging biomarker sensitive to myelin integrity, but its dependence on specialized acquisitions and reconstruction methods limits its availability for large-scale studies. We present a physics-guided, self-supervised deep learning framework for retrospectively estimating MPF directly from conventional multicontrast clinical brain MRI ($T_1$-weighted, $T_2$-weighted, and fluid-attenuated inversion recovery [FLAIR]), without paired quantitative reference scans. The framework embeds an analytical two-pool magnetization transfer signal model relating sequence-averaged radiofrequency transmit power to apparent longitudinal relaxation. To accommodate heterogeneous clinical protocols, the network conditions intermediate feature representations on acquisition parameters via adaptive instance normalization. The framework was trained and evaluated on a multisite, multivendor, multiprotocol dataset comprising 7,705 MRI sessions (3~T) from 4,489 subjects across Philips and Siemens systems. Across 1,237 test sessions, mean MPF was $0.182 \pm 0.007$ in white matter and $0.144 \pm 0.009$ in gray matter, with limited variation across vendor-by-protocol groups (between-group coefficient of variation: 2.65\% in white matter and 5.63\% in gray matter). In 70 sessions with demyelinating-disease labels, MPF was 36.2\% lower in segmented lesions than in normal-appearing white matter. As a technical demonstration, MPF maps from two different clinical protocols acquired in a healthy volunteer showed a mean absolute percentage difference of 7.7\% and a concordance correlation coefficient of $0.888$, showing good protocol-agnostic consistency. These results suggest that magnetization transfer-informed self-supervised learning enables stable estimation of MPF maps from conventional clinical MRI, providing a scalable approach for investigating demyelinating pathology in both existing imaging archives and prospectively acquired conventional clinical scans without the need for dedicated qMT acquisitions.

## 🛠️ Installation

This project is managed with **[uv](https://github.com/astral-sh/uv)**.

### Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/JelmervanL/MPF-mapping-from-conventional-MRI.git
    cd MPF-mapping-from-conventional-MRI
    ```

2.  **Initialize environment and install dependencies:**

    ```bash
    # install dependencies from pyproject.toml with uv
    uv sync
    ```

-----

## 📂 Project Structure

```text
.
├── checkpoints/trained_model/   # Put the downloaded pre-trained weights here 
├── configs/                     # Hydra configuration files
├── qmap/                        # Main source code package
│   ├── data/
│   │   ├── dataset.py           # CSV + NIfTI -> preprocessed sessions (TorchIO)
│   │   └── transforms.py        # Crop/pad, slice removal, intensity normalization
│   ├── models/
│   │   ├── networks.py          # U-Net type model, with or without AdaIN
│   │   ├── physics.py           # Two-pool apparent T1 and Bloch signal models
│   │   ├── losses.py
│   │   └── qmap_model.py        # Network -> q-maps -> synthetic contrasts -> loss
│   ├── evaluation/              # Slice-wise inference, tissue statistics
│   ├── training.py              # Training loop
│   └── testing.py               # Inference on a test set
├── scripts/
│   ├── run_full_model.sh        # Train and test the proposed model
│   └── build_cache.py           # Pre-build the preprocessing cache (optional)
├── pyproject.toml               # Project configuration and dependencies
├── train.py                     # Main entry point for training
├── test.py                      # Main entry point for inference
└── uv.lock                      # uv lockfile for reproducible environments
```

-----


## 🏃 Usage

### 1\. Data Preparation

#### MRI Preprocessing

The model expects preprocessed NIfTI files. Per the paper's methodology:

1.  **Convert** DICOM to NIfTI (e.g., using `dcm2niix`).
2.  **Resample** the 2D T2w to $1 \times 1 \text{ mm}^2$ in-plane resolution.
3.  **Register** T1w and FLAIR to the T2w space (rigid registration, ANTsPy).
4.  **Skull-strip** (HD-BET) and apply **N4 bias field correction**.
5.  **Segment** brain, WM, GM, CSF and ventricle masks (SynthSeg).


#### CSV Configuration

Each dataset needs one CSV per split (train/val/test) with one row per scan session. The sequence parameters in these CSVs are required for the Bloch signal model.

Required CSV columns:

  * **Session ID:** `ID` (used in the image and mask file names), `Manufacturer`
  * **Sequence Parameters:**
    * **T1w:** `T1w_TR` (Repetition Time), `T1w_TE` (Echo Time), `T1w_FA` (Flip Angle) `T1w_B1rms` (sequence averaged RF power).
    * **T2w:** `T2w_TR`, `T2w_TE`, `T2w_FA`, `T2w_B1rms`. 
    * **FLAIR:** `FLAIR_TR`, `FLAIR_TE`, `FLAIR_TI`, `FLAIR_FA`, `FLAIR_B1rms`. 

#### File Structure Input Data

```text
<dataset root>/
├── images_stripped/
│   ├── <ID>_FLAIR.nii.gz
│   ├── <ID>_T1w.nii.gz
│   └── <ID>_T2w.nii.gz
├── masks_tissue/
│   ├── <ID>_brain.nii.gz
│   ├── <ID>_csf.nii.gz
│   ├── <ID>_gm.nii.gz
│   ├── <ID>_ventricles.nii.gz
│   └── <ID>_wm.nii.gz
├── masks_wmh/                       # optional, lesion analysis only
│   └── <ID>_wmh.nii.gz
├── preprocessed_cache/              # created automatically 
├── train<csv_suffix>
├── val<csv_suffix>
└── test<csv_suffix>
```

#### Dataset Location

Set the data root with an environment variable (default: `data/` in the repository):

```bash
export QMAP_DATA_ROOT=/path/to/datasets
```

### 2\. Training

`scripts/run_full_model.sh` trains the proposed model and then tests it on the UMCU and MR-RATE test sets:

```bash
scripts/run_full_model.sh train
```

To train a single model:

```bash
uv run python train.py experiment=full_model      # proposed model (with AdaIN)
uv run python train.py experiment=no_adain        # without AdaIN
```

### 3\. Inference / Testing

1.  **Download Weights:** Download the pre-trained weights of the proposed model from the **Releases** page.
2.  **Place Weights:** Put the downloaded file in `checkpoints/trained_model/latest_net_G.pth`.
3.  **Run Inference:**

<!-- end list -->

```bash
# the published proposed model on both test sets
scripts/run_full_model.sh paper

# or one model and test set at a time (test_set: umcu or mrrate)
uv run python test.py experiment=full_model test_set=umcu checkpoint=paper     # pre-trained weights
uv run python test.py experiment=full_model test_set=mrrate                      # your own run in outputs/
uv run python test.py experiment=no_adain test_set=umcu                          # your own run without AdaIN
```

The results are saved in the checkpoint folder (`checkpoints/trained_model/` or `outputs/<experiment name>/`), in `test_umcu/` or `test_mrrate/`.

-----

## 📄 License

This project is licensed under a **Creative Commons Attribution-NonCommercial (CC BY-NC)** license.
