**Are Deep Generative Models Ready to Support Marine Mammal Acoustic Classifiers? A Case Study for the South Resident Killer Whale**

This repository contains all scripts and tools used in the paper above. It supports reproducible training, evaluation, and comparison of deep learning models for whale-call detection and general marine bioacoustic tasks, including:
- Generating spectrogram databases from continuous audio data
- Training a **ResNet-based classifier** for call detection
- Generating **synthetic spectrograms** with **VAE**, **GAN**, and **DDPM** models
- Evaluating detection performance using custom **segment-based precision–recall** analysis

The toolkit targets Southern Resident Killer Whale (SRKW) call synthesis and augmentation but is general enough for spectrogram-based tasks.

## Installation

1) Clone the repo
```bash
git clone https://github.com/bpadovese/srkw-augment.git
cd srkw-augment
```

2) Install dependencies
```bash
pip install -r requirements.txt
```

GPU strongly recommended.

## Repository Structure

```bash
GAugSRKW/
├── create_db.py              # Dataset creation / spectrogram DB builder
├── inference.py              # Segment-based classifier inference
├── train_classifier.py       # Training entrypoint for ResNet classifier
├── metrics.py                # Segment-based evaluation utilities
├── .gitignore
├── README.md
├── augmentation/             # Augmentation training & generation
│   ├── ddpm.py
│   ├── gans.py
│   ├── masking.py
│   └── vae.py
├── data_handling/            # Dataset and spectrogram preprocessing
│   ├── dataset.py
│   └── spec_preprocessing.py
├── dev_utils/                # Helpers, audio frontend, validation
│   ├── annotation_processing.py
│   ├── audio_processing.py
│   ├── file_management.py
│   ├── nn.py
│   └── validate_generation.py
└── ...
```

## Workflow

### 1. Prepare your audio data and annotations
- Organize files:
  ```bash
  data/
    ├── audio/
    │   ├── file1.wav
    │   ├── file2.wav
    └── annotations.csv
  ```
- `annotations.csv` columns: `filename,start,end,label`.

### 2. Create a baseline spectrogram database (no augmentation)

Assumes an annotation CSV with KW labels:
```bash
python create_db.py data/audio configs/audio_config.json \
    --annotations data/annotations.csv \
    --random_selections same 0 \
    --output dataset_images/baseline \
    --labels KW=1
```
This will:
- Read each annotation segment for label `KW`.
- Generate spectrograms per the JSON config.
- Save positives to `dataset_images/baseline/1/`.
- Create an equal number of background samples in `dataset_images/baseline/0/`.

#### Audio representation configuration
The data and inference scripts use a JSON file. Example `audio_config.json`:
```json
{
    "sr": 24000,
    "window": 0.05,
    "step": 0.0125,
    "num_filters": 128,
    "duration": 3,
    "fmin": 0,
    "fmax": 12000
}
```

### 3. Create augmented dataset (time-shifted samples)
```bash
python create_db.py data/audio configs/audio_config.json \
    --annotations data/annotations.csv \
    --annotation_step 0.5 \
    --only_augmented \
    --output dataset_images/augmented/time-shift
```
This will:
- Generate time-shifted segments with step size `0.5` s.
- Include only augmented samples.
- Save to `dataset_images/augmented/time-shift`.

### 4. Mask construction and background merging
Each mask is obtained by:
- Performing PCA background estimation (first principal component models stationary noise).
- Subtracting this reconstruction from the original spectrogram.
- Applying percentile-based thresholding to supress residual noise.

```bash
python -m augmentation.masking \
    --mode mask \
    --input_folder dataset_images/baseline/1 \
    --output_folder mask_outputs/KW_masks \
    --percentile 95 \
    --n_components 1
```
This will:
- Load each spectrogram from `dataset_images/baseline/1/`.
- Perform PCA reconstruction with one component.
- Retain only the 95th-percentile energy pixels.
- Save masks to `mask_outputs/KW_masks/mask/`.

#### Merging masks with background spectrograms
```bash
python -m augmentation.masking \
    --mode merge \
    --mask_folder mask_outputs/KW_masks/mask \
    --background_folder dataset_images/baseline/0 \
    --output_folder dataset_images/mask_augmented/1 \
    --num_samples 500
```
This will:
- Randomly select masks and background spectrograms.
- Overlay each mask onto a background.
- Generate 500 synthetic spectrograms in `dataset_images/mask_augmented/1/`.

### 5. VAE-based data augmentation
Train on baseline positives (e.g., `KW=1`):
```bash
python -m augmentation.vae \
    --dataset dataset_images/baseline/1 \
    --output_dir vae_models/KW_vae \
    --image_size 128 \
    --num_epochs 100 \
    --batch_size 64 \
    --beta 1.0
```
This will:
- Train a VAE on `dataset_images/baseline/1/`.
- Save checkpoints and samples to `vae_models/KW_vae/`.
- Output the final model as `vae_final.pth`.

Generate synthetic spectrograms:
```bash
python -m augmentation.vae \
    --generate_only \
    --pretrained_vae vae_models/KW_vae/vae_final.pth \
    --num_samples 500 \
    --output_dir dataset_images/vae_augmented/1 \
    --image_size 128
```
This will sample 500 latent vectors and save them to `dataset_images/vae_augmented/1/`.

Optional PCA-based filtering:
```bash
python -m augmentation.vae \
    --generate_only \
    --pretrained_vae vae_models/KW_vae/vae_final.pth \
    --num_samples 500 \
    --validation_dir dataset_images/baseline/1 \
    --output_dir dataset_images/vae_augmented/1
```

### 6. GAN-based data augmentation
Train the GAN:
```bash
python -m augmentation.gans \
    --dataset dataset_images/baseline/1 \
    --output_dir gan_models/KW_gan \
    --image_size 128 \
    --num_epochs 400 \
    --batch_size 64
```
This will train a generator-discriminator pair, saving sample grids to `gan_models/KW_gan/epoch_X.png` and final weights to `generator_final.pth` / `critic_final.pth`.

Generate synthetic spectrograms:
```bash
python -m augmentation.gans \
    --generate_only \
    --pretrained_generator gan_models/KW_gan/generator_final.pth \
    --num_samples 500 \
    --output_dir dataset_images/gan_augmented/1 \
    --image_size 128
```
Optional PCA filtering:
```bash
python -m augmentation.gans \
    --generate_only \
    --pretrained_generator gan_models/KW_gan/generator_final.pth \
    --num_samples 500 \
    --validation_dir dataset_images/baseline/1 \
    --output_dir dataset_images/gan_augmented/1
```

### 7. DDPM-based data augmentation
Train the DDPM:
```bash
python -m augmentation.ddpm \
    dataset_images/baseline/1 \
    --output_dir ddpm_models/KW_ddpm \
    --image_size 128 \
    --num_epochs 50 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --num_timesteps 1000
```
This will train on positives, save checkpoints to `ddpm_models/KW_ddpm/`, and periodically generate samples.

Generate synthetic spectrograms:
```bash
python -m augmentation.ddpm \
    --generate_only \
    --pretrained_model ddpm_models/KW_ddpm \
    --num_samples 500 \
    --output_dir dataset_images/ddpm_augmented/1 \
    --image_size 128 \
    --batch_size 8 \
    --num_timesteps 50
```
Optional PCA filtering:
```bash
python -m augmentation.ddpm \
    --generate_only \
    --pretrained_model ddpm_models/KW_ddpm \
    --num_samples 500 \
    --validation_dir dataset_images/baseline/1 \
    --output_dir dataset_images/ddpm_augmented/1
```

### 8. Classifier training
Expected dataset layout:
```bash
dataset_images/baseline/
├── 0/    # Background samples
└── 1/    # Positive samples (e.g., KW)
```

Train the baseline classifier:
```bash
python -m train_classifier \
    dataset_images/baseline \
    --train_set 0 1 \
    --val_set 0 1 \
    --output_folder models/baseline_classifier \
    --model_name resnet18_baseline \
    --num_epochs 20 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --input_shape 128 \
    --versioning
```
This will balance classes, train a ResNet-18, log metrics to `models/baseline_classifier/logs/`, and save `resnet18_baseline_v0.pt` (auto-incremented if `--versioning`).

### 9. Segment-wise inference on continuous audio
Run inference on trained classifier:
```bash
python -m inference \
    models/baseline_classifier/resnet18_baseline_v0.pt \
    data/audio/ \
    configs/audio_config.json \
    --output_folder detections/ \
    --input_shape 128
```
This will slice audio, convert to Mel-spectrograms, run inference, and save `detections/detections_raw.csv`.

Using a file list:
```bash
python inference.py \
    models/baseline_classifier/resnet18_baseline_v0.pt \
    data/audio \
    configs/audio_config.json \
    --file_list files_to_process.txt \
    --output_folder detections_subset/
```
`files_to_process.txt` contains one filename per line.

### 10. Evaluation metrics (threshold sweeps & segment-level PR curves)
Supports segment-based evaluation (interval overlap) using raw `inference.py` outputs.

- Thresholded evaluation:
  ```bash
  python -m metrics \
      reference_annotations.csv \
      --evaluation detections/detections_raw.csv \
      --mode thresholded \
      --threshold_min 0.4 \
      --threshold_max 0.6 \
      --threshold_inc 0.05 \
      --output_folder metrics_thresholded
  ```
  Saves `metrics_thresholded/metrics.csv`.

- Score-based PR curve:
  ```bash
  python -m metrics \
      reference_annotations.csv \
      --evaluation detections/detections_raw.csv \
      --mode score_based \
      --output_folder metrics_pr_curve/
  ```
  Saves `metrics_pr_curve/pr_curve.csv` for plotting.
