# Autoencoder-Based Visual Reinforcement Learning in CarRacing-v3

University AI-LAB project comparing two visual reinforcement learning pipelines:

1. **Raw-pixel PPO** uses Stable-Baselines3 `CnnPolicy` on RGB observations.
2. **Latent PPO** uses `MlpPolicy` on a frozen 128-dimensional autoencoder
   representation.

The project evaluates whether offline visual pretraining improves PPO sample
efficiency and performance under a controlled multi-seed protocol.

## Structure

```text
data/                 Local frame dataset
report/               LaTeX paper and final PDF
presentation/         Final PowerPoint presentation
results/              Local checkpoints, logs and generated plots
scripts/              Training, evaluation and plotting entry points
src/                  Models, environments, callbacks and datasets
```

Main entry points:

- `scripts/train.py`: raw-pixel PPO and optional frame collection.
- `scripts/train_autoencoder.py`: two-session autoencoder training.
- `scripts/train_latent.py`: PPO on frozen latent observations.
- `scripts/eval.py`: visual evaluation of a saved raw or latent policy.
- `scripts/plot.py`: per-run and multi-seed PPO plots.
- `scripts/plot_autoencoder.py`: autoencoder loss and reconstruction plots.

## Setup

The experiments used Python 3.11, Gymnasium 1.2.3 and Stable-Baselines3 2.8.0.

```powershell
conda create -n vrl python=3.11
conda activate vrl
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements.txt
```

A CPU PyTorch installation can be used when CUDA is unavailable, although raw
PPO training will be slower.

## Reproduction

Run every command from the repository root.

### 1. Collect Frames

Frame collection is integrated into raw PPO training:

```powershell
python -m scripts.train --seed 42 --save-frames
```

Frames are written to `data/frames`. The submitted experimental dataset contains
18,293 RGB PNG frames. Because `data/` is intentionally ignored by Git, it must
be included separately when full autoencoder retraining is required.

### 2. Train the Autoencoder

```powershell
python -m scripts.train_autoencoder --overwrite
```

The script reproduces the training protocol used in the project:

- fixed seed 42 and fixed 90/10 split;
- latent dimension 128;
- batch size 128;
- Adam with learning rate `1e-4`;
- two consecutive sessions of 20 epochs;
- model weights retained and Adam reinitialized before the second session.

The checkpoint and configuration are saved under
`results/shared/autoencoder/`. The repository includes the checkpoint used by
the experiments, so `--overwrite` is required for intentional full retraining.
Without that flag, the script protects the supplied model from accidental
replacement.

The shared autoencoder checkpoint is retained with the code because latent PPO
training and evaluation require it. Raw/latent PPO run directories, logs and
generated plots remain excluded from version control.

### 3. Train PPO

```powershell
python -m scripts.train --seed 42
python -m scripts.train --seed 123
python -m scripts.train --seed 456

python -m scripts.train_latent --seed 42
python -m scripts.train_latent --seed 123
python -m scripts.train_latent --seed 456
```

Each run uses two environments, two million timesteps, learning rate `1e-4`,
five deterministic evaluation episodes every 20,000 aggregate timesteps, and
evaluation base seed 10,000.

### 4. Generate Results

```powershell
python -m scripts.plot_autoencoder
python -m scripts.plot
```

## Main Results

Mean and standard deviation across training seeds 42, 123 and 456:

| Metric | Raw PPO | Latent PPO | Difference |
|---|---:|---:|---:|
| Best evaluation reward | 798.1 +/- 71.7 | **882.4 +/- 40.5** | +10.6% |
| Final five-evaluation mean | 459.2 +/- 211.3 | **685.1 +/- 127.4** | +49.2% |
| Normalized learning-curve AUC | 377.8 +/- 120.0 | **567.5 +/- 134.5** | +50.2% |
| Steps to reward 700 | 960k +/- 529k | **520k +/- 394k** | -45.8% |

Latent PPO performed better on all predefined aggregate metrics, while both
methods retained substantial variability across seeds.

## Final Materials

- Paper source: `report/autoencoder_visual_rl_report.tex`
- Paper PDF: `report/autoencoder_visual_rl_report.pdf`
- Presentation: the `.pptx` file in `presentation/`

Large datasets, PPO checkpoints, logs and generated plots are intentionally
excluded from version control. The frozen autoencoder checkpoint is the only
model artifact retained because it is a required input to the latent pipeline.
Configuration JSON files inside completed runs record the parameters used for
the controlled benchmark.
