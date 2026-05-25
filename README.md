# Autoencoder-Based Visual Reinforcement Learning in CarRacing-v3

University AI Lab project comparing two Visual Reinforcement Learning pipelines on
`Gymnasium CarRacing-v3`:

1. **Raw-pixel PPO**: PPO learns directly from RGB frames with `CnnPolicy`.
2. **Latent PPO**: PPO receives a 128-dimensional latent vector produced by a
   frozen CNN autoencoder encoder and uses `MlpPolicy`.

The project is research-oriented: the goal is not to reach state-of-the-art
performance, but to evaluate whether an autoencoder-learned visual
representation improves sample efficiency, reward stability, and final policy
performance compared to direct raw-pixel training.

## Research Question

Does an autoencoder-learned latent representation improve PPO training compared
to learning directly from raw pixels in Visual Reinforcement Learning?

The comparison focuses on:

- mean evaluation reward;
- best and final evaluation reward;
- learning curve behaviour;
- reward stability;
- qualitative autoencoder reconstruction quality.

## Project Structure

```text
.
├── data/                    # Local frame dataset, ignored by git
├── report/                  # Final report in Markdown
├── presentation/            # Final presentation PDF
├── results/                 # Local logs, plots, checkpoints, ignored by git
├── scripts/                 # Runnable training, evaluation, plotting scripts
└── src/                     # Reusable project code
    ├── callbacks/           # Stable-Baselines3 callbacks
    ├── data/                # PyTorch frame dataset
    ├── envs/                # CarRacing and latent observation wrappers
    └── models/              # CNN autoencoder model
```

Important files:

- `scripts/train.py`: trains raw-pixel PPO with `CnnPolicy`.
- `scripts/train_latent.py`: trains latent PPO with frozen encoder + `MlpPolicy`.
- `scripts/train_autoencoder.py`: trains the CNN autoencoder on saved frames.
- `scripts/plot.py`: generates PPO training/evaluation plots.
- `scripts/plot_autoencoder.py`: generates reconstruction and loss plots.
- `src/envs/latent_obs_wrapper.py`: converts RGB observations into latent vectors.
- `src/models/autoencoder.py`: CNN autoencoder architecture.
- `report/report.md`: technical report.
- `presentation/autoencoder_visual_rl_presentation.pdf`: final slide deck.

## Setup

The project was developed with Python 3.11.

Create and activate a clean environment:

```powershell
conda create -n vrl python=3.11
conda activate vrl
python -m pip install --upgrade pip
```

For a CUDA-enabled PyTorch installation, install PyTorch from the official
PyTorch index matching your CUDA version. For example, on the development
machine:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

Then install the remaining dependencies:

```powershell
pip install -r requirements.txt
```

If GPU support is not needed, the standard CPU PyTorch installation is enough.

## Running the Experiments

Run commands from the project root.

Train the raw-pixel PPO baseline:

```powershell
python -m scripts.train
```

Train the autoencoder:

```powershell
python -m scripts.train_autoencoder
```

Train PPO on latent observations:

```powershell
python -m scripts.train_latent
```

Generate PPO plots:

```powershell
python -m scripts.plot
```

Generate autoencoder plots:

```powershell
python -m scripts.plot_autoencoder
```

Evaluate a trained PPO checkpoint:

```powershell
python -m scripts.eval
```

## Method Summary

The autoencoder is trained offline on RGB frames saved during CarRacing
interaction. The encoder compresses each `96 x 96 x 3` frame into a
128-dimensional latent vector. During latent PPO training, a Gymnasium
observation wrapper applies the same preprocessing used during autoencoder
training and replaces each RGB observation with the frozen encoder output.

The raw-pixel baseline uses Stable-Baselines3 PPO with `CnnPolicy`, while the
latent agent uses PPO with `MlpPolicy`.

## Main Results

| Run | Input | Learning rate | Best eval reward | Final eval reward |
|---|---|---:|---:|---:|
| raw/v0 | RGB pixels | 3e-4 | **927.78** | 213.84 |
| raw/v2 | RGB pixels | 1e-4 | 796.82 | 531.41 |
| latent/v1 | latent 128 | 1e-4 | 899.65 | 504.64 |
| latent/v2 | latent 128 | 1e-4 | 842.06 | **805.81** |
| latent/v3 | latent 128 | 3e-4 | 739.18 | 617.91 |

The results are nuanced. Raw-pixel PPO can reach very high peak performance, but
some runs degrade strongly after reaching a good policy. Latent PPO is
competitive and one latent run achieved the best final evaluation reward, but
the latent representation did not consistently improve sample efficiency.

## Version-Control Notes

Generated data, model checkpoints, logs, plots, videos, and temporary slide
exports are ignored by git. This keeps the repository lightweight. The report
and final presentation PDF are tracked.

To reproduce plots or continue training, regenerate local outputs by running the
scripts above.
