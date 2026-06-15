# asteroids-rl

Training deep-RL agents (PPO and DQN, via Stable-Baselines3) to play **Atari Asteroids**
(`ALE/Asteroids-v5` from Gymnasium / the Arcade Learning Environment).

> 🚧 Work in progress — results table, training curves, and gameplay GIF land here
> once training runs complete.

## Results

| Agent  | Mean episode reward (30 eps) | Std |
|--------|------------------------------|-----|
| Random | 465.0 (10 eps)               | 278 |
| PPO    | _TBD_                        | —   |
| DQN    | _TBD_                        | —   |

## Setup

```bash
uv venv
uv sync
```

ROMs are installed automatically via AutoROM (`accept-rom-license` extra).

## Usage

```bash
# Random-agent baseline (sanity check + baseline reward)
python -m asteroids_rl.evaluate --random --episodes 30 --out reports/random_baseline.json

# Train PPO / DQN (Hydra configs; override any value on the CLI)
python -m asteroids_rl.train --config-name=ppo
python -m asteroids_rl.train --config-name=dqn

# Quick wiring check (no real training)
python -m asteroids_rl.train --config-name=ppo total_timesteps=10000 wandb.enabled=false

# Evaluate a trained model (prints improvement factor vs the random baseline)
python -m asteroids_rl.evaluate --model models/<run>/best/best_model.zip --episodes 30

# Record gameplay video (add --wandb to upload the clip to Weights & Biases)
python -m asteroids_rl.record --model models/<run>/best/best_model.zip --step 10000000 --wandb
```

## Development

```bash
uv run pytest
uv run ruff check src tests
uv run black --check src tests
```

## Stack

Python 3.10+ · Gymnasium + ALE-py · Stable-Baselines3 (PyTorch) · wandb ·
Hydra configs · pytest · ruff/black · GitHub Actions · Docker
