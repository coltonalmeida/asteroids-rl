# CLAUDE.md — Asteroids RL Agent

> Project context file for Claude Code. Train a reinforcement-learning agent to
> play Atari Asteroids. Optimized to be a portfolio piece employers will respect.

---

## 1. Project Goal

Train an AI agent to play **Atari Asteroids** well above random-play baseline,
using a standard, reproducible deep-RL pipeline. Ship it with clean code,
tracked experiments, evaluation metrics, and a recorded gameplay video.

**Success criteria**
- Agent mean episode reward significantly beats a random-action baseline.
- Training is reproducible from a single command + config file.
- Repo includes metrics, plots, a saved model, and a gameplay video.

## 2. Key Decision: Don't Build the Game

You do **not** need to find or build an Asteroids game on GitHub. The
**Arcade Learning Environment (ALE)**, exposed through **Farama Gymnasium**,
ships Asteroids as a ready, standardized environment: `ALE/Asteroids-v5`.
This is the exact environment used in RL research and industry, which is why
it's the right call for a portfolio. Your only job is the *training pipeline*.

## 3. Algorithm Choice

Train **PPO first** — it's more stable and faster to tune than DQN on Atari.
Then train **DQN** as a second baseline so you can *compare* — comparison
tables and plots are what make a portfolio project look rigorous.

## 4. Repository Structure

```
asteroids-rl/
├── CLAUDE.md                 # this file
├── README.md                 # results, plots, gif, how-to-run
├── pyproject.toml            # deps (uv/pip), ruff/black config
├── Dockerfile
├── .github/workflows/ci.yml  # ruff + pytest on push
├── configs/
│   ├── ppo.yaml
│   └── dqn.yaml
├── src/asteroids_rl/
│   ├── __init__.py
│   ├── env.py                # make_env(): wrappers, frame-stack, preprocessing
│   ├── train.py              # CLI entry: reads config, trains, logs to wandb
│   ├── evaluate.py           # load model, run N episodes, report mean/std
│   ├── record.py             # save gameplay video of trained agent
│   └── callbacks.py          # eval + checkpoint callbacks
├── tests/
│   └── test_env.py           # env builds, step/reset shapes correct
├── models/                   # saved checkpoints (gitignored if large)
├── videos/                   # recorded gameplay
└── reports/                  # plots, final metrics
```

## 5. Environment Setup

```bash
# Using uv (preferred)
uv venv && source .venv/bin/activate
uv pip install "stable-baselines3[extra]" "gymnasium[atari,accept-rom-license]" \
               torch wandb hydra-core pytest ruff black tensorboard
```

`stable-baselines3[extra]` pulls in the Atari preprocessing helper
`make_atari_env` plus ROM handling. Requires Python >= 3.10.

## 6. Workload Split

Your local machine is for writing code and reviewing results. The heavy lifting
runs elsewhere:

```
Your PC          →  push code, watch W&B dashboard, download milestone videos
GitHub Actions   →  runs pytest automatically on every push (free, zero local effort)
Cloud GPU        →  trains PPO/DQN, saves checkpoints, records milestone videos
W&B              →  streams live reward curves + stores checkpoint videos for download
```

You never need to run the test suite locally unless you're debugging a specific
failure — CI handles it. You never need to run long training runs locally either.

**Recommended cloud GPU providers (cheapest to most convenient):**

| Provider | Notes | Approx. cost |
|---|---|---|
| Vast.ai / RunPod | Cheapest $/hr, most control, best for long runs | ~$0.10–0.40/hr (T4 / RTX 3090) |
| Kaggle Notebooks | Free GPU quota (~30 hrs/week), no setup required | Free |
| Google Colab Pro | Easy notebooks, decent GPUs | ~$10/mo |
| Lambda Labs | Clean UI, reliable, slightly pricier | ~$0.50/hr |

For a portfolio project, **RunPod or Vast.ai** give the most control and are
most cost-effective for multi-hour PPO/DQN runs. Kaggle is a good zero-cost
option to validate the pipeline before committing to a paid run.

## 7. Milestone Video Workflow

Do **not** render video during training — it significantly slows down the run.
Instead, save checkpoints throughout training and record clean gameplay videos
from those checkpoints afterward (on the same cloud machine before it shuts down).

```python
# record_checkpoint.py
import wandb
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo

def record_checkpoint(checkpoint_path, step):
    env = RecordVideo(make_env(), video_folder=f"./videos/{step}")
    model = PPO.load(checkpoint_path)

    obs, _ = env.reset()
    for _ in range(5000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, trunc, _ = env.step(action)
        if done or trunc:
            obs, _ = env.reset()
    env.close()

    wandb.log({
        f"gameplay/step_{step}": wandb.Video(f"./videos/{step}/agent.mp4")
    })
```

Videos upload to W&B automatically and are downloadable from the dashboard
at any time. The most compelling portfolio demo is a **progression montage**:
early-checkpoint gameplay side by side with late-checkpoint gameplay.

**Suggested checkpoint schedule for the before/after demo:**

| Checkpoint | Steps | Purpose |
|---|---|---|
| Baseline | 0 (random agent) | Control — shows what chance looks like |
| Early | 500k steps | Agent is just beginning to learn |
| Mid | 3M steps | Visible improvement, not yet competent |
| Late | 10M steps | Best trained behaviour |

Training is measured in **steps** (individual frames/actions), not episodes.
Meaningful visible improvement requires millions of steps. Frame portfolio
comparisons as "early-step vs. late-step checkpoints", not "N tries vs M tries".

## 8. Implementation Plan (phased)

Phases 0 and 1 are complete. All remaining phases follow a **build-first**
structure: write and wire every module before any long training run begins.

---

### ✅ Phase 0 — Scaffolding (complete)
- GitHub repo `asteroids-rl` created, README + .gitignore committed.
- Project folder set up, venv created, skeleton committed.
- ruff/black config added, passing pytest stub wired, GitHub Actions CI live.

### ✅ Phase 1 — Environment (complete)
- `make_env()` implemented via SB3's `make_atari_env` (grayscale, 84×84,
  4-frame stack, frame-skip, sticky actions).
- Random agent sanity check added; baseline mean reward recorded.

---

### ✅ Phase 2 — Build Training Pipeline (complete)
- `train.py` reads config via Hydra (`@hydra.main`, `--config-name=ppo|dqn`),
  constructs vectorised envs (`n_envs=8`), initialises model, attaches
  callbacks, runs `.learn()`. Values overridable on the CLI (e.g.
  `total_timesteps=10000 wandb.enabled=false`).
- `callbacks.py` wires `CheckpointCallback` (periodic `.zip` saves) +
  `EvalCallback` (eval mean reward → W&B via `sync_tensorboard=True`).
- `configs/ppo.yaml` and `configs/dqn.yaml` carry all hyperparameters (§9).
- Smoke-tested at `total_timesteps=10000` for both PPO and DQN — loop runs
  end-to-end, callbacks fire, `final_model.zip` saved.

### ✅ Phase 3 — Build Evaluation & Recording Pipeline (complete)
- `evaluate.py` loads a `.zip` checkpoint, runs N episodes, prints mean ± std
  reward and the improvement factor vs the random baseline
  (`reports/random_baseline.json`, `--baseline` to override).
- `record.py` loads a checkpoint, records a full-game MP4 (`RecordVideo`), and
  optionally uploads it to W&B as a video artifact (`--wandb`, see Section 7).
- Smoke-tested both against the Phase 2 10k-step checkpoint.

---
> **All code is now written and wired. Do not proceed until both smoke-tests
> pass and CI is green. From this point on, work moves to a cloud GPU machine.**
>
> Push all code to GitHub, spin up a cloud instance (see Section 6), clone the
> repo, install deps, and run the commands below. Your local machine is now
> just a dashboard viewer.
---

### Phase 4 — Cloud Training Runs
- On the cloud machine, run PPO to 10M steps:
  ```bash
  python -m asteroids_rl.train --config-name=ppo
  ```
- Monitor live reward curves in the W&B dashboard from your PC.
- After PPO completes, run `record.py` against each milestone checkpoint
  (500k, 3M, 10M) before the instance shuts down. Videos upload to W&B.
- Run DQN to 10M steps the same way:
  ```bash
  python -m asteroids_rl.train --config-name=dqn
  ```
- Record DQN milestone videos the same way.

### Phase 5 — Evaluate & Report
- Run `evaluate.py` against the best PPO and DQN checkpoints (30 episodes each).
- Generate reward-curve comparison plots into `reports/` from W&B export.
- Download milestone videos from W&B; assemble the progression montage.

### Phase 6 — Polish
- README: results table (Random vs PPO vs DQN), training curves, gameplay GIF,
  exact reproduce commands.
- Confirm Dockerfile works end-to-end.
- Confirm CI is green on `main`.

---

## 9. Core Hyperparameters (starting points)

**PPO (Atari defaults):** `n_envs=8`, `n_steps=128`, `batch_size=256`,
`n_epochs=4`, `gamma=0.99`, `learning_rate=2.5e-4` (linearly decayed),
`clip_range=0.1`, `ent_coef=0.01`, `vf_coef=0.5`, policy=`CnnPolicy`.

**DQN (Atari defaults):** `buffer_size=100_000`, `learning_starts=100_000`,
`target_update_interval=1000`, `train_freq=4`, `gamma=0.99`,
`exploration_fraction=0.1`, `learning_rate=1e-4`, policy=`CnnPolicy`.

> Note: `CnnPolicy` is required — observations are raw pixels.

## 10. Conventions for Claude Code

- Keep env-building logic in **one** place (`env.py`) — never duplicate wrappers.
- Every training run must be driven by a **config file**, never hardcoded params.
- All runs log to **wandb** with the config attached, so experiments are
  comparable and reproducible.
- Prefer SB3's built-in helpers over custom implementations unless there's a
  concrete reason; document any deviation.

## 11. Resume / Portfolio Framing

When this is done, it demonstrates: deep RL (PPO + DQN), PyTorch, the modern
Gymnasium/ALE stack, experiment tracking (wandb), reproducible configs,
testing, CI/CD, and Docker — i.e. an *end-to-end ML engineering* project, not
just a notebook. Lead the README with the results table and the gameplay GIF.
