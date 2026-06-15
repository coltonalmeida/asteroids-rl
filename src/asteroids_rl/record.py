"""Record a gameplay video of a trained agent, optionally uploading it to W&B.

Usage:
    python -m asteroids_rl.record --model models/ppo_best/best_model.zip
    python -m asteroids_rl.record --model checkpoints/ppo_500000_steps.zip \\
        --step 500000 --wandb --wandb-project asteroids-rl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

from asteroids_rl.env import ENV_ID
from asteroids_rl.train import load_model


def _make_recorded_env(env_id: str, out_dir: str, episodes: int):
    """Raw env wrapped with RecordVideo (full-color frames), then Atari preprocessing
    so the model sees the same observations as during training.

    terminal_on_life_loss is disabled so the recording shows a full multi-life game
    (a compelling demo) rather than ending at the first life lost. Only the first
    `episodes` episodes are recorded, so the vec-env auto-reset after the final
    episode doesn't leave a spurious near-empty clip behind.
    """

    def _thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        env = RecordVideo(env, video_folder=out_dir, episode_trigger=lambda ep: ep < episodes)
        return AtariWrapper(env, terminal_on_life_loss=False)

    return _thunk


def record(model_path: str, env_id: str, out_dir: str, episodes: int, seed: int) -> list[Path]:
    vec = DummyVecEnv([_make_recorded_env(env_id, out_dir, episodes)])
    vec.seed(seed)
    vec = VecFrameStack(vec, n_stack=4)
    vec = VecTransposeImage(vec)
    model = load_model(model_path, vec)

    for _ in range(episodes):
        obs = vec.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = vec.step(np.asarray(action))
            done = bool(dones[0])
    vec.close()

    videos = sorted(Path(out_dir).glob("*.mp4"))
    print(f"Saved {len(videos)} video(s) to {Path(out_dir).resolve()}")
    return videos


def upload_to_wandb(videos: list[Path], project: str, step: int) -> None:
    """Log recorded gameplay clips to W&B as video artifacts (see CLAUDE.md §7)."""
    if not videos:
        print("No videos to upload.")
        return
    import wandb

    run = wandb.init(project=project, job_type="record", config={"step": step})
    for i, path in enumerate(videos):
        key = f"gameplay/step_{step}" if len(videos) == 1 else f"gameplay/step_{step}_ep{i}"
        run.log({key: wandb.Video(str(path), format="mp4")})
    run.finish()
    print(f"Uploaded {len(videos)} clip(s) to W&B project '{project}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--env-id", type=str, default=ENV_ID)
    parser.add_argument("--out-dir", type=str, default="videos")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", help="Upload the clip(s) to W&B")
    parser.add_argument("--wandb-project", type=str, default="asteroids-rl")
    parser.add_argument(
        "--step", type=int, default=0, help="Training step label for the W&B video key"
    )
    args = parser.parse_args()
    videos = record(args.model, args.env_id, args.out_dir, args.episodes, args.seed)
    if args.wandb:
        upload_to_wandb(videos, args.wandb_project, args.step)


if __name__ == "__main__":
    main()
