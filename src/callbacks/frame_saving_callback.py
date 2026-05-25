from pathlib import Path

import cv2
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class FrameSavingCallback(BaseCallback):
    """Save RGB observations during PPO training for the autoencoder dataset."""

    def __init__(
        self,
        frame_dir: str | Path,
        min_save_interval: int,
        max_save_interval: int,
        target_frames: int,
        min_steps_after_reset: int,
        seed: int | None = None,
    ):
        super().__init__()
        self.frame_dir = Path(frame_dir)
        self.min_save_interval = min_save_interval
        self.max_save_interval = max_save_interval
        self.target_frames = target_frames
        self.min_steps_after_reset = min_steps_after_reset
        self.rng = np.random.default_rng(seed)
        existing_frames = sorted(self.frame_dir.glob("frame_*.png"))
        self.saved_frames = len(existing_frames)
        self.steps_since_last_save = 0
        self.next_save_interval = self._sample_next_interval()
        self.steps_since_reset = None

    def _on_training_start(self) -> None:
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"Found {self.saved_frames} existing frames. "
            f"Saving up to {self.target_frames} frames "
            f"to {self.frame_dir} with random intervals "
            f"[{self.min_save_interval}, {self.max_save_interval}] "
            f"and min_steps_after_reset={self.min_steps_after_reset}."
        )

    def _on_step(self) -> bool:
        if self.saved_frames >= self.target_frames:
            return True

        observations = self.locals.get("new_obs")
        if observations is None:
            return True

        observations = np.asarray(observations)
        dones = np.asarray(self.locals.get("dones", np.zeros(len(observations), dtype=bool)))

        self._update_reset_counters(len(observations), dones)

        self.steps_since_last_save += 1
        if self.steps_since_last_save < self.next_save_interval:
            return True

        self.steps_since_last_save = 0
        self.next_save_interval = self._sample_next_interval()

        # With vectorized environments, observations have shape:
        # (n_envs, height, width, channels).
        if observations.ndim == 3:
            observations = observations[None, ...]

        for env_index, frame in enumerate(observations):
            if self.saved_frames >= self.target_frames:
                break

            if self.steps_since_reset[env_index] < self.min_steps_after_reset:
                continue

            self._save_frame(frame)
            self.saved_frames += 1

        return True

    def _sample_next_interval(self) -> int:
        return int(self.rng.integers(self.min_save_interval, self.max_save_interval + 1))

    def _update_reset_counters(self, n_envs: int, dones: np.ndarray) -> None:
        if self.steps_since_reset is None:
            self.steps_since_reset = np.full(n_envs, self.min_steps_after_reset)

        self.steps_since_reset += 1

        for env_index, done in enumerate(dones):
            if done:
                self.steps_since_reset[env_index] = 0

    def _save_frame(self, frame: np.ndarray) -> None:
        frame = np.asarray(frame)

        # SB3 may transpose image observations to channel-first format: C x H x W.
        # OpenCV expects H x W x C, so convert back when needed.
        if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
            frame = np.transpose(frame, (1, 2, 0))

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        output_path = self.frame_dir / f"frame_{self.saved_frames:06d}.png"

        # Gymnasium observations are RGB, while OpenCV writes images as BGR.
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), frame_bgr)
