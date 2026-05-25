from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage


def make_carracing_train_env(env_id, n_envs, log_dir, env_kwargs):
    """Create parallel environments used by PPO to collect training rollouts."""
    env = make_vec_env(
        env_id,
        n_envs=n_envs,
        monitor_dir=log_dir,
        env_kwargs=env_kwargs,
    )

    return VecTransposeImage(env)


def make_carracing_eval_env(env_id, log_dir, env_kwargs):
    """Create one environment used only for periodic deterministic evaluation."""
    env = make_vec_env(
        env_id,
        n_envs=1,
        monitor_dir=log_dir,
        env_kwargs=env_kwargs,
    )

    return VecTransposeImage(env)
