import os
from pathlib import Path

import wandb
from rsl_rl.env.vec_env import VecEnv
from rsl_rl.runners import OnPolicyRunner

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.tracking.rl.exporter import (
    attach_onnx_metadata,
    export_motion_policy_as_onnx,
)


class MotionTrackingOnPolicyRunner(OnPolicyRunner):
    env: RslRlVecEnvWrapper

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        registry_name: str | None = None,
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger.logger_type in ["wandb"]:
            # Use pathlib for robust cross-platform path handling
            pt_path = Path(path)
            export_dir = pt_path.parent
            filename = f"{export_dir.name}.onnx"

            if self.alg.policy.actor_obs_normalization:
                normalizer = self.alg.policy.actor_obs_normalizer
            else:
                normalizer = None
            export_motion_policy_as_onnx(
                self.env.unwrapped,
                self.alg.policy,
                normalizer=normalizer,
                path=str(export_dir),
                filename=filename,
            )
            attach_onnx_metadata(
                self.env.unwrapped,
                wandb.run.name,  # type: ignore
                path=str(export_dir),
                filename=filename,
            )
            wandb.save(str(export_dir / filename), base_path=str(export_dir))

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)  # type: ignore
                self.registry_name = None
