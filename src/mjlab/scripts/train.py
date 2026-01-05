"""Script to train RL agent with RSL-RL."""

import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import tyro
from rsl_rl.runners import OnPolicyRunner

# Monkeypatch os.symlink on Windows to support non-admin users
if os.name == "nt":
    import shutil

    _os_symlink = os.symlink

    def _symlink_with_fallback(src, dst, target_is_directory=False):
        try:
            _os_symlink(src, dst, target_is_directory)
        except OSError as e:
            if hasattr(e, "winerror") and e.winerror == 1314:
                # Fallback to copying if symlink privilege is missing.
                src_path = Path(src)
                dst_path = Path(dst)
                # If src is relative, it's likely relative to the dst's parent.
                if not src_path.is_absolute():
                    resolved_src = (dst_path.parent / src_path).resolve()
                    if resolved_src.exists():
                        src_path = resolved_src
                    else:
                        src_path = src_path.resolve()

                if src_path.is_dir():
                    shutil.copytree(str(src_path), str(dst_path))
                elif src_path.exists():
                    shutil.copy2(str(src_path), str(dst_path))
            else:
                raise

    os.symlink = _symlink_with_fallback

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import (
    list_tasks,
    load_env_cfg,
    load_rl_cfg,
    load_runner_cls,
)
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import (
    dump_yaml,
    get_checkpoint_path,
    get_wandb_checkpoint_path,
)
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class TrainConfig:
    env: ManagerBasedRlEnvCfg
    agent: RslRlOnPolicyRunnerCfg
    registry_name: str | None = None
    motion_file: str | None = None
    video: bool = False
    video_length: int = 200
    video_interval: int = 2000
    video_width: int = 320
    video_height: int = 240
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    enable_nan_guard: bool = False
    torchrunx_log_dir: str | None = None
    wandb_run_path: str | None = None
    gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

    @staticmethod
    def from_task(task_id: str) -> "TrainConfig":
        env_cfg = load_env_cfg(task_id)
        agent_cfg = load_rl_cfg(task_id)
        assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)
        return TrainConfig(env=env_cfg, agent=agent_cfg)


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible == "":
        device = "cpu"
        seed = cfg.agent.seed
        rank = 0
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        # Set EGL device to match the CUDA device.
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
        device = f"cuda:{local_rank}"
        # Set seed to have diversity in different processes.
        seed = cfg.agent.seed + local_rank

    configure_torch_backends()

    cfg.agent.seed = seed
    cfg.env.seed = seed

    print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

    registry_name: str | None = None

    # Check if this is a tracking task by checking for motion command.
    is_tracking_task = (
        cfg.env.commands is not None
        and "motion" in cfg.env.commands
        and isinstance(cfg.env.commands["motion"], MotionCommandCfg)
    )

    if is_tracking_task:
        if cfg.motion_file:
            assert cfg.env.commands is not None
            motion_cmd = cfg.env.commands["motion"]
            assert isinstance(motion_cmd, MotionCommandCfg)
            motion_cmd.motion_file = cfg.motion_file
            print(f"[INFO] Using local motion file: {cfg.motion_file}")

        elif cfg.registry_name:
            # Check if the registry name includes alias.
            # If not, append ":latest".
            registry_name = cast(str, cfg.registry_name)
            if ":" not in registry_name:
                registry_name = registry_name + ":latest"
            import wandb

            api = wandb.Api()
            artifact = api.artifact(registry_name)

            assert cfg.env.commands is not None
            motion_cmd = cfg.env.commands["motion"]
            assert isinstance(motion_cmd, MotionCommandCfg)
            artifact_dir = Path(artifact.download())
            if (artifact_dir / "motion.npz").exists():
                motion_file = artifact_dir / "motion.npz"
            else:
                # Find any .npz file in the artifact directory
                npz_files = list(artifact_dir.glob("*.npz"))
                if npz_files:
                    motion_file = npz_files[0]
                else:
                    raise FileNotFoundError(
                        "No .npz file found in artifact directory: " f"{artifact_dir}"
                    )
            motion_cmd.motion_file = str(motion_file)
        else:
            raise ValueError(
                "Must provide --registry-name or --motion-file for " "tracking tasks."
            )

    # Enable NaN guard if requested.
    if cfg.enable_nan_guard:
        cfg.env.sim.nan_guard.enabled = True
        print(
            "[INFO] NaN guard enabled, output dir: "
            f"{cfg.env.sim.nan_guard.output_dir}"
        )

    if rank == 0:
        print(f"[INFO] Logging directory: {log_dir}")
        if cfg.video:
            cfg.env.viewer.width = cfg.video_width
            cfg.env.viewer.height = cfg.video_height
            print(f"[INFO] Video resolution: {cfg.video_width}x{cfg.video_height}")

        if cfg.wandb_project:
            cfg.agent.wandb_project = cfg.wandb_project
        if cfg.wandb_run_name:
            cfg.agent.run_name = cfg.wandb_run_name

    env = ManagerBasedRlEnv(
        cfg=cfg.env,
        device=device,
        render_mode="rgb_array" if cfg.video else None,
    )

    # Go up from specific run dir to experiment dir.
    log_root_path = log_dir.parent

    resume_path: Path | None = None
    if cfg.agent.resume:
        if cfg.wandb_run_path is not None:
            # Load checkpoint from W&B.
            resume_path, was_cached = get_wandb_checkpoint_path(
                log_root_path, Path(cfg.wandb_run_path)
            )
            if rank == 0:
                run_id = resume_path.parent.name
                checkpoint_name = resume_path.name
                cached_str = "cached" if was_cached else "downloaded"
                print(
                    f"[INFO]: Loading checkpoint from W&B: {checkpoint_name} "
                    f"(run: {run_id}, {cached_str})"
                )
        else:
            # Load checkpoint from local filesystem.
            resume_path = get_checkpoint_path(
                log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
            )

    # Only record videos on rank 0 to avoid multiple workers writing to
    # the same files.
    if cfg.video and rank == 0:
        env = VideoRecorder(
            env,
            video_folder=Path(log_dir) / "videos" / "train",
            step_trigger=lambda step: step % cfg.video_interval == 0,
            video_length=cfg.video_length,
            disable_logger=True,
        )
        print("[INFO] Recording videos during training.")

    env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

    agent_cfg = asdict(cfg.agent)
    env_cfg = asdict(cfg.env)

    runner_cls = load_runner_cls(task_id)
    if runner_cls is None:
        runner_cls = OnPolicyRunner

    runner_kwargs = {}
    if is_tracking_task:
        runner_kwargs["registry_name"] = registry_name

    runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

    add_wandb_tags(cfg.agent.wandb_tags)
    runner.add_git_repo_to_log(__file__)
    if resume_path is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(str(resume_path))
        # Synchronize video step count with loaded iteration
        recorder = getattr(env, "env", env)
        if isinstance(recorder, VideoRecorder):
            # step_count is the total number of env.step() calls.
            # RSL-RL performs num_steps_per_env steps per iteration.
            recorder.step_count = (
                runner.current_learning_iteration * cfg.agent.num_steps_per_env
            )
            print(
                f"[INFO] Synced video step_count to {recorder.step_count} (iter {runner.current_learning_iteration})"
            )

    # Only write config files from rank 0 to avoid race conditions.
    if rank == 0:
        dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
        dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

    runner.learn(
        num_learning_iterations=cfg.agent.max_iterations,
        init_at_random_ep_len=True,
    )

    env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
    args = args or TrainConfig.from_task(task_id)

    # Create log directory once before launching workers.
    log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
    log_root_path.resolve()
    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.agent.run_name:
        log_dir_name += f"_{args.agent.run_name}"
    log_dir = log_root_path / log_dir_name

    # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
    selected_gpus, num_gpus = select_gpus(args.gpu_ids)

    # Set environment variables for all modes.
    if selected_gpus is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))

    if sys.platform == "win32":
        os.environ.setdefault("MUJOCO_GL", "glfw")
    else:
        os.environ["MUJOCO_GL"] = "egl"

    if num_gpus <= 1:
        # CPU or single GPU: run directly without torchrunx.
        run_train(task_id, args, log_dir)
    else:
        # Multi-GPU: use torchrunx.
        import torchrunx

        # torchrunx redirects stdout to logging.
        logging.basicConfig(level=logging.INFO)

        # Configure torchrunx logging directory.
        # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
        if "TORCHRUNX_LOG_DIR" not in os.environ:
            if args.torchrunx_log_dir is not None:
                # User specified a value via flag (could be "" to disable).
                os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
            else:
                # Default: put logs in training directory.
                os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

        print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
        torchrunx.Launcher(
            hostnames=["localhost"],
            workers_per_host=num_gpus,
            backend=None,  # Let rsl_rl handle process group initialization.
            copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
        ).run(run_train, task_id, args, log_dir)


def main():
    # Parse first argument to choose the task.
    # Import tasks to populate the registry.
    import mjlab.tasks  # noqa: F401

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
    )

    args = tyro.cli(
        TrainConfig,
        args=remaining_args,
        default=TrainConfig.from_task(chosen_task),
        prog=sys.argv[0] + f" {chosen_task}",
        config=(
            tyro.conf.AvoidSubcommands,
            tyro.conf.FlagConversionOff,
        ),
    )
    del remaining_args

    launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
    main()
