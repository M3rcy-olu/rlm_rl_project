import asyncio
import logging
from datetime import datetime
from pathlib import Path

import chz
from tinker_cookbook import cli_utils, model_info
from rlm_env import RLMDatasetBuilder
from tinker_cookbook.rl.train import AsyncConfig, Config
from tinker_cookbook.rl import train
from tinker_cookbook.hyperparam_utils import get_lr



logger = logging.getLogger(__name__)



# @chz.chz
# class StreamMinibatchConfig:
#     """
#     Configuration for training with minibatch streaming.
#     Once we have accumulated enough trajectories for a minibatch, we will
#     immediately train on them, instead of waiting for the full batch of
#     trajectories to be ready.
#     """

#     # Total number of trajectory groups across all minibatches and substeps
#     groups_per_batch: int
#     # For each substep, we will divide up the number of trajectory groups
#     # into this many minibatches.
#     # We will do num_minibatches forward_backward() passes and one optim_step()
#     # per substep.
#     num_minibatches: int

@chz.chz
class CLIConfig:
    # Model parameters
    model_name: str = "Qwen/Qwen-8B"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Training params
    seed: int = 0
    batch_size: int = 10
    max_tokens: int = 2048
    eval_every: int = 0

    group_size: int = 8
    max_trajectory_tokens: int = 32768

    stream_minibatch: bool = False
    num_minibatches: int = 4

    log_path: str | None = None
    # wandb_project: str | None = None
    # wandb_name: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

async def cli_main(cli_config: CLIConfig):
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )

    learning_rate = get_lr(cli_config.model_name)

    model_tag = cli_config.model_name.replace("/", "-")
    run_name = (
        f"rlm-{model_tag}-{cli_config.lora_rank}rank-"
        f"{learning_rate}lr-{cli_config.group_size}group-"
        f"{cli_config.batch_size}batch-seed{cli_config.seed}-"
        f"{cli_config.max_trajectory_tokens // 1024}k-"
        f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )

    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/tinker-examples/code_rl/{run_name}"

    if not Path("/tmp").exists():
        raise ValueError("/tmp does not exist")
    
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)
    
    builder = RLMDatasetBuilder(
        batch_size=cli_config.batch_size,
        group_size=cli_config.group_size,
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        seed=cli_config.seed,
        max_trajectory_tokens=cli_config.max_trajectory_tokens,
    )

    if cli_config.stream_minibatch:
        stream_minibatch_config = train.StreamMinibatchConfig(
            groups_per_batch=cli_config.batch_size,
            num_minibatches=cli_config.num_minibatches,
        )
        bs_str = f"bs{cli_config.batch_size}_stream"
    else:
        stream_minibatch_config = None
        bs_str = f"bs{cli_config.batch_size}"

    config = Config(
        learning_rate=learning_rate,
        dataset_builder=builder,
        model_name=cli_config.model_name,
        max_tokens=cli_config.max_tokens,
        eval_every=cli_config.eval_every,
        lora_rank=cli_config.lora_rank,
        stream_minibatch_config=stream_minibatch_config,
        log_path=log_path,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        # wandb_project=cli_config.wandb_project,
        # wandb_name=cli_config.wandb_name,
    )

    await train.main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))


# @chz.chz
# class Config:
#     learning_rate: float
#     dataset_builder: RLDatasetBuilder  # also determines batch size
#     model_name: str
#     max_tokens: int
#     lora_rank: int = 32
#     loss_fn: Literal["importance_sampling", "ppo"] = "importance_sampling"
#     log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
#     eval_every: int = 20
#     stream_minibatch_config: StreamMinibatchConfig | None = None

