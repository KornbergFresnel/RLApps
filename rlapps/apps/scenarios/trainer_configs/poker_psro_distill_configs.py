import os
from typing import Dict, Any

from ray.rllib.env import MultiAgentEnv
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.utils import merge_dicts

from rlapps.apps.scenarios.trainer_configs.defaults import (
    GRL_DEFAULT_POKER_MARWIL_PARAMS,
)
from rlapps.rllib_tools.models.valid_actions_fcnet import (
    get_valid_action_fcn_class_for_env,
)


def distill_psro_leduc_marwil_params(env: MultiAgentEnv) -> Dict[str, Any]:
    return merge_dicts(
        GRL_DEFAULT_POKER_MARWIL_PARAMS,
        {
            "num_gpus": float(os.getenv("WORKER_GPU_NUM", 0.0)),
            "num_workers": 2,
            "num_gpus_per_worker": float(os.getenv("WORKER_GPU_NUM", 0.0)),
            "num_envs_per_worker": 1,
            "evaluation_num_workers": 1,
            "evaluation_interval": 3,
            "evaluation_duration": 5,
            "evaluation_parallel_to_training": True,
            "evaluation_config": {"input": "sampler"},
            "framework": "torch",
            "observation_space": env.observation_space,
            "action_space": env.action_space,
            "train_batch_size": 4096,
            "model": merge_dicts(
                MODEL_DEFAULTS,
                {
                    "fcnet_activation": "relu",
                    "fcnet_hiddens": [128],
                    # "custom_model": get_valid_action_fcn_class_for_env(env=env),
                },
            ),
        },
    )


def distill_psro_kuhn_marwil_params(env: MultiAgentEnv) -> Dict[str, Any]:
    return merge_dicts(
        GRL_DEFAULT_POKER_MARWIL_PARAMS,
        {
            "num_gpus": float(os.getenv("WORKER_GPU_NUM", 0.0)),
            "num_workers": 2,
            "num_gpus_per_worker": float(os.getenv("WORKER_GPU_NUM", 0.0)),
            "num_envs_per_worker": 1,
            "evaluation_num_workers": 1,
            "evaluation_interval": 3,
            "evaluation_duration": 5,
            "evaluation_parallel_to_training": True,
            "evaluation_config": {"input": "sampler"},
            "framework": "torch",
            "observation_space": env.observation_space,
            "action_space": env.action_space,
            "train_batch_size": 4096,
            "model": merge_dicts(
                MODEL_DEFAULTS,
                {
                    "fcnet_activation": "relu",
                    "fcnet_hiddens": [128],
                    # "custom_model": get_valid_action_fcn_class_for_env(env=env),
                },
            ),
        },
    )
