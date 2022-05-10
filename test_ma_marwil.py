from collections import defaultdict
from typing import Dict, Union, List

import os
import numpy as np
import ray._private.utils

from ray.rllib.utils.typing import PolicyID, AgentID
from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.marwil import MARWILTrainer
from ray.rllib.agents.marwil import DEFAULT_CONFIG as MARIL_CONFIG
from ray.rllib.offline.json_writer import JsonWriter
from ray.rllib.utils import merge_dicts

from rlapps.envs.ma_to_single_env import SingleAgentEnv
from rlapps.envs.poker_multi_agent_env import PokerMultiAgentEnv
from rlapps.apps.scenarios.catalog import scenario_catalog
from rlapps.apps.scenarios.psro_scenario import PSROScenario
from rlapps.apps.scenarios.psro_distill_scenario import DistilledPSROScenario
from rlapps.apps.psro.general_psro_eval import run_episode


def collect(
    br_player: int,
    scenario: Union[PSROScenario, DistilledPSROScenario],
    buffer_file_path: str,
    episode_num: int = 10,
):
    env = scenario.env_class(env_config=scenario.env_config)
    trainer_config = scenario.get_trainer_config(env)
    num_players = 2

    policies = [
        scenario.policy_classes_distill["eval"](
            env.observation_space, env.action_space, with_common_config(trainer_config)
        )
        for _ in range(num_players)
    ]

    buffer_list = []
    total_payoffs_per_player = np.zeros(shape=num_players, dtype=np.float64)
    for episode_th in range(episode_num):
        payoffs_per_player_this_episode, samples = run_episode(
            env=env,
            policies_for_each_player=policies,
            store_as_offline=True,
        )

        total_payoffs_per_player += payoffs_per_player_this_episode

        payoffs_per_player = total_payoffs_per_player / (episode_th + 1)

        print(f"payoffs per player:" f"0 vs 1: " f"{payoffs_per_player}")

        buffer_list.append(samples[br_player])

    writer = JsonWriter(buffer_file_path)
    for e in buffer_list:
        writer.write(e)


def run_single_agent_il(scenario: DistilledPSROScenario, num_iterations: int):
    tmp_env = scenario.env_class(env_config=scenario.env_config)
    config = scenario.get_trainer_config_distill(tmp_env)
    br_player = 0
    policy_ids = ["distilled_policy_0", "distilled_policy_1"]

    dir_name = os.path.join(
        ray._private.utils.get_user_temp_dir(),
        scenario.name,
        f"{policy_ids[0]}_vs_{policy_ids[1]}",
    )

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    # buffer_file_path = os.path.join(dir_name)
    print("Buffer file path is:", dir_name)
    buffer_file_path = dir_name

    collect(br_player, scenario, buffer_file_path)
    print("collect done..")

    prob_list = [0.1, 0.2, 0.7]

    runtime_single_agent_env_config = {
        "env_class": scenario.env_class,
        "env_config": scenario.env_config,
        "br_player": br_player,
        "multiagent": {
            "policies": {
                "eval": (
                    scenario.policy_classes_distill["eval"],
                    tmp_env.observation_space,
                    tmp_env.action_space,
                    {},
                )
            },
            "policy_mapping_fn": lambda agent_id: "eval",
        },
    }

    config = MARIL_CONFIG.copy()
    config["num_workers"] = 2
    config["evaluation_num_workers"] = 1
    config["evaluation_interval"] = 3
    config["evaluation_duration"] = 5
    config["evaluation_parallel_to_training"] = True
    config["evaluation_config"] = {"input": "sampler"}

    config.update(
        {
            "framework": "torch",
            "input": buffer_file_path,
            "observation_space": tmp_env.observation_space,
            "action_space": tmp_env.action_space,
            "env": SingleAgentEnv,
            "env_config": runtime_single_agent_env_config,
        }
    )
    # pprint.pprint(config)
    trainer = MARWILTrainer(config=config)
    print("trainer has been created")

    min_reward = 100.0

    for i in range(num_iterations):
        results = trainer.train()
        eval_results = results.get("evaluation")
        if eval_results:
            print("iter={} R={} ".format(i, eval_results["episode_reward_mean"]))
            # Learn until some reward is reached on an actual live env.
            if eval_results["episode_reward_mean"] > min_reward:
                print("learnt!")
                break


if __name__ == "__main__":
    scenario_name = "kuhn_distill_psro_dqn"
    num_iterations = 10
    run_single_agent_il(
        scenario=scenario_catalog.get(scenario_name), num_iterations=num_iterations
    )
