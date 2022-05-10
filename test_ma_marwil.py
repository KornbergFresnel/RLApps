from collections import defaultdict
from typing import Dict, Union, List

import os
import numpy as np
import ray._private.utils

from ray.rllib.utils.typing import PolicyID, AgentID
from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.marwil import MARWILTrainer
from ray.rllib.offline.json_writer import JsonWriter
from ray.rllib.utils import merge_dicts

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
    env = PokerMultiAgentEnv(env_config=None)
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

        for v in samples[br_player]:
            buffer_list.append(v)

    writer = JsonWriter(buffer_file_path)
    for e in buffer_list[br_player]:
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

    def select_policy(agent_id):
        if agent_id == 0:
            return "distilled_policy_0"
        elif agent_id == 1:
            return "distilled_policy_1"
        else:
            raise ValueError(f"Unknown agent id: {agent_id}")

    collect(br_player, scenario, buffer_file_path)
    print("collect done..")

    prob_list = [0.1, 0.2, 0.7]

    runtime_single_agent_env_config = merge_dicts(
        scenario.single_agent_env_config,
        {
            "multiagent": {
                "policies": {
                    "metanash": (
                        scenario.policy_classes_distill["meta_nash"],
                        tmp_env.observation_space,
                        tmp_env.action_space,
                        {"prob_list": prob_list},
                    )
                }
            }
        },
    )

    config.update(
        {
            "input": buffer_file_path,
            "evaluation_interval": None,
            "input_evaluation": [],
            "postprocess_inputs": True,
            "observation_space": tmp_env.observation_space,
            "action_space": tmp_env.action_space,
            "env": scenario.single_agent_env_class,
            "env_config": runtime_single_agent_env_config,
        }
    )
    # pprint.pprint(config)
    trainer = MARWILTrainer(config=config)
    print("trainer has been created")

    min_reward = 100.0

    for i in range(num_iterations):
        results = trainer.train()
        print("+" * 10 + f"\n\titeration: {i}\n" + "+" * 10)
        print(results)

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
