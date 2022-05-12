from copy import deepcopy
from typing import List

import time
import logging
import os
import functools
import operator
import itertools
import random

import ray
import numpy as np

from ray.rllib.utils import merge_dicts
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.agents import Trainer
from ray.rllib.agents.marwil import MARWILTrainer

from rlapps.utils.strategy_spec import StrategySpec
from rlapps.utils.common import datetime_str, pretty_dict_str
from rlapps.rllib_tools.space_saving_logger import get_trainer_logger_creator
from rlapps.rllib_tools.policy_checkpoints import save_policy_checkpoint
from rlapps.algos.psro_distill.manager.manager import DistillerResult, Distiller
from rlapps.apps.scenarios.psro_distill_scenario import DistilledPSROScenario
from rlapps.envs.ma_to_single_env import SingleAgentEnv


logger = logging.getLogger(__name__)


def checkpoint_dir(trainer: Trainer):
    return os.path.join(trainer.logdir, "br_checkpoints")


class BCDistiller(Distiller):
    def __init__(self, scenario: DistilledPSROScenario) -> None:
        super().__init__()
        self.scenario = scenario

    def __call__(
        self,
        log_dir: str,
        metanash_player: int,
        prob_list_each_player: List[List[float]],
        spec_list_each_player: List[List[StrategySpec]],
        manager_metadata: dict = None,
    ) -> DistillerResult:

        should_log_result_fn = self.scenario.ray_should_log_result_filter
        env_class = self.scenario.env_class
        env_config = self.scenario.env_config

        assert len(prob_list_each_player) == self.scenario.player_num, len(
            prob_list_each_player
        )
        assert len(spec_list_each_player) == self.scenario.player_num, len(
            spec_list_each_player
        )

        tmp_env = env_class(env_config=env_config)
        stopping_condition = self.scenario.distill_get_stopping_condition()
        print_train_results = True

        joint_policy_mapping = manager_metadata["offline_dataset"][str(metanash_player)]
        # print("all mapping:", manager_metadata["offline_dataset"])
        assert (
            len(joint_policy_mapping) > 0
        ), "joint policy mapping for metanash player={} is empty: {}".format(
            metanash_player, manager_metadata["offline_dataset"]
        )
        offline_weighted_inputs = {}
        # build keys
        for prod in itertools.product(*spec_list_each_player):
            key = "&".join([e.id for e in prod])
            joint_prob = functools.reduce(
                operator.mul,
                [
                    prob_list_each_player[player_id][
                        spec_list_each_player[player_id].index(spec)
                    ]
                    for player_id, spec in enumerate(prod)
                ],
            )
            offline_weighted_inputs[joint_policy_mapping[key]] = joint_prob

        assert np.isclose(sum(offline_weighted_inputs.values()), 1.0), (
            offline_weighted_inputs,
            sum(offline_weighted_inputs.values()),
        )

        if sum(offline_weighted_inputs.values()) != 1.0:
            logger.log(logging.WARNING, "to avoid rllib error, add some noise")
            noise = 1.0 - sum(offline_weighted_inputs.values())
            rand_key = random.choice(list(offline_weighted_inputs.keys()))
            offline_weighted_inputs[rand_key] = noise

        other = 1 - metanash_player

        def select_policy(agent_id):
            if agent_id == metanash_player:
                raise RuntimeError(
                    "cannot call policy selection for best response player: {}".format(
                        agent_id
                    )
                )
            else:
                return "eval"

        runtime_single_agent_env_config = {
            "env_class": self.scenario.env_class,
            "env_config": self.scenario.env_config,
            "br_player": metanash_player,
            "multiagent": {
                "policies": {
                    "eval": (
                        self.scenario.policy_classes["eval"],
                        tmp_env.observation_space,
                        tmp_env.action_space,
                        {},
                    )
                },
                "policy_mapping_fn": select_policy,
                # FIXME(ming): model load error
                # "strategy_spec_dict": {
                #     "eval": (
                #         prob_list_each_player[other],
                #         spec_list_each_player[other],
                #     )
                # },
            },
        }

        # print("offline weighted iputs:", offline_weighted_inputs)

        training_config = {
            "input": offline_weighted_inputs,
            # "input": {"/tmp/kuhn_distill_psro_dqn/distilled_policy_0_vs_distilled_policy_1": 1.},
            "observation_space": tmp_env.observation_space,
            "action_space": tmp_env.action_space,
            "env": SingleAgentEnv,
            "env_config": runtime_single_agent_env_config,
        }

        training_config = merge_dicts(
            self.scenario.get_trainer_config_distill(tmp_env), training_config
        )

        trainer = MARWILTrainer(
            config=training_config,
            logger_creator=get_trainer_logger_creator(
                base_dir=log_dir,
                scenario_name="marwil_trainer",
                should_log_result_fn=should_log_result_fn,
                print_log_dir=False,
            ),
        )

        meta_learner_name = f"new_learner_{metanash_player}"

        def log(message):
            print(f"({meta_learner_name}): {message}")

        while True:
            train_results = trainer.train()

            eval_results = train_results.get("evaluation")
            if eval_results:
                log(
                    "iteration={} R={}".format(
                        train_results["training_iteration"],
                        eval_results["episode_reward_mean"],
                    )
                )

            # if print_train_results:
            #     log(pretty_dict_str(train_results))

            if stopping_condition.should_stop_this_iter(
                latest_trainer_result=train_results
            ):
                print("stopping condition met.")
                final_train_result = deepcopy(train_results)
                break

        strategy_id = f"distilled_{metanash_player}_{datetime_str()}"
        checkpoint_path = save_policy_checkpoint(
            trainer=trainer,
            player=metanash_player,
            save_dir=checkpoint_dir(trainer=trainer),
            policy_id_to_save=DEFAULT_POLICY_ID,
            checkpoint_name=f"{strategy_id}.h5",
            additional_data={},
        )

        distilled_strategy_spec = StrategySpec(
            strategy_id=strategy_id,
            metadata={"checkpoint_path": checkpoint_path},
        )

        # ray.kill(trainer.workers.local_worker().replay_buffer_actor)
        del trainer
        # del avg_br_reward_deque

        time.sleep(10)

        assert final_train_result is not None

        return DistillerResult(
            distilled_strategy_spec=distilled_strategy_spec,
            episodes_spent_in_solve=final_train_result["episodes_total"],
            timesteps_spent_in_solve=final_train_result["timesteps_total"],
            extra_data_to_log={},
        )
