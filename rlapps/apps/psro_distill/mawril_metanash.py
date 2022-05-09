from copy import deepcopy
from typing import List

import time
import os
import ray
import deepdish

from ray.rllib.utils import merge_dicts, try_import_torch
from ray.rllib.agents import Trainer
from ray.rllib.agents.marwil import (
    MARWILTorchPolicy,
    MARWILTrainer,
    DEFAULT_CONFIG,
    BC_DEFAULT_CONFIG,
)

from rlapps.utils.strategy_spec import StrategySpec
from rlapps.utils.common import (
    pretty_dict_str,
    datetime_str,
    ensure_dir,
    copy_attributes,
)
from rlapps.rllib_tools.space_saving_logger import get_trainer_logger_creator
from rlapps.algos.psro_distill.manager.manager import DistillerResult, Distiller
from rlapps.apps.scenarios.psro_distill_scenario import DistilledPSROScenario


def checkpoint_dir(trainer: Trainer):
    return os.path.join(trainer.logdir, "br_checkpoints")


def save_nfsp_avg_policy_checkpoint(
    trainer: Trainer,
    policy_id_to_save: str,
    save_dir: str,
    timesteps_training: int,
    episodes_training: int,
    checkpoint_name=None,
):
    policy_name = policy_id_to_save
    date_time = datetime_str()
    if checkpoint_name is None:
        checkpoint_name = f"policy_{policy_name}_{date_time}.h5"
    checkpoint_path = os.path.join(save_dir, checkpoint_name)
    br_weights = trainer.get_weights([policy_id_to_save])[policy_id_to_save]
    br_weights = {
        k.replace(".", "_dot_"): v for k, v in br_weights.items()
    }  # periods cause HDF5 NaturalNaming warnings
    ensure_dir(file_path=checkpoint_path)
    deepdish.io.save(
        path=checkpoint_path,
        data={
            "weights": br_weights,
            "date_time_str": date_time,
            "seconds_since_epoch": time.time(),
            "timesteps_training": timesteps_training,
            "episodes_training": episodes_training,
        },
    )
    return checkpoint_path


class BCDistiller(Distiller):
    def __init__(self, scenario: DistilledPSROScenario) -> None:
        super().__init__()
        self.scenario = scenario

    def __call__(
        self,
        log_dir: str,
        br_prob_list: List[float],
        br_spec_list: List[StrategySpec],
        manager_metadata: dict = None,
    ) -> DistillerResult:

        should_log_result_fn = self.scenario.ray_should_log_result_filter
        env_class = self.scenario.env_class
        env_config = self.scenario.env_config

        def assert_not_called(agent_id):
            assert False, "This function should never be called."

        def _create_base_env():
            return env_class(env_config=env_config)

        tmp_env = env_class(env_config=env_config)
        stopping_condition = self.scenario.distill_get_stopping_condition
        print_train_results = True

        training_config = merge_dicts(
            {
                "log_level": "DEBUG",
                "framework": "torch",
                "disable_env_checking": True,
                "env": self.scenario.env_class,
                "env_config": self.scenario.env_config,
                "num_gpus": 0.0,
                "num_gpus_per_worker": 0.0,
                "num_workers": 0,
                "num_envs_per_worker": 1,
                "multiagent": {
                    "policies_to_train": ["distilled_policy"],
                    "policies": {
                        "distilled_policy": (
                            self.scenario.policy_classes["distilled_policy"],
                            tmp_env.observation_space,
                            tmp_env.action_space,
                        ),
                    },
                    "policy_mapping_fn": assert_not_called,
                },
            },
        )

        trainer = MARWILTrainer(
            training_config,
            logger_creator=get_trainer_logger_creator(
                base_dir=log_dir,
                scenario_name="marwil_trainer",
                should_log_result_fn=should_log_result_fn,
            ),
        )

        while True:
            train_results = trainer.train()
            if print_train_results:
                raise NotImplementedError

            if stopping_condition.should_stop_this_iter(
                latest_trainer_result=train_results
            ):
                print("stopping condition met.")
                final_train_result = deepcopy(train_results)
                break

        if self.scenario.calculate_openspiel_metanash_at_end:
            base_env = _create_base_env()
            open_spiel_env_config = base_env.open_spiel_env_config
            openspiel_game_version = base_env.game_version
            local_avg_policy_0 = br_trainer.workers.local_worker().policy_map[
                "average_policy_0"
            ]
            local_avg_policy_1 = br_trainer.workers.local_worker().policy_map[
                "average_policy_1"
            ]
            exploitability = nxdo_nfsp_measure_exploitability_nonlstm(
                rllib_policies=[local_avg_policy_0, local_avg_policy_1],
                poker_game_version=openspiel_game_version,
                restricted_game_convertors=br_trainer.get_local_converters(),
                open_spiel_env_config=open_spiel_env_config,
                use_delegate_policy_exploration=self.scenario.allow_stochastic_best_responses,
            )
            final_train_result["avg_policy_exploitability"] = exploitability

        if "avg_policy_exploitability" in final_train_result:
            print(
                f"\n\nexploitability: {final_train_result['avg_policy_exploitability']}\n\n"
            )

        avg_policy_specs = []
        for player in range(2):
            strategy_id = f"avg_policy_player_{player}_{datetime_str()}"

            checkpoint_path = save_nfsp_avg_policy_checkpoint(
                trainer=br_trainer,
                policy_id_to_save=f"average_policy_{player}",
                save_dir=checkpoint_dir(trainer=br_trainer),
                timesteps_training=final_train_result["timesteps_total"],
                episodes_training=final_train_result["episodes_total"],
                checkpoint_name=f"{strategy_id}.h5",
            )

            avg_policy_spec = StrategySpec(
                strategy_id=strategy_id,
                metadata={
                    "checkpoint_path": checkpoint_path,
                    "delegate_policy_specs": [
                        spec.to_json()
                        for spec in player_to_base_game_action_specs[player]
                    ],
                },
            )
            avg_policy_specs.append(avg_policy_spec)

        ray.kill(avg_trainer.workers.local_worker().replay_buffer_actor)
        avg_trainer.cleanup()
        br_trainer.cleanup()
        del avg_trainer
        del br_trainer
        del avg_br_reward_deque

        time.sleep(10)

        assert final_train_result is not None
        # return avg_policy_specs, final_train_result
        # TODO(ming): pack to distilled results
        return DistillerResult(distilled_strategy_spec=distilled_strategy_spec)
