from typing import Union, Type, Dict, Any, Callable

from ray.rllib import MultiAgentEnv, Policy
from ray.rllib.agents import Trainer
from ray.rllib.utils.typing import ResultDict

from rlapps.apps.scenarios.stopping_conditions import StoppingCondition
from rlapps.apps.scenarios.psro_scenario import PSROScenario


def psro_default_log_filter(result: ResultDict) -> bool:
    return result["training_iteration"] % 10 == 0


class DistilledPSROScenario(PSROScenario):
    def __init__(
        self,
        name: str,
        ray_cluster_cpus: Union[int, float],
        ray_cluster_gpus: Union[int, float],
        ray_object_store_memory_cap_gigabytes: Union[int, float],
        env_class: Type[MultiAgentEnv],
        env_config: Dict[str, Any],
        mix_metanash_with_uniform_dist_coeff: float,
        allow_stochastic_best_responses: bool,
        trainer_class_br: Type[Trainer],
        policy_classes_br: Dict[str, Type[Policy]],
        get_trainer_config_br: Callable[[MultiAgentEnv], Dict[str, Any]],
        trainer_class_distill: Type[Trainer],
        policy_classes_distill: Dict[str, Type[Policy]],
        get_trainer_config_distill: Callable[[MultiAgentEnv], Dict[str, Any]],
        num_eval_workers: int,
        games_per_payoff_eval: int,
        p2sro: bool,
        p2sro_payoff_table_exponential_avg_coeff: Union[float, None],
        p2sro_sync_with_payoff_table_every_n_episodes: Union[int, None],
        single_agent_symmetric_game: bool,
        psro_get_stopping_condition: Callable[[], StoppingCondition],
        calc_exploitability_for_openspiel_env: bool,
        ray_should_log_result_filter: Callable[
            [ResultDict], bool
        ] = psro_default_log_filter,
    ):

        super().__init__(
            name,
            ray_cluster_cpus,
            ray_cluster_gpus,
            ray_object_store_memory_cap_gigabytes,
            env_class,
            env_config,
            mix_metanash_with_uniform_dist_coeff,
            allow_stochastic_best_responses,
            trainer_class_br,
            policy_classes_br,
            num_eval_workers,
            games_per_payoff_eval,
            p2sro,
            p2sro_payoff_table_exponential_avg_coeff,
            p2sro_sync_with_payoff_table_every_n_episodes,
            single_agent_symmetric_game,
            get_trainer_config_br,
            psro_get_stopping_condition,
            calc_exploitability_for_openspiel_env,
            ray_should_log_result_filter,
        )

        self.trainer_class_distill = trainer_class_distill
        self.policy_classes_distill = policy_classes_distill
        self.get_trainer_config_distill = get_trainer_config_distill
