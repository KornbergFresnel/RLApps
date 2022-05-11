import ray

from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.ppo import PPOTrainer, PPOTorchPolicy
from ray.rllib.agents.marwil import MARWILTrainer, MARWILTorchPolicy

from rlapps.envs.poker_multi_agent_env import PokerMultiAgentEnv
from rlapps.apps.scenarios.catalog import scenario_catalog
from rlapps.apps.scenarios.catalog.common import default_if_creating_ray_head
from rlapps.apps.scenarios.psro_distill_scenario import DistilledPSROScenario
from rlapps.apps.scenarios.stopping_conditions import *
from rlapps.apps.scenarios.trainer_configs.poker_psro_configs import *
from rlapps.apps.scenarios.trainer_configs.poker_psro_distill_configs import *
from rlapps.rllib_tools.modified_policies.simple_q_torch_policy import (
    SimpleQTorchPolicyPatched,
)

from rlapps.apps.psro_distill.mawril_metanash import BCDistiller


kuhn_distill_psro_dqn = DistilledPSROScenario(
    name="kuhn_distill_psro_dqn",
    ray_cluster_cpus=default_if_creating_ray_head(default=8),
    ray_cluster_gpus=default_if_creating_ray_head(default=0),
    ray_object_store_memory_cap_gigabytes=1,
    env_class=PokerMultiAgentEnv,
    env_config={
        "version": "kuhn_poker",
        "fixed_players": True,
    },
    mix_metanash_with_uniform_dist_coeff=0.0,
    allow_stochastic_best_responses=False,
    trainer_class_br=DQNTrainer,
    policy_classes_br={
        "metanash": SimpleQTorchPolicyPatched,
        "best_response": SimpleQTorchPolicyPatched,
        "eval": SimpleQTorchPolicyPatched,
    },
    get_trainer_config_br=psro_kuhn_dqn_params,
    trainer_class_distill=MARWILTrainer,
    policy_classes_distill={
        "distilled_policy": MARWILTorchPolicy,
        "eval": SimpleQTorchPolicyPatched,
    },
    get_trainer_config_distill=distill_psro_kuhn_marwil_params,
    get_distiller=lambda scenario: BCDistiller(scenario),
    distill_get_stopping_condition=lambda: EvalRewardStopping(
        dont_check_plateau_before_n_iteration=int(64),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_training_iteration=int(1e5),
    ),
    get_buffer_directory=lambda scenario, br_player, br_ids: os.path.join(
        ray._private.utils.get_user_temp_dir(),
        scenario.name,
        str(br_player),
        "_vs_".join(br_ids),
    ),
    num_eval_workers=8,
    games_per_payoff_eval=20000,
    p2sro=False,
    p2sro_payoff_table_exponential_avg_coeff=None,
    p2sro_sync_with_payoff_table_every_n_episodes=None,
    single_agent_symmetric_game=False,
    psro_get_stopping_condition=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_episodes=int(2e4),
        check_plateau_every_n_episodes=int(2e4),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_episodes=int(1e5),
    ),
    calc_exploitability_for_openspiel_env=True,
)


leduc_distill_psro_dqn = DistilledPSROScenario(
    name="leduc_distill_psro_dqn",
    ray_cluster_cpus=default_if_creating_ray_head(default=8),
    ray_cluster_gpus=default_if_creating_ray_head(default=0),
    ray_object_store_memory_cap_gigabytes=1,
    env_class=PokerMultiAgentEnv,
    env_config={
        "version": "leduc_poker",
        "fixed_players": True,
        "append_valid_actions_mask_to_obs": True,
    },
    mix_metanash_with_uniform_dist_coeff=0.0,
    allow_stochastic_best_responses=False,
    trainer_class_br=DQNTrainer,  # br trainer
    policy_classes_br={
        "metanash": SimpleQTorchPolicyPatched,
        "best_response": SimpleQTorchPolicyPatched,
        "eval": SimpleQTorchPolicyPatched,
    },
    get_trainer_config_br=psro_leduc_dqn_params,
    trainer_class_distill=MARWILTrainer,
    policy_classes_distill={"meta_nash": MARWILTorchPolicy, "eval": MARWILTorchPolicy},
    get_trainer_config_distill=distill_psro_leduc_marwil_params,
    get_distiller=lambda scenario: BCDistiller(scenario),
    distill_get_stopping_condition=lambda: EvalRewardStopping(
        dont_check_plateau_before_n_iteration=int(64),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_training_iteration=int(1e5),
    ),
    get_buffer_directory=lambda scenario, br_player, br_ids: os.path.join(
        ray._private.utils.get_user_temp_dir(),
        scenario.name,
        str(br_player),
        "_vs_".join(br_ids),
    ),
    num_eval_workers=8,
    games_per_payoff_eval=3000,
    p2sro=False,
    p2sro_payoff_table_exponential_avg_coeff=None,
    p2sro_sync_with_payoff_table_every_n_episodes=None,
    single_agent_symmetric_game=False,
    psro_get_stopping_condition=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_episodes=int(2e4),
        check_plateau_every_n_episodes=int(2e4),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_episodes=int(1e5),
    ),
    calc_exploitability_for_openspiel_env=True,
)

scenario_catalog.add(kuhn_distill_psro_dqn)
scenario_catalog.add(leduc_distill_psro_dqn)
