from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.ppo import PPOTrainer, PPOTorchPolicy

from rlapps.algos.nfsp_rllib.nfsp import NFSPTrainer, NFSPTorchAveragePolicy
from rlapps.envs.poker_multi_agent_env import PokerMultiAgentEnv
from rlapps.apps.nxdo.solve_restricted_game_fns import *
from rlapps.apps.scenarios.catalog import scenario_catalog
from rlapps.apps.scenarios.catalog.common import default_if_creating_ray_head
from rlapps.apps.scenarios.stopping_conditions import *
from rlapps.apps.scenarios.trainer_configs.poker_nfsp_configs import *
from rlapps.apps.scenarios.trainer_configs.poker_psro_configs import *
from rlapps.rllib_tools.models.valid_actions_fcnet import (
    get_valid_action_fcn_class_for_env,
)
from rlapps.rllib_tools.modified_policies.simple_q_torch_policy import (
    SimpleQTorchPolicyPatched,
)

# kuhn_nxdo_dqn_nfsp
scenario_catalog.add(
    NXDOScenario(
        name=f"kuhn_nxdo_dqn_nfsp",
        ray_cluster_cpus=default_if_creating_ray_head(default=8),
        ray_cluster_gpus=default_if_creating_ray_head(default=0),
        ray_object_store_memory_cap_gigabytes=1,
        use_openspiel_restricted_game=False,
        get_restricted_game_custom_model=lambda env: None,
        xdo_metanash_method="nfsp",
        allow_stochastic_best_responses=False,
        get_restricted_game_solver=lambda scenario: SolveRestrictedGameFixedRewardThreshold(
            scenario=scenario,
            br_reward_threshold=0.01,
            min_episodes=300000,
            required_fields=["avg_policy_exploitability"],
        ),
        env_class=PokerMultiAgentEnv,
        env_config={
            "version": "kuhn_poker",
            "fixed_players": True,
        },
        trainer_class_br=DQNTrainer,
        policy_classes_br={
            "metanash": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },
        get_trainer_config_br=psro_kuhn_dqn_params,
        get_stopping_condition_br=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(2e4),
            check_plateau_every_n_episodes=int(2e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(1e5),
        ),
        trainer_class_nfsp=DQNTrainer,
        avg_trainer_class_nfsp=NFSPTrainer,
        policy_classes_nfsp={
            "average_policy": NFSPTorchAveragePolicy,
            "delegate_policy": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
        },
        anticipatory_param_nfsp=0.1,
        get_trainer_config_nfsp=nfsp_kuhn_dqn_params,
        get_avg_trainer_config_nfsp=nfsp_kuhn_avg_policy_params,
        calculate_openspiel_metanash=True,
        calculate_openspiel_metanash_at_end=False,
        calc_metanash_every_n_iters=50,
        metanash_metrics_smoothing_episodes_override=50000,
    )
)

base_dqn_leduc = NXDOScenario(
    name="leduc_nxdo_dqn_nfsp",
    ray_cluster_cpus=default_if_creating_ray_head(default=8),
    ray_cluster_gpus=default_if_creating_ray_head(default=0),
    ray_object_store_memory_cap_gigabytes=1,
    use_openspiel_restricted_game=False,
    get_restricted_game_custom_model=lambda env: None,
    xdo_metanash_method="nfsp",
    allow_stochastic_best_responses=False,
    get_restricted_game_solver=lambda scenario: SolveRestrictedGameFixedRewardThreshold(
        scenario=scenario,
        br_reward_threshold=0.1,
        min_episodes=300000,
        required_fields=["avg_policy_exploitability"],
    ),
    env_class=PokerMultiAgentEnv,
    env_config={
        "version": "leduc_poker",
        "fixed_players": True,
        "append_valid_actions_mask_to_obs": True,
    },
    trainer_class_br=DQNTrainer,
    policy_classes_br={
        "metanash": NFSPTorchAveragePolicy,
        "best_response": SimpleQTorchPolicyPatched,
    },
    get_trainer_config_br=psro_leduc_dqn_params,
    get_stopping_condition_br=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_episodes=int(2e4),
        check_plateau_every_n_episodes=int(2e4),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_episodes=int(1e5),
    ),
    trainer_class_nfsp=DQNTrainer,
    avg_trainer_class_nfsp=NFSPTrainer,
    policy_classes_nfsp={
        "average_policy": NFSPTorchAveragePolicy,
        "delegate_policy": SimpleQTorchPolicyPatched,
        "best_response": SimpleQTorchPolicyPatched,
    },
    anticipatory_param_nfsp=0.1,
    get_trainer_config_nfsp=nfsp_leduc_dqn_params,
    get_avg_trainer_config_nfsp=nfsp_leduc_avg_policy_params,
    calculate_openspiel_metanash=True,
    calculate_openspiel_metanash_at_end=False,
    calc_metanash_every_n_iters=50,
    metanash_metrics_smoothing_episodes_override=50000,
)
scenario_catalog.add(base_dqn_leduc)

# va_leduc_nxdo_dqn_nfsp_fixed_timesteps_50000
# va_leduc_nxdo_dqn_nfsp_fixed_timesteps_150000
# va_leduc_nxdo_dqn_nfsp_fixed_timesteps_300000
# va_leduc_nxdo_dqn_nfsp_fixed_timesteps_600000

for nfsp_episodes in [50000, 150000, 300000, 600000]:
    for use_valid_actions_restricted_game in [True, False]:

        name = f"leduc_nxdo_dqn_nfsp_fixed_timesteps_{nfsp_episodes}"

        if use_valid_actions_restricted_game:
            name = f"va_{name}"
            get_restricted_game_custom_model = get_valid_action_fcn_class_for_env
        else:
            get_restricted_game_custom_model = lambda env: None

        scenario_catalog.add(
            base_dqn_leduc.with_updates(
                name=name,
                use_openspiel_restricted_game=use_valid_actions_restricted_game,
                get_restricted_game_custom_model=get_restricted_game_custom_model,
                get_restricted_game_solver=lambda scenario, nfsp_episodes=nfsp_episodes: SolveRestrictedGameFixedTimeToSolve(
                    scenario=scenario,
                    dont_solve_first_n_nxdo_iters=8,
                    fixed_episodes=nfsp_episodes,
                    fixed_steps=None,
                    required_fields=["avg_policy_exploitability"],
                ),
                # get_avg_trainer_config_nfsp=nfsp_leduc_avg_policy_params_improved,
            )
        )

# va_leduc_nxdo_dqn_nfsp_increasing_timesteps_1.1
# va_leduc_nxdo_dqn_nfsp_increasing_timesteps_1.5
# va_leduc_nxdo_dqn_nfsp_increasing_timesteps_2.0
# leduc_nxdo_dqn_nfsp_increasing_timesteps_2.0

for increase_multiplier in [1.1, 1.5, 2.0]:
    for use_valid_actions_restricted_game in [True, False]:
        name = (
            f"leduc_nxdo_dqn_nfsp_increasing_timesteps_{'%.1f' % increase_multiplier}"
        )

        if use_valid_actions_restricted_game:
            name = f"va_{name}"
            get_restricted_game_custom_model = get_valid_action_fcn_class_for_env
        else:
            get_restricted_game_custom_model = lambda env: None

        scenario_catalog.add(
            base_dqn_leduc.with_updates(
                name=name,
                use_openspiel_restricted_game=use_valid_actions_restricted_game,
                get_restricted_game_custom_model=get_restricted_game_custom_model,
                get_restricted_game_solver=lambda scenario, increase_multiplier=increase_multiplier: SolveRestrictedGameIncreasingTimeToSolve(
                    scenario=scenario,
                    dont_solve_first_n_nxdo_iters=7,
                    starting_episodes=50000,
                    starting_steps=None,
                    increase_multiplier=increase_multiplier,
                    required_fields=["avg_policy_exploitability"],
                ),
                # get_avg_trainer_config_nfsp=nfsp_leduc_avg_policy_params_improved,
            )
        )

# va_20_clone_leduc_nxdo_dqn_nfsp_dynamic_threshold_1_aggressive
# 20_clone_leduc_nxdo_dqn_nfsp_dynamic_threshold_1_aggressive
# va_40_clone_leduc_nxdo_dqn_nfsp_dynamic_threshold_1_aggressive
# 40_clone_leduc_nxdo_dqn_nfsp_dynamic_threshold_1_aggressive
# va_80_clone_leduc_nxdo_dqn_nfsp_dynamic_threshold_1_aggressive
# 80_clone_leduc_nxdo_dqn_nfsp_dynamic_threshold_1_aggressive
for dummy_action_multiplier in [20, 40, 80]:
    for use_valid_actions_restricted_game in [True, False]:
        name = f"{dummy_action_multiplier}_clone_leduc_nxdo_dqn_nfsp_dynamic_threshold_1_aggressive"

        if use_valid_actions_restricted_game:
            name = f"va_{name}"
            get_restricted_game_custom_model = get_valid_action_fcn_class_for_env
        else:
            get_restricted_game_custom_model = lambda env: None

        scenario_catalog.add(
            NXDOScenario(
                name=name,
                ray_cluster_cpus=default_if_creating_ray_head(default=8),
                ray_cluster_gpus=default_if_creating_ray_head(default=0),
                ray_object_store_memory_cap_gigabytes=1,
                use_openspiel_restricted_game=use_valid_actions_restricted_game,
                get_restricted_game_custom_model=get_restricted_game_custom_model,
                xdo_metanash_method="nfsp",
                allow_stochastic_best_responses=False,
                get_restricted_game_solver=lambda scenario: SolveRestrictedGameDynamicRewardThreshold1(
                    scenario=scenario,
                    dont_solve_first_n_nxdo_iters=7,
                    starting_rew_threshold=1.0,
                    min_rew_threshold=0.05,
                    min_episodes=50000,
                    epsilon=0.05,
                    required_fields=["avg_policy_exploitability"],
                ),
                env_class=PokerMultiAgentEnv,
                env_config={
                    "version": "leduc_poker",
                    "fixed_players": True,
                    "append_valid_actions_mask_to_obs": True,
                    "dummy_action_multiplier": dummy_action_multiplier,
                },
                trainer_class_br=DQNTrainer,
                policy_classes_br={
                    "metanash": NFSPTorchAveragePolicy,
                    "best_response": SimpleQTorchPolicyPatched,
                },
                get_trainer_config_br=psro_leduc_dqn_params,
                get_stopping_condition_br=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
                    br_policy_id="best_response",
                    dont_check_plateau_before_n_episodes=int(2e4),
                    check_plateau_every_n_episodes=int(2e4),
                    minimum_reward_improvement_otherwise_plateaued=0.01,
                    max_train_episodes=int(1e5),
                ),
                trainer_class_nfsp=DQNTrainer,
                avg_trainer_class_nfsp=NFSPTrainer,
                policy_classes_nfsp={
                    "average_policy": NFSPTorchAveragePolicy,
                    "delegate_policy": SimpleQTorchPolicyPatched,
                    "best_response": SimpleQTorchPolicyPatched,
                },
                anticipatory_param_nfsp=0.1,
                get_trainer_config_nfsp=nfsp_leduc_dqn_params,
                get_avg_trainer_config_nfsp=nfsp_leduc_avg_policy_params,
                calculate_openspiel_metanash=True,
                calculate_openspiel_metanash_at_end=False,
                calc_metanash_every_n_iters=50,
                metanash_metrics_smoothing_episodes_override=50000,
            )
        )

scenario_catalog.add(
    NXDOScenario(
        name=f"kuhn_no_limit_leduc_nxdo_ppo_nfsp_dynamic_threshold_1_warm_start_7",
        ray_cluster_cpus=default_if_creating_ray_head(default=8),
        ray_cluster_gpus=default_if_creating_ray_head(default=0),
        ray_object_store_memory_cap_gigabytes=1,
        use_openspiel_restricted_game=False,
        get_restricted_game_custom_model=lambda env: None,
        xdo_metanash_method="nfsp",
        allow_stochastic_best_responses=False,
        get_restricted_game_solver=lambda scenario: SolveRestrictedGameFixedRewardThreshold(
            scenario=scenario,
            br_reward_threshold=0.01,
            min_episodes=300000,
            required_fields=["avg_policy_exploitability"],
        ),
        env_class=PokerMultiAgentEnv,
        env_config={
            "version": "kuhn_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": False,
            "continuous_action_space": True,
        },
        trainer_class_br=PPOTrainer,
        policy_classes_br={
            "metanash": NFSPTorchAveragePolicy,
            "best_response": PPOTorchPolicy,
        },
        get_trainer_config_br=psro_leduc_ppo_params,
        get_stopping_condition_br=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(1e4),
            check_plateau_every_n_episodes=int(1e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(1e5),
        ),
        trainer_class_nfsp=DQNTrainer,
        avg_trainer_class_nfsp=NFSPTrainer,
        policy_classes_nfsp={
            "average_policy": NFSPTorchAveragePolicy,
            "delegate_policy": PPOTorchPolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },
        anticipatory_param_nfsp=0.1,
        get_trainer_config_nfsp=nfsp_leduc_dqn_params_two_layers_no_valid_actions_model,
        get_avg_trainer_config_nfsp=nfsp_leduc_avg_policy_params_two_layers_no_valid_actions_model,
        calculate_openspiel_metanash=True,
        calculate_openspiel_metanash_at_end=False,
        calc_metanash_every_n_iters=50,
        metanash_metrics_smoothing_episodes_override=50000,
    )
)

scenario_catalog.add(
    NXDOScenario(
        name=f"kuhn_no_limit_leduc_nxdo_ppo_nfsp_dynamic_threshold_1_warm_start_7_stochastic",
        ray_cluster_cpus=default_if_creating_ray_head(default=8),
        ray_cluster_gpus=default_if_creating_ray_head(default=0),
        ray_object_store_memory_cap_gigabytes=1,
        use_openspiel_restricted_game=False,
        get_restricted_game_custom_model=lambda env: None,
        xdo_metanash_method="nfsp",
        allow_stochastic_best_responses=True,
        get_restricted_game_solver=lambda scenario: SolveRestrictedGameFixedRewardThreshold(
            scenario=scenario,
            br_reward_threshold=0.01,
            min_episodes=300000,
            required_fields=["avg_policy_exploitability"],
        ),
        env_class=PokerMultiAgentEnv,
        env_config={
            "version": "kuhn_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": False,
            "continuous_action_space": True,
        },
        trainer_class_br=PPOTrainer,
        policy_classes_br={
            "metanash": NFSPTorchAveragePolicy,
            "best_response": PPOTorchPolicy,
        },
        get_trainer_config_br=psro_leduc_ppo_params,
        get_stopping_condition_br=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(1e4),
            check_plateau_every_n_episodes=int(1e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(1e5),
        ),
        trainer_class_nfsp=DQNTrainer,
        avg_trainer_class_nfsp=NFSPTrainer,
        policy_classes_nfsp={
            "average_policy": NFSPTorchAveragePolicy,
            "delegate_policy": PPOTorchPolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },
        anticipatory_param_nfsp=0.1,
        get_trainer_config_nfsp=nfsp_leduc_dqn_params_two_layers_no_valid_actions_model,
        get_avg_trainer_config_nfsp=nfsp_leduc_avg_policy_params_two_layers_no_valid_actions_model,
        calculate_openspiel_metanash=True,
        calculate_openspiel_metanash_at_end=False,
        calc_metanash_every_n_iters=50,
        metanash_metrics_smoothing_episodes_override=50000,
    )
)

# 12_no_limit_leduc_nxdo_dqn_nfsp_dynamic_threshold_1_aggressive
# 30_no_limit_leduc_nxdo_dqn_nfsp_dynamic_threshold_1_aggressive
# 60_no_limit_leduc_nxdo_dqn_nfsp_dynamic_threshold_1_aggressive
# 100_no_limit_leduc_nxdo_dqn_nfsp_dynamic_threshold_1_aggressive
# 1000_no_limit_leduc_nxdo_dqn_nfsp_dynamic_threshold_1_aggressive


# 1000_no_limit_leduc_nxdo_ppo_nfsp_dynamic_threshold_1_warm_start_15_stochastic
# 1000_no_limit_leduc_nxdo_ppo_nfsp_dynamic_threshold_1_warm_start_15
# 1000_no_limit_leduc_nxdo_ppo_nfsp_dynamic_threshold_1_warm_start_7_stochastic
# 1000_no_limit_leduc_nxdo_ppo_nfsp_dynamic_threshold_1_warm_start_7


for stack_size in [12, 30, 60, 100, 1000]:
    for warm_start_iters in [0, 1, 4, 7, 15]:
        for allow_stochastic_best_responses in [True, False]:
            stochastic_str = "_stochastic" if allow_stochastic_best_responses else ""

            scenario_catalog.add(
                NXDOScenario(
                    name=f"{stack_size}_no_limit_leduc_nxdo_dqn_nfsp_dynamic_threshold_1_warm_start_{warm_start_iters}{stochastic_str}",
                    ray_cluster_cpus=default_if_creating_ray_head(default=8),
                    ray_cluster_gpus=default_if_creating_ray_head(default=0),
                    ray_object_store_memory_cap_gigabytes=1,
                    use_openspiel_restricted_game=False,
                    get_restricted_game_custom_model=lambda env: None,
                    xdo_metanash_method="nfsp",
                    allow_stochastic_best_responses=allow_stochastic_best_responses,
                    get_restricted_game_solver=lambda scenario, warm_start_iters=warm_start_iters: SolveRestrictedGameDynamicRewardThreshold1(
                        scenario=scenario,
                        dont_solve_first_n_nxdo_iters=warm_start_iters,
                        starting_rew_threshold=1.0,
                        min_rew_threshold=0.05,
                        min_episodes=50000,
                        epsilon=0.05,
                        required_fields=[],
                    ),
                    env_class=PokerMultiAgentEnv,
                    env_config={
                        "version": "universal_poker",
                        "fixed_players": True,
                        "append_valid_actions_mask_to_obs": True,
                        "universal_poker_stack_size": stack_size,
                    },
                    trainer_class_br=DQNTrainer,
                    policy_classes_br={
                        "metanash": NFSPTorchAveragePolicy,
                        "best_response": SimpleQTorchPolicyPatched,
                    },
                    get_trainer_config_br=psro_leduc_dqn_params,
                    get_stopping_condition_br=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
                        br_policy_id="best_response",
                        dont_check_plateau_before_n_episodes=int(2e4),
                        check_plateau_every_n_episodes=int(2e4),
                        minimum_reward_improvement_otherwise_plateaued=0.01,
                        max_train_episodes=int(1e5),
                    ),
                    trainer_class_nfsp=DQNTrainer,
                    avg_trainer_class_nfsp=NFSPTrainer,
                    policy_classes_nfsp={
                        "average_policy": NFSPTorchAveragePolicy,
                        "delegate_policy": SimpleQTorchPolicyPatched,
                        "best_response": SimpleQTorchPolicyPatched,
                    },
                    anticipatory_param_nfsp=0.1,
                    get_trainer_config_nfsp=nfsp_leduc_dqn_params,
                    get_avg_trainer_config_nfsp=nfsp_leduc_avg_policy_params,
                    calculate_openspiel_metanash=False,
                    calculate_openspiel_metanash_at_end=False,
                    calc_metanash_every_n_iters=50,
                    metanash_metrics_smoothing_episodes_override=50000,
                )
            )

            ppo_no_limit = NXDOScenario(
                name=f"{stack_size}_no_limit_leduc_nxdo_ppo_nfsp_dynamic_threshold_1_warm_start_{warm_start_iters}{stochastic_str}",
                ray_cluster_cpus=default_if_creating_ray_head(default=8),
                ray_cluster_gpus=default_if_creating_ray_head(default=0),
                ray_object_store_memory_cap_gigabytes=1,
                use_openspiel_restricted_game=False,
                get_restricted_game_custom_model=lambda env: None,
                xdo_metanash_method="nfsp",
                allow_stochastic_best_responses=allow_stochastic_best_responses,
                get_restricted_game_solver=lambda scenario, warm_start_iters=warm_start_iters: SolveRestrictedGameDynamicRewardThreshold1(
                    scenario=scenario,
                    dont_solve_first_n_nxdo_iters=warm_start_iters,
                    starting_rew_threshold=1.0,
                    min_rew_threshold=0.05,
                    min_episodes=50000,
                    epsilon=0.05,
                    required_fields=[],
                ),
                env_class=PokerMultiAgentEnv,
                env_config={
                    "version": "universal_poker",
                    "fixed_players": True,
                    "append_valid_actions_mask_to_obs": False,
                    "universal_poker_stack_size": stack_size,
                    "continuous_action_space": True,
                },
                trainer_class_br=PPOTrainer,
                policy_classes_br={
                    "metanash": NFSPTorchAveragePolicy,
                    "best_response": PPOTorchPolicy,
                },
                get_trainer_config_br=psro_leduc_ppo_params,
                get_stopping_condition_br=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
                    br_policy_id="best_response",
                    dont_check_plateau_before_n_episodes=int(2e4),
                    check_plateau_every_n_episodes=int(2e4),
                    minimum_reward_improvement_otherwise_plateaued=0.01,
                    max_train_episodes=int(1e5),
                ),
                trainer_class_nfsp=DQNTrainer,
                avg_trainer_class_nfsp=NFSPTrainer,
                policy_classes_nfsp={
                    "average_policy": NFSPTorchAveragePolicy,
                    "delegate_policy": PPOTorchPolicy,
                    "best_response": SimpleQTorchPolicyPatched,
                },
                anticipatory_param_nfsp=0.1,
                get_trainer_config_nfsp=nfsp_leduc_dqn_params_two_layers_no_valid_actions_model,
                get_avg_trainer_config_nfsp=nfsp_leduc_avg_policy_params_two_layers_no_valid_actions_model,
                calculate_openspiel_metanash=False,
                calculate_openspiel_metanash_at_end=False,
                calc_metanash_every_n_iters=50,
                metanash_metrics_smoothing_episodes_override=50000,
            )
            scenario_catalog.add(ppo_no_limit)

            scenario_catalog.add(
                ppo_no_limit.with_updates(
                    name=f"{stack_size}_no_limit_leduc_nxdo_ppo_nfsp_binary_threshold_warm_start_{warm_start_iters}{stochastic_str}",
                    get_restricted_game_solver=lambda scenario, warm_start_iters=warm_start_iters: SolveRestrictedGameDynamicRewardThreshold1(
                        scenario=scenario,
                        dont_solve_first_n_nxdo_iters=warm_start_iters,
                        starting_rew_threshold=0.0,
                        min_rew_threshold=0.0,
                        min_episodes=100000,
                        epsilon=0.05,
                        required_fields=[],
                    ),
                )
            )

            scenario_catalog.add(
                ppo_no_limit.with_updates(
                    name=f"{stack_size}_3_round_no_limit_leduc_nxdo_ppo_nfsp_dynamic_threshold_1_warm_start_{warm_start_iters}{stochastic_str}",
                    env_config={
                        "version": "universal_poker",
                        "fixed_players": True,
                        "append_valid_actions_mask_to_obs": False,
                        "universal_poker_stack_size": stack_size,
                        "continuous_action_space": True,
                        "universal_poker_num_rounds": 3,
                    },
                )
            )
            scenario_catalog.add(
                ppo_no_limit.with_updates(
                    name=f"{stack_size}_4_round_no_limit_leduc_nxdo_ppo_nfsp_dynamic_threshold_1_warm_start_{warm_start_iters}{stochastic_str}",
                    env_config={
                        "version": "universal_poker",
                        "fixed_players": True,
                        "append_valid_actions_mask_to_obs": False,
                        "universal_poker_stack_size": stack_size,
                        "continuous_action_space": True,
                        "universal_poker_num_rounds": 4,
                    },
                )
            )
            scenario_catalog.add(
                ppo_no_limit.with_updates(
                    name=f"{stack_size}_6_rank_no_limit_leduc_nxdo_ppo_nfsp_dynamic_threshold_1_warm_start_{warm_start_iters}{stochastic_str}",
                    env_config={
                        "version": "universal_poker",
                        "fixed_players": True,
                        "append_valid_actions_mask_to_obs": False,
                        "universal_poker_stack_size": stack_size,
                        "continuous_action_space": True,
                        "universal_poker_num_ranks": 6,
                    },
                )
            )
