import json
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from threading import RLock
from termcolor import colored
from typing import List, Tuple, Union, Dict

from rlapps.utils.common import datetime_str, ensure_dir, check_if_jsonable
from rlapps.utils.strategy_spec import StrategySpec

from rlapps.algos.p2sro.p2sro_manager import P2SROManager
from rlapps.algos.p2sro.p2sro_manager.utils import (
    get_latest_metanash_strategies,
    PolicySpecDistribution,
)


class PSRODistillManager(P2SROManager):
    def __init__(
        self,
        n_players,
        is_two_player_symmetric_zero_sum: bool,
        do_external_payoff_evals_for_new_fixed_policies: bool,
        games_per_external_payoff_eval: int,
        eval_dispatcher_port: int = 4536,
        payoff_table_exponential_average_coeff: float = None,
        get_manager_logger=None,
        log_dir: str = None,
        manager_metadata: dict = None,
    ):
        super().__init__(
            n_players,
            is_two_player_symmetric_zero_sum,
            do_external_payoff_evals_for_new_fixed_policies,
            games_per_external_payoff_eval,
            eval_dispatcher_port,
            payoff_table_exponential_average_coeff,
            get_manager_logger,
            log_dir,
            manager_metadata,
        )

        self._episodes_total = 0
        self._timesteps_total = 0
        self._restricted_game_episodes_this_iter = 0
        self._restricted_game_timesteps_this_iter = 0

    def submit_new_active_policy_metadata(
        self, player, policy_num, metadata_dict
    ) -> StrategySpec:
        return super().submit_new_active_policy_metadata(
            player, policy_num, metadata_dict
        )

    def distill_meta_nash(
        self,
        br_player: int,
        active_policy_num: int,
        br_spec_lists_for_each_player,
        one_agent_plays_all_sides: bool = False,
        mix_metanash_with_uniform_dist_coeff: float = 0.0,
    ):
        latest_strategies: Dict[
            int, PolicySpecDistribution
        ] = get_latest_metanash_strategies(
            self._payoff_table,
            as_player=1 if one_agent_plays_all_sides else br_player,
            as_policy_num=active_policy_num,
            fictitious_play_iters=2000,
            mix_with_uniform_dist_coeff=mix_metanash_with_uniform_dist_coeff,
            print_matrix=True,
        )

        if latest_strategies is None:
            raise NotImplementedError
        else:
            print(
                "Distilling strategy spec with meta nash: {}".format(latest_strategies)
            )
            distillation_results = self._distiller(
                log_dir=self.log_dir,
                br_spec_lists_for_each_player=br_spec_lists_for_each_player,
                manager_meta_data=self.get_manager_metadata(),
            )

            self._restricted_game_episodes_this_iter += (
                distillation_results.episodes_spent_in_solve
            )
            self._restricted_game_timesteps_this_iter += (
                distillation_results.timesteps_spent_in_solve
            )

            self._episodes_total += (
                self._br_episodes_this_iter + self._restricted_game_episodes_this_iter
            )
            self._timesteps_total += (
                self._br_timesteps_this_iter + self._restricted_game_timesteps_this_iter
            )
