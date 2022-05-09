import json
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from threading import RLock
from termcolor import colored
from typing import List, Tuple, Union, Dict, Callable

from rlapps.utils.common import datetime_str, ensure_dir, check_if_jsonable
from rlapps.utils.strategy_spec import StrategySpec

from rlapps.algos.p2sro.p2sro_manager import P2SROManager
from rlapps.algos.p2sro.eval_dispatcher import EvalDispatcherWithServer, EvalResult
from rlapps.algos.p2sro.p2sro_manager.utils import (
    get_latest_metanash_strategies,
    PolicySpecDistribution,
)


class DistillerResult:
    def __init__(
        self,
        distilled_strategy_spec: StrategySpec,
        episodes_spent_in_solve: int,
        timesteps_spent_in_solve: int,
        extra_data_to_log: dict,
    ):
        self.distilled_strategy_spec = distilled_strategy_spec
        self.episodes_spent_in_solve = episodes_spent_in_solve
        self.timesteps_spent_in_solve = timesteps_spent_in_solve
        self.extra_data_to_log = extra_data_to_log


class Distiller(ABC):
    @abstractmethod
    def __call__(
        self,
        log_dir: str,
        br_prob_list: List[float],
        br_spec_list: List[StrategySpec],
        manager_metadata: dict = None,
    ) -> DistillerResult:
        """Execution meta strategy distillation.

        Args:
            log_dir (str): _description_
            br_prob_list (List[float]): _description_
            br_spec_list (List[StrategySpec]): _description_
            manager_metadata (dict, optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_

        Returns:
            DistillerResult: _description_
        """


class PSRODistillManager(P2SROManager):
    def __init__(
        self,
        n_players: int,
        is_two_player_symmetric_zero_sum: bool,
        do_external_payoff_evals_for_new_fixed_policies: bool,
        games_per_external_payoff_eval: int,
        distiller: Distiller,
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
        self._distiller = distiller

    def distill_meta_nash(
        self, probs_list: List[float], strategy_spec_list: List[StrategySpec]
    ) -> StrategySpec:
        """Distill a meta strategy to a single strategy spec.

        Args:
            probs_list (List[float]): A list of probs.
            strategy_spec_list (List[StrategySpec]): A list of strategy specs.

        Raises:
            NotImplementedError: _description_
        """

        print("Distilling meta strategy ...")

        distillation_results = self._distiller(
            log_dir=self.log_dir,
            br_prob_list=probs_list,
            br_spec_list=strategy_spec_list,
            manager_metadata=self.get_manager_metadata(),
        )

        # TODO(ming): print results and return strategy spec
        return distillation_results.distilled_strategy_spec