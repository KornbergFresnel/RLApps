import json
import os
from abc import ABC, abstractmethod
from typing import List

from rlapps.utils.strategy_spec import StrategySpec
from rlapps.algos.p2sro.p2sro_manager import P2SROManager
from rlapps.algos.p2sro.eval_dispatcher import EvalResult


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
        metanash_player: int,
        prob_list_each_player: List[List[float]],
        spec_list_each_player: List[List[StrategySpec]],
        manager_metadata: dict = None,
    ) -> DistillerResult:
        """Execution meta strategy distillation.

        Args:
            log_dir (str): _description_
            prob_list (List[float]): _description_
            spec_list (List[StrategySpec]): _description_
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
        self.manager_metadata["offline_dataset"] = {0: {}, 1: {}}

    def _on_finished_eval_result(self, eval_result: EvalResult):
        # updtate offline dataset path recording
        with self._modification_lock:
            eval_result_should_override_previous_results = False

            # print(f"current pending spec matchups: {self._pending_spec_matchups_for_new_fixed_policies}")

            # Check if we're waiting on this eval matchup to get the final payoff results for an active policy that
            # we're waiting to move to fixed.
            keys_to_remove = []
            for (
                key,
                matchups_for_fixed_policy,
            ) in self._pending_spec_matchups_for_new_fixed_policies.items():
                player, new_fixed_policy_num = key
                if (
                    eval_result.policy_specs_for_each_player
                    in matchups_for_fixed_policy
                ):
                    # We were waiting on this matchup, so remove it from the set of matchups we're waiting on.
                    matchups_for_fixed_policy.remove(
                        eval_result.policy_specs_for_each_player
                    )

                    # This is a final payoff eval for a fixed policy against fixed policy opponent(s).
                    # Override any other data currently on the payoff table for the same matchup.
                    eval_result_should_override_previous_results = True

                    if len(matchups_for_fixed_policy) == 0:
                        # We're no longer waiting on any eval results to move this policy to fixed.
                        keys_to_remove.append(key)

                        # Move it to fixed.
                        self._player_stats[player].move_active_policy_to_fixed(
                            policy_num=new_fixed_policy_num
                        )
                        fixed_policy_spec = (
                            self._payoff_table.get_spec_for_player_and_pure_strat_index(
                                player=player, pure_strat_index=new_fixed_policy_num
                            )
                        )
                        self._manager_logger.on_active_policy_moved_to_fixed(
                            player=player,
                            policy_num=new_fixed_policy_num,
                            fixed_policy_spec=fixed_policy_spec,
                        )
            for key in keys_to_remove:
                del self._pending_spec_matchups_for_new_fixed_policies[key]
            # print(f"new pending spec matchups: {self._pending_spec_matchups_for_new_fixed_policies}")

            # path like: /${user_temp}/${scenario_name}/${player_0_strategy_id}_vs_${player_1_strategy_id}/${timestamp}
            assert (
                len(eval_result.buffer_file_path) > 0
            ), "accepted buffer file path is: {}".format(eval_result.buffer_file_path)
            # update manager meta data
            key = "&".join(
                [spec.id for spec in eval_result.policy_specs_for_each_player]
            )
            for player in [0, 1]:
                dirname = eval_result.buffer_file_path[str(player)]
                if key not in self.manager_metadata["offline_dataset"][player]:
                    self.manager_metadata["offline_dataset"][player][key] = dirname

            # Add this payoff result to our payoff table.
            self.submit_empirical_payoff_result(
                policy_specs_for_each_player=eval_result.policy_specs_for_each_player,
                payoffs_for_each_player=eval_result.payoff_for_each_player,
                games_played=eval_result.games_played,
                override_all_previous_results=eval_result_should_override_previous_results,
            )
