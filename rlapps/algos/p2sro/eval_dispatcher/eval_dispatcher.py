from queue import Queue, Empty
from threading import RLock
from typing import List, Tuple

from rlapps.utils.strategy_spec import StrategySpec


class _EvalQueue(object):
    def __init__(self, drop_duplicates, game_is_two_player_symmetric):
        self._drop_duplicates = drop_duplicates
        self._game_is_two_player_symmetric = game_is_two_player_symmetric
        self._queue = Queue()
        self._unique_set = set()
        self._lock = RLock()

    def get(self):
        with self._lock:
            try:
                policy_spec_tuple: Tuple[StrategySpec] = self._queue.get_nowait()
            except Empty:
                return None
            if self._drop_duplicates:
                # Allow this matchup to be added to the top of the queue now. Remove it from the unique values we have.
                self._unique_set.remove(policy_spec_tuple)
                if (
                    self._game_is_two_player_symmetric
                    and policy_spec_tuple[::-1] in self._unique_set
                ):
                    # Remove any matchup with the policies for players flipped since that's equivalent.
                    self._unique_set.remove(policy_spec_tuple[::-1])
            return policy_spec_tuple

    def put(self, policy_spec_tuple: Tuple[StrategySpec]):
        with self._lock:
            if self._drop_duplicates:
                if policy_spec_tuple in self._unique_set:
                    # Do nothing if we already have the same matchup between policy specs.
                    return
                elif (
                    self._game_is_two_player_symmetric
                    and policy_spec_tuple[::-1] in self._unique_set
                ):
                    # Do nothing if we have a matchup with the policies for players flipped since that's equivalent.
                    return
                self._unique_set.add(policy_spec_tuple)
            self._queue.put(policy_spec_tuple)

    def __len__(self):
        with self._lock:
            return self._queue.qsize()


class EvalResult(object):
    def __init__(
        self,
        policy_specs_for_each_player,
        payoff_for_each_player,
        games_played,
        buffer_file_path,
    ):
        self.policy_specs_for_each_player = policy_specs_for_each_player
        self.payoff_for_each_player = payoff_for_each_player
        self.games_played = games_played
        # buffer file path is actually a dict
        self.buffer_file_path = buffer_file_path


class EvalDispatcher(object):
    def __init__(
        self,
        games_per_eval: int,
        game_is_two_player_symmetric: bool,
        drop_duplicate_requests: bool,
    ):
        self._games_per_eval = games_per_eval
        self._eval_queue = _EvalQueue(
            drop_duplicates=drop_duplicate_requests,
            game_is_two_player_symmetric=game_is_two_player_symmetric,
        )
        self._on_eval_result_callbacks = []

    def submit_eval_request(self, policy_specs_for_each_player: Tuple[StrategySpec]):
        self._eval_queue.put(policy_spec_tuple=policy_specs_for_each_player)

    def take_eval_job(self):
        policy_specs_for_each_player_tuple = self._eval_queue.get()
        return policy_specs_for_each_player_tuple, self._games_per_eval

    def submit_eval_job_result(
        self,
        policy_specs_for_each_player_tuple,
        payoffs_for_each_player: List[float],
        games_played,
        buffer_file_path: str = "",
    ):
        eval_result = EvalResult(
            policy_specs_for_each_player=policy_specs_for_each_player_tuple,
            payoff_for_each_player=payoffs_for_each_player,
            games_played=games_played,
            buffer_file_path=buffer_file_path,
        )
        for callback in self._on_eval_result_callbacks:
            callback(eval_result)

    def add_on_eval_result_callback(self, on_eval_result):
        self._on_eval_result_callbacks.append(on_eval_result)
