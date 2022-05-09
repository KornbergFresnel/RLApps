import json
import logging
from concurrent import futures
from typing import Tuple, List, Dict, Callable, Union, Any

import grpc

from ray.rllib.agents import Trainer
from ray.rllib.evaluation import RolloutWorker

from rlapps.utils.strategy_spec import StrategySpec
from rlapps.algos.psro_distill.manager.manager import Distiller, PSRODistillManager
from rlapps.algos.psro_distill.manager.protobuf.manager_pb2_grpc import (
    PSRODistillManagerStub,
    add_PSRODistillManagerServicer_to_server,
)
from rlapps.algos.p2sro.p2sro_manager.protobuf import p2sro_manager_pb2 as psro_proto
from rlapps.algos.psro_distill.manager.protobuf import manager_pb2 as psro_distill_proto
from rlapps.algos.psro_distill.manager.protobuf.manager_pb2_grpc import (
    PSRODistillManagerServicer,
)
from rlapps.algos.p2sro.p2sro_manager.remote import (
    RemoteP2SROManagerClient,
    _P2SROMangerServerServicerImpl,
)
from rlapps.algos.p2sro.p2sro_manager.utils import (
    get_latest_metanash_strategies,
    PolicySpecDistribution,
    SpecDistributionInterface,
)


GRPC_MAX_MESSAGE_LENGTH = 1048576 * 40  # 40MiB

logger = logging.getLogger(__name__)


class _PSRODistillMangerServerServicerImpl(PSRODistillManagerServicer):
    def __init__(self, manager: PSRODistillManager):
        self._manager = manager
        _P2SROMangerServerServicerImpl.__init__(self, manager)

    def GetDistilledMetaNash(
        self, request: psro_distill_proto.PolicySpecJsonList, context
    ):
        prob_list = request.policy_prob_list
        strategy_spec_list = json.load(request.policy_spec_json_list)
        result = self._manager.distill_meta_nash(prob_list, strategy_spec_list)
        return psro_proto.PolicySpecJson(policy_spec_json=result.to_json())


class PSRODistillManagerWithServer(PSRODistillManager):
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
        port: int = 4535,
    ):
        super().__init__(
            n_players=n_players,
            is_two_players_symmetric_zero_sum=is_two_player_symmetric_zero_sum,
            do_external_payoff_evals_for_new_fixed_policies=do_external_payoff_evals_for_new_fixed_policies,
            game_per_external_payoff_eval=games_per_external_payoff_eval,
            eval_distpatcher_port=eval_dispatcher_port,
            distiler=distiller,
            payoff_table_exponential_average_coeff=payoff_table_exponential_average_coeff,
            get_manager_logger=get_manager_logger,
            log_dir=log_dir,
            manager_metadata=manager_metadata,
        )

        self._grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=1),
            options=[
                ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH),
                ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH),
            ],
        )

        servicer = _PSRODistillMangerServerServicerImpl(
            manager=self, stop_server_fn=self.stop_server
        )
        add_PSRODistillManagerServicer_to_server(
            servicer=servicer, server=self._grpc_server
        )
        address = f"[::]:{port}"
        self._grpc_server.add_insecure_port(address)
        self._grpc_server.start()  # does not block
        logger.info(f"PSRODistill Manager gRPC server listening at {address}")

    def wait_for_server_termination(self):
        self._grpc_server.wait_for_termination()

    def stop_server(self):
        self._grpc_server.stop(grace=0)


class RemotePSRODistillManagerClient(RemoteP2SROManagerClient):
    """
    GRPC client for a PSRODistillManager server.
    Behaves exactly like a local PSRODistillManager but actually is connecting to a remote PSRODistill Manager on another
    process or computer.
    """

    def __init__(
        self, n_players: int, port: int = 4535, remote_server_host: str = "127.0.0.1"
    ):
        self._stub = PSRODistillManagerStub(
            channel=grpc.insecure_channel(
                target=f"{remote_server_host}:{port}",
                options=[
                    ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH),
                    ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH),
                ],
            )
        )
        server_is_same_num_players: psro_proto.Confirmation = (
            self._stub.CheckNumPlayers(psro_proto.NumPlayers(num_players=n_players))
        )
        if not server_is_same_num_players.result:
            raise ValueError(
                "Remote P2SROManger server has a different number of players than this one."
            )
        self._n_players = n_players

    def distill_meta_nash(
        self, prob_to_strategy_specs: Dict[float, StrategySpec]
    ) -> StrategySpec:
        probs = list(prob_to_strategy_specs.keys())
        specs = list(prob_to_strategy_specs.values())

        request = psro_distill_proto.PolicySpecJsonList(
            policy_prob_list=probs, policy_spec_json_list=json.dump(specs)
        )

        json_str = self._stub.GetDistilledMetaNash(request)
        distilled_strategy_spec = StrategySpec.from_json(json_str)
        return distilled_strategy_spec


def update_all_workers_to_latest_metanash(
    trainer: Trainer,
    br_player: int,
    metanash_player: int,
    p2sro_manager: RemotePSRODistillManagerClient,
    active_policy_num: int,
    mix_metanash_with_uniform_dist_coeff: float,
    one_agent_plays_all_sides: bool = False,
):
    """Compute meta strategies, and send them to trigger policy distillation.

    Args:
        trainer (Trainer): _description_
        br_player (int): _description_
        metanash_player (int): _description_
        p2sro_manager (RemotePSRODistillManagerClient): _description_
        active_policy_num (int): _description_
        mix_metanash_with_uniform_dist_coeff (float): _description_
        one_agent_plays_all_sides (bool, optional): _description_. Defaults to False.
    """

    # compute latest meta nash
    (
        latest_payoff_table,
        active_policy_nums,
        fixed_policy_nums,
    ) = p2sro_manager.get_copy_of_latest_data()
    latest_strategies: Dict[
        int, PolicySpecDistribution
    ] = get_latest_metanash_strategies(
        payoff_table=latest_payoff_table,
        as_player=1 if one_agent_plays_all_sides else br_player,
        as_policy_num=active_policy_num,
        fictitious_play_iters=2000,
        mix_with_uniform_dist_coeff=mix_metanash_with_uniform_dist_coeff,
    )

    if latest_strategies is None:
        opponent_policy_distribution = None
    else:
        opponent_player = 0 if one_agent_plays_all_sides else metanash_player
        print(
            f"latest payoff matrix for player {opponent_player}:\n"
            f"{latest_payoff_table.get_payoff_matrix_for_player(player=opponent_player)}"
        )
        print(
            f"metanash for player {opponent_player}: "
            f"{latest_strategies[opponent_player].probabilities_for_each_strategy()}"
        )

        # get the strategy distribution for the opposing player.
        opponent_policy_distribution = latest_strategies[opponent_player]

        # double check that these policy specs are for the opponent player
        assert (
            opponent_player
            in opponent_policy_distribution.sample_policy_spec()
            .get_pure_strat_indexes()
            .keys()
        )

    # TODO(ming): send training request to manager
    prob_to_strategy_specs: Dict[float, StrategySpec] = latest_payoff_table[
        opponent_player
    ].probs_to_specs

    distilled_strategy_spec = p2sro_manager.distill_meta_nash(prob_to_strategy_specs)

    def _set_opponent_policy_for_worker(worker: RolloutWorker):
        worker.opponent_policy_distribution = SpecDistributionInterface(
            {1.0: distilled_strategy_spec}
        )

    # update all workers with newest meta policy
    trainer.workers.foreach_worker(_set_opponent_policy_for_worker)
