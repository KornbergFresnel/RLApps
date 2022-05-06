import json
import logging
import traceback
from concurrent import futures
from typing import Tuple, List, Dict, Callable, Union

import grpc
from google.protobuf.empty_pb2 import Empty

from rlapps.utils.strategy_spec import StrategySpec
from rlapps.algos.psro_distill.manager.manager import PSRODistillManager
from rlapps.algos.psro_distill.manager.protobuf.manager_pb2 import *
from rlapps.algos.psro_distill.manager.protobuf.manager_pb2_grpc import (
    PSRODistillManagerServicer,
    add_PSRODistillManagerServicer_to_server,
    PSRODistillManagerStub,
)

GRPC_MAX_MESSAGE_LENGTH = 1048576 * 40  # 40MiB

logger = logging.getLogger(__name__)


class _PSRODistillMangerServerServicerImpl(PSRODistillManagerServicer):
    def __init__(self, manager: PSRODistillManager, stop_server_fn: Callable):
        self._manager = manager
        self._stop_server_fn = stop_server_fn

    def GetLogDir(self, request, context):
        return PSRODistillString(string=self._manager.get_log_dir())

    def GetManagerMetaData(self, request, context):
        metadata = self._manager.get_manager_metadata()
        return PSRODistillMetadata(json_metadata=json.dumps(metadata))

    def ClaimNewActivePolicyForPlayer(self, request: PSRODistillPlayer, context):
        out = self._manager.claim_new_active_policy_for_player(player=request.player)

        metanash_specs_for_players, delegate_specs_for_players, policy_num = out

        assert len(metanash_specs_for_players) == self._manager.n_players()
        assert len(delegate_specs_for_players) == self._manager.n_players()

        if policy_num is None:
            return PSRODistillNewBestResponseParams(policy_num=-1)

        response = PSRODistillNewBestResponseParams(policy_num=policy_num)

        for player, spec_for_player in metanash_specs_for_players.items():
            if spec_for_player is not None:
                response.metanash_specs_for_players.policy_spec_list.append(
                    PSRODistillPolicySpecJson(
                        policy_spec_json=spec_for_player.to_json()
                    )
                )

        response_delegate_spec_lists_for_other_players = []
        for player, player_delegate_spec_list in delegate_specs_for_players.items():
            player_delegate_json_spec_list = PSRODistillPolicySpecList()
            player_delegate_json_spec_list.policy_spec_list.extend(
                [
                    PSRODistillPolicySpecJson(policy_spec_json=spec.to_json())
                    for spec in player_delegate_spec_list
                ]
            )
            response_delegate_spec_lists_for_other_players.append(
                player_delegate_json_spec_list
            )
        response.delegate_specs_for_players.extend(
            response_delegate_spec_lists_for_other_players
        )

        return response

    def SubmitFinalBRPolicy(self, request: PSRODistillPolicyMetadataRequest, context):
        with self._manager.modification_lock:
            try:
                self._manager.submit_final_br_policy(
                    player=request.player,
                    policy_num=request.policy_num,
                    metadata_dict=json.loads(request.metadata_json),
                )
            except Exception as err:
                print(f"{type(err)}: {err}")
                traceback.print_exc()
                print("Submitting BR failed, shutting down manager.")
                self._stop_server_fn()

        return PSRODistillConfirmation(result=True)

    def IsPolicyFixed(self, request: PSRODistillPlayerAndPolicyNum, context):
        is_policy_fixed = self._manager.is_policy_fixed(
            player=request.player, policy_num=request.policy_num
        )
        return PSRODistillConfirmation(result=is_policy_fixed)


class PSRODistillManagerWithServer(PSRODistillManager):
    def __init__(
        self,
        solve_restricted_game: SolveRestrictedGame,
        n_players: int = 2,
        log_dir: str = None,
        manager_metadata: dict = None,
        port: int = 4545,
    ):
        super(PSRODistillManagerWithServer, self).__init__(
            solve_restricted_game=solve_restricted_game,
            n_players=n_players,
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


class RemotePSRODistillManagerClient(PSRODistillManager):

    # noinspection PyMissingConstructor
    def __init__(self, n_players, port=4545, remote_server_host="127.0.0.1"):
        self._stub = PSRODistillManagerStub(
            channel=grpc.insecure_channel(
                target=f"{remote_server_host}:{port}",
                options=[
                    ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH),
                    ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH),
                ],
            )
        )
        self._n_players = n_players

    def n_players(self) -> int:
        return self._n_players

    def get_log_dir(self) -> str:
        return self._stub.GetLogDir(Empty()).string

    def get_manager_metadata(self) -> dict:
        response: PSRODistillMetadata = self._stub.GetManagerMetaData(Empty())
        return json.loads(response.json_metadata)

    def claim_new_active_policy_for_player(
        self, player
    ) -> Union[
        Tuple[Dict[int, StrategySpec], Dict[int, List[StrategySpec]], int],
        Tuple[None, None, None],
    ]:
        request = PSRODistillPlayer(player=player)
        response: PSRODistillNewBestResponseParams = (
            self._stub.ClaimNewActivePolicyForPlayer(request)
        )

        if response.policy_num == -1:
            return None, None, None

        assert len(response.metanash_specs_for_players.policy_spec_list) in [
            self.n_players(),
            0,
        ]
        assert len(response.delegate_specs_for_players) in [self.n_players(), 0]

        metanash_json_specs_for_other_players = [
            elem.policy_spec_json
            for elem in response.metanash_specs_for_players.policy_spec_list
        ]

        metanash_specs_for_players = {
            player: StrategySpec.from_json(json_spec)
            for player, json_spec in enumerate(metanash_json_specs_for_other_players)
        }

        delegate_json_spec_lists_for_other_players = [
            [elem.policy_spec_json for elem in player_delegate_list.policy_spec_list]
            for player_delegate_list in response.delegate_specs_for_players
        ]
        delegate_specs_for_players = {
            player: [
                StrategySpec.from_json(json_spec)
                for json_spec in player_delegate_json_list
            ]
            for player, player_delegate_json_list in enumerate(
                delegate_json_spec_lists_for_other_players
            )
        }

        if len(metanash_specs_for_players) == 0:
            metanash_specs_for_players = None

        if len(delegate_specs_for_players) == 0:
            delegate_specs_for_players = None

        return (
            metanash_specs_for_players,
            delegate_specs_for_players,
            response.policy_num,
        )

    def submit_final_br_policy(self, player, policy_num, metadata_dict):
        try:
            metadata_json = json.dumps(obj=metadata_dict)
        except (TypeError, OverflowError) as json_err:
            raise ValueError(
                f"metadata_dict must be JSON serializable."
                f"When attempting to serialize, got this error:\n{json_err}"
            )

        request = PSRODistillPolicyMetadataRequest(
            player=player, policy_num=policy_num, metadata_json=metadata_json
        )
        self._stub.SubmitFinalBRPolicy(request)

    def is_policy_fixed(self, player, policy_num):
        response: PSRODistillConfirmation = self._stub.IsPolicyFixed(
            PSRODistillPlayerAndPolicyNum(player=player, policy_num=policy_num)
        )
        return response.result
