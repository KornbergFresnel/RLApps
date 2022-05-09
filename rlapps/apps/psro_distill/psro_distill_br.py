import argparse
import logging
import os

from ray.rllib.utils import try_import_torch

from rlapps.utils.port_listings import get_client_port_for_service
from rlapps.algos.psro_distill.manager import RemotePSRODistillManagerClient
from rlapps.algos.psro_distill.manager.remote import (
    update_all_workers_to_latest_metanash,
)
from rlapps.apps import GRL_SEED
from rlapps.apps.psro.general_psro_br import train_psro_best_response

torch, _ = try_import_torch()

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--player", type=int)
    parser.add_argument("--scenario", type=str)
    parser.add_argument("--use_prev_brs", default=False, action="store_true")
    parser.add_argument("--psro_port", type=int, required=False, default=None)
    parser.add_argument("--psro_host", type=str, required=False, default="localhost")
    commandline_args = parser.parse_args()

    scenario_name = commandline_args.scenario
    use_prev_brs = commandline_args.use_prev_brs

    psro_host = commandline_args.psro_host
    psro_port = commandline_args.psro_port
    if psro_port is None:
        psro_port = get_client_port_for_service(
            service_name=f"seed_{GRL_SEED}_{scenario_name}"
        )

    manager_log_dir = RemotePSRODistillManagerClient(
        n_players=2,
        port=os.getenv("P2SRO_PORT", psro_port),
        remote_server_host=psro_host,
    ).get_log_dir()
    results_dir = os.path.join(
        manager_log_dir, f"learners_player_{commandline_args.player}/"
    )
    print(f"results dir is {results_dir}")

    previous_br_checkpoint_path = None
    while True:
        # Train a br for the specified player, then repeat.
        result = train_psro_best_response(
            player=commandline_args.player,
            results_dir=results_dir,
            scenario_name=scenario_name,
            psro_manager_port=psro_port,
            psro_manager_host=psro_host,
            print_train_results=True,
            previous_br_checkpoint_path=previous_br_checkpoint_path,
            remote_manager_client=RemotePSRODistillManagerClient,
            metanash_update_procedure=update_all_workers_to_latest_metanash,
        )
        if use_prev_brs:
            previous_br_checkpoint_path = result
