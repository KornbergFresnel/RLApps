from collections import defaultdict
from typing import Any, Tuple, Dict

import argparse
import logging
import time
import os

import numpy as np
import ray

from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter

from rlapps.algos.p2sro.eval_dispatcher.remote import RemoteEvalDispatcherClient
from rlapps.apps import GRL_SEED
from rlapps.apps.scenarios.catalog import scenario_catalog
from rlapps.apps.scenarios.psro_scenario import PSROScenario
from rlapps.apps.scenarios.ray_setup import init_ray_for_scenario
from rlapps.rllib_tools.policy_checkpoints import load_pure_strat
from rlapps.utils.port_listings import get_client_port_for_service


def store_frames_to_builders(
    prep,
    agent_step,
    eps_id,
    agent_batch_dict: Dict[int, Dict[str, Any]],
    obs,
    reward,
    done,
    action,
    next_obs,
    infos,
):
    agents = list(agent_batch_dict.keys())
    # CUR_OBS, ACTION, REWERAD, DONE, INFOS
    for k, _obs in obs.items():
        if k not in agents:
            continue
        agent_batch_dict[k][SampleBatch.CUR_OBS].append(prep.transform(_obs))
    for k, _r in reward.items():
        if k not in agents:
            continue
        agent_batch_dict[k][SampleBatch.REWARDS].append(_r)
    if done["__all__"]:
        for agent in agents:
            agent_batch_dict[agent][SampleBatch.DONES].append(True)
            if len(agent_batch_dict[agent][SampleBatch.DONES]) > len(
                agent_batch_dict[agent][SampleBatch.CUR_OBS]
            ):
                # shift
                for k in [SampleBatch.REWARDS, SampleBatch.DONES]:
                    agent_batch_dict[agent][k] = agent_batch_dict[agent][k][1:]
    else:
        for k, _done in done.items():
            if k not in agents:
                continue
            agent_batch_dict[k][SampleBatch.DONES].append(_done)
    for k, _action in action.items():
        if k not in agents:
            continue
        agent_batch_dict[k][SampleBatch.ACTIONS].append(_action)
    for k, _info in infos.items():
        if k not in agents:
            continue
        agent_batch_dict[k][SampleBatch.INFOS].append(_info)


def run_episode(
    env,
    policies_for_each_player,
    store_as_offline: bool = False,
) -> Tuple[np.ndarray, Dict[int, SampleBatch]]:
    num_players = len(policies_for_each_player)

    if store_as_offline:
        agent_batch_dict = {i: defaultdict(lambda: []) for i in range(num_players)}
        prep = get_preprocessor(env.observation_space)(env.observation_space)
    else:
        agent_batch_dict = None
        prep = None

    obs = env.reset()
    dones = {}
    game_length = 0
    policy_states = [policy.get_initial_state() for policy in policies_for_each_player]
    # agent_to_pid = dict(zip(range(num_players), policy_id_for_each_player))

    payoffs_per_player_this_episode = np.zeros(shape=num_players, dtype=np.float64)
    agent_step = defaultdict(lambda: 0)
    episode_id = int(time.time())

    while True:
        if "__all__" in dones:
            if dones["__all__"]:
                break

        action_dict = {}
        action_info_dict = {}

        for player in range(num_players):
            if player in obs:
                action_index, new_policy_state, action_info = policies_for_each_player[
                    player
                ].compute_single_action(obs=obs[player], state=policy_states[player])
                policy_states[player] = new_policy_state
                action_dict[player] = action_index
                action_info_dict[player] = action_info
                agent_step[player] += 1

        new_obs, rewards, dones, infos = env.step(action_dict=action_dict)

        for player in range(num_players):
            payoff_so_far = payoffs_per_player_this_episode[player]
            payoffs_per_player_this_episode[player] = payoff_so_far + rewards.get(
                player, 0.0
            )

        if store_as_offline:
            # parse frame to save for each player
            store_frames_to_builders(
                prep,
                agent_step,
                episode_id,
                agent_batch_dict,
                obs,
                rewards,
                dones,
                action_dict,
                new_obs,
                infos,
            )

        obs = new_obs
        game_length += 1

    if store_as_offline:
        # TODO(ming): note since rllib does not support multiagent json reader yet, I return a dict of batches.
        #   see: https://github.com/ray-project/ray/issues/24283
        samples = {k: SampleBatch(**v) for k, v in agent_batch_dict.items()}
    else:
        samples = None

    return payoffs_per_player_this_episode, samples


@ray.remote(num_cpus=0, num_gpus=0)
def run_poker_evaluation_loop(
    scenario_name: str,
    eval_dispatcher_port: int,
    eval_dispatcher_host: str,
    store_as_offline: bool = False,
):
    scenario: PSROScenario = scenario_catalog.get(scenario_name=scenario_name)
    if not isinstance(scenario, PSROScenario):
        raise TypeError(
            f"Only instances of {PSROScenario} can be used here. {scenario.name} is a {type(scenario)}."
        )

    eval_dispatcher = RemoteEvalDispatcherClient(
        port=eval_dispatcher_port, remote_server_host=eval_dispatcher_host
    )

    env = scenario.env_class(env_config=scenario.env_config)
    num_players = 2

    trainer_config = scenario.get_trainer_config(env)
    trainer_config["explore"] = scenario.allow_stochastic_best_responses

    policies = [
        scenario.policy_classes["eval"](
            env.observation_space, env.action_space, with_common_config(trainer_config)
        )
        for _ in range(num_players)
    ]

    buffer_list = []

    while True:
        (
            policy_specs_for_each_player,
            required_games_to_play,
        ) = eval_dispatcher.take_eval_job()

        if policy_specs_for_each_player is None:
            time.sleep(2)
        else:
            if len(policy_specs_for_each_player) != 2:
                raise NotImplementedError(
                    f"This evaluation code only supports two player games. "
                    f"{len(policy_specs_for_each_player)} players were requested."
                )

            # print(f"Got eval matchup:")
            # for spec in policy_specs_for_each_player:
            #     print(f"spec: {spec.to_json()}")
            for policy, spec in zip(policies, policy_specs_for_each_player):
                load_pure_strat(policy=policy, pure_strat_spec=spec)

            total_payoffs_per_player = np.zeros(shape=num_players, dtype=np.float64)

            # max_reward = None
            # min_reward = None
            # time_since_last_output = time.time()
            for game in range(required_games_to_play):
                # if game % 1000 == 0:
                #     now = time.time()
                #     print(f"{policy_specs_for_each_player[0].id} vs "
                #           f"{policy_specs_for_each_player[1].id}: "
                #           f"{game}/{required_games_to_play} games played, {now - time_since_last_output} seconds")
                #     time_since_last_output = now

                payoffs_per_player_this_episode, samples = run_episode(
                    env=env,
                    policies_for_each_player=policies,
                    store_as_offline=store_as_offline,
                )

                if samples is not None:
                    buffer_list.append(samples)

                total_payoffs_per_player += payoffs_per_player_this_episode

                # if max_reward is None or max(payoffs_per_player_this_episode) > max_reward:
                #     max_reward = max(payoffs_per_player_this_episode)
                # if min_reward is None or min(payoffs_per_player_this_episode) < min_reward:
                #     min_reward = min(payoffs_per_player_this_episode)

            payoffs_per_player = total_payoffs_per_player / required_games_to_play

            print(
                f"payoffs per player:"
                f"{policy_specs_for_each_player[0].id} vs "
                f"{policy_specs_for_each_player[1].id}: "
                f"{payoffs_per_player}"
            )

            # save buffers
            if len(buffer_list) > 0:
                dir_name = os.path.join(
                    ray._private.utils.get_user_temp_dir(),
                    scenario_name,
                    f"{policy_specs_for_each_player[0].id}_vs_{policy_specs_for_each_player[1].id}",
                )

                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                buffer_file_path = os.path.join(dir_name, str(time.time()))
                writer = JsonWriter(buffer_file_path)
                for e in buffer_list:
                    writer.write(e)
            else:
                buffer_file_path = ""

            eval_dispatcher.submit_eval_job_result(
                policy_specs_for_each_player_tuple=policy_specs_for_each_player,
                payoffs_for_each_player=payoffs_per_player,
                games_played=required_games_to_play,
                buffer_file_path=buffer_file_path,
            )


def launch_evals(
    scenario_name: str,
    eval_dispatcher_port: int,
    eval_dispatcher_host: str,
    block=True,
    ray_head_address=None,
):

    scenario: PSROScenario = scenario_catalog.get(scenario_name=scenario_name)

    init_ray_for_scenario(
        scenario=scenario, head_address=ray_head_address, logging_level=logging.INFO
    )

    num_workers = scenario.num_eval_workers
    evaluator_refs = [
        run_poker_evaluation_loop.remote(
            scenario_name, eval_dispatcher_port, eval_dispatcher_host
        )
        for _ in range(num_workers)
    ]
    if block:
        ray.wait(evaluator_refs, num_returns=num_workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str)
    parser.add_argument("--ray_head", type=str, required=False, default=None)
    parser.add_argument("--eval_port", type=int, required=False, default=None)
    parser.add_argument("--eval_host", type=str, required=False, default="localhost")
    commandline_args = parser.parse_args()

    scenario_name = commandline_args.scenario
    eval_port = commandline_args.eval_port
    if eval_port is None:
        eval_port = get_client_port_for_service(
            service_name=f"seed_{GRL_SEED}_{scenario_name}_evals"
        )

    launch_evals(
        scenario_name=scenario_name,
        eval_dispatcher_port=eval_port,
        eval_dispatcher_host=commandline_args.eval_host,
        block=True,
        ray_head_address=commandline_args.ray_head,
    )
