from typing import Any, Tuple, Dict

import gym

from dataclasses import dataclass

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import PolicyID, AgentID
from ray.rllib.policy import Policy
from ray.rllib.agents.dqn import DQNTorchPolicy

from rlapps.envs.poker_multi_agent_env import PokerMultiAgentEnv
from rlapps.rllib_tools.policy_checkpoints import create_get_pure_strat_cached
from rlapps.algos.p2sro.p2sro_manager.utils import SpecDistributionInterface


@dataclass
class Frame:
    observation: Any
    prev_action: Any
    prev_reward: float


class SingleAgentEnv(gym.Env):
    def __init__(self, env_config) -> None:
        self.ma_env: MultiAgentEnv = env_config["env_class"](
            env_config=env_config["env_config"]
        )

        assert isinstance(self.ma_env, MultiAgentEnv)
        assert "multiagent" in env_config

        policy_spec = env_config["multiagent"]["policies"]
        policy_mapping_func = env_config["multiagent"]["policy_mapping_fn"]

        agent_ids = self.ma_env.get_agent_ids()
        assert env_config["br_player"] in agent_ids, (
            env_config["br_player"],
            agent_ids,
        )

        agent_to_policy_id = {
            aid: policy_mapping_func(aid)
            for aid in agent_ids
            if aid != env_config["br_player"]
        }
        active_policy_ids = set(agent_to_policy_id.values())
        load_from_spec = create_get_pure_strat_cached({})

        fixed_policies: Dict[PolicyID, Policy] = {}
        spec_distribution: Dict[PolicyID, SpecDistributionInterface] = {}
        for policy_id in active_policy_ids:
            policy_cls, obs_space, act_space, custom_config = policy_spec[policy_id]

            policy = policy_cls(obs_space, act_space, custom_config)
            fixed_policies[policy_id] = policy

            if "strategy_spec_dict" in env_config["multiagent"]:
                prob_list, spec_list = env_config["multiagent"]["strategy_spec_dict"][
                    policy_id
                ]
                spec_distribution[policy_id] = SpecDistributionInterface(
                    dict(zip(prob_list, spec_list))
                )

        self.br_player = env_config["br_player"]
        self.agent_ids = agent_ids
        self.load_from_spec = load_from_spec
        self.fixed_policies: Dict[PolicyID, Policy] = fixed_policies
        self.agent_to_policy_id = agent_to_policy_id
        self.observation_space = self.ma_env.observation_space
        self.action_space = self.ma_env.action_space
        self.last_frame = {
            agent: Frame(
                self.observation_space.sample(), self.action_space.sample(), 0.0
            )
            for agent in agent_ids
        }
        self.step_cnt = 0.0
        self.spec_distribution = spec_distribution

    def _step_until_br_met(self, new_obs, actions, rewards, dones, infos):
        for k in new_obs.keys():
            self.last_frame[k].observation = new_obs[k]
            self.last_frame[k].prev_action = actions.get(k, self.action_space.sample())
            self.last_frame[k].prev_reward = rewards[k]

        if self.br_player not in new_obs:
            # step untile br_player met
            actions = {}
            for agent, obs in new_obs.items():
                frame = self.last_frame[agent]
                policy = self.fixed_policies[self.agent_to_policy_id[agent]]
                actions[agent], _, _ = policy.compute_single_action(
                    frame.observation,
                    prev_action=frame.prev_action,
                    prev_reward=frame.prev_reward,
                )

            new_obs, rewards, dones, infos = self.ma_env.step(actions)
            self.step_cnt += 1
            return self._step_until_br_met(new_obs, actions, rewards, dones, infos)
        else:
            obs = new_obs[self.br_player]
            reward = rewards[self.br_player]
            done = dones.get(self.br_player, dones["__all__"])
            info = infos.get(self.br_player, {})
            return obs, reward, done, info

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        actions = {k: frame.prev_action for k, frame in self.last_frame.items()}
        actions = {self.br_player: action}
        new_obs, rewards, dones, infos = self.ma_env.step(actions)
        self.step_cnt += 1
        new_obs, rewards, dones, infos = self._step_until_br_met(
            new_obs, actions, rewards, dones, infos
        )

        return new_obs, rewards, dones, infos

    def reset(self) -> Any:
        obs = self.ma_env.reset()
        infos = {}
        actions = {}
        rewards = {}
        dones = {}
        self.step_cnt = 0

        for policy_id, spec_distribution in self.spec_distribution.items():
            spec = spec_distribution.sample_policy_spec()
            self.load_from_spec(self.fixed_policies[policy_id], spec)

        if self.br_player in obs:
            return obs[self.br_player]

        for agent, _obs in obs.items():
            pid = self.agent_to_policy_id[agent]
            actions[agent], _, _ = self.fixed_policies[pid].compute_single_action(_obs)
            rewards[agent] = 0.0
            dones[agent] = False
            infos[agent] = {}

        obs, _, _, _ = self._step_until_br_met(obs, actions, rewards, dones, infos)
        return obs


if __name__ == "__main__":
    maenv_config = {}
    tmp_env = PokerMultiAgentEnv(env_config=maenv_config)
    action_space = tmp_env.action_space
    obs_space = tmp_env.observation_space
    agent_ids = list(tmp_env.get_agent_ids())

    env_config = {
        "env_class": PokerMultiAgentEnv,
        "env_config": maenv_config,
        "br_player": agent_ids[0],
        "multiagent": {
            "policies": {
                "dqn": (
                    DQNTorchPolicy,
                    tmp_env.observation_space,
                    tmp_env.action_space,
                    {},
                )
            },
            "policy_mapping_fn": lambda agent_id: "dqn",
        },
    }

    env = SingleAgentEnv(env_config)
    done = False
    obs = env.reset()
    step = 0

    while not done:
        obs, reward, done, info = env.step(action_space.sample())
        step += 1
        print(f"br: {agent_ids[0]} reward: {reward}")

    print("--------- total env step: {}".format(env.step_cnt))
