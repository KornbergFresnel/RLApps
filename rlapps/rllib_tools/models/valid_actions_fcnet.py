import logging
from typing import Type, Union

from gym.spaces import Box, Discrete

from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.env import MultiAgentEnv

from rlapps.envs.valid_actions_multi_agent_env import ValidActionsMultiAgentEnv

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

ILLEGAL_ACTION_LOGITS_PENALTY = -1e24


def get_valid_action_fcn_class(
    obs_len: int, action_space_n: int, dummy_actions_multiplier: int = 1
) -> Type[FullyConnectedNetwork]:
    class ValidActionFullyConnectedNetwork(FullyConnectedNetwork, nn.Module):
        @override(FullyConnectedNetwork)
        def __init__(self, obs_space, action_space, num_outputs, model_config, name):
            obs_space = Box(low=0.0, high=1.0, shape=(obs_len,))
            FullyConnectedNetwork.__init__(
                self,
                obs_space=obs_space,
                action_space=action_space,
                num_outputs=num_outputs,
                model_config=model_config,
                name=name,
            )

        @override(FullyConnectedNetwork)
        def forward(self, input_dict, state, seq_lens):
            obs = input_dict["obs_flat"].float()

            non_dummy_action_space_n = action_space_n // dummy_actions_multiplier

            assert obs.shape[1] == obs_len + non_dummy_action_space_n, (
                f"obs shape with valid action fc net is {obs.shape}\n"
                f"obs_len without actions: {obs_len}\n"
                f"non_dummy_action_space_n: {non_dummy_action_space_n}\n"
                f"action space n: {action_space_n}\n"
                f"obs: {obs}"
            )

            obs = obs[:, :-non_dummy_action_space_n]
            self.valid_actions_mask = input_dict["obs_flat"][
                :, -non_dummy_action_space_n:
            ].repeat(1, dummy_actions_multiplier)

            self._last_flat_in = obs.reshape(obs.shape[0], -1)
            self._features = self._hidden_layers(self._last_flat_in)
            logits = self._logits(self._features) if self._logits else self._features

            if self.free_log_std:
                raise NotImplementedError

            illegal_actions = 1 - self.valid_actions_mask
            illegal_logit_penalties = illegal_actions * ILLEGAL_ACTION_LOGITS_PENALTY

            masked_logits = (logits * self.valid_actions_mask) + illegal_logit_penalties

            return masked_logits, state

        @override(FullyConnectedNetwork)
        def value_function(self):
            if not self._value_branch_separate:
                raise NotImplementedError
            return super(ValidActionFullyConnectedNetwork, self).value_function()

    return ValidActionFullyConnectedNetwork


def get_valid_action_fcn_class_for_env(
    env: MultiAgentEnv, policy_role: str = None, force_action_space_n: int = None
) -> Union[None, Type[FullyConnectedNetwork]]:

    if not isinstance(env, ValidActionsMultiAgentEnv):
        raise TypeError(
            "Valid actions fcns are made only to work with subclasses of ValidActionsMultiAgentEnv. "
            f"This env is a {type(env)}."
        )
    if policy_role is None:
        action_space = (
            force_action_space_n if force_action_space_n else env.action_space
        )
        observation_length = env.orig_observation_length
    else:
        # In some multiagent settings, different spaces may be defined for different policies in a dictionary.
        action_space = (
            force_action_space_n
            if force_action_space_n
            else env.action_space[policy_role]
        )
        observation_length = env.orig_observation_length[policy_role]
    if len(env.observation_space.shape) != 1:
        raise ValueError(
            f"Valid action fcn models require a 1D observation space (len(observation_space.shape) == 1). "
            f"The shape of this observation space is {env.observation_space.shape}"
        )
    if isinstance(action_space, Discrete):
        if hasattr(env, "dummy_action_multiplier"):
            dummy_action_multiplier = env.dummy_action_multiplier
        else:
            dummy_action_multiplier = 1
        custom_model = get_valid_action_fcn_class(
            obs_len=observation_length,
            action_space_n=action_space.n,
            dummy_actions_multiplier=dummy_action_multiplier,
        )
    else:
        custom_model = None
    return custom_model
