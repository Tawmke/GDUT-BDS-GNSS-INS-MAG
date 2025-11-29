import sys
import time
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
import pathlib
import io
from torch import nn
from gym import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.save_util import load_from_zip_file, recursive_getattr, recursive_setattr, save_to_zip_file
from model.common.buffers_RBO import RecurrentDictRolloutBuffer_RSP_RBO
from model.common.RSP_module import MLP
from model.common.utils import *

from sb3_contrib.common.recurrent.buffers import RecurrentDictRolloutBuffer, RecurrentRolloutBuffer
# from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from model.common.policies_RSP import RecurrentActorCriticPolicy_RSP
from sb3_contrib.common.recurrent.type_aliases import RNNStates
#from sb3_contrib.ppo_recurrent.policies import CnnLstmPolicy, MlpLstmPolicy, MultiInputLstmPolicy
from model.ppo_rec.policies import CnnLstmPolicy, MlpLstmPolicy, MultiInputLstmPolicy, MlpLstmPolicy_RSP

class RecurrentPPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)
    with support for recurrent policies (LSTM).

    Based on the original Stable Baselines 3 implementation.

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpLstmPolicy_RSP": MlpLstmPolicy_RSP,
        "CnnLstmPolicy": CnnLstmPolicy,
        "MultiInputLstmPolicy": MultiInputLstmPolicy
    }

    def __init__(
        self,
        policy: Union[str, Type[RecurrentActorCriticPolicy_RSP]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 128, # 128
        batch_size: Optional[int] = 128, #128
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        noisescale: float = 0.5,
        beta_robust: float = 0.01,
        learning_rate_predictor = 5e-5,
        seq_embed_mode='learned',
        detach = 'nodetach',
        need_RBO = True
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            create_eval_env=create_eval_env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self._last_lstm_states = None
        # add noise for robust BO
        self.addnoise = 'UnifNoise'
        self.noisescale = noisescale
        self.beta_robust = beta_robust
        self.tau = 0.01
        self.embedding_size = 50
        self.pred_seq_len = 128 # 128
        self.use_target_network = False # if use targetnetwork for RSP
        self.soft_target_tau = 1e-2
        self.seq_embed_mode = seq_embed_mode
        self.cl_coef = 0.5 # transform 更新比例
        self.alpha = 0.05 # 更新cl_loss用
        self.n_aug = 2 # 更新cl_loss用
        self.momentum = 0.001 # 更新target_network
        self.learning_rate_predictor = learning_rate_predictor
        self._lam_scale, self._gamma_transfer, self._gamma_scale = None, None, None
        self.need_RBO = need_RBO
        self.sample_times = 7
        self.detach = detach

        if 'ATF_trig' in self.policy_kwargs:
            self.ATF_trig=policy_kwargs['ATF_trig']
            policy_kwargs.pop('ATF_trig')

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = (
            RecurrentDictRolloutBuffer_RSP_RBO if isinstance(self.observation_space, gym.spaces.Dict) else RecurrentRolloutBuffer
        )

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            detach=self.detach,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

        # We assume that LSTM for the actor and the critic
        # have the same architecture
        lstm = self.policy.lstm_actor

        ################ set up module for RSP##############
        seq_len_model = self.pred_seq_len * 2 if self.seq_embed_mode == "fourier" else self.pred_seq_len # 傅里叶变化的序列要乘2
        def build_networks():
            trunk = nn.Sequential( # 表示学习，embedding
                nn.Linear(self.policy.lstm_output_dim, self.embedding_size),
                nn.LayerNorm(self.embedding_size),
                nn.Tanh()).to(self.device)
            seq_predictor = MLP( # 序列预测模块，相当于论文的Z
                self.embedding_size + self.action_space.shape[1],
                1+seq_len_model, #r, q
                hidden_layers=[256,256],
                activation='relu').to(self.device)
            return trunk, seq_predictor

        self.trunk, self.seq_predictor = build_networks()
        if self.use_target_network:
            target_networks = build_networks()
            self.target_trunk = target_networks[0]
            self.target_seq_predictor = target_networks[1]
            self._update_target(1)
        else:
            self.target_trunk = self.trunk
            self.target_seq_predictor = self.seq_predictor

        # self.lam_scale, self.gamma_transfer, self.gamma_scale = generate_coef(128, "learned")  # 初始化学习转换,learned mode
        # self.lam_scale, self.gamma_transfer, self.gamma_scale = self.lam_scale.to(self.device), self.gamma_transfer.to(self.device), self.gamma_scale.to(self.device)
        # set optimizer for Z and transfer
        lstm_parameters = self.policy.lstm_actor.parameters()
        lstm_parameters_critic = self.policy.lstm_critic.parameters()  # 注意是否选择ac共享lstm层，这里的话就是分开
        ATF1_parameters = self.policy.features_extractor.parameters()
        trunk_parameters = self.trunk.parameters()
        seq_predictor_parameters = self.seq_predictor.parameters()

        if self.seq_embed_mode == 'learned':
            self.transfer = transfer() # learned transfer
            self.lam_scale, self.gamma_transfer, self.gamma_scale = self.transfer.lam_scale.to(self.device), self.transfer.gamma_transfer.to(self.device), self.transfer.gamma_scale.to(self.device)
            allparameters = list(lstm_parameters) + list(lstm_parameters_critic) + list(ATF1_parameters) + list(trunk_parameters) \
                            + list(seq_predictor_parameters) + list(self.transfer.parameters())
        else:
            self.reward_lambda, self.gamma_transfer = generate_coef(device=self.device, gamma=self.gamma, scale=1, seq_len=self.pred_seq_len, mode=self.seq_embed_mode)
            allparameters = list(lstm_parameters) + list(lstm_parameters_critic) + list(ATF1_parameters) + list(trunk_parameters) \
                            + list(seq_predictor_parameters)

        self.predictor_optimizer = th.optim.Adam(allparameters, lr=self.learning_rate_predictor, **self.policy.optimizer_kwargs) # 重新构建adam优化器

        if not isinstance(self.policy, RecurrentActorCriticPolicy_RSP):
            raise ValueError("Policy must subclass RecurrentActorCriticPolicy")

        single_hidden_state_shape = (lstm.num_layers, self.n_envs, lstm.hidden_size)
        # hidden and cell states for actor and critic
        self._last_lstm_states = RNNStates(
            (
                th.zeros(single_hidden_state_shape).to(self.device),
                th.zeros(single_hidden_state_shape).to(self.device),
            ),
            (
                th.zeros(single_hidden_state_shape).to(self.device),
                th.zeros(single_hidden_state_shape).to(self.device),
            ),
        )

        hidden_state_buffer_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            hidden_state_buffer_shape,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def _setup_learn(
        self,
        total_timesteps: int,
        eval_env: Optional[GymEnv],
        callback: MaybeCallback = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "RecurrentPPO",
    ) -> Tuple[int, BaseCallback]:
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param eval_env: Environment to use for evaluation.
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param eval_freq: How many steps between evaluations
        :param n_eval_episodes: How many episodes to play per evaluation
        :param log_path: Path to a folder where the evaluations will be saved
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :return:
        """

        total_timesteps, callback = super()._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            log_path,
            reset_num_timesteps,
            tb_log_name,
        )
        return total_timesteps, callback

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert isinstance(
            rollout_buffer, (RecurrentRolloutBuffer, RecurrentDictRolloutBuffer_RSP_RBO)
        ), f"{rollout_buffer} doesn't support recurrent policy"

        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        lstm_states = deepcopy(self._last_lstm_states)
        lstm_states_noise = deepcopy(self._last_lstm_states)

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                episode_starts = th.tensor(self._last_episode_starts).float().to(self.device)
                actions, values, log_probs, lstm_states = self.policy.forward(obs_tensor, lstm_states, episode_starts)

                # Add noise to observation for computing Softmin operator
                if self.need_RBO:
                    values_noise_list_1 = []
                    values_noise_list_2 = []
                    noisevec = np.random.uniform(-self.noisescale, self.noisescale,size=self.sample_times)
                    for idx in range(self.sample_times): # important sampling for softmin operator
                        if self.addnoise == 'UnifNoise':
                            last_obs_noise = {'gnss': self._last_obs['gnss'] + noisevec[idx],'pos': self._last_obs['pos']}
                        # elif self.addnoise == 'GN':
                        #     sigma = sigma_compute(self._last_obs, SNRdB=self.SNR)
                        #     mean, std = 0, 1  # mean and standard deviation
                        #     mask = np.random.uniform(0, 1) < 0.5
                        #     last_obs_noise = {'gnss': self._last_obs['gnss'] + np.random.normal(mean, std,len(self._last_obs)) * sigma * mask,
                        #         'pos': self._last_obs['pos']}
                        # elif self.addnoise == 'RAS-S':
                        #     #random_number_2 = np.random.uniform(self.args.min, self.args.max)
                        #     last_obs_noise = {'gnss': self._last_obs['gnss'] * np.random.uniform(-self.noisescale, self.noisescale),
                        #         'pos': self._last_obs['pos']}
                        obs_noise_tensor = obs_as_tensor(last_obs_noise, self.device)
                        actions_noise, values_noise, log_probs_noise, lstm_states_noise_tmp = self.policy.forward(obs_noise_tensor, lstm_states_noise, episode_starts)
                        values_noise = values_noise.clone().cpu().numpy()
                        #print(f'v={values_noise}')
                        values_noise_1 = values_noise * np.exp(-self.tau * values_noise)
                        values_noise_2 = np.exp(-self.tau * values_noise)
                        #print(f'v1={values_noise_1}')
                        #print(values_noise_2)
                        values_noise_list_1.append(values_noise_1)
                        values_noise_list_2.append(values_noise_2)
                    lstm_states_noise = lstm_states_noise_tmp
                    softmean_values_noise = np.mean(values_noise_list_1)/np.mean(values_noise_list_2)
                    softmean_values_noise = th.from_numpy(np.array([softmean_values_noise])).to(device=self.device)
                    self.logger.record('values/softmean_values', th.mean(softmean_values_noise).item())
                else:
                    softmean_values_noise = th.from_numpy(np.array([0])).to(device=self.device)

            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            # predict next action for RSP
            next_obs_tensor = obs_as_tensor(new_obs, self.device)
            next_episode_starts = th.tensor(dones).float().to(self.device)
            nextactions, nextvalues, nextlog_probs, _ = self.policy.forward(next_obs_tensor, lstm_states, next_episode_starts)
            nextactions = nextactions.detach().cpu().numpy()

            # recording rewards per step # remote edition 20221021
            self.logger.record("values/rewards per step", np.mean(rewards))

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done_ in enumerate(dones):
                if (
                    done_
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_lstm_state = (
                            lstm_states.vf[0][:, idx : idx + 1, :],
                            lstm_states.vf[1][:, idx : idx + 1, :],
                        )
                        # terminal_lstm_state = None
                        episode_starts = th.tensor([False]).float().to(self.device)
                        terminal_value = self.policy.predict_values(terminal_obs, terminal_lstm_state, episode_starts)[0]
                    rewards[idx] += self.gamma * terminal_value

            # can not find mask 221021 (no mask in standard ppo
            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                lstm_states=self._last_lstm_states,
                value_noise = softmean_values_noise,
                nextobs = new_obs,
                nextaction = nextactions
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_lstm_states = lstm_states

        with th.no_grad():
            # Compute value for the last timestep
            episode_starts = th.tensor(dones).float().to(self.device)

            if self.need_RBO:
                values_noise_list_1 = []
                values_noise_list_2 = []
                noisevec = np.random.uniform(-self.noisescale, self.noisescale, size=self.sample_times)
                for idx in range(self.sample_times):
                    if self.addnoise == 'UnifNoise':
                        new_obs_noise = {'gnss': new_obs['gnss'] + noisevec[idx],'pos': new_obs['pos']}
                    # elif self.addnoise == 'GN':
                    #     sigma = sigma_compute(new_obs, SNRdB=self.SNR)
                    #     mean, std = 0, 1  # mean and standard deviation
                    #     mask = np.random.uniform(0, 1) < 0.5
                    #     new_obs_noise = {'gnss': new_obs['gnss'] + np.random.normal(mean, std, len(self._last_obs)) * sigma * mask,
                    #                      'pos': new_obs['pos']}
                    # elif self.addnoise == 'RAS-S':
                    #     # random_number_2 = np.random.uniform(self.args.min, self.args.max)
                    #     new_obs_noise = {'gnss': new_obs['gnss'] * np.random.uniform(-self.noisescale, self.noisescale),
                    #                      'pos': new_obs['pos']}

                    values_noise = self.policy.predict_values(obs_as_tensor(new_obs_noise, self.device),lstm_states_noise.vf, episode_starts)
                    values_noise = values_noise.clone().cpu().numpy()
                    values_noise_1 = values_noise * np.exp(-self.tau * values_noise)
                    values_noise_2 = np.exp(-self.tau * values_noise)
                    values_noise_list_1.append(values_noise_1)
                    values_noise_list_2.append(values_noise_2)
                values_noise = np.mean(values_noise_list_1) / np.mean(values_noise_list_2)
                values_noise = th.from_numpy(np.array([values_noise])).to(device=self.device)
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device), lstm_states.vf, episode_starts)

        # rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        if self.need_RBO:
            rollout_buffer.compute_returns_and_robust_advantage(last_values=values, last_values_noise=values_noise, dones=dones, beta=self.beta_robust) # compute robust Bellman operator
        else:
            rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        predictor_losses, cl_losses = [], [] # set loss for RSP

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Convert mask from float to bool
                mask = rollout_data.mask > 1e-8

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )

                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob) # 新旧策略比

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.mean(th.min(policy_loss_1, policy_loss_2)[mask])

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()[mask]).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf)
                # Value loss using the TD(gae_lambda) target
                # Mask padded sequences
                value_loss = th.mean(((rollout_data.returns - values_pred) ** 2)[mask])

                # added recordings 20221021
                self.logger.record('values/perepoch_values', th.mean(values[mask]).item())
                self.logger.record('values/perepoch_values_pred', np.float(th.mean(values_pred[mask])))
                self.logger.record('values/perepoch_returns', th.mean(rollout_data.returns[mask]).item())

                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob[mask])
                else:
                    entropy_loss = -th.mean(entropy[mask])

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(((th.exp(log_ratio) - 1) - log_ratio)[mask]).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm) # 梯度裁剪防止梯度爆炸
                self.policy.optimizer.step()

                #### update predictor module for robust representation learning
                features = self.policy.extract_features(rollout_data.observations)
                latent_pi, _ = self.policy._process_sequence(features, rollout_data.lstm_states.pi, rollout_data.episode_starts, self.policy.lstm_actor)
                latent_vf, _ = self.policy._process_sequence(features, rollout_data.lstm_states.vf, rollout_data.episode_starts, self.policy.lstm_critic) # 不共享lstm层

                nextfeatures = self.policy.extract_features(rollout_data.nextobservations)
                next_latent_pi, _ = self.policy._process_sequence(nextfeatures, rollout_data.lstm_states.pi,rollout_data.episode_starts, self.policy.lstm_actor)
                next_latent_vf, _ = self.policy._process_sequence(nextfeatures, rollout_data.lstm_states.vf,rollout_data.episode_starts, self.policy.lstm_critic) # 不共享lstm层

                if self.seq_embed_mode == 'learned':
                    pred_loss_actor, pred_loss, cl_loss = self.compute_auxiliary_loss(latent_pi, rollout_data.actions, rollout_data.rewards,next_latent_pi, rollout_data.nextactions)
                    pred_loss_critic, pred_loss, cl_loss = self.compute_auxiliary_loss(latent_vf, rollout_data.actions,rollout_data.rewards, next_latent_vf,rollout_data.nextactions)
                else:
                    pred_loss_actor = self.compute_auxiliary_loss(latent_pi, rollout_data.actions,rollout_data.rewards,next_latent_pi,rollout_data.nextactions)
                    pred_loss_critic = self.compute_auxiliary_loss(latent_vf, rollout_data.actions,rollout_data.rewards,next_latent_vf,rollout_data.nextactions)
                pred_loss_ac = pred_loss_actor + pred_loss_critic
                # Optimization step
                self.predictor_optimizer.zero_grad()
                pred_loss_ac.backward()
                lstm_parameters = self.policy.lstm_actor.parameters()
                lstm_parameters_critic = self.policy.lstm_critic.parameters()  # 注意是否选择ac共享lstm层，这里的话就是分开
                ATF1_parameters = self.policy.features_extractor.parameters()
                trunk_parameters = self.trunk.parameters()
                seq_predictor_parameters = self.seq_predictor.parameters()
                if self.seq_embed_mode == 'learned':
                    allparameters = list(lstm_parameters) + list(lstm_parameters_critic) + list(ATF1_parameters) + list(trunk_parameters) \
                                    + list(seq_predictor_parameters) + list(self.transfer.parameters())
                    # logger record for RSP
                    predictor_losses.append(pred_loss.item())
                    cl_losses.append(cl_loss.item())
                else:
                    allparameters = list(lstm_parameters) + list(lstm_parameters_critic) + list(ATF1_parameters) + list(trunk_parameters) \
                                    + list(seq_predictor_parameters)
                    predictor_losses.append(pred_loss_ac.item())
                    cl_losses.append(0)
                th.nn.utils.clip_grad_norm_(allparameters, self.max_grad_norm)
                self.predictor_optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/predictor_loss", np.mean(predictor_losses))
        self.logger.record("train/cl_loss", np.mean(cl_losses))
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        # added recordings 20221021
        self.logger.record('values/values', th.mean(values[mask]).item())
        self.logger.record('values/values_pred', np.float(th.mean(values_pred[mask])))
        self.logger.record('values/returns', th.mean(rollout_data.returns[mask]).item())

    def _update_target(self, tau):
        if not self.use_target_network:
            return
        if tau == 1:
            copy_model_params_from_to(self.trunk, self.target_trunk)
            copy_model_params_from_to(self.seq_predictor, self.target_seq_predictor)
        else:
            soft_update_from_to(self.trunk, self.target_trunk, tau)
            soft_update_from_to(self.seq_predictor, self.target_seq_predictor, tau)

    def compute_auxiliary_loss(self, obs, a, r, next_obs, next_a, n_step=0, log=False, frame=None):
        self._update_target(self.soft_target_tau)
        # predict
        h = self.trunk(obs)
        pred_feature = th.cat([h, a], dim=-1)
        pred_rq = self.seq_predictor(pred_feature)
        pred_r = pred_rq[:, :1]
        pred_q = pred_rq[:, 1:]
        with th.no_grad():
            next_h = self.target_trunk(next_obs)
            next_pred_feature = th.cat([next_h, next_a], dim=-1)
            next_pred_rq = self.target_seq_predictor(next_pred_feature)
            next_target_q = next_pred_rq[:, 1:]
        if self.seq_embed_mode == 'learned':
            if self._lam_scale is None:
                self._lam_scale = self.lam_scale.detach()
                self._gamma_transfer = self.gamma_transfer.detach()
                self._gamma_scale = self.gamma_scale.detach()
            else:
                mm = self.momentum  # 什么动量操作
                self._lam_scale = self.lam_scale.detach() * mm + (1 - mm) * self._lam_scale
                self._gamma_transfer = self.gamma_transfer.detach() * mm + (1 - mm) * self._gamma_transfer
                self._gamma_scale = self.gamma_scale.detach() * mm + (1 - mm) * self._gamma_scale

            r = r[0:self.pred_seq_len]
            next_target_q = next_target_q[0:self.pred_seq_len]
            target_q, target_r, target_next = td_style(r, next_target_q, self.gamma,
                                                       self.lam_scale, self.gamma_transfer, self.gamma_scale, tmp=1, control=1)  # 计算预测目标
        else:
            r = r[0:self.pred_seq_len]
            next_target_q = next_target_q[0:self.pred_seq_len]
            target_q = td_style_fixed(r, next_target_q, self.reward_lambda, self.gamma_transfer, self.seq_embed_mode)  # 计算预测目标

        target_q[..., -1] = 1
        pred_r = pred_r[0:self.pred_seq_len]
        pred_q = pred_q[0:self.pred_seq_len]
        pred_loss, r_loss, q_loss = self._compute_predict_loss(pred_r, r.detach(), pred_q, target_q.detach())
        # _, _, q_loss =self._compute_predict_loss(
        #             pred_r, r.detach(), pred_q.detach(), target_q)
        if self.cl_coef > 0 and self.seq_embed_mode == 'learned':  # 自动学习权重
            cl_loss, _, accuracy = compute_cl_loss(pred_q,target_q, self.alpha, self.n_aug, self.device)
            pred_loss_all = pred_loss + cl_loss * self.cl_coef
            return pred_loss_all, pred_loss, cl_loss
        else:
            return pred_loss

    def _compute_predict_loss(self, r, tar_r, q, tar_q):
        r_loss = F.mse_loss(r.squeeze(1), tar_r)
        q_loss = F.mse_loss(q, tar_q)
        loss = r_loss + q_loss
        return loss, r_loss, q_loss

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "RecurrentPPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "RecurrentPPO":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name)

        callback.on_training_start(locals(), globals())

        while iteration < total_timesteps:  #改：self.num_timesteps——>iteration
            #if iteration % 3 == 0:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(iteration, total_timesteps) #改：self.num_timesteps——>iteration

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=iteration) #改：self.num_timesteps——>iteration

            self.train()

        callback.on_training_end()

        return self

    def load(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ):
        """
        Load the model from a zip-file.
        Warning: ``load`` re-creates the model from scratch, it does not update it in-place!
        For an in-place load use ``set_parameters`` instead.

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param print_system_info: Whether to print system info from the saved model
            and the current system info (useful to debug loading issues)
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See https://github.com/DLR-RM/stable-baselines3/issues/597
        :param kwargs: extra arguments to change the model when loading
        :return: new model instance with loaded parameters
        """
        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        # put state_dicts back in place
        self.set_parameters(load_path_or_dict=params, exact_match=True, device=device)

        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(self, name + ".data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if self.use_sde:
            self.policy.reset_noise()  # pytype: disable=attribute-error
        # return model

    def set_parameters(
        self,
        load_path_or_dict: Union[str, Dict[str, Dict]],
        exact_match: bool = True,
        device: Union[th.device, str] = "auto",
    ) -> None:
        """
        Load parameters from a given zip-file or a nested dictionary containing parameters for
        different modules (see ``get_parameters``).

        :param load_path_or_iter: Location of the saved data (path or file-like, see ``save``), or a nested
            dictionary containing nn.Module parameters used by the policy. The dictionary maps
            object names to a state-dictionary returned by ``torch.nn.Module.state_dict()``.
        :param exact_match: If True, the given parameters should include parameters for each
            module and each of their parameters, otherwise raises an Exception. If set to False, this
            can be used to update only specific parameters.
        :param device: Device on which the code should run.
        """
        params = None
        if isinstance(load_path_or_dict, dict):
            params = load_path_or_dict
        else:
            _, params, _ = load_from_zip_file(load_path_or_dict, device=device)

        # Keep track which objects were updated.
        # `_get_torch_save_params` returns [params, other_pytorch_variables].
        # We are only interested in former here.
        objects_needing_update = set(self._get_torch_save_params()[0])
        updated_objects = set()

        for name in params:
            attr = None
            try:
                attr = recursive_getattr(self, name)
            except Exception as e:
                # What errors recursive_getattr could throw? KeyError, but
                # possible something else too (e.g. if key is an int?).
                # Catch anything for now.
                raise ValueError(f"Key {name} is an invalid object name.") from e

            if isinstance(attr, th.optim.Optimizer):
                # Optimizers do not support "strict" keyword...
                # Seems like they will just replace the whole
                # optimizer state with the given one.
                # On top of this, optimizer state-dict
                # seems to change (e.g. first ``optim.step()``),
                # which makes comparing state dictionary keys
                # invalid (there is also a nesting of dictionaries
                # with lists with dictionaries with ...), adding to the
                # mess.
                #
                # TL;DR: We might not be able to reliably say
                # if given state-dict is missing keys.
                #
                # Solution: Just load the state-dict as is, and trust
                # the user has provided a sensible state dictionary.
                attr.load_state_dict(params[name])
            else:
                # Assume attr is th.nn.Module
                attr.load_state_dict(params[name], strict=exact_match)
            updated_objects.add(name)

        if exact_match and updated_objects != objects_needing_update:
            raise ValueError(
                "Names of parameters do not match agents' parameters: "
                f"expected {objects_needing_update}, got {updated_objects}"
            )

def sigma_compute(train_data,SNRdB= 40):
    # data_mean = [np.mean(train_data[0], axis=0), np.mean(train_data[1], axis=0)]
    # data_std = [np.std(train_data[0], axis=0), np.std(train_data[1], axis=0)]
    train_data = np.array(train_data)
    data_mean = np.mean(train_data)
    data_std = np.std(train_data)
    a_noise = np.sqrt(np.power(10, (SNRdB / 10)))
    SNRsigma = np.sqrt(np.power(data_std, 2) + np.power(data_mean, 2)) / a_noise
    return SNRsigma

def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau)

def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def randn(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = None
    return th.randn(*args, **kwargs, device=torch_device)

def generate_coef(device=0, gamma=0.99, seq_len=128,  mode="learned", scale=1):
    if mode == 'learned':
        assert mode in ['learned', 'random', 'fourier']
        lam_scale = randn(seq_len)
        gamma_transfer = randn(seq_len, seq_len)
        gamma_scale = randn(seq_len,)
        lam_scale = nn.Parameter(lam_scale)
        # gamma_transfer = nn.Parameter(gamma_transfer)
        gamma_scale = nn.Parameter(gamma_scale)
        return lam_scale, gamma_transfer, gamma_scale
    elif mode == 'fourier':
        fourier_gamma = np.arange(seq_len)/seq_len*2*np.pi #np.arange生成一个在指定范围内以固定间隔递增的数组
        real = np.cos(fourier_gamma)
        image = np.sin(-fourier_gamma)
        fourier_gamma = np.stack([real, image])
        gamma = (fourier_gamma*gamma)
        reward_lambda = np.ones(seq_len)*scale
        gamma = th.from_numpy(gamma).float().to(device)
        reward_lambda = th.from_numpy(reward_lambda).float().to(device)
        return reward_lambda, gamma
    elif mode == "direct":
        return 1, gamma

def td_style(reward, next_v, gamma, lam_scale, gamma_transfer, gamma_scale, tmp=1, control=True):
    if control:
        gamma_transfer = th.softmax(gamma_transfer/tmp, dim=0) # softmax:输出的所有元素都介于 0 和 1 之间
        gamma_scale = gamma*th.sigmoid(gamma_scale)
    # lam_scale = F.softplus(lam_scale)+1
    # clip some data
    target_r = reward*lam_scale
    target_next = th.mm(next_v, gamma_transfer)*gamma_scale # 对应于收缩因子操作
    target_q = target_r+target_next
    return target_q, target_r, target_next

def td_style_fixed(reward, next_v, reward_lambda, gamma, mode="fourier"):
    if mode == "fourier":
        k = gamma.shape[-1]
        next_v = next_v.view(-1,2,k)
        real_part = next_v * gamma #点乘
        real_part = real_part[:,0]-real_part[:,1]
        real_part = real_part + reward_lambda*reward
        gamma = gamma[[1,0]] # 反转tensor元素的顺序
        image_part = next_v * gamma
        image_part = image_part.sum(dim=1)
        v = torch.cat([real_part, image_part], dim=-1)
    elif mode == "laplace":
        v = gamma*next_v+reward_lambda*reward
    elif mode == "direct":
        v = torch.cat([reward.unsqueeze(1), next_v[:,:-1]*gamma],dim=-1)
    elif mode == "random":
        v = torch.mm(reward, reward_lambda) + torch.mm(next_v, gamma)
    else:
        raise NotImplementedError
    return v

def compute_cl_loss(e1, e2, alpha, n_aug, device):
    e1_norm = th.norm(e1, dim=-1, p=2, keepdim=True)
    e1 = e1 / e1_norm
    e2_norm = th.norm(e2, dim=-1, p=2, keepdim=True)
    e2 = e2 / e2_norm
    similarity = th.mm(e1, th.t(e2))
    similarity = similarity/alpha
    with th.no_grad():
        pred_prob = th.softmax(similarity, dim=-1)
        target_prob = eye(len(similarity))
        if n_aug==2:
            aug_p = eye(len(similarity))
            p1, p2 = torch.chunk(aug_p,2,dim=-1)
            aug_p = torch.cat([p2, p1], dim=-1)
            target_prob = (target_prob+aug_p)/2
        accuracy = (pred_prob * target_prob.to(device)).sum(-1)
        diff = pred_prob-target_prob.to(device)
    loss = (similarity*diff).sum(-1).mean()
    return loss, pred_prob, accuracy

class transfer(nn.Module):
    def __init__(self):
        super(transfer, self).__init__()
        self.lam_scale, self.gamma_transfer, self.gamma_scale = generate_coef(seq_len=128, mode="learned")