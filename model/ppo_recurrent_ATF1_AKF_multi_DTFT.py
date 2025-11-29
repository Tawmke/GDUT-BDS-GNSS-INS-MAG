import sys
import time
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Type, Union, Iterable, List
import pathlib
import io
import gym
import numpy as np
import torch as th
import torch.nn.functional
from gym import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.save_util import load_from_zip_file, recursive_getattr, recursive_setattr, save_to_zip_file
from model.common.RSP_module import Prediction,Projection,Projection2

from sb3_contrib.common.recurrent.buffers import RecurrentDictRolloutBuffer, RecurrentRolloutBuffer
from model.common.buffers_AKF_multi import RecurrentRolloutBuffer_AKF_multi, RecurrentDictRolloutBuffer_AKF_multi_DTFT
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from sb3_contrib.ppo_recurrent.policies import CnnLstmPolicy, MlpLstmPolicy, MultiInputLstmPolicy

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
        "MlpLstmPolicy": MlpLstmPolicy,
        "CnnLstmPolicy": CnnLstmPolicy,
        "MultiInputLstmPolicy": MultiInputLstmPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[RecurrentActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 128,
        batch_size: Optional[int] = 128,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs_dic: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        use_projection: bool= True,
        cosine_similarity: bool=True,
        use_target_network: bool=True,
        learning_rate_predictor: Union[float, Schedule] = 8e-5,
        normalizer: Union[th.device, str] = 'layer',
        dimstate: Union[th.device, str] = 'lospos',
        _init_setup_model: bool = True,
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
            policy_kwargs=policy_kwargs_dic['Q_policy'],
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
        self._last_lstm_states_Q = None
        self._last_lstm_states_R = None
        self.policy_kwargs_R = policy_kwargs_dic['R_policy']
        self.use_projection = use_projection
        self.use_target_network = use_target_network
        self.cosine_similarity = cosine_similarity
        self.soft_target_tau = 1e-2
        self.pred_seq_len = 128
        self.dim_discretize = 128
        self.trun = 15
        self.normalizer = normalizer
        self.learning_rate_predictor = learning_rate_predictor
        self.dimstate = dimstate

        if 'ATF_trig' in self.policy_kwargs:
            self.ATF_trig=policy_kwargs_dic['R_policy']['ATF_trig']
            policy_kwargs_dic['R_policy'].pop('ATF_trig')
            policy_kwargs_dic['Q_policy'].pop('ATF_trig')

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = (
            RecurrentDictRolloutBuffer_AKF_multi_DTFT if isinstance(self.observation_space, gym.spaces.Dict) else RecurrentRolloutBuffer
        )
        # define feature dim
        # Q_obs_space = spaces.Dict({'pos': self.observation_space['pos'], 'Q_noise': self.observation_space['Q_noise']})
        # R_obs_space = spaces.Dict({'gnss': self.observation_space['gnss'],
        #                            'pos': self.observation_space['pos'], 'R_noise': self.observation_space['R_noise']})
        actionspace = spaces.Box(low=-100, high=100, shape=(1, int(self.action_space.shape[1]*0.5)), dtype=np.float)
        # define policy Q
        self.policy_Q = self.policy_class(
            self.observation_space,
            actionspace,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy_Q = self.policy_Q.to(self.device)

        lstm_Q = self.policy_Q.lstm_actor

        if not isinstance(self.policy_Q, RecurrentActorCriticPolicy):
            raise ValueError("Policy must subclass RecurrentActorCriticPolicy")

        single_hidden_state_shape = (lstm_Q.num_layers, self.n_envs, lstm_Q.hidden_size)
        # hidden and cell states for actor and critic
        self._last_lstm_states_Q = RNNStates(
            (
                th.zeros(single_hidden_state_shape).to(self.device),
                th.zeros(single_hidden_state_shape).to(self.device),
            ),
            (
                th.zeros(single_hidden_state_shape).to(self.device),
                th.zeros(single_hidden_state_shape).to(self.device),
            ),
        )

        # define policy for policy_R
        self.policy_R = self.policy_class(
            self.observation_space,
            actionspace,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs_R,  # pytype:disable=not-instantiable
        )
        self.policy_R = self.policy_R.to(self.device)

        # We assume that LSTM for the actor and the critic
        # have the same architecture
        lstm_R = self.policy_R.lstm_actor

        # setting the lr scheduler
        self.scheduler_Q = th.optim.lr_scheduler.StepLR(self.policy_Q.optimizer, step_size=5e4, gamma=0.9)
        self.scheduler_R = th.optim.lr_scheduler.StepLR(self.policy_R.optimizer, step_size=5e4, gamma=0.9)

        if not isinstance(self.policy_R, RecurrentActorCriticPolicy):
            raise ValueError("Policy must subclass RecurrentActorCriticPolicy")

        single_hidden_state_shape = (lstm_R.num_layers, self.n_envs, lstm_R.hidden_size)
        # hidden and cell states for actor and critic
        self._last_lstm_states_R = RNNStates(
            (
                th.zeros(single_hidden_state_shape).to(self.device),
                th.zeros(single_hidden_state_shape).to(self.device),
            ),
            (
                th.zeros(single_hidden_state_shape).to(self.device),
                th.zeros(single_hidden_state_shape).to(self.device),
            ),
        )

        hidden_state_buffer_shape = (self.n_steps, lstm_Q.num_layers, self.n_envs, lstm_Q.hidden_size)

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

        ################ set up module for DTFT ##############
        self.end = int(self.dim_discretize * 0.5 + 1)
        features_dim_gnss = self.observation_space['gnss'].shape[-1]
        features_dim_pos = self.observation_space['pos'].shape[-1]
        # features_dim_Q = self.observation_space['Q_noise'].shape[-1]
        # features_dim_R = self.observation_space['R_noise'].shape[-1]
        if self.dimstate == 'lospos':
            self.dim_state = features_dim_pos + features_dim_gnss #  + features_dim_Q +  features_dim_R # 观测空间维度
        elif self.dimstate == 'los':
            self.dim_state = features_dim_gnss
        elif self.dimstate == 'pos':
            self.dim_state = features_dim_pos

            # prediction for policy Q
        input_dim_projection = (self.end  - 2 * self.trun) * self.dim_state  # self.end * self.dim_state
        self.prediction_Q = Prediction(dim_input=self.policy_Q.lstm_output_dim,dim_discretize=self.end, dim_state=self.dim_state, normalizer=self.normalizer).to(self.device)
        if self.use_target_network:
            self.target_prediction_Q = Prediction(dim_input=self.policy_Q.lstm_output_dim,dim_discretize=self.end, dim_state=self.dim_state, normalizer=self.normalizer).to(self.device)
        if self.use_projection == True:
            self.projection_Q = Projection(input_dim = input_dim_projection,normalizer=self.normalizer).to(self.device) # 两层的mlp映射
            self.projection2_Q = Projection2(input_dim = 256, normalizer=self.normalizer).to(self.device)
            if self.use_target_network:
                self.target_projection_Q = Projection(input_dim=input_dim_projection,normalizer=self.normalizer).to(self.device)  # 两层的mlp映射
                self.target_projection2_Q = Projection2(input_dim=256, normalizer=self.normalizer).to(self.device)

        # prediction for policy R
        self.prediction_R = Prediction(dim_input=self.policy_R.lstm_output_dim, dim_discretize=self.end, dim_state=self.dim_state, normalizer=self.normalizer).to(self.device)
        if self.use_target_network:
            self.target_prediction_R = Prediction(dim_input=self.policy_R.lstm_output_dim,dim_discretize=self.end, dim_state=self.dim_state, normalizer=self.normalizer).to(self.device)
        if self.use_projection == True:
            self.projection_R = Projection(input_dim=input_dim_projection, normalizer=self.normalizer).to(self.device)  # 两层的mlp映射
            self.projection2_R = Projection2(input_dim=256, normalizer=self.normalizer).to(self.device)
            if self.use_target_network:
                self.target_projection_R = Projection(input_dim=input_dim_projection,normalizer=self.normalizer).to(self.device)  # 两层的mlp映射
                self.target_projection2_R = Projection2(input_dim=256, normalizer=self.normalizer).to(self.device)
        self._update_target(1) # update target model

        ratio = 2 * np.pi / self.dim_discretize # 2 pi / 128
        con = th.tensor([k * ratio for k in range(self.end)])
        self.Gamma_re =  self.gamma * th.diag(th.cos(con)).to(self.device)
        self.Gamma_im = -self.gamma * th.diag(th.sin(con)).to(self.device)

        # set optimizer for projection and prediction
        lstm_parameters = self.policy_Q.lstm_actor.parameters()
        lstm_parameters_critic = self.policy_Q.lstm_critic.parameters()  # 注意是否选择ac共享lstm层，这里的话就是分开
        ATF1_parameters = self.policy_Q.features_extractor.parameters()
        prediction_parameters_Q = self.prediction_Q.parameters()
        allparameters_Q = list(lstm_parameters) + list(lstm_parameters_critic) + list(ATF1_parameters) + list(prediction_parameters_Q)
        if self.use_projection == True:
            projection_parameters_Q = self.projection_Q.parameters()
            projection2_parameters_Q = self.projection2_Q.parameters()
            allparameters_Q = allparameters_Q + list(projection_parameters_Q) + list(projection2_parameters_Q)
        self.predictor_optimizer_Q = th.optim.Adam(allparameters_Q, lr=self.learning_rate_predictor, **self.policy_Q.optimizer_kwargs)

        lstm_parameters = self.policy_R.lstm_actor.parameters()
        lstm_parameters_critic = self.policy_R.lstm_critic.parameters()  # 注意是否选择ac共享lstm层，这里的话就是分开
        ATF1_parameters = self.policy_R.features_extractor.parameters()
        prediction_parameters_R = self.prediction_R.parameters()
        allparameters_R = list(lstm_parameters) + list(lstm_parameters_critic) + list(ATF1_parameters) + list(prediction_parameters_R)
        if self.use_projection == True:
            projection_parameters_R = self.projection_Q.parameters()
            projection2_parameters_R = self.projection2_Q.parameters()
            allparameters_R = allparameters_R + list(projection_parameters_R) + list(projection2_parameters_R)
        self.predictor_optimizer_R = th.optim.Adam(allparameters_R, lr=self.learning_rate_predictor, **self.policy_R.optimizer_kwargs)

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
            rollout_buffer, (RecurrentRolloutBuffer, RecurrentDictRolloutBuffer, RecurrentDictRolloutBuffer_AKF_multi_DTFT)
        ), f"{rollout_buffer} doesn't support recurrent policy"

        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy_Q.set_training_mode(False)
        self.policy_R.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy_Q.reset_noise(env.num_envs)
            self.policy_R.reset_noise(env.num_envs)

        callback.on_rollout_start()

        lstm_states_Q = deepcopy(self._last_lstm_states_Q)
        lstm_states_R = deepcopy(self._last_lstm_states_R)

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy_Q.reset_noise(env.num_envs)
                self.policy_R.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                episode_starts = th.tensor(self._last_episode_starts).float().to(self.device)
                actions_Q, values_Q , log_probs_Q, lstm_states_Q = self.policy_Q.forward(obs_tensor, lstm_states_Q, episode_starts)
                actions_R, values_R, log_probs_R, lstm_states_R = self.policy_R.forward(obs_tensor, lstm_states_R, episode_starts)
                actions = th.cat((actions_Q, actions_R), dim=1)
            actions = actions.cpu().numpy()
            values = (values_Q + values_R) * 0.5
            log_probs = (log_probs_Q + log_probs_R) * 0.5 # how to concat policy logprob ?

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            # predict next action for RSP
            next_obs_tensor = obs_as_tensor(new_obs, self.device)
            next_episode_starts = th.tensor(dones).float().to(self.device)
            nextactions_Q, nextvalues_Q, nextlog_probs_Q, _ = self.policy_Q.forward(next_obs_tensor, lstm_states_Q, next_episode_starts)
            nextactions_R, nextvalues_R, nextlog_probs_R, _ = self.policy_R.forward(next_obs_tensor, lstm_states_R, next_episode_starts)
            nextactions = th.cat((nextactions_Q, nextactions_R), dim=1)
            nextactions = nextactions.detach().cpu().numpy()

            # recording rewards per step # remote edition 20221021
            self.logger.record("values/rewards per step", np.mean(rewards))
            self.logger.record("values/positioning error perstep", infos[0]['error'])
            self.logger.record('values/perepoch_logprob', log_probs.cpu().numpy().item())

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
                    terminal_obs = self.policy_R.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_lstm_state = (
                            lstm_states_R.vf[0][:, idx : idx + 1, :],
                            lstm_states_R.vf[1][:, idx : idx + 1, :],
                        )
                        # terminal_lstm_state = None
                        episode_starts = th.tensor([False]).float().to(self.device)
                        terminal_value_R = self.policy_R.predict_values(terminal_obs, terminal_lstm_state, episode_starts)[0]

                    terminal_obs = self.policy_Q.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_lstm_state = (
                            lstm_states_Q.vf[0][:, idx : idx + 1, :],
                            lstm_states_Q.vf[1][:, idx : idx + 1, :],
                        )
                        # terminal_lstm_state = None
                        episode_starts = th.tensor([False]).float().to(self.device)
                        terminal_value_Q = self.policy_Q.predict_values(terminal_obs, terminal_lstm_state, episode_starts)[0]

                    rewards[idx] += self.gamma * (terminal_value_Q + terminal_value_R) * 0.5

            # can not find mask 221021 (no mask in standard ppo
            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                lstm_states_Q=self._last_lstm_states_Q,
                lstm_states_R=self._last_lstm_states_R,
                nextobs=new_obs,
                nextaction=nextactions
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_lstm_states_Q = lstm_states_Q
            self._last_lstm_states_R = lstm_states_R

        with th.no_grad():
            # Compute value for the last timestep
            episode_starts = th.tensor(dones).float().to(self.device)
            values_Q = self.policy_Q.predict_values(obs_as_tensor(new_obs, self.device), lstm_states_Q.vf, episode_starts)
            values_R = self.policy_R.predict_values(obs_as_tensor(new_obs, self.device), lstm_states_R.vf, episode_starts)
            values = (values_Q + values_R) * 0.5

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy_Q.set_training_mode(True)
        self.policy_R.set_training_mode(True)
        # Update optimizer learning rate
        # self._update_learning_rate(self.policy_Q.optimizer)
        # self._update_learning_rate(self.policy_R.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        predictor_losses_R, predictor_losses_Q = [],[] # record predictor Q and R loss here

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

                len_action = int(actions.shape[1]/2)
                actions_Q, actions_R  = actions[:,:len_action], actions[:,-len_action:]
                # Convert mask from float to bool
                mask = rollout_data.mask > 1e-8

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy_Q.reset_noise(self.batch_size)
                    self.policy_R.reset_noise(self.batch_size)

                values_Q, log_prob_Q, entropy_Q = self.policy_Q.evaluate_actions(
                    rollout_data.observations,
                    actions_Q,
                    rollout_data.lstm_states_Q,
                    rollout_data.episode_starts,
                )

                values_R, log_prob_R, entropy_R = self.policy_R.evaluate_actions(
                    rollout_data.observations,
                    actions_R,
                    rollout_data.lstm_states_R,
                    rollout_data.episode_starts,
                )

                values = (values_Q + values_R) * 0.5
                log_prob = (log_prob_Q + log_prob_R) * 0.5
                entropy = (entropy_Q + entropy_R) * 0.5

                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

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
                    values_pred = rollout_data.old_values + th.clamp(values - rollout_data.old_values, -clip_range_vf, clip_range_vf)
                # Value loss using the TD(gae_lambda) target
                # Mask padded sequences
                value_loss = th.mean(((rollout_data.returns - values_pred) ** 2)[mask])

                # added recordings 20221021
                self.logger.record('values/perepoch_values', th.mean(values[mask]).item())
                self.logger.record('values/perepoch_values_pred', np.float(th.mean(values_pred[mask])))
                self.logger.record('values/perepoch_returns', th.mean(rollout_data.returns[mask]).item())
                self.logger.record("train/learning_rate", self.policy_Q.optimizer.state_dict()['param_groups'][0]['lr'])

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
                # if value_loss.item() > 3e3: # 如果value loss过大不稳定，不训练
                #     print(f'value loss={value_loss.item()},continue')
                #     print(self.env.buf_infos[0]['tripIDnum'])
                #     continue
                self.policy_Q.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy_Q.parameters(), self.max_grad_norm)
                self.policy_Q.optimizer.step()

                self.policy_R.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy_R.parameters(), self.max_grad_norm)
                self.policy_R.optimizer.step()
                # lr updating
                self.scheduler_Q.step()
                self.scheduler_R.step()

                #### update predictor module for DTFT learning
                # representation of ot and ot+1
                features_R = self.policy_R.extract_features(rollout_data.observations)
                features_Q = self.policy_Q.extract_features(rollout_data.observations)
                latent_pi_R, _ = self.policy_R._process_sequence(features_R, rollout_data.lstm_states_R.pi,
                                                             rollout_data.episode_starts, self.policy_R.lstm_actor)
                latent_vf_R, _ = self.policy_R._process_sequence(features_R, rollout_data.lstm_states_R.vf,
                                                             rollout_data.episode_starts, self.policy_R.lstm_critic)
                latent_pi_Q, _ = self.policy_Q._process_sequence(features_Q, rollout_data.lstm_states_Q.pi,
                                                             rollout_data.episode_starts, self.policy_Q.lstm_actor)
                latent_vf_Q, _ = self.policy_Q._process_sequence(features_Q, rollout_data.lstm_states_Q.vf,
                                                             rollout_data.episode_starts, self.policy_Q.lstm_critic)

                nextfeatures_R = self.policy_R.extract_features(rollout_data.nextobservations)
                nextfeatures_Q = self.policy_Q.extract_features(rollout_data.nextobservations)
                next_latent_pi_R, _ = self.policy_R._process_sequence(nextfeatures_R, rollout_data.lstm_states_R.pi,
                                                             rollout_data.episode_starts, self.policy_R.lstm_actor)
                next_latent_vf_R, _ = self.policy_R._process_sequence(nextfeatures_R, rollout_data.lstm_states_R.vf,
                                                             rollout_data.episode_starts, self.policy_R.lstm_critic)
                next_latent_pi_Q, _ = self.policy_Q._process_sequence(nextfeatures_Q, rollout_data.lstm_states_Q.pi,
                                                             rollout_data.episode_starts, self.policy_Q.lstm_actor)
                next_latent_vf_Q, _ = self.policy_Q._process_sequence(nextfeatures_Q, rollout_data.lstm_states_Q.vf,
                                                             rollout_data.episode_starts, self.policy_Q.lstm_critic)
                # compute target and loss
                pred_loss_actor_R = self.compute_auxiliary_loss(latent_pi_R, rollout_data.actions, next_latent_pi_R,
                                                        rollout_data.nextactions, rollout_data.nextobservations,
                                                                rollout_data.episode_starts, 'R')
                pred_loss_critic_R = self.compute_auxiliary_loss(latent_vf_R, rollout_data.actions, next_latent_vf_R,
                                                        rollout_data.nextactions, rollout_data.nextobservations,
                                                                rollout_data.episode_starts, 'R')
                pred_loss_ac_R = pred_loss_actor_R + pred_loss_critic_R
                predictor_losses_R.append(th.mean(pred_loss_ac_R).item())
                pred_loss_actor_Q = self.compute_auxiliary_loss(latent_pi_Q, rollout_data.actions, next_latent_pi_Q,
                                                                rollout_data.nextactions, rollout_data.nextobservations,
                                                                rollout_data.episode_starts, 'Q')
                pred_loss_critic_Q = self.compute_auxiliary_loss(latent_vf_Q, rollout_data.actions, next_latent_vf_Q,
                                                                 rollout_data.nextactions,rollout_data.nextobservations,
                                                                 rollout_data.episode_starts, 'Q')
                pred_loss_ac_Q = pred_loss_actor_Q + pred_loss_critic_Q
                predictor_losses_Q.append(th.mean(pred_loss_ac_Q).item())
                # Optimization step (R)
                self.predictor_optimizer_R.zero_grad()
                pred_loss_ac_R.backward()
                allparameters_R = self.obtain_parameters(self.policy_R,self.prediction_R,self.projection_R,self.projection2_R)
                th.nn.utils.clip_grad_norm_(allparameters_R, self.max_grad_norm)
                self.predictor_optimizer_R.step()
                # Optimization step (R)
                self.predictor_optimizer_Q.zero_grad()
                pred_loss_ac_Q.backward()
                allparameters_Q = self.obtain_parameters(self.policy_Q, self.prediction_Q, self.projection_Q,
                                                         self.projection2_Q)
                th.nn.utils.clip_grad_norm_(allparameters_Q, self.max_grad_norm)
                self.predictor_optimizer_Q.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/predictor_loss_R", np.mean(predictor_losses_R))
        self.logger.record("train/predictor_loss_Q", np.mean(predictor_losses_Q))
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy_Q, "log_std"):
            self.logger.record("train/std_Q", th.exp(self.policy_Q.log_std).mean().item())
        if hasattr(self.policy_R, "log_std"):
            self.logger.record("train/std_R", th.exp(self.policy_R.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        # added recordings 20221021
        self.logger.record('values/values', th.mean(values[mask]).item())
        self.logger.record('values/values_pred', np.float(th.mean(values_pred[mask])))
        self.logger.record('values/returns_target', th.mean(rollout_data.returns[mask]).item())

    def compute_auxiliary_loss(self, latent, a, next_latent, next_a, next_obs, dones, label, n_step=0, log=False, frame=None):
        batch_size = latent.shape[0]
        self._update_target(self.soft_target_tau) # update parameter of target model
        # predict
        if label == 'R':
            predicted_re, predicted_im = self.prediction_R(latent)
            next_predicted_re, next_predicted_im = self.target_prediction_R(next_latent)
            target_projection = self.target_projection_R
            target_projection2 = self.target_projection2_R
            projection = self.projection_R
            projection2 = self.projection2_R
        elif label == 'Q':
            predicted_re, predicted_im = self.prediction_Q(latent)
            next_predicted_re, next_predicted_im = self.target_prediction_Q(next_latent)
            target_projection = self.target_projection_Q
            target_projection2 = self.target_projection2_Q
            projection = self.projection_Q
            projection2 = self.projection2_Q
        predicted_re = th.reshape(predicted_re, [batch_size, self.end, self.dim_state])
        predicted_im = th.reshape(predicted_im, [batch_size, self.end, self.dim_state])
        next_predicted_re = th.reshape(next_predicted_re, [batch_size, self.end, self.dim_state])
        next_predicted_im = th.reshape(next_predicted_im, [batch_size, self.end, self.dim_state])
        predicted_re, predicted_im = predicted_re[:self.pred_seq_len], predicted_im[:self.pred_seq_len]
        next_predicted_re, next_predicted_im = next_predicted_re[:self.pred_seq_len], next_predicted_im[:self.pred_seq_len]
        # compute target
        dones = dones[:self.pred_seq_len]
        dones = dones.unsqueeze(-1).unsqueeze(-1).repeat(1, self.end, self.dim_state)
        if self.dimstate == 'lospos':
            next_obs = th.cat((next_obs['gnss'], next_obs['pos']), dim=-1) # note that which features do you use for dtft
        elif self.dimstate == 'los':
            next_obs = next_obs['gnss']
        elif self.dimstate == 'pos':
            next_obs = next_obs['pos']
        next_obs = next_obs[:self.pred_seq_len]
        next_obs_redim = next_obs[:, :self.dim_state].repeat(1, self.end, 1)

        y_target_re = next_obs_redim + (th.matmul(self.Gamma_re, next_predicted_re)-th.matmul(self.Gamma_im, next_predicted_im))*(1-dones)
        y_re = predicted_re
        pred_re_loss = self.loss(y_target_re, y_re, projection,projection2, target_projection, target_projection2)

        y_target_im = (th.matmul(self.Gamma_im, next_predicted_re) + th.matmul(self.Gamma_re, next_predicted_im)) * (1 - dones)
        y_im = predicted_im
        pred_im_loss = self.loss(y_target_im, y_im, projection,projection2, target_projection, target_projection2)

        return pred_re_loss + pred_im_loss

    def loss(self,y_target, y, projection,projection2, target_projection, target_projection2):
        # trun = 15 # 为什么一部分project一部分不project
        if self.use_projection == True:
            y_target2 = target_projection(y_target[:,self.trun:self.end-self.trun,:]) # [:,trun:self.end-trun,:]
            con = target_projection2(y_target2)
            y2 = projection(y[:,self.trun:self.end-self.trun,:])
            y2 = projection2(y2)
        else:
            y_target2 = y_target[:,self.trun:self.end-self.trun,:]
            y2 = y[:,self.trun:self.end-self.trun,:]
        if self.cosine_similarity == True: # 考虑是否使用余弦相似，或者是否考虑直接用mse，直接不用projection
            # 思想：前端和后端分别对应高维和低维分量，保留信息直接坐cosine，中间用投影层进行非线性变换防止过拟合
            loss1 = torch.nn.functional.cosine_similarity(y_target[:, 0:self.trun, :].view(y_target.size(0),-1), y[:, 0:self.trun, :].view(y.size(0),-1),dim=-1)
            loss2 = torch.nn.functional.cosine_similarity(y_target2, y2,dim=-1)
            loss3 = torch.nn.functional.cosine_similarity(y_target[:,self.end-self.trun:self.end,:].view(y_target.size(0),-1), y[:,self.end-self.trun:self.end,:].view(y.size(0),-1),dim=-1)
            loss = (loss1 + loss2 + loss3).mean()
        else:
            loss = torch.nn.functional.mse_loss(y_target2 - y2)

        return loss

    def obtain_parameters(self,policy,prediction,projection,projection2):
        lstm_parameters_actor = policy.lstm_actor.parameters()
        lstm_parameters_critic = policy.lstm_critic.parameters()  # 注意是否选择ac共享lstm层，这里的话就是分开
        ATF1_parameters = policy.features_extractor.parameters()
        prediction_parameters = prediction.parameters()
        projection_parameters = projection.parameters()
        projection2_parameters = projection2.parameters()
        allparameters = list(lstm_parameters_actor) + list(lstm_parameters_critic) + list(ATF1_parameters) + \
                        list(prediction_parameters) + list(projection_parameters) + list(projection2_parameters)
        return allparameters

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
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while iteration < total_timesteps:  #改：self.num_timesteps——>iteration
            #if iteration % 3 == 0:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(iteration, total_timesteps)
            print(f'{self.logger.dir}')
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
                self.logger.dump(step=iteration)

            self.train()

        callback.on_training_end()

        return self

    def _update_target(self, tau): # update target prediction
        if not self.use_target_network:
            return
        if tau == 1:
            copy_model_params_from_to(self.prediction_Q, self.target_prediction_Q)
            copy_model_params_from_to(self.prediction_R, self.target_prediction_R)
            if self.use_projection:
                copy_model_params_from_to(self.projection_Q, self.target_projection_Q)
                copy_model_params_from_to(self.projection_R, self.target_projection_R)
                copy_model_params_from_to(self.projection2_Q, self.target_projection2_Q)
                copy_model_params_from_to(self.projection2_R, self.target_projection2_R)

        else:
            soft_update_from_to(self.prediction_Q, self.target_prediction_Q, tau)
            soft_update_from_to(self.prediction_R, self.target_prediction_R, tau)
            if self.use_projection:
                soft_update_from_to(self.projection_Q, self.target_projection_Q, tau)
                soft_update_from_to(self.projection_R, self.target_projection_R, tau)
                soft_update_from_to(self.projection2_Q, self.target_projection2_Q, tau)
                soft_update_from_to(self.projection2_R, self.target_projection2_R, tau)

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        action_Q, _states = self.policy_Q.predict(observation, state, episode_start, deterministic)
        action_R, _states = self.policy_R.predict(observation, state, episode_start, deterministic)
        action = np.concatenate((action_Q,action_R),axis=1)
        return action, _states

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

    def get_parameters(self) -> Dict[str, Dict]:
        """
        Return the parameters of the agent. This includes parameters from different networks, e.g.
        critics (value functions) and policies (pi functions).

        :return: Mapping of from names of the objects to PyTorch state-dicts.
        """
        state_dicts_names, _ = self._get_torch_save_params()
        params = {}
        for name in state_dicts_names:
            attr = recursive_getattr(self, name)
            # Retrieve state dict
            params[name] = attr.state_dict()
        return params

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy_Q", "policy_Q.optimizer", "policy_R", "policy_R.optimizer",]
        return state_dicts, []

    def save(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Save all the attributes of the object and the model parameters in a zip-file.

        :param path: path to the file where the rl agent should be saved
        :param exclude: name of parameters that should be excluded in addition to the default ones
        :param include: name of parameters that might be excluded but should be included anyway
        """
        # Copy parameter list so we don't mutate the original dict
        data = self.__dict__.copy()

        # Exclude is union of specified parameters (if any) and standard exclusions
        if exclude is None:
            exclude = []
        exclude = set(exclude).union(self._excluded_save_params())

        # Do not exclude params if they are specifically included
        if include is not None:
            exclude = exclude.difference(include)

        state_dicts_names, torch_variable_names = self._get_torch_save_params()
        all_pytorch_variables = state_dicts_names + torch_variable_names
        for torch_var in all_pytorch_variables:
            # We need to get only the name of the top most module as we'll remove that
            var_name = torch_var.split(".")[0]
            # Any params that are in the save vars must not be saved by data
            exclude.add(var_name)

        # Remove parameter entries of parameters which are to be excluded
        for param_name in exclude:
            data.pop(param_name, None)

        # Build dict of torch variables
        pytorch_variables = None
        if torch_variable_names is not None:
            pytorch_variables = {}
            for name in torch_variable_names:
                attr = recursive_getattr(self, name)
                pytorch_variables[name] = attr

        # Build dict of state_dicts
        params_to_save = self.get_parameters()

        save_to_zip_file(path, data=data, params=params_to_save, pytorch_variables=pytorch_variables)

def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau)